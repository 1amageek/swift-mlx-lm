import Metal

struct STAFSpecializedWeightStoreBuilder {
    let device: MTLDevice
    private let accessPolicyResolver: ProjectionWeightAccessPolicyResolver

    init(
        device: MTLDevice,
        accessPolicyResolver: ProjectionWeightAccessPolicyResolver = ProjectionWeightAccessPolicyResolver()
    ) {
        self.device = device
        self.accessPolicyResolver = accessPolicyResolver
    }

    func prepare(
        store: STAFWeightStore?,
        entries: [DispatchEntry]
    ) throws -> STAFWeightStore? {
        guard var store else { return nil }

        for request in specializedRequests(in: entries, store: store) {
            guard request.preferredLayout != .rowMajor else { continue }
            if store.bufferAccess(for: request.tensorName, layout: request.preferredLayout) != nil {
                continue
            }
            let access = try makeSpecializedAccess(for: request, store: store)
            store = store.registeringSpecializedBufferAccess(access, for: request)
        }

        return store
    }

    private func specializedRequests(
        in entries: [DispatchEntry],
        store: STAFWeightStore
    ) -> [STAFWeightAccessRequest] {
        var requests: [STAFWeightAccessRequest] = []
        var seen: Set<STAFSpecializedWeightKey> = []

        for entry in entries {
            guard case .projection(let projection, _) = entry.kind,
                  let binding = entry.parameterBindings.first(where: { $0.role == projection.field }) else {
                continue
            }
            let request = accessPolicyResolver.accessRequest(
                for: entry,
                role: projection.field,
                binding: binding,
                executionPhase: .decode,
                stafWeightStore: store
            )
            let key = STAFSpecializedWeightKey(
                tensorName: binding.tensorName,
                layout: request.preferredLayout
            )
            if seen.insert(key).inserted {
                requests.append(request)
            }
        }

        return requests
    }

    func makeSpecializedAccess(
        for request: STAFWeightAccessRequest,
        store: STAFWeightStore
    ) throws -> STAFWeightBufferAccess {
        switch request.preferredLayout {
        case .rowMajor:
            guard let access = store.bufferAccess(for: request.tensorName, layout: .rowMajor) else {
                throw MetalCompilerError.deviceSetupFailed("Missing row-major access for \(request.tensorName)")
            }
            return access
        case .blockedRows4Tiles128:
            return try makeBlockedRowsTiles128Access(
                for: request.tensorName,
                store: store,
                rowsPerBlock: 4,
                layout: .blockedRows4Tiles128
            )
        case .blockedRows8Tiles128:
            return try makeBlockedRowsTiles128Access(
                for: request.tensorName,
                store: store,
                rowsPerBlock: 8,
                layout: .blockedRows8Tiles128
            )
        }
    }

    func makeBlockedRowsTiles128Access(
        for tensorName: String,
        store: STAFWeightStore,
        rowsPerBlock: Int,
        layout: STAFWeightLayout
    ) throws -> STAFWeightBufferAccess {
        guard let rowMajorAccess = store.bufferAccess(for: tensorName, layout: .rowMajor),
              let entry = store.entries[tensorName] else {
            throw MetalCompilerError.deviceSetupFailed("Missing source tensor for specialized layout: \(tensorName)")
        }
        guard entry.shape.count >= 2 else {
            throw MetalCompilerError.deviceSetupFailed("Unsupported tensor rank for specialized layout: \(tensorName)")
        }

        let outputDimension = entry.shape[0]
        let inputDimension = entry.shape[1]
        let tileElements = 128
        let elementSize = MemoryLayout<UInt16>.size
        let totalElements = outputDimension * inputDimension
        let totalBytes = totalElements * elementSize

        guard outputDimension.isMultiple(of: rowsPerBlock),
              inputDimension.isMultiple(of: tileElements) else {
            throw MetalCompilerError.deviceSetupFailed("Unsupported blocked layout dimensions for \(tensorName)")
        }
        guard rowMajorAccess.offset >= 0,
              rowMajorAccess.offset + totalBytes <= rowMajorAccess.buffer.length else {
            throw MetalCompilerError.deviceSetupFailed("Specialized layout source out of bounds for \(tensorName)")
        }
        guard let packedBuffer = device.makeBuffer(length: totalBytes, options: [.storageModeShared]) else {
            throw MetalCompilerError.deviceSetupFailed("Failed to allocate specialized layout buffer for \(tensorName)")
        }

        let source = rowMajorAccess.buffer.contents()
            .advanced(by: rowMajorAccess.offset)
            .assumingMemoryBound(to: UInt16.self)
        let destination = packedBuffer.contents().assumingMemoryBound(to: UInt16.self)

        var destinationIndex = 0
        for rowBlock in stride(from: 0, to: outputDimension, by: rowsPerBlock) {
            for base in stride(from: 0, to: inputDimension, by: tileElements) {
                for rowInBlock in 0..<rowsPerBlock {
                    let row = rowBlock + rowInBlock
                    let sourceIndex = row * inputDimension + base
                    memcpy(
                        destination.advanced(by: destinationIndex),
                        source.advanced(by: sourceIndex),
                        tileElements * elementSize
                    )
                    destinationIndex += tileElements
                }
            }
        }
        let labelSuffix: String
        switch layout {
        case .rowMajor:
            labelSuffix = "rowMajor"
        case .blockedRows4Tiles128:
            labelSuffix = "blockedRows4Tiles128"
        case .blockedRows8Tiles128:
            labelSuffix = "blockedRows8Tiles128"
        }
        packedBuffer.label = "\(tensorName)::\(labelSuffix)"
        return STAFWeightBufferAccess(
            buffer: packedBuffer,
            offset: 0,
            size: totalBytes,
            layout: layout
        )
    }

    func makeBlockedRows8Tiles128Access(
        for tensorName: String,
        store: STAFWeightStore
    ) throws -> STAFWeightBufferAccess {
        try makeBlockedRowsTiles128Access(
            for: tensorName,
            store: store,
            rowsPerBlock: 8,
            layout: .blockedRows8Tiles128
        )
    }
}
