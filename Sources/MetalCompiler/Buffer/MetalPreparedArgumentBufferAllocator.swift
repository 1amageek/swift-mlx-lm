import Metal

struct MetalPreparedArgumentBufferAllocator: Sendable {
    let device: MTLDevice
    let argumentBufferIndex: Int

    init(device: MTLDevice, argumentBufferIndex: Int = 30) {
        self.device = device
        self.argumentBufferIndex = argumentBufferIndex
    }

    func makeBindingTables(
        from tables: [MetalBindingTable]
    ) throws -> [MetalBindingTable] {
        try tables.map(materializeBindingTable(from:))
    }

    private func materializeBindingTable(
        from table: MetalBindingTable
    ) throws -> MetalBindingTable {
        guard case .argumentTable(let argumentTable) = table.bufferBindings else {
            return table
        }
        guard case .planned = argumentTable.encodingState else {
            return table
        }
        guard try canMaterialize(argumentTable) else {
            return table
        }

        let encoder = try makeArgumentEncoder(for: argumentTable.layout)
        guard let argumentBuffer = device.makeBuffer(
            length: encoder.encodedLength,
            options: .storageModeShared)
        else {
            throw MetalCompilerError.deviceSetupFailed(
                "Cannot allocate prepared argument buffer for layout \(argumentTable.layout.id)")
        }
        argumentBuffer.label = "swift-lm.argtable.layout\(argumentTable.layout.id)"
        encoder.setArgumentBuffer(argumentBuffer, offset: 0)
        for binding in argumentTable.bindings {
            encoder.setBuffer(binding.buffer, offset: binding.offset, index: binding.index)
        }

        let preparedBindings = MetalArgumentTableBindings(
            layout: argumentTable.layout,
            bindings: argumentTable.bindings,
            encodingState: .prepared(
                buffer: argumentBuffer,
                index: argumentBufferIndex,
                offset: 0))
        return MetalBindingTable(
            bufferBindings: .argumentTable(preparedBindings),
            constantBindings: table.constantBindings)
    }

    private func canMaterialize(
        _ argumentTable: MetalArgumentTableBindings
    ) throws -> Bool {
        let layoutIndices = Set(argumentTable.layout.indices)
        for binding in argumentTable.bindings {
            guard layoutIndices.contains(binding.index) else {
                throw MetalCompilerError.deviceSetupFailed(
                    "Argument table layout \(argumentTable.layout.id) does not contain binding index \(binding.index)")
            }
            guard binding.offset >= 0 else {
                throw MetalCompilerError.deviceSetupFailed(
                    "Argument table layout \(argumentTable.layout.id) has negative offset \(binding.offset) for binding index \(binding.index)")
            }
            guard binding.offset < binding.buffer.length else {
                throw MetalCompilerError.deviceSetupFailed(
                    "Argument table layout \(argumentTable.layout.id) offset \(binding.offset) exceeds buffer length \(binding.buffer.length) for binding index \(binding.index)")
            }
            // MTLArgumentEncoder validates buffer offsets with abort() in debug
            // builds. Non-zero slice offsets are safely handled by the MTL4
            // direct-address path, so leave those tables unmaterialized.
            guard binding.offset == 0 else {
                return false
            }
        }
        return true
    }

    private func makeArgumentEncoder(
        for layout: MetalArgumentTableLayout
    ) throws -> MTLArgumentEncoder {
        let descriptors = layout.indices.map { index in
            let descriptor = MTLArgumentDescriptor()
            descriptor.index = index
            descriptor.dataType = .pointer
            descriptor.access = .readWrite
            return descriptor
        }
        guard let encoder = device.makeArgumentEncoder(arguments: descriptors) else {
            throw MetalCompilerError.deviceSetupFailed(
                "Cannot create argument encoder for layout \(layout.id)")
        }
        return encoder
    }
}
