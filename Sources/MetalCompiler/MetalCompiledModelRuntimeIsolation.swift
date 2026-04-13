import Metal

extension MetalKVCache {
    init(
        keys: MTLBuffer,
        values: MTLBuffer,
        specification: KVCacheSpecification,
        length: Int,
        rotorParameters: MTLBuffer?,
        qjlMatrix: MTLBuffer?,
        qjlResidualK: MTLBuffer?,
        numRotorGroups: Int,
        qjlDimension: Int
    ) {
        self.keys = keys
        self.values = values
        self.specification = specification
        self.length = length
        self.rotorParameters = rotorParameters
        self.qjlMatrix = qjlMatrix
        self.qjlResidualK = qjlResidualK
        self.numRotorGroups = numRotorGroups
        self.qjlDimension = qjlDimension
    }
}

public extension MetalCompiledModel {
    func makeRuntimeIsolatedCopy(device: MTLDevice) throws -> MetalCompiledModel {
        var cloner = MetalCompiledModelRuntimeCloner(device: device)
        let decodePlan = try cloner.cloneDecodePlan(decodePlan)
        let prefillPlan = try prefillPlan.map { try cloner.clonePrefillPlan($0) }
        return MetalCompiledModel(
            decodePlan: decodePlan,
            prefillPlan: prefillPlan,
            auxiliaryPipelines: auxiliaryPipelines
        )
    }
}

public extension MetalPrefillPlan {
    func makeRuntimeIsolatedCopy(device: MTLDevice) throws -> MetalPrefillPlan {
        var cloner = MetalCompiledModelRuntimeCloner(device: device)
        return try cloner.clonePrefillPlan(self)
    }
}

private struct MetalCompiledModelRuntimeCloner {
    private let device: MTLDevice
    private var runtimeBufferReplacements: [ObjectIdentifier: MTLBuffer] = [:]

    init(device: MTLDevice) {
        self.device = device
    }

    mutating func cloneDecodePlan(_ plan: MetalDispatchPlan) throws -> MetalDispatchPlan {
        let buffers = try cloneDecodeBuffers(plan.buffers)
        let steps = try plan.steps.map { try cloneDispatchStep($0) }
        return MetalDispatchPlan(
            steps: steps,
            buffers: buffers,
            unfusedEntryCount: plan.unfusedEntryCount,
            fusedEntryCount: plan.fusedEntryCount,
            quantizationPlan: plan.quantizationPlan,
            supplementalResidencyBuffers: supplementalResidencyBuffers(from: steps)
        )
    }

    mutating func clonePrefillPlan(_ plan: MetalPrefillPlan) throws -> MetalPrefillPlan {
        let buffers = try clonePrefillBuffers(plan.buffers)
        let steps = try plan.steps.map { try clonePrefillStep($0) }
        return MetalPrefillPlan(
            steps: steps,
            buffers: buffers,
            slotDimension: plan.slotDimension,
            maximumSequenceLength: plan.maximumSequenceLength,
            stepCount: plan.stepCount,
            usesMPP: plan.usesMPP,
            quantizationPlan: plan.quantizationPlan,
            finalHiddenBuffer: mappedBuffer(plan.finalHiddenBuffer),
            finalHiddenBaseOffset: plan.finalHiddenBaseOffset,
            finalHiddenRowStride: plan.finalHiddenRowStride,
            supplementalResidencyBuffers: supplementalResidencyBuffers(from: steps)
        )
    }

    private mutating func cloneDecodeBuffers(_ buffers: MetalBufferSet) throws -> MetalBufferSet {
        let kvCache: MetalKVCache? = if let kvCache = buffers.kvCache {
            try cloneKVCache(kvCache)
        } else {
            nil
        }
        let convState: MTLBuffer? = if let convState = buffers.convState {
            try clonedRuntimeBuffer(convState)
        } else {
            nil
        }
        let recurrentState: MTLBuffer? = if let recurrentState = buffers.recurrentState {
            try clonedRuntimeBuffer(recurrentState)
        } else {
            nil
        }
        let perLayerInputs: MTLBuffer? = if let perLayerInputs = buffers.perLayerInputs {
            try clonedRuntimeBuffer(perLayerInputs)
        } else {
            nil
        }
        return MetalBufferSet(
            bufferPrecision: buffers.bufferPrecision,
            hidden: try clonedRuntimeBuffer(buffers.hidden),
            residual: try clonedRuntimeBuffer(buffers.residual),
            scratch: try clonedRuntimeBuffer(buffers.scratch),
            weights: buffers.weights,
            kvCache: kvCache,
            convState: convState,
            recurrentState: recurrentState,
            convStateDimension: buffers.convStateDimension,
            convStateKernelSize: buffers.convStateKernelSize,
            recurrentStateBytesPerLayer: buffers.recurrentStateBytesPerLayer,
            perLayerInputs: perLayerInputs,
            perLayerInputDimension: buffers.perLayerInputDimension,
            perLayerInputLayerCount: buffers.perLayerInputLayerCount,
            logits: try clonedRuntimeBuffer(buffers.logits),
            position: try clonedRuntimeBuffer(buffers.position),
            ropePositionAxes: try clonedRuntimeBuffer(buffers.ropePositionAxes),
            tokenIn: try clonedRuntimeBuffer(buffers.tokenIn),
            tokenOut: try clonedRuntimeBuffer(buffers.tokenOut)
        )
    }

    private mutating func clonePrefillBuffers(_ buffers: PrefillBufferSet) throws -> PrefillBufferSet {
        let kvCache: MetalKVCache? = if let kvCache = buffers.kvCache {
            try cloneKVCache(kvCache)
        } else {
            nil
        }
        let convState: MTLBuffer? = if let convState = buffers.convState {
            try clonedRuntimeBuffer(convState)
        } else {
            nil
        }
        let recurrentState: MTLBuffer? = if let recurrentState = buffers.recurrentState {
            try clonedRuntimeBuffer(recurrentState)
        } else {
            nil
        }
        let perLayerInputs: MTLBuffer? = if let perLayerInputs = buffers.perLayerInputs {
            try clonedRuntimeBuffer(perLayerInputs)
        } else {
            nil
        }
        let dequantScratch: MTLBuffer? = if let dequantScratch = buffers.dequantScratch {
            try clonedRuntimeBuffer(dequantScratch)
        } else {
            nil
        }
        return PrefillBufferSet(
            bufferPrecision: buffers.bufferPrecision,
            hidden: try clonedRuntimeBuffer(buffers.hidden),
            residual: try clonedRuntimeBuffer(buffers.residual),
            scratch: try clonedRuntimeBuffer(buffers.scratch),
            weights: buffers.weights,
            kvCache: kvCache,
            convState: convState,
            recurrentState: recurrentState,
            convStateDimension: buffers.convStateDimension,
            convStateKernelSize: buffers.convStateKernelSize,
            recurrentStateBytesPerLayer: buffers.recurrentStateBytesPerLayer,
            perLayerInputs: perLayerInputs,
            perLayerInputDimension: buffers.perLayerInputDimension,
            perLayerInputLayerCount: buffers.perLayerInputLayerCount,
            logits: try clonedRuntimeBuffer(buffers.logits),
            tokenIDs: try clonedRuntimeBuffer(buffers.tokenIDs),
            positions: try clonedRuntimeBuffer(buffers.positions),
            ropePositionAxes: try clonedRuntimeBuffer(buffers.ropePositionAxes),
            tokenOut: try clonedRuntimeBuffer(buffers.tokenOut),
            dequantScratch: dequantScratch,
            runtimeConstantBuffer: try clonedRuntimeBuffer(buffers.runtimeConstantBuffer)
        )
    }

    private mutating func cloneKVCache(_ kvCache: MetalKVCache) throws -> MetalKVCache {
        let qjlResidualK: MTLBuffer? = if let qjlResidualK = kvCache.qjlResidualK {
            try clonedRuntimeBuffer(qjlResidualK)
        } else {
            nil
        }
        return MetalKVCache(
            keys: try clonedRuntimeBuffer(kvCache.keys),
            values: try clonedRuntimeBuffer(kvCache.values),
            specification: kvCache.specification,
            length: 0,
            rotorParameters: kvCache.rotorParameters,
            qjlMatrix: kvCache.qjlMatrix,
            qjlResidualK: qjlResidualK,
            numRotorGroups: kvCache.numRotorGroups,
            qjlDimension: kvCache.qjlDimension
        )
    }

    private mutating func cloneDispatchStep(_ step: MetalDispatchStep) throws -> MetalDispatchStep {
        MetalDispatchStep(
            descriptor: clonedDescriptor(step.descriptor),
            bindings: try clonedBindingTable(step.bindings),
            bufferAccesses: clonedBufferAccesses(step.bufferAccesses),
            metadata: step.metadata
        )
    }

    private mutating func clonePrefillStep(_ step: MetalPrefillStep) throws -> MetalPrefillStep {
        MetalPrefillStep(
            descriptor: clonedDescriptor(step.descriptor),
            bindings: try clonedBindingTable(step.bindings),
            mode: step.mode,
            sequenceLengthPolicy: step.sequenceLengthPolicy,
            positionBufferIndex: step.positionBufferIndex,
            perPositionStrides: step.perPositionStrides,
            metadata: step.metadata
        )
    }

    private mutating func clonedBindingTable(_ table: MetalBindingTable) throws -> MetalBindingTable {
        MetalBindingTable(
            bufferBindings: try clonedBufferBindingSet(table.bufferBindings),
            constantBindings: clonedConstantBindingSet(table.constantBindings)
        )
    }

    private mutating func clonedBufferBindingSet(_ bindings: MetalBufferBindingSet) throws -> MetalBufferBindingSet {
        switch bindings {
        case .inline(let inlineBindings):
            return .inline(inlineBindings.map(clonedBufferBinding))
        case .argumentTable(let table):
            let clonedBindings = table.bindings.map(clonedBufferBinding)
            let clonedState = try clonedArgumentTableEncodingState(
                table.encodingState,
                layout: table.layout,
                bindings: clonedBindings
            )
            return .argumentTable(MetalArgumentTableBindings(
                layout: table.layout,
                bindings: clonedBindings,
                encodingState: clonedState
            ))
        }
    }

    private func clonedConstantBindingSet(_ bindings: MetalConstantBindingSet) -> MetalConstantBindingSet {
        switch bindings {
        case .inline(let inlineBindings):
            return .inline(inlineBindings)
        case .resident(let resident):
            return .resident(MetalResidentConstantBindings(
                buffer: mappedBuffer(resident.buffer),
                bindings: resident.bindings.map(clonedConstantBufferBinding)
            ))
        case .mixed(let mixedBindings):
            return .mixed(mixedBindings.map { binding in
                switch binding {
                case .inline(let bytes):
                    return .inline(bytes)
                case .buffer(let bufferBinding):
                    return .buffer(clonedConstantBufferBinding(bufferBinding))
                }
            })
        }
    }

    private func clonedConstantBufferBinding(_ binding: MetalConstantBufferBinding) -> MetalConstantBufferBinding {
        MetalConstantBufferBinding(
            index: binding.index,
            buffer: mappedBuffer(binding.buffer),
            offset: binding.offset,
            length: binding.length
        )
    }

    private func clonedBufferBinding(_ binding: MetalBufferBinding) -> MetalBufferBinding {
        MetalBufferBinding(
            index: binding.index,
            buffer: mappedBuffer(binding.buffer),
            offset: binding.offset
        )
    }

    private mutating func clonedArgumentTableEncodingState(
        _ state: MetalArgumentTableEncodingState,
        layout: MetalArgumentTableLayout,
        bindings: [MetalBufferBinding]
    ) throws -> MetalArgumentTableEncodingState {
        switch state {
        case .planned:
            return .planned
        case .prepared(_, let index, let offset):
            return .prepared(
                buffer: try makeArgumentBuffer(layout: layout, bindings: bindings, labelPrefix: "swift-lm.argtable"),
                index: index,
                offset: offset
            )
        case .encoded(_, let index, let offset):
            return .encoded(
                buffer: try makeArgumentBuffer(layout: layout, bindings: bindings, labelPrefix: "swift-lm.argtable.encoded"),
                index: index,
                offset: offset
            )
        }
    }

    private func clonedDescriptor(_ descriptor: MetalDispatchDescriptor) -> MetalDispatchDescriptor {
        MetalDispatchDescriptor(
            pipeline: descriptor.pipeline,
            gridSize: descriptor.gridSize,
            threadgroupSize: descriptor.threadgroupSize,
            threadgroupMemoryLength: descriptor.threadgroupMemoryLength,
            barrierPolicy: clonedBarrierPolicy(descriptor.barrierPolicy)
        )
    }

    private func clonedBarrierPolicy(_ policy: MetalBarrierPolicy) -> MetalBarrierPolicy {
        // Visibility options are value types — no resource identity to remap.
        policy
    }

    private func clonedBufferAccesses(_ accesses: MetalBufferAccesses) -> MetalBufferAccesses {
        MetalBufferAccesses(
            reads: Set(accesses.reads.map(clonedBufferRegion)),
            writes: Set(accesses.writes.map(clonedBufferRegion))
        )
    }

    private func clonedBufferRegion(_ region: BufferRegion) -> BufferRegion {
        BufferRegion(buffer: mappedBuffer(region.rawBuffer), offset: region.offset)
    }

    private mutating func clonedRuntimeBuffer(_ buffer: MTLBuffer) throws -> MTLBuffer {
        let identity = ObjectIdentifier(buffer)
        if let existing = runtimeBufferReplacements[identity] {
            return existing
        }

        let bufferName = buffer.label ?? "unnamed"
        guard let cloned = device.makeBuffer(length: buffer.length, options: resourceOptions(for: buffer)) else {
            throw MetalCompilerError.bufferAllocationFailed(
                "Cannot allocate runtime-isolated buffer for \(bufferName)"
            )
        }
        cloned.label = buffer.label
        runtimeBufferReplacements[identity] = cloned
        return cloned
    }

    private func mappedBuffer(_ buffer: MTLBuffer) -> MTLBuffer {
        runtimeBufferReplacements[ObjectIdentifier(buffer)] ?? buffer
    }

    private func makeArgumentBuffer(
        layout: MetalArgumentTableLayout,
        bindings: [MetalBufferBinding],
        labelPrefix: String
    ) throws -> MTLBuffer {
        let encoder = try makeArgumentEncoder(layout: layout)
        guard let argumentBuffer = device.makeBuffer(
            length: encoder.encodedLength,
            options: .storageModeShared
        ) else {
            throw MetalCompilerError.deviceSetupFailed(
                "Cannot allocate argument buffer for layout \(layout.id)"
            )
        }
        argumentBuffer.label = "\(labelPrefix).layout\(layout.id)"
        encoder.setArgumentBuffer(argumentBuffer, offset: 0)
        for binding in bindings {
            encoder.setBuffer(binding.buffer, offset: binding.offset, index: binding.index)
        }
        return argumentBuffer
    }

    private func makeArgumentEncoder(layout: MetalArgumentTableLayout) throws -> MTLArgumentEncoder {
        let descriptors = layout.indices.map { index in
            let descriptor = MTLArgumentDescriptor()
            descriptor.index = index
            descriptor.dataType = .pointer
            descriptor.access = .readWrite
            return descriptor
        }
        guard let encoder = device.makeArgumentEncoder(arguments: descriptors) else {
            throw MetalCompilerError.deviceSetupFailed(
                "Cannot create argument encoder for layout \(layout.id)"
            )
        }
        return encoder
    }

    private func resourceOptions(for buffer: MTLBuffer) -> MTLResourceOptions {
        var options: MTLResourceOptions = []
        switch buffer.storageMode {
        case .shared:
            options.insert(.storageModeShared)
        case .private:
            options.insert(.storageModePrivate)
            if buffer.hazardTrackingMode == .untracked {
                options.insert(.hazardTrackingModeUntracked)
            }
        case .memoryless:
            options.insert(.storageModeMemoryless)
        case .managed:
            options.insert(.storageModeManaged)
        @unknown default:
            options.insert(.storageModeShared)
        }

        if buffer.cpuCacheMode == .writeCombined {
            options.insert(.cpuCacheModeWriteCombined)
        }
        return options
    }

    private func supplementalResidencyBuffers(from steps: [MetalDispatchStep]) -> [MTLBuffer] {
        var seen = Set<ObjectIdentifier>()
        var buffers: [MTLBuffer] = []
        for step in steps {
            for buffer in step.bindings.ownedResidencyBuffers {
                let identity = ObjectIdentifier(buffer)
                if seen.insert(identity).inserted {
                    buffers.append(buffer)
                }
            }
        }
        return buffers
    }

    private func supplementalResidencyBuffers(from steps: [MetalPrefillStep]) -> [MTLBuffer] {
        var seen = Set<ObjectIdentifier>()
        var buffers: [MTLBuffer] = []
        for step in steps {
            for buffer in step.bindings.ownedResidencyBuffers {
                let identity = ObjectIdentifier(buffer)
                if seen.insert(identity).inserted {
                    buffers.append(buffer)
                }
            }
        }
        return buffers
    }
}
