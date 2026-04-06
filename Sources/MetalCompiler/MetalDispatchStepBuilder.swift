import Metal

struct MetalDispatchStepBuilder {
    func buildDecodePlan(
        fusedEntries: [DispatchEntry],
        unfusedCount: Int,
        bufferSet: MetalBufferSet,
        slotDimension: Int,
        stafWeightStore: STAFWeightStore?,
        hiddenSize: Int,
        accessPolicyResolver: ProjectionWeightAccessPolicyResolver,
        planBuildContext: PlanBuildContext,
        argumentEncoders: [String: MTLArgumentEncoder],
        resolveDispatch: (DispatchEntry) throws -> (
            name: String,
            pipeline: MTLComputePipelineState,
            config: (grid: MTLSize, threadgroup: MTLSize, sharedMemoryBytes: Int)
        )
    ) throws -> MetalDispatchPlan {
        let constantAllocator = MetalConstantBindingAllocator(device: planBuildContext.device)
        let argumentAllocator = MetalArgumentBindingAllocator()
        let preparedArgumentAllocator = MetalPreparedArgumentBufferAllocator(device: planBuildContext.device)

        var steps: [MetalDispatchStep] = []
        var routingPlanner = DecodeRoutingPlanner(
            bufferSet: bufferSet,
            stafWeightStore: stafWeightStore,
            hiddenSize: hiddenSize,
            slotDimension: slotDimension,
            accessPolicyResolver: accessPolicyResolver
        )

        for entry in fusedEntries {
            let resolved = try resolveDispatch(entry)
            let bindings = routingPlanner.bindings(for: entry)
            steps.append(MetalDispatchStep(
                pipeline: resolved.pipeline,
                gridSize: resolved.config.grid,
                threadgroupSize: resolved.config.threadgroup,
                bufferBindings: bindings.buffers,
                bytesBindings: bindings.bytes,
                threadgroupMemoryLength: resolved.config.sharedMemoryBytes,
                sync: .bufferBarrier,
                bufferAccesses: Self.decodeBufferAccesses(
                    for: entry,
                    buffers: bindings.buffers),
                metadata: MetalDispatchStepMetadata(
                    kernelName: resolved.name,
                    layerIndex: entry.layerIndex
                )
            ))
        }

        let residentSteps = try makeResidentConstantSteps(steps, allocator: constantAllocator)
        let argumentTableSteps = makeArgumentTableSteps(residentSteps, allocator: argumentAllocator)
        let preparedArgumentSteps = try makePreparedArgumentTableSteps(
            argumentTableSteps,
            allocator: preparedArgumentAllocator
        )
        let encodedArgumentSteps = try makeEncodedArgumentTableSteps(
            preparedArgumentSteps,
            pipelineCache: planBuildContext.pipelineCache,
            argumentEncoders: argumentEncoders
        )
        let optimizedBarrierSteps = Self.optimizeDecodeBarrierPolicies(encodedArgumentSteps)

        return MetalDispatchPlan(
            steps: optimizedBarrierSteps,
            buffers: bufferSet,
            unfusedEntryCount: unfusedCount,
            fusedEntryCount: fusedEntries.count
        )
    }

    private func makeResidentConstantSteps(
        _ steps: [MetalDispatchStep],
        allocator: MetalConstantBindingAllocator
    ) throws -> [MetalDispatchStep] {
        let bindingTables = steps.map(\.bindings)
        let residentBindings = try allocator.makeBindingTables(from: bindingTables)
        return zip(steps, residentBindings).map { step, bindings in
            MetalDispatchStep(
                descriptor: step.descriptor,
                bindings: bindings,
                bufferAccesses: step.bufferAccesses,
                metadata: step.metadata
            )
        }
    }

    private func makeArgumentTableSteps(
        _ steps: [MetalDispatchStep],
        allocator: MetalArgumentBindingAllocator
    ) -> [MetalDispatchStep] {
        let bindingTables = steps.map(\.bindings)
        let plannedBindings = allocator.makeBindingTables(from: bindingTables)
        return zip(steps, plannedBindings).map { step, bindings in
            MetalDispatchStep(
                descriptor: step.descriptor,
                bindings: bindings,
                bufferAccesses: step.bufferAccesses,
                metadata: step.metadata
            )
        }
    }

    private func makePreparedArgumentTableSteps(
        _ steps: [MetalDispatchStep],
        allocator: MetalPreparedArgumentBufferAllocator
    ) throws -> [MetalDispatchStep] {
        let bindingTables = steps.map(\.bindings)
        let preparedBindings = try allocator.makeBindingTables(from: bindingTables)
        return zip(steps, preparedBindings).map { step, bindings in
            MetalDispatchStep(
                descriptor: step.descriptor,
                bindings: bindings,
                bufferAccesses: step.bufferAccesses,
                metadata: step.metadata
            )
        }
    }

    private func makeEncodedArgumentTableSteps(
        _ steps: [MetalDispatchStep],
        pipelineCache: [String: MTLComputePipelineState],
        argumentEncoders: [String: MTLArgumentEncoder]
    ) throws -> [MetalDispatchStep] {
        try steps.map { step in
            guard
                let kernelLabel = step.pipeline.label,
                let variantKernelName = Self.encodedArgumentTableKernelName(
                    for: kernelLabel,
                    bindings: step.bindings
                ),
                let variantPipeline = pipelineCache[variantKernelName],
                let argumentEncoder = argumentEncoders[variantKernelName],
                case .argumentTable(let table) = step.bindings.bufferBindings,
                case .prepared(_, let index, let offset) = table.encodingState
            else {
                return step
            }

            guard let encodedArgumentBuffer = variantPipeline.device.makeBuffer(
                length: argumentEncoder.encodedLength,
                options: .storageModeShared
            ) else {
                throw MetalCompilerError.deviceSetupFailed(
                    "Cannot allocate encoded argument buffer for \(variantKernelName)"
                )
            }
            encodedArgumentBuffer.label =
                "swift-lm.argtable.encoded.\(variantKernelName).layout\(table.layout.id)"
            argumentEncoder.setArgumentBuffer(encodedArgumentBuffer, offset: 0)
            for binding in table.bindings {
                argumentEncoder.setBuffer(binding.buffer, offset: binding.offset, index: binding.index)
            }
            encodedArgumentBuffer.didModifyRange(0..<argumentEncoder.encodedLength)

            let encodedBindings = MetalBindingTable(
                bufferBindings: .argumentTable(MetalArgumentTableBindings(
                    layout: table.layout,
                    bindings: table.bindings,
                    encodingState: .encoded(
                        buffer: encodedArgumentBuffer,
                        index: index,
                        offset: offset
                    )
                )),
                constantBindings: Self.constantBindingsForEncodedVariant(
                    step.bindings.constantBindings,
                    variantKernelName: variantKernelName
                )
            )
            let encodedDescriptor = MetalDispatchDescriptor(
                pipeline: variantPipeline,
                gridSize: step.gridSize,
                threadgroupSize: step.threadgroupSize,
                threadgroupMemoryLength: step.threadgroupMemoryLength,
                barrierPolicy: step.barrierPolicy
            )
            return MetalDispatchStep(
                descriptor: encodedDescriptor,
                bindings: encodedBindings,
                bufferAccesses: step.bufferAccesses,
                metadata: MetalDispatchStepMetadata(
                    kernelName: variantKernelName,
                    layerIndex: step.metadata.layerIndex
                )
            )
        }
    }

    private static func constantBindingsForEncodedVariant(
        _ bindings: MetalConstantBindingSet,
        variantKernelName: String
    ) -> MetalConstantBindingSet {
        if (
            variantKernelName.hasPrefix("gemv_2048_sq") ||
            variantKernelName.hasPrefix("gemv_2048_6144")
        ) && variantKernelName.hasSuffix("_argbuf") {
            return .inline([])
        }
        return bindings
    }

    private static func decodeBufferAccesses(
        for entry: DispatchEntry,
        buffers: [(index: Int, buffer: MTLBuffer, offset: Int)]
    ) -> MetalBufferAccesses {
        let mapped = buffers.map { MetalBufferBinding(index: $0.index, buffer: $0.buffer, offset: $0.offset) }

        func binding(_ index: Int) -> MTLBuffer? {
            mapped.first(where: { $0.index == index })?.buffer
        }

        func bindings(in indices: some Sequence<Int>) -> [MTLBuffer] {
            indices.compactMap(binding(_:))
        }

        switch entry.kind {
        case .projection:
            return MetalBufferAccesses(
                readBuffers: bindings(in: [0, 1]),
                writeBuffers: bindings(in: [2])
            )
        case .fusedSwiGLUProjection:
            return MetalBufferAccesses(
                readBuffers: bindings(in: [0, 1, 2]),
                writeBuffers: bindings(in: [3])
            )
        case .fusedCopyNorm, .fusedResidualAddCopyNorm, .fusedResidualAddNorm:
            return MetalBufferAccesses(
                readBuffers: bindings(in: [0, 1, 2]),
                writeBuffers: bindings(in: [3])
            )
        case .structuralCopy:
            return MetalBufferAccesses(
                readBuffers: bindings(in: [0]),
                writeBuffers: bindings(in: [1])
            )
        case .structuralAdd:
            return MetalBufferAccesses(
                readBuffers: bindings(in: [0, 1]),
                writeBuffers: bindings(in: [2])
            )
        case .batchedProjection(let batched):
            let count = batched.projections.count
            return MetalBufferAccesses(
                readBuffers: bindings(in: 0..<(1 + count)),
                writeBuffers: bindings(in: (1 + count)..<(1 + 2 * count))
            )
        case .batchedFragment, .fragment:
            return MetalBufferAccesses.conservative(mapped)
        }
    }

    private static func optimizeDecodeBarrierPolicies(
        _ steps: [MetalDispatchStep]
    ) -> [MetalDispatchStep] {
        var pendingWrites = Set<ObjectIdentifier>()
        return steps.map { step in
            let requiresBarrier = step.bufferAccesses.requiresBarrier(after: pendingWrites)
            let barrierPolicy: MetalBarrierPolicy = requiresBarrier ? .bufferBarrier : .none
            let descriptor = MetalDispatchDescriptor(
                pipeline: step.pipeline,
                gridSize: step.gridSize,
                threadgroupSize: step.threadgroupSize,
                threadgroupMemoryLength: step.threadgroupMemoryLength,
                barrierPolicy: barrierPolicy
            )

            if requiresBarrier {
                pendingWrites = step.bufferAccesses.writes
            } else {
                pendingWrites.formUnion(step.bufferAccesses.writes)
            }

            return MetalDispatchStep(
                descriptor: descriptor,
                bindings: step.bindings,
                bufferAccesses: step.bufferAccesses,
                metadata: step.metadata
            )
        }
    }

    private static func encodedArgumentTableKernelName(
        for kernelName: String,
        bindings: MetalBindingTable
    ) -> String? {
        guard case .argumentTable(let table) = bindings.bufferBindings else {
            return nil
        }
        switch table.layout.indices {
        case [0, 1]:
            switch kernelName {
            case "argmax":
                return MetalKernelNameResolver.argumentTableVariantKernelName(for: kernelName)
            case "rms_norm", "rms_norm_bf16":
                return MetalKernelNameResolver.argumentTableVariantKernelName(for: kernelName)
            case "qk_rms_norm", "qk_rms_norm_bf16":
                return MetalKernelNameResolver.argumentTableVariantKernelName(for: kernelName)
            default:
                return nil
            }
        case [0, 1, 2]:
            switch kernelName {
            case "embedding_lookup", "embedding_lookup_bf16":
                return MetalKernelNameResolver.argumentTableVariantKernelName(for: kernelName)
            default:
                if kernelName.hasPrefix("gemv_2048_sq")
                    || kernelName.hasPrefix("gemv_2048_6144")
                    || kernelName == "gemv_8192_tiled"
                    || kernelName == "gemv_8192_tiled_bf16"
                    || kernelName == "gemv"
                    || kernelName == "gemv_bf16"
                    || kernelName == "gemv_vocab"
                    || kernelName == "gemv_vocab_bf16"
                {
                    return MetalKernelNameResolver.argumentTableVariantKernelName(for: kernelName)
                }
                switch kernelName {
                case "residual_add":
                    return MetalKernelNameResolver.argumentTableVariantKernelName(for: kernelName)
                case "rope":
                    return MetalKernelNameResolver.argumentTableVariantKernelName(for: kernelName)
                default:
                    return nil
                }
            }
        case [0, 1, 2, 3]:
            switch kernelName {
            case let name where name.hasPrefix("fused_copy_rms_norm"):
                return MetalKernelNameResolver.argumentTableVariantKernelName(for: kernelName)
            case let name where name.hasPrefix("fused_residual_add_copy_rms_norm"):
                return MetalKernelNameResolver.argumentTableVariantKernelName(for: kernelName)
            case let name where name.hasPrefix("fused_residual_add_rms_norm"):
                return MetalKernelNameResolver.argumentTableVariantKernelName(for: kernelName)
            case "fused_swiglu_projection_2048", "fused_swiglu_projection_2048_bf16":
                return MetalKernelNameResolver.argumentTableVariantKernelName(for: kernelName)
            case "conv_state_update", "conv_state_update_bf16":
                return MetalKernelNameResolver.argumentTableVariantKernelName(for: kernelName)
            case "batched_qk_rms_norm_2", "batched_qk_rms_norm_bf16_2":
                return MetalKernelNameResolver.argumentTableVariantKernelName(for: kernelName)
            default:
                return nil
            }
        case [0, 1, 2, 3, 4]:
            switch kernelName {
            case "batched_gemv2", "batched_gemv2_bf16":
                return MetalKernelNameResolver.argumentTableVariantKernelName(for: kernelName)
            default:
                return nil
            }
        default:
            if table.layout.indices == [0, 1, 2, 3, 4, 5, 6] {
                switch kernelName {
                case "flash_attn_decode":
                    return MetalKernelNameResolver.argumentTableVariantKernelName(for: kernelName)
                case "batched_gemv3", "batched_gemv3_bf16":
                    return MetalKernelNameResolver.argumentTableVariantKernelName(for: kernelName)
                default:
                    return nil
                }
            }
            if table.layout.indices == [0, 1, 2, 3, 4, 5, 6, 7, 8] {
                switch kernelName {
                case "batched_gemv4", "batched_gemv4_bf16":
                    return MetalKernelNameResolver.argumentTableVariantKernelName(for: kernelName)
                default:
                    return nil
                }
            }
            return nil
        }
    }
}

struct DecodeRoutingPlanner {
    let bufferSet: MetalBufferSet
    let stafWeightStore: STAFWeightStore?
    let hiddenSize: Int
    let slotDimension: Int
    let accessPolicyResolver: ProjectionWeightAccessPolicyResolver
    private let elementSize = MemoryLayout<Float16>.size
    private var kvCacheIndex: Int = 0
    private var routingState = BufferRoutingState()

    init(
        bufferSet: MetalBufferSet,
        stafWeightStore: STAFWeightStore?,
        hiddenSize: Int,
        slotDimension: Int,
        accessPolicyResolver: ProjectionWeightAccessPolicyResolver
    ) {
        self.bufferSet = bufferSet
        self.stafWeightStore = stafWeightStore
        self.hiddenSize = hiddenSize
        self.slotDimension = slotDimension
        self.accessPolicyResolver = accessPolicyResolver
    }

    mutating func bindings(
        for entry: DispatchEntry
    ) -> (
        buffers: [(index: Int, buffer: MTLBuffer, offset: Int)],
        bytes: [(index: Int, value: [UInt8])]
    ) {
        let weightResolver = WeightResolver(
            entry: entry,
            stafWeightStore: stafWeightStore,
            fallbackBuffer: bufferSet.hidden,
            logsMisses: true,
            executionPhase: .decode,
            accessPolicyResolver: accessPolicyResolver
        )

        func fusedNormBindings(dimension: Int, epsilon: Float) -> (
            buffers: [(index: Int, buffer: MTLBuffer, offset: Int)],
            bytes: [(index: Int, value: [UInt8])]
        ) {
            let (weightBuffer, weightOffset) = weightResolver.resolve(role: "scale")
            routingState.lastOutputIsHidden = false
            routingState.currentInputOffset = 0
            routingState.projectionIndex = 0
            return (
                buffers: [
                    (0, bufferSet.hidden, 0),
                    (1, bufferSet.residual, 0),
                    (2, weightBuffer, weightOffset),
                    (3, bufferSet.scratch, 0),
                ],
                bytes: [
                    uint32Binding(4, UInt32(dimension)),
                    floatBinding(5, epsilon),
                ]
            )
        }

        switch entry.kind {
        case .fusedCopyNorm(let fusedOperation):
            return fusedNormBindings(
                dimension: fusedOperation.dimension,
                epsilon: fusedOperation.epsilon
            )

        case .fusedResidualAddCopyNorm(let fusedOperation):
            return fusedNormBindings(
                dimension: fusedOperation.dimension,
                epsilon: fusedOperation.epsilon
            )

        case .projection(let projection, let isOutput):
            let (weightBuffer, weightOffset) = weightResolver.resolve(role: projection.field)

            let inputBuffer: MTLBuffer
            let inputOffset: Int
            if routingState.lastOutputIsHidden {
                inputBuffer = bufferSet.hidden
                inputOffset = 0
            } else {
                inputBuffer = bufferSet.scratch
                inputOffset = routingState.currentInputOffset
            }

            let outputBuffer: MTLBuffer
            let outputOffset: Int

            if isOutput && projection.outputDimension > hiddenSize {
                outputBuffer = bufferSet.logits
                outputOffset = 0
                routingState.lastOutputIsHidden = false
            } else if isOutput {
                outputBuffer = bufferSet.hidden
                outputOffset = 0
                routingState.lastOutputIsHidden = true
            } else {
                let scratchSlot = routingState.projectionIndex + 1
                outputBuffer = bufferSet.scratch
                outputOffset = scratchSlot * slotDimension * elementSize
                routingState.lastOutputIsHidden = false
            }

            routingState.projectionIndex += 1

            return (
                buffers: [
                    (0, inputBuffer, inputOffset),
                    (1, weightBuffer, weightOffset),
                    (2, outputBuffer, outputOffset),
                ],
                bytes: [
                    uint32Binding(3, UInt32(projection.inputDimension)),
                    uint32Binding(4, UInt32(projection.outputDimension)),
                ]
            )

        case .fusedSwiGLUProjection(let fusedOperation):
            let (gateWeightBuffer, gateWeightOffset) = weightResolver.resolve(role: fusedOperation.gateField)
            let (upWeightBuffer, upWeightOffset) = weightResolver.resolve(role: fusedOperation.upField)
            let family = FusedSwiGLUProjectionFamily.resolve(
                inputDimension: fusedOperation.inputDimension,
                outputDimension: fusedOperation.outputDimension
            )

            let inputBuffer: MTLBuffer
            let inputOffset: Int
            if routingState.lastOutputIsHidden {
                inputBuffer = bufferSet.hidden
                inputOffset = 0
            } else {
                inputBuffer = bufferSet.scratch
                inputOffset = routingState.currentInputOffset
            }

            routingState.lastOutputIsHidden = false
            routingState.currentInputOffset = slotDimension * elementSize
            routingState.projectionIndex = 0
            let bytes: [(index: Int, value: [UInt8])]
            switch family {
            case .generic:
                bytes = [
                    uint32Binding(4, UInt32(fusedOperation.inputDimension)),
                    uint32Binding(5, UInt32(fusedOperation.outputDimension)),
                ]
            case .input2048Dense:
                bytes = [
                    uint32Binding(4, UInt32(fusedOperation.outputDimension)),
                ]
            }

            return (
                buffers: [
                    (0, inputBuffer, inputOffset),
                    (1, gateWeightBuffer, gateWeightOffset),
                    (2, upWeightBuffer, upWeightOffset),
                    (3, bufferSet.scratch, slotDimension * elementSize),
                ],
                bytes: bytes
            )

        case .structuralCopy(let dimension):
            routingState.projectionIndex = 0
            return (
                buffers: [
                    (0, bufferSet.hidden, 0),
                    (1, bufferSet.residual, 0),
                ],
                bytes: [
                    uint32Binding(2, UInt32(dimension)),
                ]
            )

        case .structuralAdd(let dimension):
            routingState.lastOutputIsHidden = true
            return (
                buffers: [
                    (0, bufferSet.hidden, 0),
                    (1, bufferSet.residual, 0),
                    (2, bufferSet.hidden, 0),
                ],
                bytes: [
                    uint32Binding(3, UInt32(dimension)),
                ]
            )

        case .fragment(let fragment):
            let bindingContext = BufferBindingContext(
                bufferSet: bufferSet,
                slotDimension: slotDimension,
                elementSize: elementSize,
                layerIndex: entry.layerIndex,
                kvCacheIndex: kvCacheIndex,
                convLayerIndex: routingState.convLayerIndex,
                recurrentLayerIndex: routingState.recurrentLayerIndex,
                resolveWeight: weightResolver.resolve
            )
            let bindings = fragment.decodeBindings(context: bindingContext)
            if bindings.resetsProjectionIndex {
                routingState.projectionIndex = 0
                if !bindings.outputIsHidden {
                    routingState.currentInputOffset = 0
                }
            }
            if bindings.consumesKVCacheLayer { kvCacheIndex += 1 }
            if bindings.consumesConvLayer { routingState.convLayerIndex += 1 }
            if bindings.consumesRecurrentLayer { routingState.recurrentLayerIndex += 1 }
            routingState.lastOutputIsHidden = bindings.outputIsHidden
            return (buffers: bindings.buffers, bytes: bindings.bytes)

        case .fusedResidualAddNorm(let fusedOperation):
            let (weightBuffer, weightOffset) = weightResolver.resolve(role: "scale")
            routingState.lastOutputIsHidden = false
            routingState.currentInputOffset = 0
            routingState.projectionIndex = 0
            return (
                buffers: [
                    (0, bufferSet.hidden, 0),
                    (1, bufferSet.residual, 0),
                    (2, weightBuffer, weightOffset),
                    (3, bufferSet.scratch, 0),
                ],
                bytes: [
                    uint32Binding(4, UInt32(fusedOperation.dimension)),
                    floatBinding(5, fusedOperation.epsilon),
                ]
            )

        case .batchedProjection(let batched):
            let inputBuffer: MTLBuffer
            let inputOffset: Int
            if routingState.lastOutputIsHidden {
                inputBuffer = bufferSet.hidden
                inputOffset = 0
            } else {
                inputBuffer = bufferSet.scratch
                inputOffset = 0
            }

            let count = batched.projections.count
            var bufferBindings: [(index: Int, buffer: MTLBuffer, offset: Int)] = [
                (0, inputBuffer, inputOffset),
            ]
            var bytesBindings: [(index: Int, value: [UInt8])] = []

            for (i, proj) in batched.projections.enumerated() {
                let (weightBuf, weightOff) = weightResolver.resolve(role: proj.field)
                bufferBindings.append((1 + i, weightBuf, weightOff))
            }

            for i in 0..<count {
                let scratchSlot = routingState.projectionIndex + 1
                let outputOffset = scratchSlot * slotDimension * elementSize
                bufferBindings.append((1 + count + i, bufferSet.scratch, outputOffset))
                routingState.projectionIndex += 1
            }

            let bytesStart = 1 + 2 * count
            bytesBindings.append(uint32Binding(bytesStart, UInt32(batched.inputDimension)))
            for (i, proj) in batched.projections.enumerated() {
                bytesBindings.append(uint32Binding(bytesStart + 1 + i, UInt32(proj.outputDimension)))
            }

            routingState.lastOutputIsHidden = false
            return (buffers: bufferBindings, bytes: bytesBindings)

        case .batchedFragment(let batch):
            let slotBytes = slotDimension * elementSize
            var bufferBindings: [(index: Int, buffer: MTLBuffer, offset: Int)] = []
            var bytesBindings: [(index: Int, value: [UInt8])] = []

            for i in 0..<batch.fragments.count {
                let scratchSlotIndex = 1 + i
                bufferBindings.append((i, bufferSet.scratch, scratchSlotIndex * slotBytes))
            }

            for (i, frag) in batch.fragments.enumerated() {
                if let weightSlot = frag.weightSlots.first {
                    let role = weightSlot.field ?? "weight"
                    let (weightBuffer, weightOffset) = weightResolver.resolve(role: role)
                    bufferBindings.append((batch.fragments.count + i, weightBuffer, weightOffset))
                }
            }

            let bytesStart = 2 * batch.fragments.count
            for (i, frag) in batch.fragments.enumerated() {
                if case .perHead(let headCount) = frag.dispatchDimension {
                    bytesBindings.append(uint32Binding(bytesStart + i, UInt32(headCount)))
                }
            }

            if case .perHead = batch.dispatchDimension,
               let firstFrag = batch.fragments.first,
               case .perHead(let firstHeadCount) = firstFrag.dispatchDimension
            {
                let headDimension = hiddenSize / firstHeadCount
                bytesBindings.append(uint32Binding(bytesStart + batch.fragments.count, UInt32(headDimension)))
                let epsilon = batch.fragments.first?.normEpsilon ?? 1e-6
                bytesBindings.append(floatBinding(bytesStart + batch.fragments.count + 1, epsilon))
            }

            return (buffers: bufferBindings, bytes: bytesBindings)
        }
    }
}
