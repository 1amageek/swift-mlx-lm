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
        var quantizationEntries: [MetalQuantizationPlanEntry] = []
        var routingPlanner = DecodeRoutingPlanner(
            bufferSet: bufferSet,
            stafWeightStore: stafWeightStore,
            hiddenSize: hiddenSize,
            slotDimension: slotDimension,
            fallbackWeightFormat: planBuildContext.kernelContext.weightFormat,
            minimumFallbackLength: max(
                hiddenSize * hiddenSize,
                hiddenSize * slotDimension
            ) * planBuildContext.kernelContext.weightFormat.storageByteSize,
            accessPolicyResolver: accessPolicyResolver
        )

        for entry in fusedEntries {
            let resolved = try resolveDispatch(entry)
            routingPlanner.lastFragmentWriteBufferIndices = nil
            let bindings = routingPlanner.bindings(for: entry)
            let writeIndices = routingPlanner.lastFragmentWriteBufferIndices
            let bufferAccessPattern = Self.decodeBufferAccessPattern(
                for: entry,
                buffers: bindings.buffers,
                writeBufferIndices: writeIndices
            )
            let weightTensorName = Self.primaryWeightTensorName(for: entry)
            Self.recordQuantizationEntries(
                for: entry,
                selectedKernelName: resolved.name,
                stafWeightStore: stafWeightStore,
                accessPolicyResolver: accessPolicyResolver,
                fallbackSchemeIdentifier: planBuildContext.compileContext.weightFormat.schemeIdentifier,
                into: &quantizationEntries
            )
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
                    buffers: bindings.buffers,
                    writeBufferIndices: writeIndices),
                metadata: MetalDispatchStepMetadata(
                    kernelName: resolved.name,
                    entryIndex: entry.index,
                    layerIndex: entry.layerIndex,
                    weightTensorName: weightTensorName,
                    bufferAccessPattern: bufferAccessPattern
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
        let supplementalResidencyBuffers = Self.supplementalResidencyBuffers(in: optimizedBarrierSteps)

        return MetalDispatchPlan(
            steps: optimizedBarrierSteps,
            buffers: bufferSet,
            unfusedEntryCount: unfusedCount,
            fusedEntryCount: fusedEntries.count,
            quantizationPlan: MetalQuantizationPlan(
                capabilities: planBuildContext.quantizationCapabilities,
                entries: quantizationEntries
            ),
            supplementalResidencyBuffers: supplementalResidencyBuffers
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
                    entryIndex: step.metadata.entryIndex,
                    layerIndex: step.metadata.layerIndex,
                    weightTensorName: step.metadata.weightTensorName,
                    bufferAccessPattern: step.metadata.bufferAccessPattern
                )
            )
        }
    }

    private static func primaryWeightTensorName(for entry: DispatchEntry) -> String? {
        if let linear = entry.fragment as? LinearFragment {
            return entry.parameterBindings.first(where: { $0.role == linear.field })?.tensorName
        }
        return nil
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

    private static func supplementalResidencyBuffers(
        in steps: [MetalDispatchStep]
    ) -> [MTLBuffer] {
        var seen = Set<ObjectIdentifier>()
        var buffers: [MTLBuffer] = []
        for step in steps {
            for buffer in step.bindings.ownedResidencyBuffers {
                let identifier = ObjectIdentifier(buffer as AnyObject)
                guard seen.insert(identifier).inserted else { continue }
                buffers.append(buffer)
            }
        }
        return buffers
    }

    private static func decodeBufferAccesses(
        for entry: DispatchEntry,
        buffers: [(index: Int, buffer: MTLBuffer, offset: Int)],
        writeBufferIndices: Set<Int>? = nil
    ) -> MetalBufferAccesses {
        let mapped = buffers.map { MetalBufferBinding(index: $0.index, buffer: $0.buffer, offset: $0.offset) }

        func bindingTuple(_ index: Int) -> (buffer: MTLBuffer, offset: Int)? {
            mapped.first(where: { $0.index == index }).map { ($0.buffer, $0.offset) }
        }

        func bindingTuples(in indices: some Sequence<Int>) -> [(buffer: MTLBuffer, offset: Int)] {
            indices.compactMap(bindingTuple(_:))
        }

        if entry.fragment is LinearFragment {
            return MetalBufferAccesses(
                readBuffers: bindingTuples(in: [0, 1]),
                writeBuffers: bindingTuples(in: [2])
            )
        }
        if let writeBufferIndices {
            let allRegions = Set(mapped.map { BufferRegion(buffer: $0.buffer, offset: $0.offset) })
            let writeRegions = Set(
                buffers.filter { writeBufferIndices.contains($0.index) }
                    .map { BufferRegion(buffer: $0.buffer, offset: $0.offset) }
            )
            return MetalBufferAccesses(reads: allRegions, writes: writeRegions)
        }
        return MetalBufferAccesses.conservative(mapped)
    }

    private static func decodeBufferAccessPattern(
        for entry: DispatchEntry,
        buffers: [(index: Int, buffer: MTLBuffer, offset: Int)],
        writeBufferIndices: Set<Int>? = nil
    ) -> MetalDispatchStepMetadata.BufferAccessPattern? {
        if entry.fragment is LinearFragment {
            return .init(reads: [0, 1], writes: [2])
        }
        let bindingIndices = Set(buffers.map(\.index))
        guard let writeBufferIndices else {
            return nil
        }
        return .init(reads: bindingIndices, writes: writeBufferIndices)
    }

    private static func optimizeDecodeBarrierPolicies(
        _ steps: [MetalDispatchStep]
    ) -> [MetalDispatchStep] {
        var pendingReads = Set<BufferRegion>()
        var pendingWrites = Set<BufferRegion>()
        return steps.map { step in
            let requiresBarrier = step.bufferAccesses.requiresBarrier(
                after: pendingReads,
                pendingWrites: pendingWrites
            )
            let barrierPolicy: MetalBarrierPolicy
            if requiresBarrier {
                let visibility: MTL4VisibilityOptions =
                    MetalBufferAccesses.pendingWritesInvolveSharedBuffer(pendingWrites)
                    ? .device : []
                barrierPolicy = .barrier(visibility: visibility)
            } else {
                barrierPolicy = .none
            }
            let descriptor = MetalDispatchDescriptor(
                pipeline: step.pipeline,
                gridSize: step.gridSize,
                threadgroupSize: step.threadgroupSize,
                threadgroupMemoryLength: step.threadgroupMemoryLength,
                barrierPolicy: barrierPolicy
            )

            if requiresBarrier {
                pendingReads = step.bufferAccesses.reads
                pendingWrites = step.bufferAccesses.writes
            } else {
                pendingReads.formUnion(step.bufferAccesses.reads)
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

    private static func recordQuantizationEntries(
        for entry: DispatchEntry,
        selectedKernelName: String,
        stafWeightStore: STAFWeightStore?,
        accessPolicyResolver: ProjectionWeightAccessPolicyResolver,
        fallbackSchemeIdentifier: QuantizationSchemeIdentifier,
        into entries: inout [MetalQuantizationPlanEntry]
    ) {
        let fragment = entry.fragment
        if let linear = fragment as? LinearFragment {
            let descriptor = resolveWeightDescriptor(
                role: linear.field,
                entry: entry,
                executionPhase: .decode,
                stafWeightStore: stafWeightStore,
                accessPolicyResolver: accessPolicyResolver,
                fallbackSchemeIdentifier: fallbackSchemeIdentifier
            )
            entries.append(
                MetalQuantizationPlanEntry(
                    entryIndex: entry.index,
                    layerIndex: entry.layerIndex,
                    tensorName: descriptor.tensorName,
                    path: .decodeProjection,
                    schemeIdentifier: descriptor.schemeIdentifier,
                    layout: descriptor.layout,
                    kernelFamily: .classify(kernelName: selectedKernelName, usesMPP: false),
                    usedFallback: descriptor.usedFallback,
                    fallbackReason: descriptor.fallbackReason
                )
            )
        } else if let batched = fragment as? BatchedProjection {
            for projection in batched.projections {
                let descriptor = resolveWeightDescriptor(
                    role: projection.field,
                    entry: entry,
                    executionPhase: .decode,
                    stafWeightStore: stafWeightStore,
                    accessPolicyResolver: accessPolicyResolver,
                    fallbackSchemeIdentifier: fallbackSchemeIdentifier
                )
                entries.append(
                    MetalQuantizationPlanEntry(
                        entryIndex: entry.index,
                        layerIndex: entry.layerIndex,
                        tensorName: descriptor.tensorName,
                        path: .decodeProjection,
                        schemeIdentifier: descriptor.schemeIdentifier,
                        layout: descriptor.layout,
                        kernelFamily: .classify(kernelName: selectedKernelName, usesMPP: false),
                        usedFallback: descriptor.usedFallback,
                        fallbackReason: descriptor.fallbackReason
                    )
                )
            }
        } else if fragment is GatherFragment {
            let descriptor = resolveWeightDescriptor(
                role: "embedding_table",
                entry: entry,
                executionPhase: .decode,
                stafWeightStore: stafWeightStore,
                accessPolicyResolver: accessPolicyResolver,
                fallbackSchemeIdentifier: fallbackSchemeIdentifier
            )
            entries.append(
                MetalQuantizationPlanEntry(
                    entryIndex: entry.index,
                    layerIndex: entry.layerIndex,
                    tensorName: descriptor.tensorName,
                    path: .embeddingLookup,
                    schemeIdentifier: descriptor.schemeIdentifier,
                    layout: descriptor.layout,
                    kernelFamily: .classify(kernelName: selectedKernelName, usesMPP: false),
                    usedFallback: descriptor.usedFallback,
                    fallbackReason: descriptor.fallbackReason
                )
            )
        }
    }

    private static func resolveWeightDescriptor(
        role: String,
        entry: DispatchEntry,
        executionPhase: STAFWeightExecutionPhase,
        stafWeightStore: STAFWeightStore?,
        accessPolicyResolver: ProjectionWeightAccessPolicyResolver,
        fallbackSchemeIdentifier: QuantizationSchemeIdentifier
    ) -> DecodeWeightDescriptor {
        guard let binding = entry.parameterBindings.first(where: { $0.role == role }) else {
            return DecodeWeightDescriptor(
                tensorName: nil,
                schemeIdentifier: fallbackSchemeIdentifier,
                layout: .rowMajor,
                usedFallback: true,
                fallbackReason: .missingTensorBinding
            )
        }
        guard let stafWeightStore else {
            return DecodeWeightDescriptor(
                tensorName: binding.tensorName,
                schemeIdentifier: fallbackSchemeIdentifier,
                layout: .rowMajor,
                usedFallback: true,
                fallbackReason: .missingWeightStore
            )
        }

        let request = accessPolicyResolver.accessRequest(
            for: entry,
            role: role,
            binding: binding,
            executionPhase: executionPhase,
            stafWeightStore: stafWeightStore
        )
        let layout = stafWeightStore.resolvedBufferAccess(for: request)?.layout ?? request.preferredLayout
        guard let tensorEntry = stafWeightStore.entries[binding.tensorName] else {
            return DecodeWeightDescriptor(
                tensorName: binding.tensorName,
                schemeIdentifier: fallbackSchemeIdentifier,
                layout: layout,
                usedFallback: true,
                fallbackReason: .missingTensorMetadata
            )
        }
        return DecodeWeightDescriptor(
            tensorName: binding.tensorName,
            schemeIdentifier: tensorEntry.schemeIdentifier,
            layout: layout,
            usedFallback: false,
            fallbackReason: nil
        )
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
            case "residual_add_inplace":
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
            case let name where name.hasSuffix("glu_projection_2048") || name.hasSuffix("glu_projection_2048_bf16"):
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
            if table.layout.indices == [0, 1, 2, 3, 4, 5, 6, 17, 18, 19] {
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

private struct DecodeWeightDescriptor {
    let tensorName: String?
    let schemeIdentifier: QuantizationSchemeIdentifier
    let layout: STAFWeightLayout
    let usedFallback: Bool
    let fallbackReason: MetalQuantizationFallbackReason?
}

struct DecodeRoutingPlanner {
    let bufferSet: MetalBufferSet
    let stafWeightStore: STAFWeightStore?
    let hiddenSize: Int
    let slotDimension: Int
    let fallbackWeightFormat: WeightFormat
    let minimumFallbackLength: Int
    let accessPolicyResolver: ProjectionWeightAccessPolicyResolver
    private let elementSize: Int
    private var kvCacheIndex: Int = 0
    private var routingState = BufferRoutingState()
    private var activeCompositeID: Int?
    private var compositeInputSource: (buffer: MTLBuffer, offset: Int)?

    /// Write buffer indices from the most recent fragment binding.
    /// Set by `bindings(for:)` when entry is .fragment.
    /// nil for non-fragment entries or when fragment does not declare write indices.
    var lastFragmentWriteBufferIndices: Set<Int>?

    init(
        bufferSet: MetalBufferSet,
        stafWeightStore: STAFWeightStore?,
        hiddenSize: Int,
        slotDimension: Int,
        fallbackWeightFormat: WeightFormat,
        minimumFallbackLength: Int,
        accessPolicyResolver: ProjectionWeightAccessPolicyResolver
    ) {
        self.bufferSet = bufferSet
        self.stafWeightStore = stafWeightStore
        self.hiddenSize = hiddenSize
        self.slotDimension = slotDimension
        self.fallbackWeightFormat = fallbackWeightFormat
        self.minimumFallbackLength = minimumFallbackLength
        self.accessPolicyResolver = accessPolicyResolver
        self.elementSize = bufferSet.bufferPrecision.byteSize
    }

    mutating func bindings(
        for entry: DispatchEntry
    ) -> (
        buffers: [(index: Int, buffer: MTLBuffer, offset: Int)],
        bytes: [(index: Int, value: [UInt8])]
    ) {
        updateCompositeInputSource(for: entry)

        let weightResolver = WeightResolver(
            entry: entry,
            stafWeightStore: stafWeightStore,
            executionPhase: .decode,
            accessPolicyResolver: accessPolicyResolver
        )

        if let linear = entry.fragment as? LinearFragment {
            let projection = linear
            let isOutput = linear.isOutput
            let (weightBuffer, weightOffset) = weightResolver.resolve(role: projection.field)

            let inputBuffer: MTLBuffer
            let inputOffset: Int
            if !isOutput, let compositeInputSource {
                inputBuffer = compositeInputSource.buffer
                inputOffset = compositeInputSource.offset
            } else if routingState.lastOutputIsHidden {
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
                routingState.currentInputOffset = outputOffset
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
        } else {
            let fragment = entry.fragment
            let resolvedKVCacheIndex = fragment.kvCacheIndexOverride ?? kvCacheIndex
            let currentInputBuffer: MTLBuffer
            let currentInputOffset: Int
            if routingState.lastOutputIsHidden {
                currentInputBuffer = bufferSet.hidden
                currentInputOffset = 0
            } else {
                currentInputBuffer = bufferSet.scratch
                currentInputOffset = routingState.currentInputOffset
            }
            let bindingContext = BufferBindingContext(
                bufferSet: bufferSet,
                slotDimension: slotDimension,
                elementSize: elementSize,
                currentInputBuffer: currentInputBuffer,
                currentInputOffset: currentInputOffset,
                layerIndex: entry.layerIndex,
                kvCacheIndex: resolvedKVCacheIndex,
                convLayerIndex: routingState.convLayerIndex,
                recurrentLayerIndex: routingState.recurrentLayerIndex,
                projectionIndex: routingState.projectionIndex,
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
            // Advance projection index for projection-type fragments
            if bindings.projectionSlotsConsumed > 0 {
                routingState.projectionIndex += bindings.projectionSlotsConsumed
                routingState.currentInputOffset = routingState.projectionIndex * slotDimension * elementSize
            }
            if bindings.resetsProjectionIndex {
                refreshCompositeInputSource()
            }
            lastFragmentWriteBufferIndices = bindings.writeBufferIndices
            return (buffers: bindings.buffers, bytes: bindings.bytes)
        }
    }

    private mutating func updateCompositeInputSource(for entry: DispatchEntry) {
        guard activeCompositeID != entry.compositeID else { return }
        activeCompositeID = entry.compositeID
        refreshCompositeInputSource()
    }

    private mutating func refreshCompositeInputSource() {
        if routingState.lastOutputIsHidden {
            compositeInputSource = (bufferSet.hidden, 0)
        } else {
            compositeInputSource = (bufferSet.scratch, routingState.currentInputOffset)
        }
    }
}
