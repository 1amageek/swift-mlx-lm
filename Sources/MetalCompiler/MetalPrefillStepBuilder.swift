import Metal

struct MetalPrefillStepBuilder {
    func buildPrefillPlan(
        fusedEntries: [DispatchEntry],
        buffers: PrefillBufferSet,
        slotDimension: Int,
        maximumSequenceLength: Int,
        hiddenSize: Int,
        scratchElementSize: Int,
        usesMPP: Bool,
        planBuildContext: PlanBuildContext,
        resolveDispatch: @escaping (DispatchEntry) throws -> (
            name: String,
            pipeline: MTLComputePipelineState,
            config: (grid: MTLSize, threadgroup: MTLSize, sharedMemoryBytes: Int)
        )
    ) throws -> MetalPrefillPlan {
        let constantAllocator = MetalConstantBindingAllocator(device: planBuildContext.device)
        var steps: [MetalPrefillStep] = []
        var planner = PrefillStepPlanner(
            buffers: buffers,
            stafWeightStore: planBuildContext.stafWeightStore,
            hiddenSize: hiddenSize,
            slotDimension: slotDimension,
            maximumSequenceLength: maximumSequenceLength,
            scratchElementSize: scratchElementSize,
            usesMPP: usesMPP,
            planBuildContext: planBuildContext,
            resolveDispatch: resolveDispatch
        )

        for entry in fusedEntries {
            let prefillSteps = try planner.buildSteps(for: entry)
            steps.append(contentsOf: prefillSteps)
        }

        let residentSteps = try makeResidentConstantSteps(steps, allocator: constantAllocator)
        let optimizedSteps = Self.optimizePrefillBarrierPolicies(residentSteps)
        let supplementalResidencyBuffers = Self.supplementalResidencyBuffers(in: optimizedSteps)
        let finalHiddenSource = planner.finalHiddenSource()
        return MetalPrefillPlan(
            steps: optimizedSteps,
            buffers: buffers,
            slotDimension: slotDimension,
            maximumSequenceLength: maximumSequenceLength,
            stepCount: optimizedSteps.count,
            usesMPP: usesMPP,
            quantizationPlan: planner.makeQuantizationPlan(),
            finalHiddenBuffer: finalHiddenSource.buffer,
            finalHiddenBaseOffset: finalHiddenSource.offset,
            finalHiddenRowStride: finalHiddenSource.rowStride,
            supplementalResidencyBuffers: supplementalResidencyBuffers
        )
    }

    /// Offset-aware buffer region for precise hazard detection.
    /// Distinguishes scratch[0] from scratch[1] on the same MTLBuffer.
    /// Eliminate unnecessary memory barriers between prefill steps using
    /// offset-aware buffer region tracking.
    ///
    /// Each step's `metadata.bufferAccessPattern` declares which binding indices are
    /// reads vs writes. Steps without a declared pattern are treated conservatively
    /// (all bindings as both read and written).
    static func optimizePrefillBarrierPolicies(
        _ steps: [MetalPrefillStep]
    ) -> [MetalPrefillStep] {
        var pendingReads = Set<BufferRegion>()
        var pendingWrites = Set<BufferRegion>()
        return steps.map { step in
            let accesses = resolveBufferRegions(for: step)
            if step.mode == .lastToken {
                pendingReads = accesses.reads
                pendingWrites = accesses.writes
                return step
            }
            let requiresBarrier = accesses.requiresBarrier(
                after: pendingReads,
                pendingWrites: pendingWrites
            )
            let newBarrierPolicy: MetalBarrierPolicy
            if requiresBarrier {
                let visibility: MTL4VisibilityOptions =
                    MetalBufferAccesses.pendingWritesInvolveSharedBuffer(pendingWrites)
                    ? .device : []
                newBarrierPolicy = .barrier(visibility: visibility)
            } else {
                newBarrierPolicy = .none
            }

            if requiresBarrier {
                pendingReads = accesses.reads
                pendingWrites = accesses.writes
            } else {
                pendingReads.formUnion(accesses.reads)
                pendingWrites.formUnion(accesses.writes)
            }

            guard newBarrierPolicy != step.barrierPolicy else { return step }

            let descriptor = MetalDispatchDescriptor(
                pipeline: step.pipeline,
                gridSize: step.gridSize,
                threadgroupSize: step.threadgroupSize,
                threadgroupMemoryLength: step.threadgroupMemoryLength,
                barrierPolicy: newBarrierPolicy
            )
            return MetalPrefillStep(
                descriptor: descriptor,
                bindings: step.bindings,
                mode: step.mode,
                sequenceLengthPolicy: step.sequenceLengthPolicy,
                positionBufferIndex: step.positionBufferIndex,
                perPositionStrides: step.perPositionStrides,
                metadata: step.metadata
            )
        }
    }

    private static func supplementalResidencyBuffers(
        in steps: [MetalPrefillStep]
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

    /// Convert a step's declared buffer access pattern into concrete buffer regions.
    /// Falls back to treating all bindings as read+written when no pattern is declared.
    private static func resolveBufferRegions(
        for step: MetalPrefillStep
    ) -> MetalBufferAccesses {
        let buffers = step.bindings.buffers

        func regions(for indices: Set<Int>) -> Set<BufferRegion> {
            Set(buffers.filter { indices.contains($0.index) }
                .map { BufferRegion(buffer: $0.buffer, offset: $0.offset) })
        }

        if let pattern = step.metadata.bufferAccessPattern {
            return MetalBufferAccesses(
                reads: regions(for: pattern.readIndices),
                writes: regions(for: pattern.writeIndices))
        }

        // Conservative fallback: treat all bindings as both read and written.
        return MetalBufferAccesses.conservative(buffers)
    }

    private func makeResidentConstantSteps(
        _ steps: [MetalPrefillStep],
        allocator: MetalConstantBindingAllocator
    ) throws -> [MetalPrefillStep] {
        let bindingTables = steps.map(\.bindings)
        let residentBindings = try allocator.makeBindingTables(from: bindingTables)
        return zip(steps, residentBindings).map { step, bindings in
            MetalPrefillStep(
                descriptor: step.descriptor,
                bindings: bindings,
                mode: step.mode,
                sequenceLengthPolicy: step.sequenceLengthPolicy,
                positionBufferIndex: step.positionBufferIndex,
                perPositionStrides: step.perPositionStrides,
                metadata: step.metadata
            )
        }
    }
}

private struct PrefillStepPlanner {
    let buffers: PrefillBufferSet
    let stafWeightStore: STAFWeightStore?
    let hiddenSize: Int
    let slotDimension: Int
    let maximumSequenceLength: Int
    let scratchElementSize: Int
    let usesMPP: Bool
    let planBuildContext: PlanBuildContext
    let fallbackWeightFormat: WeightFormat
    let minimumFallbackLength: Int
    let resolveDispatch: (DispatchEntry) throws -> (
        name: String,
        pipeline: MTLComputePipelineState,
        config: (grid: MTLSize, threadgroup: MTLSize, sharedMemoryBytes: Int)
    )
    var kvCacheIndex: Int = 0
    var routingState = BufferRoutingState()
    var outputHeadInputSource: (buffer: MTLBuffer, offset: Int, rowStride: Int)?
    var activeCompositeID: Int?
    var compositeInputSource: (buffer: MTLBuffer, offset: Int)?
    var quantizationEntries: [MetalQuantizationPlanEntry] = []

    init(
        buffers: PrefillBufferSet,
        stafWeightStore: STAFWeightStore?,
        hiddenSize: Int,
        slotDimension: Int,
        maximumSequenceLength: Int,
        scratchElementSize: Int,
        usesMPP: Bool,
        planBuildContext: PlanBuildContext,
        resolveDispatch: @escaping (DispatchEntry) throws -> (
            name: String,
            pipeline: MTLComputePipelineState,
            config: (grid: MTLSize, threadgroup: MTLSize, sharedMemoryBytes: Int)
        )
    ) {
        self.buffers = buffers
        self.stafWeightStore = stafWeightStore
        self.hiddenSize = hiddenSize
        self.slotDimension = slotDimension
        self.maximumSequenceLength = maximumSequenceLength
        self.scratchElementSize = scratchElementSize
        self.usesMPP = usesMPP
        self.planBuildContext = planBuildContext
        self.fallbackWeightFormat = planBuildContext.kernelContext.weightFormat
        self.minimumFallbackLength = max(
            hiddenSize * hiddenSize,
            hiddenSize * slotDimension
        ) * planBuildContext.kernelContext.weightFormat.storageByteSize
        self.resolveDispatch = resolveDispatch
    }

    private func annotate(
        _ steps: [MetalPrefillStep],
        entryIndex: Int,
        layerIndex: Int?
    ) -> [MetalPrefillStep] {
        steps.map { step in
            MetalPrefillStep(
                descriptor: step.descriptor,
                bindings: step.bindings,
                mode: step.mode,
                sequenceLengthPolicy: step.sequenceLengthPolicy,
                positionBufferIndex: step.positionBufferIndex,
                perPositionStrides: step.perPositionStrides,
                metadata: MetalDispatchStepMetadata(
                    kernelName: step.metadata.kernelName,
                    entryIndex: entryIndex,
                    layerIndex: layerIndex,
                    weightTensorName: step.metadata.weightTensorName,
                    bufferAccessPattern: step.metadata.bufferAccessPattern
                )
            )
        }
    }

    private func fragmentKernelContext(
        for fragment: any PrimitiveMetalKernelFragment,
        entry: DispatchEntry
    ) -> KernelContext {
        let weightFormatResolver = KernelWeightFormatResolver(stafWeightStore: stafWeightStore)
        return KernelContext(
            bufferPrecision: planBuildContext.kernelContext.bufferPrecision,
            weightFormat: weightFormatResolver.resolve(forFragment: fragment, entry: entry)
        )
    }

    mutating func buildSteps(for entry: DispatchEntry) throws -> [MetalPrefillStep] {
        updateCompositeInputSource(for: entry)

        let weightResolver = WeightResolver(
            entry: entry,
            stafWeightStore: stafWeightStore,
            fallbackBuffer: buffers.hidden,
            fallbackWeightFormat: fallbackWeightFormat,
            minimumFallbackLength: minimumFallbackLength,
            logsMisses: false,
            executionPhase: .prefill,
            accessPolicyResolver: planBuildContext.compileContext.accessPolicyResolver
        )

        switch entry.kind {
        case .fragment(let frag):
            let pipelineCache = planBuildContext.pipelineCache
            let kernelContext = fragmentKernelContext(for: frag, entry: entry)
            let resolvedKVCacheIndex = frag.kvCacheIndexOverride ?? kvCacheIndex
            let currentInputBuffer: MTLBuffer
            let currentInputOffset: Int
            if routingState.lastOutputIsHidden {
                currentInputBuffer = buffers.hidden
                currentInputOffset = 0
            } else {
                currentInputBuffer = buffers.scratch
                currentInputOffset = routingState.currentInputOffset
            }
            let prefillContext = PrefillBindingContext(
                buffers: buffers,
                slotDimension: slotDimension,
                scratchElementSize: scratchElementSize,
                maximumSequenceLength: maximumSequenceLength,
                currentInputBuffer: currentInputBuffer,
                currentInputOffset: currentInputOffset,
                layerIndex: entry.layerIndex,
                kvCacheIndex: resolvedKVCacheIndex,
                convLayerIndex: routingState.convLayerIndex,
                recurrentLayerIndex: routingState.recurrentLayerIndex,
                kernelContext: kernelContext,
                resolveWeight: weightResolver.resolve,
                getPipeline: { name in
                    guard let pipeline = pipelineCache[name] else {
                        let relatedKernelNames = pipelineCache.keys
                            .filter {
                                $0.contains("embedding_lookup")
                                    || $0.contains("rms_norm_seq")
                                    || $0.contains("qk_rms_norm_seq")
                            }
                            .sorted()
                        if !relatedKernelNames.isEmpty {
                            print("[Compiler] missing prefill kernel '\(name)'; related compiled kernels: \(relatedKernelNames)")
                        }
                        throw MetalCompilerError.kernelNotFound(name)
                    }
                    return pipeline
                }
            )
            if let reduction = frag as? Reduction,
               shouldCaptureResidualInput(for: reduction.weightRole),
               currentInputBuffer === buffers.hidden,
               currentInputOffset == 0
            {
                var steps: [MetalPrefillStep] = []
                steps.append(try makeHiddenToResidualCopyStep(
                    dimension: reduction.dimension,
                    entry: entry
                ))
                steps.append(contentsOf: try buildNormToHiddenStep(
                    inputBuffer: buffers.residual,
                    inputOffset: 0,
                    dimension: reduction.dimension,
                    epsilon: reduction.epsilon,
                    weightRole: reduction.weightRole,
                    weightBias: reduction.weightBias,
                    entry: entry
                ))
                routingState.projectionIndex = 0
                routingState.lastOutputIsHidden = true
                routingState.currentInputOffset = 0
                refreshCompositeInputSource()
                return annotate(steps, entryIndex: entry.index, layerIndex: entry.layerIndex)
            }
            let result = try frag.prefillSteps(context: prefillContext)
            if frag is GatherFragment, let selectedKernelName = result.steps.first?.pipeline.label {
                let descriptor = resolveProjectionWeightDescriptor(role: "embedding_table", entry: entry)
                quantizationEntries.append(
                    MetalQuantizationPlanEntry(
                        entryIndex: entry.index,
                        layerIndex: entry.layerIndex,
                        tensorName: descriptor.tensorName,
                        path: .embeddingLookup,
                        schemeIdentifier: descriptor.schemeIdentifier,
                        layout: descriptor.layout,
                        kernelFamily: .classify(
                            kernelName: selectedKernelName,
                            usesMPP: false
                        ),
                        usedFallback: descriptor.usedFallback,
                        fallbackReason: descriptor.fallbackReason
                    )
                )
            }
            if result.resetsProjectionIndex {
                routingState.projectionIndex = 0
                if !result.outputIsHidden {
                    routingState.currentInputOffset = 0
                }
            }
            if result.consumesKVCacheLayer { kvCacheIndex += 1 }
            if result.consumesConvLayer { routingState.convLayerIndex += 1 }
            if result.consumesRecurrentLayer { routingState.recurrentLayerIndex += 1 }
            routingState.lastOutputIsHidden = result.outputIsHidden
            if result.resetsProjectionIndex {
                refreshCompositeInputSource()
            }
            return annotate(result.steps, entryIndex: entry.index, layerIndex: entry.layerIndex)

        case .batchedProjection(let batched):
            let inputBuffer: MTLBuffer
            let inputOffset: Int
            if routingState.lastOutputIsHidden {
                inputBuffer = buffers.hidden
                inputOffset = 0
            } else {
                inputBuffer = buffers.scratch
                inputOffset = routingState.currentInputOffset
            }
            let scratchSlotSize = slotDimension * scratchElementSize * maximumSequenceLength
            let inputRowStride = inputBuffer === buffers.hidden
                ? (buffers.hidden.length / max(maximumSequenceLength, 1)) / scratchElementSize
                : batched.projections[0].inputDimension

            // Try direct quantized GEMM: single dispatch for all projections
            let firstDescriptor = resolveProjectionWeightDescriptor(
                role: batched.projections[0].field, entry: entry
            )
            if let batchedGEMM = resolveBatchedQuantizedGEMM(
                   for: firstDescriptor.schemeIdentifier, count: batched.projections.count
               ),
               let batchedPipeline = planBuildContext.pipelineCache[batchedGEMM.kernelName] {

                let count = batched.projections.count

                // Buffer layout: input(0), weight0..N-1(1..N), output0..N-1(N+1..2N)
                var bufferBindings: [(Int, MTLBuffer, Int)] = [(0, inputBuffer, inputOffset)]
                var totalOutputDim = 0
                var lastOutputOffset = routingState.currentInputOffset

                for (i, projection) in batched.projections.enumerated() {
                    let (weightBuffer, weightOffset) = weightResolver.resolve(role: projection.field)
                    bufferBindings.append((1 + i, weightBuffer, weightOffset))

                    let outputOffset = (routingState.projectionIndex + 1) * scratchSlotSize
                    lastOutputOffset = outputOffset
                    routingState.projectionIndex += 1
                    bufferBindings.append((1 + count + i, buffers.scratch, outputOffset))

                    totalOutputDim += projection.outputDimension
                }

                // Bytes layout: inputDim(2N+1), outDim0..N-1(2N+2..3N+1), seqLen(3N+2), rowStride(3N+3)
                let dimBase = 1 + 2 * count
                var bytesBindings: [(index: Int, value: [UInt8])] = [
                    uint32Binding(dimBase, UInt32(batched.projections[0].inputDimension)),
                ]
                for (i, projection) in batched.projections.enumerated() {
                    bytesBindings.append(uint32Binding(dimBase + 1 + i, UInt32(projection.outputDimension)))
                }
                let seqLenIndex = dimBase + 1 + count
                bytesBindings.append(uint32Binding(seqLenIndex, UInt32(maximumSequenceLength)))
                bytesBindings.append(uint32Binding(seqLenIndex + 1, UInt32(inputRowStride)))

                // Grid covers all output rows across all projections
                let simdWidth = max(batchedPipeline.threadExecutionWidth, 1)
                let rowsPerThreadgroup = 2
                let threads = min(
                    simdWidth * rowsPerThreadgroup,
                    batchedPipeline.maxTotalThreadsPerThreadgroup
                )
                let gridSize = MTLSize(
                    width: (totalOutputDim + rowsPerThreadgroup - 1) / rowsPerThreadgroup,
                    height: maximumSequenceLength,
                    depth: 1
                )
                let threadgroupSize = MTLSize(width: threads, height: 1, depth: 1)

                // Threadgroup memory: input tile for quantized block unpacking
                let threadgroupMemoryLength = batchedGEMM.threadgroupMemoryLength

                // Buffer access pattern: reads input + all weights, writes all outputs
                let readIndices = Set(0...count)
                let writeIndices = Set((count + 1)...(2 * count))

                let step = MetalPrefillStep(
                    pipeline: batchedPipeline,
                    gridSize: gridSize,
                    threadgroupSize: threadgroupSize,
                    bufferBindings: bufferBindings,
                    bytesBindings: bytesBindings,
                    threadgroupMemoryLength: threadgroupMemoryLength,
                    sync: .bufferBarrier,
                    mode: .batch,
                    sequenceLengthPolicy: .bindAndAdjustGridHeight(index: seqLenIndex),
                    positionBufferIndex: nil,
                    perPositionStrides: [:],
                    metadata: .init(
                        kernelName: batchedGEMM.kernelName,
                        entryIndex: entry.index,
                        weightTensorName: nil,
                        bufferAccessPattern: .init(reads: readIndices, writes: writeIndices)
                    )
                )

                routingState.lastOutputIsHidden = false
                routingState.currentInputOffset = lastOutputOffset
                return annotate([step], entryIndex: entry.index, layerIndex: entry.layerIndex)
            }

            // Fallback: expand to individual projection steps
            var steps: [MetalPrefillStep] = []
            steps.reserveCapacity(batched.projections.count)
            var lastOutputOffset = routingState.currentInputOffset
            for projection in batched.projections {
                let projInputRowStride = inputBuffer === buffers.hidden
                    ? (buffers.hidden.length / max(maximumSequenceLength, 1)) / scratchElementSize
                    : projection.inputDimension
                let resolved = try resolveDispatch(
                    DispatchEntry(
                        index: entry.index,
                        kind: .projection(
                            MetalProjection(
                                field: projection.field,
                                inputDimension: projection.inputDimension,
                                outputDimension: projection.outputDimension
                            ),
                            isOutput: false
                        ),
                        parameterBindings: entry.parameterBindings,
                        layerIndex: entry.layerIndex
                    )
                )
                let (weightBuffer, weightOffset) = weightResolver.resolve(role: projection.field)
                let weightTensorName = entry.parameterBindings.first(where: { $0.role == projection.field })?.tensorName
                let quantizationDescriptor = resolveProjectionWeightDescriptor(role: projection.field, entry: entry)
                let outputOffset = (routingState.projectionIndex + 1) * scratchSlotSize
                lastOutputOffset = outputOffset
                routingState.projectionIndex += 1

                // Prefer direct quantized GEMM (dequant in registers) when available.
                // Falls back to dequant→AMX when no direct kernel exists.
                let directGEMM = resolveDirectQuantizedGEMM(for: quantizationDescriptor.schemeIdentifier)
                let useDirectQuantizedGEMM = directGEMM.flatMap {
                    planBuildContext.pipelineCache[$0.kernelName]
                } != nil

                let canDequantForAMX = quantizationDescriptor.schemeIdentifier.isWeightQuantized
                    && buffers.dequantScratch != nil
                    && dequantKernelName(for: quantizationDescriptor.schemeIdentifier) != nil
                let usesMPPForStep = usesMPP
                    && projInputRowStride == projection.inputDimension
                    && (!quantizationDescriptor.schemeIdentifier.isWeightQuantized || canDequantForAMX)

                if !useDirectQuantizedGEMM, canDequantForAMX && usesMPPForStep,
                   let dequantName = dequantKernelName(for: quantizationDescriptor.schemeIdentifier),
                   let dequantPipeline = planBuildContext.pipelineCache[dequantName],
                   let dequantScratch = buffers.dequantScratch {
                    steps.append(
                        MetalPrefillStep(
                            pipeline: dequantPipeline,
                            gridSize: MTLSize(width: projection.outputDimension, height: 1, depth: 1),
                            threadgroupSize: MTLSize(width: 256, height: 1, depth: 1),
                            bufferBindings: [
                                (0, weightBuffer, weightOffset),
                                (1, dequantScratch, 0),
                            ],
                            bytesBindings: [
                                uint32Binding(2, UInt32(projection.inputDimension)),
                                uint32Binding(3, UInt32(projection.outputDimension)),
                            ],
                            threadgroupMemoryLength: 0,
                            sync: .bufferBarrier,
                            mode: .batch,
                            sequenceLengthPolicy: .none,
                            positionBufferIndex: nil,
                            perPositionStrides: [:],
                            metadata: .init(
                                kernelName: dequantName,
                                entryIndex: entry.index,
                                weightTensorName: weightTensorName,
                                bufferAccessPattern: .init(reads: [0], writes: [1])
                            )
                        )
                    )
                }

                let selectedPipeline: MTLComputePipelineState
                let selectedKernelName: String
                if useDirectQuantizedGEMM,
                   let resolved = directGEMM,
                   let directPipeline = planBuildContext.pipelineCache[resolved.kernelName] {
                    selectedPipeline = directPipeline
                    selectedKernelName = resolved.kernelName
                } else if canDequantForAMX && usesMPPForStep,
                   let mppPipeline = planBuildContext.pipelineCache["gemm_bf16_f32s"] {
                    selectedPipeline = mppPipeline
                    selectedKernelName = "gemm_bf16_f32s"
                } else if !usesMPPForStep,
                   let naivePipeline = planBuildContext.pipelineCache["naive::\(resolved.name)"] {
                    selectedPipeline = naivePipeline
                    selectedKernelName = "naive::\(resolved.name)"
                } else {
                    selectedPipeline = resolved.pipeline
                    selectedKernelName = resolved.name
                }

                let gemmWeightBuffer: MTLBuffer
                let gemmWeightOffset: Int
                if useDirectQuantizedGEMM {
                    gemmWeightBuffer = weightBuffer
                    gemmWeightOffset = weightOffset
                } else if canDequantForAMX && usesMPPForStep, let dequantScratch = buffers.dequantScratch {
                    gemmWeightBuffer = dequantScratch
                    gemmWeightOffset = 0
                } else {
                    gemmWeightBuffer = weightBuffer
                    gemmWeightOffset = weightOffset
                }

                let gridSize: MTLSize
                let threadgroupSize: MTLSize
                if usesMPPForStep && !useDirectQuantizedGEMM {
                    let simdWidth = selectedPipeline.threadExecutionWidth
                    gridSize = MTLSize(
                        width: (projection.outputDimension + 31) / 32,
                        height: (maximumSequenceLength + 63) / 64,
                        depth: 1
                    )
                    threadgroupSize = MTLSize(width: simdWidth * 4, height: 1, depth: 1)
                } else {
                    let simdWidth = max(selectedPipeline.threadExecutionWidth, 1)
                    let rowsPerThreadgroup = 2
                    let threads = min(
                        simdWidth * rowsPerThreadgroup,
                        selectedPipeline.maxTotalThreadsPerThreadgroup
                    )
                    gridSize = MTLSize(
                        width: (projection.outputDimension + rowsPerThreadgroup - 1) / rowsPerThreadgroup,
                        height: maximumSequenceLength,
                        depth: 1
                    )
                    threadgroupSize = MTLSize(width: threads, height: 1, depth: 1)
                }

                let gemmPattern = MetalDispatchStepMetadata.BufferAccessPattern(reads: [0, 1], writes: [2])
                recordProjectionQuantization(
                    entry: entry,
                    descriptor: quantizationDescriptor,
                    mode: .batch,
                    inputRowStride: projInputRowStride,
                    inputDimension: projection.inputDimension,
                    selectedKernelName: selectedKernelName,
                    usesMPPForStep: usesMPPForStep
                )
                steps.append(
                    MetalPrefillStep(
                        pipeline: selectedPipeline,
                        gridSize: gridSize,
                        threadgroupSize: threadgroupSize,
                        bufferBindings: [
                            (0, inputBuffer, inputOffset),
                            (1, gemmWeightBuffer, gemmWeightOffset),
                            (2, buffers.scratch, outputOffset),
                        ],
                        bytesBindings: [
                            uint32Binding(3, UInt32(projection.inputDimension)),
                            uint32Binding(4, UInt32(projection.outputDimension)),
                            uint32Binding(5, UInt32(maximumSequenceLength)),
                            uint32Binding(6, UInt32(projInputRowStride)),
                        ],
                        threadgroupMemoryLength: useDirectQuantizedGEMM
                            ? (directGEMM?.threadgroupMemoryLength ?? 0)
                            : (usesMPPForStep ? 0 : resolved.config.sharedMemoryBytes),
                        sync: .bufferBarrier,
                        mode: .batch,
                        sequenceLengthPolicy: usesMPPForStep && !useDirectQuantizedGEMM
                            ? .bindAndAdjustGridHeightTiled(index: 5, tileHeight: 64)
                            : .bindAndAdjustGridHeight(index: 5),
                        positionBufferIndex: nil,
                        perPositionStrides: [:],
                        metadata: .init(
                            kernelName: selectedKernelName,
                            entryIndex: entry.index,
                            weightTensorName: weightTensorName,
                            bufferAccessPattern: gemmPattern
                        )
                    )
                )
            }

            routingState.lastOutputIsHidden = false
            routingState.currentInputOffset = lastOutputOffset
            return annotate(steps, entryIndex: entry.index, layerIndex: entry.layerIndex)

        case .batchedFragment(let batch):
            // Fast path: batched QK norm → single dispatch
            if batch.fragments.count == 2,
               let qNorm = batch.fragments[0] as? QKNormFragment,
               let kNorm = batch.fragments[1] as? QKNormFragment,
               let batchedStep = try makeBatchedQKNormStep(
                   qNorm: qNorm, kNorm: kNorm, entry: entry) {
                return annotate([batchedStep], entryIndex: entry.index, layerIndex: entry.layerIndex)
            }
            // Fallback: decompose to individual fragments
            var steps: [MetalPrefillStep] = []
            for (i, frag) in batch.fragments.enumerated() {
                let singleEntry = DispatchEntry(
                    index: entry.index + i,
                    kind: .fragment(frag),
                    parameterBindings: entry.parameterBindings,
                    layerIndex: entry.layerIndex
                )
                let fragSteps = try buildSteps(for: singleEntry)
                steps.append(contentsOf: fragSteps)
            }
            return annotate(steps, entryIndex: entry.index, layerIndex: entry.layerIndex)

        case .fusedResidualAddNorm(let fusedOp):
            // Try single-dispatch fused kernel for prefill
            if let fusedStep = try makeFusedResidualAddNormStep(fusedOp: fusedOp, entry: entry) {
                routingState.lastOutputIsHidden = true
                routingState.currentInputOffset = 0
                routingState.projectionIndex = 0
                refreshCompositeInputSource()
                return annotate([fusedStep], entryIndex: entry.index, layerIndex: entry.layerIndex)
            }
            // Fallback: decompose to individual dispatches
            var steps: [MetalPrefillStep] = []

            // If preNorm is present, emit in-place norm on hidden first
            if let preNorm = fusedOp.preNorm {
                let preNormEntry = DispatchEntry(
                    index: entry.index,
                    kind: entry.kind,
                    parameterBindings: preNorm.parameterBindings,
                    layerIndex: entry.layerIndex)
                steps.append(contentsOf: try buildNormToHiddenStep(
                    inputBuffer: buffers.hidden,
                    inputOffset: 0,
                    dimension: fusedOp.dimension,
                    epsilon: preNorm.epsilon,
                    weightRole: "scale",
                    weightBias: preNorm.weightBias,
                    entry: preNormEntry
                ))
            }

            let addEntry = DispatchEntry(
                index: entry.index,
                kind: .structuralAdd(dimension: fusedOp.dimension),
                parameterBindings: [],
                layerIndex: entry.layerIndex
            )
            steps.append(contentsOf: try buildSteps(for: addEntry))
            steps.append(try makeHiddenToResidualCopyStep(
                dimension: fusedOp.dimension,
                entry: entry
            ))
            steps.append(contentsOf: try buildNormToHiddenStep(
                inputBuffer: buffers.residual,
                inputOffset: 0,
                dimension: fusedOp.dimension,
                epsilon: fusedOp.epsilon,
                weightRole: "scale",
                weightBias: fusedOp.weightBias,
                entry: entry
            ))
            // Standalone sequence RMSNorm kernels use logical `dimension`
            // row strides, which matches `hidden` but not `scratch`'s
            // slotDimension stride. Keep normalized activations in hidden
            // for subsequent projections / output-head routing.
            routingState.lastOutputIsHidden = true
            routingState.currentInputOffset = 0
            routingState.projectionIndex = 0
            refreshCompositeInputSource()
            return annotate(steps, entryIndex: entry.index, layerIndex: entry.layerIndex)

        case .fusedSwiGLUProjection(let fusedOp):
            let batchedEntry = DispatchEntry(
                index: entry.index,
                kind: .batchedProjection(BatchedProjection(projections: [
                    .init(field: fusedOp.gateField, inputDimension: fusedOp.inputDimension, outputDimension: fusedOp.outputDimension),
                    .init(field: fusedOp.upField, inputDimension: fusedOp.inputDimension, outputDimension: fusedOp.outputDimension),
                ])),
                parameterBindings: entry.parameterBindings,
                layerIndex: entry.layerIndex
            )
            let elementwiseKind: ElementwiseFragment.ElementwiseKind = switch fusedOp.activation {
            case .silu: .swiglu
            case .geluTanh: .geluGated
            }
            let swigluEntry = DispatchEntry(
                index: entry.index + 1,
                kind: .fragment(ElementwiseFragment(count: fusedOp.outputDimension, kind: elementwiseKind)),
                parameterBindings: entry.parameterBindings,
                layerIndex: entry.layerIndex
            )

            var steps: [MetalPrefillStep] = []
            for decomposed in [batchedEntry, swigluEntry] {
                let built = try buildSteps(for: decomposed)
                steps.append(contentsOf: built)
            }
            return annotate(steps, entryIndex: entry.index, layerIndex: entry.layerIndex)

        case .fusedCopyNorm(let fusedOp):
            // Try single-dispatch fused kernel for prefill
            if let fusedStep = try makeFusedCopyNormStep(fusedOp: fusedOp, entry: entry) {
                routingState.lastOutputIsHidden = true
                routingState.currentInputOffset = 0
                routingState.projectionIndex = 0
                refreshCompositeInputSource()
                return annotate([fusedStep], entryIndex: entry.index, layerIndex: entry.layerIndex)
            }
            // Fallback: decompose to copy + norm (2 dispatches)
            var steps: [MetalPrefillStep] = []
            let copyEntry = DispatchEntry(
                index: entry.index,
                kind: .structuralCopy(dimension: fusedOp.dimension),
                parameterBindings: [],
                layerIndex: entry.layerIndex
            )
            steps.append(contentsOf: try buildSteps(for: copyEntry))
            steps.append(contentsOf: try buildNormToHiddenStep(
                inputBuffer: buffers.residual,
                inputOffset: 0,
                dimension: fusedOp.dimension,
                epsilon: fusedOp.epsilon,
                weightRole: "scale",
                weightBias: 0,
                entry: entry
            ))
            routingState.lastOutputIsHidden = true
            routingState.currentInputOffset = 0
            routingState.projectionIndex = 0
            refreshCompositeInputSource()
            return annotate(steps, entryIndex: entry.index, layerIndex: entry.layerIndex)

        case .fusedResidualAddCopyNorm(let fusedOp):
            // Try single-dispatch fused kernel for prefill
            if let fusedStep = try makeFusedResidualAddCopyNormStep(fusedOp: fusedOp, entry: entry) {
                routingState.lastOutputIsHidden = true
                routingState.currentInputOffset = 0
                routingState.projectionIndex = 0
                refreshCompositeInputSource()
                return annotate([fusedStep], entryIndex: entry.index, layerIndex: entry.layerIndex)
            }
            // Fallback: decompose to individual dispatches
            var steps: [MetalPrefillStep] = []

            // If preNorm is present, emit in-place norm on hidden first
            if let preNorm = fusedOp.preNorm {
                let preNormEntry = DispatchEntry(
                    index: entry.index,
                    kind: entry.kind,
                    parameterBindings: preNorm.parameterBindings,
                    layerIndex: entry.layerIndex)
                steps.append(contentsOf: try buildNormToHiddenStep(
                    inputBuffer: buffers.hidden,
                    inputOffset: 0,
                    dimension: fusedOp.dimension,
                    epsilon: preNorm.epsilon,
                    weightRole: "scale",
                    weightBias: preNorm.weightBias,
                    entry: preNormEntry
                ))
            }

            let addEntry = DispatchEntry(
                index: entry.index,
                kind: .structuralAdd(dimension: fusedOp.dimension),
                parameterBindings: [],
                layerIndex: entry.layerIndex
            )
            let copyEntry = DispatchEntry(
                index: entry.index + 1,
                kind: .structuralCopy(dimension: fusedOp.dimension),
                parameterBindings: [],
                layerIndex: entry.layerIndex
            )
            steps.append(contentsOf: try buildSteps(for: addEntry))
            steps.append(contentsOf: try buildSteps(for: copyEntry))
            steps.append(contentsOf: try buildNormToHiddenStep(
                inputBuffer: buffers.residual,
                inputOffset: 0,
                dimension: fusedOp.dimension,
                epsilon: fusedOp.epsilon,
                weightRole: "scale",
                weightBias: fusedOp.weightBias,
                entry: entry
            ))
            routingState.lastOutputIsHidden = true
            routingState.currentInputOffset = 0
            routingState.projectionIndex = 0
            refreshCompositeInputSource()
            return annotate(steps, entryIndex: entry.index, layerIndex: entry.layerIndex)

        case .projection(let projection, let isOutput):
            let resolved = try resolveDispatch(entry)
            let (weightBuffer, weightOffset) = weightResolver.resolve(role: projection.field)
            let weightTensorName = entry.parameterBindings.first(where: { $0.role == projection.field })?.tensorName
            let quantizationDescriptor = resolveProjectionWeightDescriptor(role: projection.field, entry: entry)

            let inputBuffer: MTLBuffer
            let inputOffset: Int
            if !isOutput, let compositeInputSource {
                inputBuffer = compositeInputSource.buffer
                inputOffset = compositeInputSource.offset
            } else if routingState.lastOutputIsHidden {
                inputBuffer = buffers.hidden
                inputOffset = 0
            } else {
                inputBuffer = buffers.scratch
                inputOffset = routingState.currentInputOffset
            }

            let outputBuffer: MTLBuffer
            let outputOffset: Int
            let mode: PrefillStepMode
            let seqLenValue: UInt32
            let scratchSlotSize = slotDimension * scratchElementSize * maximumSequenceLength
            let inputRowStride = inputBuffer === buffers.hidden
                ? (buffers.hidden.length / max(maximumSequenceLength, 1)) / scratchElementSize
                : projection.inputDimension

            if isOutput && projection.outputDimension > hiddenSize {
                let inputRowStride = inputBuffer === buffers.hidden
                    ? buffers.hidden.length / max(maximumSequenceLength, 1)
                    : projection.inputDimension * scratchElementSize
                outputHeadInputSource = (
                    buffer: inputBuffer,
                    offset: inputOffset,
                    rowStride: inputRowStride
                )
                outputBuffer = buffers.logits
                outputOffset = 0
                mode = .lastToken
                seqLenValue = 1
                routingState.lastOutputIsHidden = false
                routingState.currentInputOffset = 0
            } else if isOutput {
                outputBuffer = buffers.hidden
                outputOffset = 0
                mode = .batch
                seqLenValue = UInt32(maximumSequenceLength)
                routingState.lastOutputIsHidden = true
                routingState.currentInputOffset = 0
            } else {
                let scratchSlot = routingState.projectionIndex + 1
                outputBuffer = buffers.scratch
                outputOffset = scratchSlot * scratchSlotSize
                mode = .batch
                seqLenValue = UInt32(maximumSequenceLength)
                routingState.lastOutputIsHidden = false
                routingState.currentInputOffset = outputOffset
            }
            routingState.projectionIndex += 1

            var perPositionStrides: [Int: Int] = [:]
            if mode == .lastToken {
                let inputRowStride = inputBuffer === buffers.hidden
                    ? buffers.hidden.length / max(maximumSequenceLength, 1)
                    : projection.inputDimension * scratchElementSize
                perPositionStrides[0] = inputRowStride
            }
            // Prefer direct quantized GEMM (dequant in registers) when available.
            // Falls back to dequant→AMX when no direct kernel exists.
            let directGEMM = resolveDirectQuantizedGEMM(for: quantizationDescriptor.schemeIdentifier)
            let useDirectQuantizedGEMM = directGEMM.flatMap {
                planBuildContext.pipelineCache[$0.kernelName]
            } != nil

            let canDequantForAMX = quantizationDescriptor.schemeIdentifier.isWeightQuantized
                && buffers.dequantScratch != nil
                && dequantKernelName(for: quantizationDescriptor.schemeIdentifier) != nil
            let usesMPPForStep = usesMPP
                && mode == .batch
                && inputRowStride == projection.inputDimension
                && (!quantizationDescriptor.schemeIdentifier.isWeightQuantized || canDequantForAMX)

            // Emit dequant step (skipped when direct quantized GEMM is active)
            var dequantSteps: [MetalPrefillStep] = []
            if !useDirectQuantizedGEMM, canDequantForAMX && usesMPPForStep,
               let dequantName = dequantKernelName(for: quantizationDescriptor.schemeIdentifier),
               let dequantPipeline = planBuildContext.pipelineCache[dequantName],
               let dequantScratch = buffers.dequantScratch {
                dequantSteps.append(
                    MetalPrefillStep(
                        pipeline: dequantPipeline,
                        gridSize: MTLSize(width: projection.outputDimension, height: 1, depth: 1),
                        threadgroupSize: MTLSize(width: 256, height: 1, depth: 1),
                        bufferBindings: [
                            (0, weightBuffer, weightOffset),
                            (1, dequantScratch, 0),
                        ],
                        bytesBindings: [
                            uint32Binding(2, UInt32(projection.inputDimension)),
                            uint32Binding(3, UInt32(projection.outputDimension)),
                        ],
                        threadgroupMemoryLength: 0,
                        sync: .bufferBarrier,
                        mode: .batch,
                        sequenceLengthPolicy: .none,
                        positionBufferIndex: nil,
                        perPositionStrides: [:],
                        metadata: .init(
                            kernelName: dequantName,
                            entryIndex: entry.index,
                            weightTensorName: weightTensorName,
                            bufferAccessPattern: .init(reads: [0], writes: [1])
                        )
                    )
                )
            }

            // Resolve GEMM pipeline
            let selectedPipeline: MTLComputePipelineState
            let selectedKernelName: String
            if useDirectQuantizedGEMM,
               let resolvedGEMM = directGEMM,
               let directPipeline = planBuildContext.pipelineCache[resolvedGEMM.kernelName] {
                selectedPipeline = directPipeline
                selectedKernelName = resolvedGEMM.kernelName
            } else if canDequantForAMX && usesMPPForStep,
               let mppPipeline = planBuildContext.pipelineCache["gemm_bf16_f32s"] {
                selectedPipeline = mppPipeline
                selectedKernelName = "gemm_bf16_f32s"
            } else if !usesMPPForStep,
               let naivePipeline = planBuildContext.pipelineCache["naive::\(resolved.name)"] {
                selectedPipeline = naivePipeline
                selectedKernelName = "naive::\(resolved.name)"
            } else {
                selectedPipeline = resolved.pipeline
                selectedKernelName = resolved.name
            }

            // GEMM weight source: original packed weights (direct) or dequant scratch (BF16)
            let gemmWeightBuffer: MTLBuffer
            let gemmWeightOffset: Int
            if useDirectQuantizedGEMM {
                gemmWeightBuffer = weightBuffer
                gemmWeightOffset = weightOffset
            } else if canDequantForAMX && usesMPPForStep, let dequantScratch = buffers.dequantScratch {
                gemmWeightBuffer = dequantScratch
                gemmWeightOffset = 0
            } else {
                gemmWeightBuffer = weightBuffer
                gemmWeightOffset = weightOffset
            }

            let gridSize: MTLSize
            let threadgroupSize: MTLSize
            if usesMPPForStep && !useDirectQuantizedGEMM {
                let simdWidth = selectedPipeline.threadExecutionWidth
                gridSize = MTLSize(
                    width: (projection.outputDimension + 31) / 32,
                    height: (maximumSequenceLength + 63) / 64,
                    depth: 1
                )
                threadgroupSize = MTLSize(width: simdWidth * 4, height: 1, depth: 1)
            } else if mode == .batch {
                let simdWidth = max(selectedPipeline.threadExecutionWidth, 1)
                let rowsPerThreadgroup = 2
                let threads = min(
                    simdWidth * rowsPerThreadgroup,
                    selectedPipeline.maxTotalThreadsPerThreadgroup
                )
                gridSize = MTLSize(
                    width: (projection.outputDimension + rowsPerThreadgroup - 1) / rowsPerThreadgroup,
                    height: maximumSequenceLength,
                    depth: 1
                )
                threadgroupSize = MTLSize(width: threads, height: 1, depth: 1)
            } else if mode == .lastToken {
                gridSize = MTLSize(width: resolved.config.grid.width, height: 1, depth: 1)
                threadgroupSize = resolved.config.threadgroup
            } else {
                gridSize = MTLSize(
                    width: resolved.config.grid.width,
                    height: maximumSequenceLength,
                    depth: 1
                )
                threadgroupSize = resolved.config.threadgroup
            }

            // GEMM: reads input[0] + weight[1], writes output[2]
            let gemmPattern = MetalDispatchStepMetadata.BufferAccessPattern(reads: [0, 1], writes: [2])
            recordProjectionQuantization(
                entry: entry,
                descriptor: quantizationDescriptor,
                mode: mode,
                inputRowStride: inputRowStride,
                inputDimension: projection.inputDimension,
                selectedKernelName: selectedKernelName,
                usesMPPForStep: usesMPPForStep
            )
            return dequantSteps + [MetalPrefillStep(
                pipeline: selectedPipeline,
                gridSize: gridSize,
                threadgroupSize: threadgroupSize,
                bufferBindings: [
                    (0, inputBuffer, inputOffset),
                    (1, gemmWeightBuffer, gemmWeightOffset),
                    (2, outputBuffer, outputOffset),
                ],
                bytesBindings: [
                    uint32Binding(3, UInt32(projection.inputDimension)),
                    uint32Binding(4, UInt32(projection.outputDimension)),
                    uint32Binding(5, seqLenValue),
                    uint32Binding(6, UInt32(inputRowStride)),
                ],
                threadgroupMemoryLength: useDirectQuantizedGEMM
                    ? (directGEMM?.threadgroupMemoryLength ?? 0)
                    : (usesMPPForStep ? 0 : resolved.config.sharedMemoryBytes),
                sync: .bufferBarrier,
                mode: mode,
                sequenceLengthPolicy: mode == .batch
                    ? (usesMPPForStep && !useDirectQuantizedGEMM
                        ? .bindAndAdjustGridHeightTiled(index: 5, tileHeight: 64)
                        : .bindAndAdjustGridHeight(index: 5))
                    : .none,
                positionBufferIndex: nil,
                perPositionStrides: perPositionStrides,
                metadata: .init(
                    kernelName: selectedKernelName,
                    entryIndex: entry.index,
                    weightTensorName: weightTensorName,
                    bufferAccessPattern: gemmPattern
                )
            )]

        case .structuralCopy(let dimension):
            let resolved = try resolveDispatch(entry)
            routingState.projectionIndex = 0
            routingState.currentInputOffset = 0

            // copy: reads source[0], writes destination[1]
            let copyPattern = MetalDispatchStepMetadata.BufferAccessPattern(reads: [0], writes: [1])
            return [MetalPrefillStep(
                pipeline: resolved.pipeline,
                gridSize: MTLSize(width: resolved.config.grid.width, height: maximumSequenceLength, depth: 1),
                threadgroupSize: resolved.config.threadgroup,
                bufferBindings: [
                    (0, buffers.hidden, 0),
                    (1, buffers.residual, 0),
                ],
                bytesBindings: [
                    uint32Binding(2, UInt32(dimension)),
                    uint32Binding(3, UInt32(maximumSequenceLength)),
                ],
                threadgroupMemoryLength: 0,
                sync: .bufferBarrier,
                mode: .batch,
                sequenceLengthPolicy: .bindAndAdjustGridHeight(index: 3),
                positionBufferIndex: nil,
                perPositionStrides: [:],
                metadata: .init(entryIndex: entry.index, bufferAccessPattern: copyPattern)
            )]

        case .structuralAdd(let dimension):
            let resolved = try resolveDispatch(entry)
            let inputBuffer: MTLBuffer
            let inputOffset: Int
            if routingState.lastOutputIsHidden {
                inputBuffer = buffers.hidden
                inputOffset = 0
            } else {
                inputBuffer = buffers.scratch
                inputOffset = routingState.currentInputOffset
            }
            routingState.lastOutputIsHidden = true
            routingState.currentInputOffset = 0

            if inputBuffer === buffers.hidden, inputOffset == 0 {
                guard let inplacePipeline = planBuildContext.pipelineCache["residual_add_inplace_seq_f32"] else {
                    throw MetalCompilerError.kernelNotFound("residual_add_inplace_seq_f32")
                }
                let addPattern = MetalDispatchStepMetadata.BufferAccessPattern(reads: [0, 1], writes: [0])
                return [MetalPrefillStep(
                    pipeline: inplacePipeline,
                    gridSize: MTLSize(width: resolved.config.grid.width, height: maximumSequenceLength, depth: 1),
                    threadgroupSize: resolved.config.threadgroup,
                    bufferBindings: [
                        (0, buffers.hidden, 0),
                        (1, buffers.residual, 0),
                    ],
                    bytesBindings: [
                        uint32Binding(2, UInt32(dimension)),
                        uint32Binding(3, UInt32(maximumSequenceLength)),
                    ],
                    threadgroupMemoryLength: 0,
                    sync: .bufferBarrier,
                    mode: .batch,
                    sequenceLengthPolicy: .bindAndAdjustGridHeight(index: 3),
                    positionBufferIndex: nil,
                    perPositionStrides: [:],
                    metadata: .init(
                        kernelName: "residual_add_inplace_seq_f32",
                        entryIndex: entry.index,
                        bufferAccessPattern: addPattern
                    )
                )]
            }

            // add: reads operands[0,1], writes result[2]
            let addPattern = MetalDispatchStepMetadata.BufferAccessPattern(reads: [0, 1], writes: [2])
            return [MetalPrefillStep(
                pipeline: resolved.pipeline,
                gridSize: MTLSize(width: resolved.config.grid.width, height: maximumSequenceLength, depth: 1),
                threadgroupSize: resolved.config.threadgroup,
                bufferBindings: [
                    (0, inputBuffer, inputOffset),
                    (1, buffers.residual, 0),
                    (2, buffers.hidden, 0),
                ],
                bytesBindings: [
                    uint32Binding(3, UInt32(dimension)),
                    uint32Binding(4, UInt32(maximumSequenceLength)),
                ],
                threadgroupMemoryLength: 0,
                sync: .bufferBarrier,
                mode: .batch,
                sequenceLengthPolicy: .bindAndAdjustGridHeight(index: 4),
                positionBufferIndex: nil,
                perPositionStrides: [:],
                metadata: .init(entryIndex: entry.index, bufferAccessPattern: addPattern)
            )]
        }
    }

    private mutating func updateCompositeInputSource(for entry: DispatchEntry) {
        guard activeCompositeID != entry.compositeID else { return }
        activeCompositeID = entry.compositeID
        refreshCompositeInputSource()
    }

    private mutating func refreshCompositeInputSource() {
        if routingState.lastOutputIsHidden {
            compositeInputSource = (buffers.hidden, 0)
        } else {
            compositeInputSource = (buffers.scratch, routingState.currentInputOffset)
        }
    }

    private func buildNormToHiddenStep(
        inputBuffer: MTLBuffer,
        inputOffset: Int,
        dimension: Int,
        epsilon: Float,
        weightRole: String,
        weightBias: Float,
        entry: DispatchEntry
    ) throws -> [MetalPrefillStep] {
        let weightResolver = WeightResolver(
            entry: entry,
            stafWeightStore: stafWeightStore,
            fallbackBuffer: buffers.hidden,
            fallbackWeightFormat: fallbackWeightFormat,
            minimumFallbackLength: minimumFallbackLength,
            logsMisses: false,
            executionPhase: .prefill,
            accessPolicyResolver: planBuildContext.compileContext.accessPolicyResolver
        )

        let normKernelName = Reduction(
            dimension: dimension,
            epsilon: epsilon,
            weightRole: weightRole,
            weightBias: weightBias
        )
            .kernelName(context: planBuildContext.kernelContext)
        guard let pipeline = planBuildContext.pipelineCache[normKernelName] else {
            throw MetalCompilerError.kernelNotFound(normKernelName)
        }
        let simdWidth = pipeline.threadExecutionWidth
        let clamped = min(max(dimension, 1), 1024)
        let rounded = ((clamped + simdWidth - 1) / simdWidth) * simdWidth
        let threads = min(rounded, pipeline.maxTotalThreadsPerThreadgroup)

        let (weightBuffer, weightOffset) = weightResolver.resolve(role: weightRole)

        // norm: reads input[0] + weight[1], writes output[2]
        let normPattern = MetalDispatchStepMetadata.BufferAccessPattern(reads: [0, 1], writes: [2])
        return [MetalPrefillStep(
            pipeline: pipeline,
            gridSize: MTLSize(width: maximumSequenceLength, height: 1, depth: 1),
            threadgroupSize: MTLSize(width: threads, height: 1, depth: 1),
            bufferBindings: [
                (0, inputBuffer, inputOffset),
                (1, weightBuffer, weightOffset),
                (2, buffers.hidden, 0),
            ],
            bytesBindings: [
                uint32Binding(3, UInt32(dimension)),
                floatBinding(4, epsilon),
                floatBinding(5, weightBias),
                uint32Binding(6, UInt32(maximumSequenceLength)),
            ],
            threadgroupMemoryLength: 0,
            sync: .bufferBarrier,
            mode: .batch,
            sequenceLengthPolicy: .bind(index: 6),
            positionBufferIndex: nil,
            perPositionStrides: [:],
            metadata: .init(
                entryIndex: entry.index,
                weightTensorName: entry.parameterBindings.first(where: { $0.role == weightRole })?.tensorName,
                bufferAccessPattern: normPattern
            )
        )]
    }

    private func makeHiddenToResidualCopyStep(
        dimension: Int,
        entry: DispatchEntry
    ) throws -> MetalPrefillStep {
        let resolved = try resolveDispatch(
            DispatchEntry(
                index: entry.index,
                kind: .structuralCopy(dimension: dimension),
                parameterBindings: [],
                layerIndex: entry.layerIndex
            )
        )
        let copyPattern = MetalDispatchStepMetadata.BufferAccessPattern(reads: [0], writes: [1])
        return MetalPrefillStep(
            pipeline: resolved.pipeline,
            gridSize: MTLSize(width: resolved.config.grid.width, height: maximumSequenceLength, depth: 1),
            threadgroupSize: resolved.config.threadgroup,
            bufferBindings: [
                (0, buffers.hidden, 0),
                (1, buffers.residual, 0),
            ],
            bytesBindings: [
                uint32Binding(2, UInt32(dimension)),
                uint32Binding(3, UInt32(maximumSequenceLength)),
            ],
            threadgroupMemoryLength: 0,
            sync: .bufferBarrier,
            mode: .batch,
            sequenceLengthPolicy: .bindAndAdjustGridHeight(index: 3),
            positionBufferIndex: nil,
            perPositionStrides: [:],
            metadata: .init(entryIndex: entry.index, bufferAccessPattern: copyPattern)
        )
    }

    private func shouldCaptureResidualInput(for weightRole: String) -> Bool {
        switch weightRole {
        case "input_layernorm", "pre_feedforward_layernorm", "operator_norm":
            return true
        default:
            return false
        }
    }

    // MARK: - Fused Prefill Steps (Single-Dispatch)

    /// Resolve the sequence-fused kernel name for a given fused operation type.
    private func sequenceFusedKernelName(prefix: String) -> String {
        switch fallbackWeightFormat {
        case .bfloat16:
            return "\(prefix)_seq_bf16_f32"
        default:
            return "\(prefix)_seq_f32"
        }
    }

    /// Resolve norm weight buffer and compute threadgroup size for fused kernels.
    private func resolveFusedNormParameters(
        dimension: Int,
        entry: DispatchEntry
    ) -> (weightBuffer: MTLBuffer, weightOffset: Int, threads: Int)? {
        let weightResolver = WeightResolver(
            entry: entry,
            stafWeightStore: stafWeightStore,
            fallbackBuffer: buffers.hidden,
            fallbackWeightFormat: fallbackWeightFormat,
            minimumFallbackLength: minimumFallbackLength,
            logsMisses: false,
            executionPhase: .prefill,
            accessPolicyResolver: planBuildContext.compileContext.accessPolicyResolver
        )
        let (weightBuffer, weightOffset) = weightResolver.resolve(role: "scale")
        let simdWidth = 32
        let clamped = min(max(dimension, 1), 1024)
        let rounded = ((clamped + simdWidth - 1) / simdWidth) * simdWidth
        let threads = min(rounded, 1024)
        return (weightBuffer, weightOffset, threads)
    }

    /// Build a single fused copy + RMSNorm step for prefill.
    /// Returns nil if the fused kernel is not available in the pipeline cache.
    private func makeFusedCopyNormStep(
        fusedOp: FusedCopyNorm,
        entry: DispatchEntry
    ) throws -> MetalPrefillStep? {
        let kernelName = sequenceFusedKernelName(prefix: "fused_copy_rms_norm")
        guard let pipeline = planBuildContext.pipelineCache[kernelName],
              let params = resolveFusedNormParameters(dimension: fusedOp.dimension, entry: entry)
        else { return nil }

        // Reads hidden[0] + weight[2], writes hidden[0] + residual[1]
        let pattern = MetalDispatchStepMetadata.BufferAccessPattern(reads: [0, 2], writes: [0, 1])
        return MetalPrefillStep(
            pipeline: pipeline,
            gridSize: MTLSize(width: maximumSequenceLength, height: 1, depth: 1),
            threadgroupSize: MTLSize(width: params.threads, height: 1, depth: 1),
            bufferBindings: [
                (0, buffers.hidden, 0),
                (1, buffers.residual, 0),
                (2, params.weightBuffer, params.weightOffset),
            ],
            bytesBindings: [
                uint32Binding(3, UInt32(fusedOp.dimension)),
                floatBinding(4, fusedOp.epsilon),
                uint32Binding(5, UInt32(maximumSequenceLength)),
                floatBinding(6, fusedOp.weightBias),
            ],
            threadgroupMemoryLength: 0,
            sync: .bufferBarrier,
            mode: .batch,
            sequenceLengthPolicy: .bind(index: 5),
            positionBufferIndex: nil,
            perPositionStrides: [:],
            metadata: .init(
                kernelName: kernelName,
                entryIndex: entry.index,
                weightTensorName: entry.parameterBindings.first(where: { $0.role == "scale" })?.tensorName,
                bufferAccessPattern: pattern
            )
        )
    }

    /// Build a single fused residualAdd + copy + RMSNorm step for prefill.
    /// When `fusedOp.preNorm` is non-nil, uses the 4→1 preNorm variant.
    /// Returns nil if the fused kernel is not available in the pipeline cache.
    private func makeFusedResidualAddCopyNormStep(
        fusedOp: FusedResidualAddCopyNorm,
        entry: DispatchEntry
    ) throws -> MetalPrefillStep? {
        if let preNorm = fusedOp.preNorm {
            return try makeFusedPreNormResidualAddCopyNormStep(
                fusedOp: fusedOp, preNorm: preNorm, entry: entry)
        }

        let kernelName = sequenceFusedKernelName(prefix: "fused_residual_add_copy_rms_norm")
        guard let pipeline = planBuildContext.pipelineCache[kernelName],
              let params = resolveFusedNormParameters(dimension: fusedOp.dimension, entry: entry)
        else { return nil }

        // Reads hidden[0] + residual[1] + weight[2], writes hidden[0] + residual[1]
        let pattern = MetalDispatchStepMetadata.BufferAccessPattern(reads: [0, 1, 2], writes: [0, 1])
        return MetalPrefillStep(
            pipeline: pipeline,
            gridSize: MTLSize(width: maximumSequenceLength, height: 1, depth: 1),
            threadgroupSize: MTLSize(width: params.threads, height: 1, depth: 1),
            bufferBindings: [
                (0, buffers.hidden, 0),
                (1, buffers.residual, 0),
                (2, params.weightBuffer, params.weightOffset),
            ],
            bytesBindings: [
                uint32Binding(3, UInt32(fusedOp.dimension)),
                floatBinding(4, fusedOp.epsilon),
                uint32Binding(5, UInt32(maximumSequenceLength)),
                floatBinding(6, fusedOp.weightBias),
            ],
            threadgroupMemoryLength: 0,
            sync: .bufferBarrier,
            mode: .batch,
            sequenceLengthPolicy: .bind(index: 5),
            positionBufferIndex: nil,
            perPositionStrides: [:],
            metadata: .init(
                kernelName: kernelName,
                entryIndex: entry.index,
                weightTensorName: entry.parameterBindings.first(where: { $0.role == "scale" })?.tensorName,
                bufferAccessPattern: pattern
            )
        )
    }

    /// Build a fused preNorm + residualAdd + copy + outputNorm step (4→1).
    ///
    /// Buffer layout: [0]=hidden, [1]=residual, [2]=preNormWeight, [3]=outputNormWeight
    /// Bytes: [4]=dimension, [5]=preNormEpsilon, [6]=seqLen, [7]=preNormWeightBias,
    ///        [8]=outputEpsilon, [9]=outputWeightBias
    private func makeFusedPreNormResidualAddCopyNormStep(
        fusedOp: FusedResidualAddCopyNorm,
        preNorm: FusedResidualAddCopyNorm.PreNorm,
        entry: DispatchEntry
    ) throws -> MetalPrefillStep? {
        let kernelName = sequenceFusedKernelName(prefix: "fused_pre_norm_residual_add_copy_rms_norm")
        guard let pipeline = planBuildContext.pipelineCache[kernelName] else { return nil }

        // Resolve preNorm weight from preNorm's parameter bindings
        let preNormWeightResolver = WeightResolver(
            entry: DispatchEntry(
                index: entry.index,
                kind: entry.kind,
                parameterBindings: preNorm.parameterBindings,
                layerIndex: entry.layerIndex),
            stafWeightStore: stafWeightStore,
            fallbackBuffer: buffers.hidden,
            fallbackWeightFormat: fallbackWeightFormat,
            minimumFallbackLength: minimumFallbackLength,
            logsMisses: false,
            executionPhase: .prefill,
            accessPolicyResolver: planBuildContext.compileContext.accessPolicyResolver
        )
        let (preNormWeightBuffer, preNormWeightOffset) = preNormWeightResolver.resolve(role: "scale")

        // Resolve output norm weight from the entry's parameter bindings
        guard let outputParams = resolveFusedNormParameters(dimension: fusedOp.dimension, entry: entry)
        else { return nil }

        let simdWidth = 32
        let clamped = min(max(fusedOp.dimension, 1), 1024)
        let rounded = ((clamped + simdWidth - 1) / simdWidth) * simdWidth
        let threads = min(rounded, 1024)

        // Reads hidden[0] + residual[1] + preNormWeight[2] + outputWeight[3],
        // writes hidden[0] + residual[1]
        let pattern = MetalDispatchStepMetadata.BufferAccessPattern(reads: [0, 1, 2, 3], writes: [0, 1])
        return MetalPrefillStep(
            pipeline: pipeline,
            gridSize: MTLSize(width: maximumSequenceLength, height: 1, depth: 1),
            threadgroupSize: MTLSize(width: threads, height: 1, depth: 1),
            bufferBindings: [
                (0, buffers.hidden, 0),
                (1, buffers.residual, 0),
                (2, preNormWeightBuffer, preNormWeightOffset),
                (3, outputParams.weightBuffer, outputParams.weightOffset),
            ],
            bytesBindings: [
                uint32Binding(4, UInt32(fusedOp.dimension)),
                floatBinding(5, preNorm.epsilon),
                uint32Binding(6, UInt32(maximumSequenceLength)),
                floatBinding(7, preNorm.weightBias),
                floatBinding(8, fusedOp.epsilon),
                floatBinding(9, fusedOp.weightBias),
            ],
            threadgroupMemoryLength: 0,
            sync: .bufferBarrier,
            mode: .batch,
            sequenceLengthPolicy: .bind(index: 6),
            positionBufferIndex: nil,
            perPositionStrides: [:],
            metadata: .init(
                kernelName: kernelName,
                entryIndex: entry.index,
                weightTensorName: entry.parameterBindings.first(where: { $0.role == "scale" })?.tensorName,
                bufferAccessPattern: pattern
            )
        )
    }

    /// Build a single fused residualAdd + RMSNorm step for prefill (no copy to residual).
    /// When `fusedOp.preNorm` is non-nil, uses the 3→1 preNorm variant (no copy).
    /// Returns nil if the fused kernel is not available in the pipeline cache.
    private func makeFusedResidualAddNormStep(
        fusedOp: FusedResidualAddNorm,
        entry: DispatchEntry
    ) throws -> MetalPrefillStep? {
        if let preNorm = fusedOp.preNorm {
            return try makeFusedPreNormResidualAddNormStep(
                fusedOp: fusedOp, preNorm: preNorm, entry: entry)
        }

        let kernelName = sequenceFusedKernelName(prefix: "fused_residual_add_rms_norm")
        guard let pipeline = planBuildContext.pipelineCache[kernelName],
              let params = resolveFusedNormParameters(dimension: fusedOp.dimension, entry: entry)
        else { return nil }

        // Reads hidden[0] + residual[1] + weight[2], writes hidden[0]
        let pattern = MetalDispatchStepMetadata.BufferAccessPattern(reads: [0, 1, 2], writes: [0])
        return MetalPrefillStep(
            pipeline: pipeline,
            gridSize: MTLSize(width: maximumSequenceLength, height: 1, depth: 1),
            threadgroupSize: MTLSize(width: params.threads, height: 1, depth: 1),
            bufferBindings: [
                (0, buffers.hidden, 0),
                (1, buffers.residual, 0),
                (2, params.weightBuffer, params.weightOffset),
            ],
            bytesBindings: [
                uint32Binding(3, UInt32(fusedOp.dimension)),
                floatBinding(4, fusedOp.epsilon),
                uint32Binding(5, UInt32(maximumSequenceLength)),
                floatBinding(6, fusedOp.weightBias),
            ],
            threadgroupMemoryLength: 0,
            sync: .bufferBarrier,
            mode: .batch,
            sequenceLengthPolicy: .bind(index: 5),
            positionBufferIndex: nil,
            perPositionStrides: [:],
            metadata: .init(
                kernelName: kernelName,
                entryIndex: entry.index,
                weightTensorName: entry.parameterBindings.first(where: { $0.role == "scale" })?.tensorName,
                bufferAccessPattern: pattern
            )
        )
    }

    /// Build a fused preNorm + residualAdd + outputNorm step (3→1, no copy).
    ///
    /// Buffer layout: [0]=hidden, [1]=residual, [2]=preNormWeight, [3]=outputNormWeight
    /// Bytes: [4]=dimension, [5]=preNormEpsilon, [6]=seqLen, [7]=preNormWeightBias,
    ///        [8]=outputEpsilon, [9]=outputWeightBias
    private func makeFusedPreNormResidualAddNormStep(
        fusedOp: FusedResidualAddNorm,
        preNorm: FusedResidualAddCopyNorm.PreNorm,
        entry: DispatchEntry
    ) throws -> MetalPrefillStep? {
        let kernelName = sequenceFusedKernelName(prefix: "fused_pre_norm_residual_add_rms_norm")
        guard let pipeline = planBuildContext.pipelineCache[kernelName] else { return nil }

        // Resolve preNorm weight from preNorm's parameter bindings
        let preNormWeightResolver = WeightResolver(
            entry: DispatchEntry(
                index: entry.index,
                kind: entry.kind,
                parameterBindings: preNorm.parameterBindings,
                layerIndex: entry.layerIndex),
            stafWeightStore: stafWeightStore,
            fallbackBuffer: buffers.hidden,
            fallbackWeightFormat: fallbackWeightFormat,
            minimumFallbackLength: minimumFallbackLength,
            logsMisses: false,
            executionPhase: .prefill,
            accessPolicyResolver: planBuildContext.compileContext.accessPolicyResolver
        )
        let (preNormWeightBuffer, preNormWeightOffset) = preNormWeightResolver.resolve(role: "scale")

        // Resolve output norm weight from the entry's parameter bindings
        guard let outputParams = resolveFusedNormParameters(dimension: fusedOp.dimension, entry: entry)
        else { return nil }

        let simdWidth = 32
        let clamped = min(max(fusedOp.dimension, 1), 1024)
        let rounded = ((clamped + simdWidth - 1) / simdWidth) * simdWidth
        let threads = min(rounded, 1024)

        // Reads hidden[0] + residual[1] + preNormWeight[2] + outputWeight[3],
        // writes hidden[0]
        let pattern = MetalDispatchStepMetadata.BufferAccessPattern(reads: [0, 1, 2, 3], writes: [0])
        return MetalPrefillStep(
            pipeline: pipeline,
            gridSize: MTLSize(width: maximumSequenceLength, height: 1, depth: 1),
            threadgroupSize: MTLSize(width: threads, height: 1, depth: 1),
            bufferBindings: [
                (0, buffers.hidden, 0),
                (1, buffers.residual, 0),
                (2, preNormWeightBuffer, preNormWeightOffset),
                (3, outputParams.weightBuffer, outputParams.weightOffset),
            ],
            bytesBindings: [
                uint32Binding(4, UInt32(fusedOp.dimension)),
                floatBinding(5, preNorm.epsilon),
                uint32Binding(6, UInt32(maximumSequenceLength)),
                floatBinding(7, preNorm.weightBias),
                floatBinding(8, fusedOp.epsilon),
                floatBinding(9, fusedOp.weightBias),
            ],
            threadgroupMemoryLength: 0,
            sync: .bufferBarrier,
            mode: .batch,
            sequenceLengthPolicy: .bind(index: 6),
            positionBufferIndex: nil,
            perPositionStrides: [:],
            metadata: .init(
                kernelName: kernelName,
                entryIndex: entry.index,
                weightTensorName: entry.parameterBindings.first(where: { $0.role == "scale" })?.tensorName,
                bufferAccessPattern: pattern
            )
        )
    }

    /// Build a single batched QK norm step for prefill.
    ///
    /// Merges Q and K per-head RMS norm into a single dispatch.
    /// Grid: (qHeadCount + kHeadCount, sequenceLength, 1)
    ///
    /// Buffer layout: [0]=qData, [1]=kData, [2]=qWeight, [3]=kWeight
    /// Bytes: [4]=qHeadCount, [5]=kHeadCount, [6]=headDimension, [7]=epsilon,
    ///        [8]=weightBias, [9]=sequenceLength, [10]=qTotalDim, [11]=kTotalDim
    private func makeBatchedQKNormStep(
        qNorm: QKNormFragment,
        kNorm: QKNormFragment,
        entry: DispatchEntry
    ) throws -> MetalPrefillStep? {
        let kernelName = fallbackWeightFormat == .bfloat16
            ? "batched_qk_rms_norm_seq_bf16_f32"
            : "batched_qk_rms_norm_seq_f32"
        guard let pipeline = planBuildContext.pipelineCache[kernelName] else { return nil }

        let scratchSlotSize = slotDimension * scratchElementSize * maximumSequenceLength

        // Resolve Q and K weights
        let qWeightResolver = WeightResolver(
            entry: entry,
            stafWeightStore: stafWeightStore,
            fallbackBuffer: buffers.hidden,
            fallbackWeightFormat: fallbackWeightFormat,
            minimumFallbackLength: minimumFallbackLength,
            logsMisses: false,
            executionPhase: .prefill,
            accessPolicyResolver: planBuildContext.compileContext.accessPolicyResolver
        )
        let (qWeightBuffer, qWeightOffset) = qWeightResolver.resolve(role: qNorm.weightRole)
        let (kWeightBuffer, kWeightOffset) = qWeightResolver.resolve(role: kNorm.weightRole)

        let totalHeadCount = qNorm.headCount + kNorm.headCount
        let threads = min(32, pipeline.maxTotalThreadsPerThreadgroup)

        let qTotalDimension = qNorm.headCount * qNorm.headDimension
        let kTotalDimension = kNorm.headCount * kNorm.headDimension

        // Reads qData[0] + kData[1] + qWeight[2] + kWeight[3], writes qData[0] + kData[1]
        let pattern = MetalDispatchStepMetadata.BufferAccessPattern(reads: [0, 1, 2, 3], writes: [0, 1])
        return MetalPrefillStep(
            pipeline: pipeline,
            gridSize: MTLSize(width: totalHeadCount, height: maximumSequenceLength, depth: 1),
            threadgroupSize: MTLSize(width: threads, height: 1, depth: 1),
            bufferBindings: [
                (0, buffers.scratch, qNorm.scratchSlotIndex * scratchSlotSize),
                (1, buffers.scratch, kNorm.scratchSlotIndex * scratchSlotSize),
                (2, qWeightBuffer, qWeightOffset),
                (3, kWeightBuffer, kWeightOffset),
            ],
            bytesBindings: [
                uint32Binding(4, UInt32(qNorm.headCount)),
                uint32Binding(5, UInt32(kNorm.headCount)),
                uint32Binding(6, UInt32(qNorm.headDimension)),
                floatBinding(7, qNorm.epsilon),
                floatBinding(8, qNorm.weightBias),
                uint32Binding(9, UInt32(maximumSequenceLength)),
                uint32Binding(10, UInt32(qTotalDimension)),
                uint32Binding(11, UInt32(kTotalDimension)),
            ],
            threadgroupMemoryLength: 0,
            sync: .bufferBarrier,
            mode: .batch,
            sequenceLengthPolicy: .bindAndAdjustGridHeight(index: 9),
            positionBufferIndex: nil,
            perPositionStrides: [:],
            metadata: .init(
                kernelName: kernelName,
                entryIndex: entry.index,
                weightTensorName: entry.parameterBindings.first(where: { $0.role == qNorm.weightRole })?.tensorName,
                bufferAccessPattern: pattern
            )
        )
    }

    func finalHiddenSource() -> (buffer: MTLBuffer, offset: Int, rowStride: Int) {
        if let outputHeadInputSource {
            return outputHeadInputSource
        }
        if routingState.lastOutputIsHidden {
            let rowStride = buffers.hidden.length / max(maximumSequenceLength, 1)
            return (buffers.hidden, 0, rowStride)
        }
        // Scratch is laid out using the slot dimension for every token row.
        // The hidden vector may occupy only a prefix of that row, but per-token
        // addressing must still advance by the full slot stride.
        let rowStride = slotDimension * scratchElementSize
        return (buffers.scratch, routingState.currentInputOffset, rowStride)
    }

    mutating func makeQuantizationPlan() -> MetalQuantizationPlan {
        MetalQuantizationPlan(
            capabilities: planBuildContext.quantizationCapabilities,
            entries: quantizationEntries
        )
    }

    private mutating func recordProjectionQuantization(
        entry: DispatchEntry,
        descriptor: ProjectionWeightDescriptor,
        mode: PrefillStepMode,
        inputRowStride: Int,
        inputDimension: Int,
        selectedKernelName: String,
        usesMPPForStep: Bool
    ) {
        let fallbackReason = resolveProjectionFallbackReason(
            descriptor: descriptor,
            mode: mode,
            inputRowStride: inputRowStride,
            inputDimension: inputDimension,
            usesMPPForStep: usesMPPForStep
        )
        quantizationEntries.append(
            MetalQuantizationPlanEntry(
                entryIndex: entry.index,
                layerIndex: entry.layerIndex,
                tensorName: descriptor.tensorName,
                path: .prefillProjection,
                schemeIdentifier: descriptor.schemeIdentifier,
                layout: descriptor.layout,
                kernelFamily: .classify(
                    kernelName: selectedKernelName,
                    usesMPP: usesMPPForStep
                ),
                usedFallback: descriptor.usedFallback || fallbackReason != nil,
                fallbackReason: descriptor.fallbackReason ?? fallbackReason
            )
        )
    }

    private func resolveProjectionWeightDescriptor(
        role: String,
        entry: DispatchEntry
    ) -> ProjectionWeightDescriptor {
        guard let binding = entry.parameterBindings.first(where: { $0.role == role }) else {
            return ProjectionWeightDescriptor(
                tensorName: nil,
                schemeIdentifier: fallbackSchemeIdentifier,
                layout: .rowMajor,
                usedFallback: true,
                fallbackReason: .missingTensorBinding
            )
        }
        guard let stafWeightStore else {
            return ProjectionWeightDescriptor(
                tensorName: binding.tensorName,
                schemeIdentifier: fallbackSchemeIdentifier,
                layout: .rowMajor,
                usedFallback: true,
                fallbackReason: .missingWeightStore
            )
        }

        let request = planBuildContext.compileContext.accessPolicyResolver.accessRequest(
            for: entry,
            role: role,
            binding: binding,
            executionPhase: .prefill,
            stafWeightStore: stafWeightStore
        )
        let layout = stafWeightStore.resolvedBufferAccess(for: request)?.layout ?? request.preferredLayout
        guard let tensorEntry = stafWeightStore.entries[binding.tensorName] else {
            return ProjectionWeightDescriptor(
                tensorName: binding.tensorName,
                schemeIdentifier: fallbackSchemeIdentifier,
                layout: layout,
                usedFallback: true,
                fallbackReason: .missingTensorMetadata
            )
        }
        return ProjectionWeightDescriptor(
            tensorName: binding.tensorName,
            schemeIdentifier: tensorEntry.schemeIdentifier,
            layout: layout,
            usedFallback: false,
            fallbackReason: nil
        )
    }

    private func resolveProjectionFallbackReason(
        descriptor: ProjectionWeightDescriptor,
        mode: PrefillStepMode,
        inputRowStride: Int,
        inputDimension: Int,
        usesMPPForStep: Bool
    ) -> MetalQuantizationFallbackReason? {
        if let fallbackReason = descriptor.fallbackReason {
            return fallbackReason
        }
        if mode == .lastToken {
            return .lastTokenProjectionUsesDecodeKernel
        }
        if inputRowStride != inputDimension {
            return .inputStrideMismatch
        }
        guard !descriptor.schemeIdentifier.isWeightQuantized else {
            return nil
        }
        guard !usesMPPForStep else {
            return nil
        }
        switch planBuildContext.quantizationCapabilities.prefillProjectionAcceleration {
        case .disabledByEnvironment:
            return .disabledByEnvironment
        case .unavailable:
            return .unavailableAcceleration
        case .enabled:
            return nil
        }
    }

    private var fallbackSchemeIdentifier: QuantizationSchemeIdentifier {
        switch fallbackWeightFormat {
        case .float16:
            return .fp16RowMajor
        case .bfloat16:
            return .bf16RowMajor
        case .float32:
            return .fp32RowMajor
        case .quantized4Bit(let groupSize):
            switch groupSize {
            case 64:
                return .q4Group64ScaleF16
            case 128:
                return .q4Group128ScaleF16
            default:
                return .passthrough
            }
        case .quantized8Bit(let groupSize):
            switch groupSize {
            case 32:
                return .q8Group32ScaleF16
            case 64:
                return .q8Group64ScaleF16
            case 128:
                return .q8Group128ScaleF16
            default:
                return .passthrough
            }
        }
    }
}

private struct ProjectionWeightDescriptor {
    let tensorName: String?
    let schemeIdentifier: QuantizationSchemeIdentifier
    let layout: STAFWeightLayout
    let usedFallback: Bool
    let fallbackReason: MetalQuantizationFallbackReason?
}

// MARK: - Q4 Dequant → AMX Helpers

/// Dequant kernel name for the given quantization scheme.
/// Returns nil for non-Q4 schemes.
private func dequantKernelName(for scheme: QuantizationSchemeIdentifier) -> String? {
    switch scheme {
    case .q4Group64ScaleF16: return "dequant_q4_g64_bf16"
    case .q4Group128ScaleF16: return "dequant_q4_g128_bf16"
    default: return nil
    }
}

/// Resolved direct quantized GEMM kernel — reads packed weights in registers,
/// bypassing the dequant→AMX two-step pipeline.
private struct DirectQuantizedGEMM {
    let kernelName: String
    let threadgroupMemoryLength: Int
}

/// Resolve the direct quantized GEMM kernel for a single projection.
/// Returns nil when no direct kernel exists for this scheme (falls back to dequant→AMX).
private func resolveDirectQuantizedGEMM(
    for scheme: QuantizationSchemeIdentifier
) -> DirectQuantizedGEMM? {
    switch scheme {
    case .q4Group64ScaleF16:
        return DirectQuantizedGEMM(
            kernelName: "gemv_q4_g64",
            threadgroupMemoryLength: max(64 * 2, 256) * MemoryLayout<Float>.size)
    case .q4Group128ScaleF16:
        return DirectQuantizedGEMM(
            kernelName: "gemv_q4_g128",
            threadgroupMemoryLength: max(128 * 2, 256) * MemoryLayout<Float>.size)
    // Q8: no multi-row GEMM kernel yet (GEMV lacks sequenceLength/inputRowStride).
    // Add cases here when Q8 GEMM kernels are implemented.
    default:
        return nil
    }
}

/// Resolve the batched quantized GEMM kernel for multi-projection dispatch.
/// Returns nil when no batched kernel exists for this scheme.
private func resolveBatchedQuantizedGEMM(
    for scheme: QuantizationSchemeIdentifier,
    count: Int
) -> DirectQuantizedGEMM? {
    switch scheme {
    case .q4Group64ScaleF16:
        return DirectQuantizedGEMM(
            kernelName: "batched_gemm_q4_g64_\(count)",
            threadgroupMemoryLength: max(64 * 2, 256) * MemoryLayout<Float>.size)
    case .q4Group128ScaleF16:
        return DirectQuantizedGEMM(
            kernelName: "batched_gemm_q4_g128_\(count)",
            threadgroupMemoryLength: max(128 * 2, 256) * MemoryLayout<Float>.size)
    default:
        return nil
    }
}
