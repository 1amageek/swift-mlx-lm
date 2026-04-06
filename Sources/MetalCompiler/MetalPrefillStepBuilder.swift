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
        return MetalPrefillPlan(
            steps: optimizedSteps,
            buffers: buffers,
            maximumSequenceLength: maximumSequenceLength,
            stepCount: optimizedSteps.count
        )
    }

    /// Offset-aware buffer region for precise hazard detection.
    /// Distinguishes scratch[0] from scratch[1] on the same MTLBuffer.
    private struct BufferRegion: Hashable {
        let buffer: ObjectIdentifier
        let offset: Int
    }

    /// Eliminate unnecessary memory barriers between prefill steps using
    /// offset-aware buffer region tracking.
    ///
    /// Each step's `metadata.bufferAccessPattern` declares which binding indices are
    /// reads vs writes. Steps without a declared pattern are treated conservatively
    /// (all bindings as both read and written).
    static func optimizePrefillBarrierPolicies(
        _ steps: [MetalPrefillStep]
    ) -> [MetalPrefillStep] {
        var pendingWrites = Set<BufferRegion>()
        return steps.map { step in
            let accesses = resolveBufferRegions(for: step)
            let requiresBarrier = !pendingWrites.isDisjoint(with: accesses.reads.union(accesses.writes))
            let newBarrierPolicy: MetalBarrierPolicy = requiresBarrier ? .bufferBarrier : .none

            if requiresBarrier {
                pendingWrites = accesses.writes
            } else {
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

    /// Convert a step's declared buffer access pattern into concrete buffer regions.
    /// Falls back to treating all bindings as read+written when no pattern is declared.
    private static func resolveBufferRegions(
        for step: MetalPrefillStep
    ) -> (reads: Set<BufferRegion>, writes: Set<BufferRegion>) {
        let buffers = step.bindings.buffers

        func regions(for indices: Set<Int>) -> Set<BufferRegion> {
            Set(buffers.filter { indices.contains($0.index) }
                .map { BufferRegion(buffer: ObjectIdentifier($0.buffer), offset: $0.offset) })
        }

        if let pattern = step.metadata.bufferAccessPattern {
            return (reads: regions(for: pattern.readIndices), writes: regions(for: pattern.writeIndices))
        }

        // Conservative fallback: treat all bindings as both read and written.
        let all = Set(buffers.map {
            BufferRegion(buffer: ObjectIdentifier($0.buffer), offset: $0.offset)
        })
        return (reads: all, writes: all)
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
    let resolveDispatch: (DispatchEntry) throws -> (
        name: String,
        pipeline: MTLComputePipelineState,
        config: (grid: MTLSize, threadgroup: MTLSize, sharedMemoryBytes: Int)
    )
    var kvCacheIndex: Int = 0
    var routingState = BufferRoutingState()

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
        self.resolveDispatch = resolveDispatch
    }

    private func annotate(
        _ steps: [MetalPrefillStep],
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
                    layerIndex: layerIndex
                )
            )
        }
    }

    mutating func buildSteps(for entry: DispatchEntry) throws -> [MetalPrefillStep] {
        let weightResolver = WeightResolver(
            entry: entry,
            stafWeightStore: stafWeightStore,
            fallbackBuffer: buffers.hidden,
            logsMisses: false,
            executionPhase: .prefill,
            accessPolicyResolver: planBuildContext.compileContext.accessPolicyResolver
        )

        switch entry.kind {
        case .fragment(let frag):
            let pipelineCache = planBuildContext.pipelineCache
            let prefillContext = PrefillBindingContext(
                buffers: buffers,
                slotDimension: slotDimension,
                scratchElementSize: scratchElementSize,
                maximumSequenceLength: maximumSequenceLength,
                layerIndex: entry.layerIndex,
                kvCacheIndex: kvCacheIndex,
                convLayerIndex: routingState.convLayerIndex,
                recurrentLayerIndex: routingState.recurrentLayerIndex,
                kernelContext: planBuildContext.kernelContext,
                resolveWeight: weightResolver.resolve,
                getPipeline: { name in
                    guard let pipeline = pipelineCache[name] else {
                        throw MetalCompilerError.kernelNotFound(name)
                    }
                    return pipeline
                }
            )
            let result = try frag.prefillSteps(context: prefillContext)
            if result.resetsProjectionIndex { routingState.projectionIndex = 0 }
            if result.consumesKVCacheLayer { kvCacheIndex += 1 }
            if result.consumesConvLayer { routingState.convLayerIndex += 1 }
            if result.consumesRecurrentLayer { routingState.recurrentLayerIndex += 1 }
            routingState.lastOutputIsHidden = result.outputIsHidden
            return annotate(result.steps, layerIndex: entry.layerIndex)

        case .batchedProjection(let batched):
            var steps: [MetalPrefillStep] = []
            for (i, proj) in batched.projections.enumerated() {
                let singleProjection = MetalProjection(
                    field: proj.field,
                    inputDimension: proj.inputDimension,
                    outputDimension: proj.outputDimension
                )
                let singleEntry = DispatchEntry(
                    index: entry.index + i,
                    kind: .projection(singleProjection, isOutput: false),
                    parameterBindings: entry.parameterBindings,
                    layerIndex: entry.layerIndex
                )
                let projSteps = try buildSteps(for: singleEntry)
                steps.append(contentsOf: projSteps)
            }
            return annotate(steps, layerIndex: entry.layerIndex)

        case .batchedFragment(let batch):
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
            return annotate(steps, layerIndex: entry.layerIndex)

        case .fusedResidualAddNorm(let fusedOp):
            let addEntry = DispatchEntry(
                index: entry.index,
                kind: .structuralAdd(dimension: fusedOp.dimension),
                parameterBindings: [],
                layerIndex: entry.layerIndex
            )
            let normEntry = DispatchEntry(
                index: entry.index + 1,
                kind: .fragment(Reduction(dimension: fusedOp.dimension, epsilon: fusedOp.epsilon)),
                parameterBindings: entry.parameterBindings,
                layerIndex: entry.layerIndex
            )
            var steps: [MetalPrefillStep] = []
            for decomposed in [addEntry, normEntry] {
                let built = try buildSteps(for: decomposed)
                steps.append(contentsOf: built)
            }
            return annotate(steps, layerIndex: entry.layerIndex)

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
            let swigluEntry = DispatchEntry(
                index: entry.index + 1,
                kind: .fragment(ElementwiseFragment(count: fusedOp.outputDimension, kind: .swiglu)),
                parameterBindings: entry.parameterBindings,
                layerIndex: entry.layerIndex
            )

            var steps: [MetalPrefillStep] = []
            for decomposed in [batchedEntry, swigluEntry] {
                let built = try buildSteps(for: decomposed)
                steps.append(contentsOf: built)
            }
            return annotate(steps, layerIndex: entry.layerIndex)

        case .fusedCopyNorm(let fusedOp):
            var steps: [MetalPrefillStep] = []
            let copyEntry = DispatchEntry(
                index: entry.index,
                kind: .structuralCopy(dimension: fusedOp.dimension),
                parameterBindings: [],
                layerIndex: entry.layerIndex
            )
            steps.append(contentsOf: try buildSteps(for: copyEntry))
            steps.append(contentsOf: try buildNormToScratchStep(
                dimension: fusedOp.dimension,
                epsilon: fusedOp.epsilon,
                entry: entry
            ))
            routingState.lastOutputIsHidden = false
            routingState.projectionIndex = 0
            return annotate(steps, layerIndex: entry.layerIndex)

        case .fusedResidualAddCopyNorm(let fusedOp):
            var steps: [MetalPrefillStep] = []
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
            steps.append(contentsOf: try buildNormToScratchStep(
                dimension: fusedOp.dimension,
                epsilon: fusedOp.epsilon,
                entry: entry
            ))
            routingState.lastOutputIsHidden = false
            routingState.projectionIndex = 0
            return annotate(steps, layerIndex: entry.layerIndex)

        case .projection(let projection, let isOutput):
            let resolved = try resolveDispatch(entry)
            let (weightBuffer, weightOffset) = weightResolver.resolve(role: projection.field)

            let inputBuffer: MTLBuffer
            let inputOffset: Int
            if routingState.lastOutputIsHidden {
                inputBuffer = buffers.hidden
                inputOffset = 0
            } else {
                inputBuffer = buffers.scratch
                inputOffset = 0
            }

            let outputBuffer: MTLBuffer
            let outputOffset: Int
            let mode: PrefillStepMode
            let seqLenValue: UInt32
            let scratchSlotSize = slotDimension * scratchElementSize * maximumSequenceLength

            if isOutput && projection.outputDimension > hiddenSize {
                outputBuffer = buffers.logits
                outputOffset = 0
                mode = .lastToken
                seqLenValue = 1
                routingState.lastOutputIsHidden = false
            } else if isOutput {
                outputBuffer = buffers.hidden
                outputOffset = 0
                mode = .batch
                seqLenValue = UInt32(maximumSequenceLength)
                routingState.lastOutputIsHidden = true
            } else {
                let scratchSlot = routingState.projectionIndex + 1
                outputBuffer = buffers.scratch
                outputOffset = scratchSlot * scratchSlotSize
                mode = .batch
                seqLenValue = UInt32(maximumSequenceLength)
                routingState.lastOutputIsHidden = false
            }
            routingState.projectionIndex += 1

            var perPositionStrides: [Int: Int] = [:]
            if mode == .lastToken {
                perPositionStrides[0] = projection.inputDimension * scratchElementSize
            }

            let gridSize: MTLSize
            let threadgroupSize: MTLSize
            if usesMPP && mode == .batch {
                let simdWidth = resolved.pipeline.threadExecutionWidth
                gridSize = MTLSize(
                    width: (projection.outputDimension + 31) / 32,
                    height: (maximumSequenceLength + 63) / 64,
                    depth: 1
                )
                threadgroupSize = MTLSize(width: simdWidth * 4, height: 1, depth: 1)
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
            return [MetalPrefillStep(
                pipeline: resolved.pipeline,
                gridSize: gridSize,
                threadgroupSize: threadgroupSize,
                bufferBindings: [
                    (0, inputBuffer, inputOffset),
                    (1, weightBuffer, weightOffset),
                    (2, outputBuffer, outputOffset),
                ],
                bytesBindings: [
                    uint32Binding(3, UInt32(projection.inputDimension)),
                    uint32Binding(4, UInt32(projection.outputDimension)),
                    uint32Binding(5, seqLenValue),
                ],
                threadgroupMemoryLength: (usesMPP && mode == .batch) ? 0 : resolved.config.sharedMemoryBytes,
                sync: .bufferBarrier,
                mode: mode,
                sequenceLengthPolicy: mode == .batch
                    ? (usesMPP
                        ? .bindAndAdjustGridHeightTiled(index: 5, tileHeight: 64)
                        : .bindAndAdjustGridHeight(index: 5))
                    : .none,
                positionBufferIndex: nil,
                perPositionStrides: perPositionStrides,
                metadata: .init(bufferAccessPattern: gemmPattern)
            )]

        case .structuralCopy(let dimension):
            let resolved = try resolveDispatch(entry)
            routingState.projectionIndex = 0

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
                metadata: .init(bufferAccessPattern: copyPattern)
            )]

        case .structuralAdd(let dimension):
            let resolved = try resolveDispatch(entry)
            routingState.lastOutputIsHidden = true

            // add: reads operands[0,1], writes result[2]
            let addPattern = MetalDispatchStepMetadata.BufferAccessPattern(reads: [0, 1], writes: [2])
            return [MetalPrefillStep(
                pipeline: resolved.pipeline,
                gridSize: MTLSize(width: resolved.config.grid.width, height: maximumSequenceLength, depth: 1),
                threadgroupSize: resolved.config.threadgroup,
                bufferBindings: [
                    (0, buffers.hidden, 0),
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
                metadata: .init(bufferAccessPattern: addPattern)
            )]
        }
    }

    private func buildNormToScratchStep(
        dimension: Int,
        epsilon: Float,
        entry: DispatchEntry
    ) throws -> [MetalPrefillStep] {
        let weightResolver = WeightResolver(
            entry: entry,
            stafWeightStore: stafWeightStore,
            fallbackBuffer: buffers.hidden,
            logsMisses: false,
            executionPhase: .prefill,
            accessPolicyResolver: planBuildContext.compileContext.accessPolicyResolver
        )

        let normKernelName = Reduction(dimension: dimension, epsilon: epsilon)
            .kernelName(context: planBuildContext.kernelContext)
        guard let pipeline = planBuildContext.pipelineCache[normKernelName] else {
            throw MetalCompilerError.kernelNotFound(normKernelName)
        }
        let simdWidth = pipeline.threadExecutionWidth
        let clamped = min(max(dimension, 1), 1024)
        let rounded = ((clamped + simdWidth - 1) / simdWidth) * simdWidth
        let threads = min(rounded, pipeline.maxTotalThreadsPerThreadgroup)

        let (weightBuffer, weightOffset) = weightResolver.resolve(role: "scale")

        // norm: reads input[0] + weight[1], writes output[2]
        let normPattern = MetalDispatchStepMetadata.BufferAccessPattern(reads: [0, 1], writes: [2])
        return [MetalPrefillStep(
            pipeline: pipeline,
            gridSize: MTLSize(width: maximumSequenceLength, height: 1, depth: 1),
            threadgroupSize: MTLSize(width: threads, height: 1, depth: 1),
            bufferBindings: [
                (0, buffers.hidden, 0),
                (1, weightBuffer, weightOffset),
                (2, buffers.scratch, 0),
            ],
            bytesBindings: [
                uint32Binding(3, UInt32(dimension)),
                floatBinding(4, epsilon),
                uint32Binding(5, UInt32(maximumSequenceLength)),
            ],
            threadgroupMemoryLength: 0,
            sync: .bufferBarrier,
            mode: .batch,
            sequenceLengthPolicy: .bindAndAdjustGridHeight(index: 5),
            positionBufferIndex: nil,
            perPositionStrides: [:],
            metadata: .init(bufferAccessPattern: normPattern)
        )]
    }
}
