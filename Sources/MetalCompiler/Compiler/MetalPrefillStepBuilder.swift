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

        let correctedSteps = try Self.insertDecodeEquivalentSequenceStorageRoundingIfNeeded(
            steps,
            buffers: buffers,
            slotDimension: slotDimension,
            hiddenSize: hiddenSize,
            maximumSequenceLength: maximumSequenceLength,
            planBuildContext: planBuildContext
        )
        let residentSteps = try makeResidentConstantSteps(correctedSteps, allocator: constantAllocator)
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

    private static func insertDecodeEquivalentSequenceStorageRoundingIfNeeded(
        _ steps: [MetalPrefillStep],
        buffers: PrefillBufferSet,
        slotDimension: Int,
        hiddenSize: Int,
        maximumSequenceLength: Int,
        planBuildContext: PlanBuildContext
    ) throws -> [MetalPrefillStep] {
        guard buffers.bufferPrecision.isPrefillSequencePrecision else {
            return steps
        }
        let kernelName: String
        switch planBuildContext.compileContext.decodeBufferPrecision {
        case .float16:
            kernelName = "round_f16_seq_f32"
        case .bfloat16:
            kernelName = "round_bf16_seq_f32"
        case .float32, .float32Decode:
            return steps
        }
        guard let pipeline = planBuildContext.pipelineCache[kernelName] else {
            throw MetalCompilerError.kernelNotFound(kernelName)
        }

        var corrected: [MetalPrefillStep] = []
        corrected.reserveCapacity(steps.count * 2)
        for step in steps {
            corrected.append(step)
            corrected.append(contentsOf: try makeSequenceStorageRoundingSteps(
                after: step,
                kernelName: kernelName,
                pipeline: pipeline,
                buffers: buffers,
                slotDimension: slotDimension,
                hiddenSize: hiddenSize,
                maximumSequenceLength: maximumSequenceLength
            ))
        }
        return corrected
    }

    private static func makeSequenceStorageRoundingSteps(
        after step: MetalPrefillStep,
        kernelName: String,
        pipeline: MTLComputePipelineState,
        buffers: PrefillBufferSet,
        slotDimension: Int,
        hiddenSize: Int,
        maximumSequenceLength: Int
    ) throws -> [MetalPrefillStep] {
        if shouldPreserveFloat32SequenceStorage(after: step) {
            return []
        }
        guard let bufferAccessPattern = step.metadata.bufferAccessPattern else {
            if step.bufferBindings.contains(where: {
                isSequenceActivationBuffer($0.buffer, buffers: buffers)
            }) {
                let producer = step.metadata.kernelName ?? step.pipeline.label ?? "<unknown>"
                throw MetalCompilerError.deviceSetupFailed(
                    "Sequence storage rounding requires buffer access metadata for \(producer)"
                )
            }
            return []
        }
        let writeIndices = bufferAccessPattern.writeIndices
        guard !writeIndices.isEmpty else {
            return []
        }

        var roundedRegions = Set<BufferRegion>()
        var roundSteps: [MetalPrefillStep] = []
        for binding in step.bufferBindings where writeIndices.contains(binding.index) {
            guard let elementCount = float16RoundElementCount(
                buffer: binding.buffer,
                offset: binding.offset,
                buffers: buffers,
                slotDimension: slotDimension,
                hiddenSize: hiddenSize,
                maximumSequenceLength: maximumSequenceLength
            ), elementCount > 0 else {
                continue
            }
            let region = BufferRegion(buffer: binding.buffer, offset: binding.offset)
            guard roundedRegions.insert(region).inserted else { continue }

            let threads = min(
                max(pipeline.threadExecutionWidth, 1) * 4,
                pipeline.maxTotalThreadsPerThreadgroup
            )
            let groups = (elementCount + threads - 1) / threads
            roundSteps.append(MetalPrefillStep(
                pipeline: pipeline,
                gridSize: MTLSize(width: max(groups, 1), height: 1, depth: 1),
                threadgroupSize: MTLSize(width: max(threads, 1), height: 1, depth: 1),
                bufferBindings: [(0, binding.buffer, binding.offset)],
                bytesBindings: [uint32Binding(1, UInt32(elementCount))],
                threadgroupMemoryLength: 0,
                sync: .bufferBarrier,
                mode: .batch,
                sequenceLengthPolicy: .none,
                positionBufferIndex: nil,
                perPositionStrides: [:],
                metadata: .init(
                    kernelName: kernelName,
                    entryIndex: step.metadata.entryIndex,
                    layerIndex: step.metadata.layerIndex,
                    bufferAccessPattern: .init(reads: [0], writes: [0])
                )
            ))
        }
        return roundSteps
    }

    private static func shouldPreserveFloat32SequenceStorage(after step: MetalPrefillStep) -> Bool {
        false
    }

    private static func float16RoundElementCount(
        buffer: MTLBuffer,
        offset: Int,
        buffers: PrefillBufferSet,
        slotDimension: Int,
        hiddenSize: Int,
        maximumSequenceLength: Int
    ) -> Int? {
        let availableElements = max(0, (buffer.length - offset) / MemoryLayout<Float>.stride)
        if buffer === buffers.scratch {
            return min(slotDimension * maximumSequenceLength, availableElements)
        }
        if buffer === buffers.hidden || buffer === buffers.residual {
            return min(hiddenSize * maximumSequenceLength, availableElements)
        }
        return nil
    }

    private static func isSequenceActivationBuffer(
        _ buffer: MTLBuffer,
        buffers: PrefillBufferSet
    ) -> Bool {
        buffer === buffers.scratch || buffer === buffers.hidden || buffer === buffers.residual
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

    private var needsDecodeEquivalentSequenceProjectionMath: Bool {
        buffers.bufferPrecision.isPrefillSequencePrecision
            && (buffers.convState != nil || buffers.recurrentState != nil)
    }

    private func decodeEquivalentSequenceGEMVKernelName(
        for descriptor: ProjectionWeightDescriptor
    ) -> String? {
        guard needsDecodeEquivalentSequenceProjectionMath else { return nil }
        switch descriptor.schemeIdentifier {
        case .bf16RowMajor:
            return "gemv_seq_bf16_f32s"
        case .fp16RowMajor:
            return "gemv_seq_f32s"
        default:
            return nil
        }
    }

    private func decodeEquivalentBatchedSequenceGEMVKernelName(
        for descriptor: ProjectionWeightDescriptor,
        count: Int
    ) -> String? {
        guard needsDecodeEquivalentSequenceProjectionMath else { return nil }
        guard count >= 2 && count <= 4 else { return nil }
        switch descriptor.schemeIdentifier {
        case .bf16RowMajor:
            return "batched_gemv\(count)_seq_bf16_f32s"
        case .fp16RowMajor:
            return "batched_gemv\(count)_seq_f32s"
        case .fp32RowMajor:
            return "batched_gemv\(count)_seq_fp32_f32s"
        default:
            return nil
        }
    }

    mutating func buildSteps(for entry: DispatchEntry) throws -> [MetalPrefillStep] {
        updateCompositeInputSource(for: entry)

        let weightResolver = WeightResolver(
            entry: entry,
            stafWeightStore: stafWeightStore,
            executionPhase: .prefill,
            accessPolicyResolver: planBuildContext.compileContext.accessPolicyResolver
        )

        if let linear = entry.fragment as? LinearFragment {
            let projection = linear
            let isOutput = linear.isOutput
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
                : slotDimension

            if isOutput && projection.outputDimension > hiddenSize {
                let inputRowStride = inputBuffer === buffers.hidden
                    ? buffers.hidden.length / max(maximumSequenceLength, 1)
                    : slotDimension * scratchElementSize
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
                    : slotDimension * scratchElementSize
                perPositionStrides[0] = inputRowStride
            }
            // Prefer direct quantized GEMM (dequant in registers) when available.
            // Falls back to dequant→AMX when no direct kernel exists.
            let directGEMM = resolveDirectQuantizedGEMM(for: quantizationDescriptor.schemeIdentifier)
            let useDirectQuantizedGEMM = directGEMM.flatMap {
                planBuildContext.pipelineCache[$0.kernelName]
            } != nil
            let sequenceGEMVKernelName = mode == .batch
                ? decodeEquivalentSequenceGEMVKernelName(for: quantizationDescriptor)
                : nil
            let usesSequenceGEMVForStep = sequenceGEMVKernelName != nil

            let canDequantForAMX = quantizationDescriptor.schemeIdentifier.isWeightQuantized
                && buffers.dequantScratch != nil
                && dequantKernelName(for: quantizationDescriptor.schemeIdentifier) != nil
            let usesMPPForStep = usesMPP
                && !usesSequenceGEMVForStep
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
            if let sequenceGEMVKernelName {
                guard let sequencePipeline = planBuildContext.pipelineCache[sequenceGEMVKernelName] else {
                    throw MetalCompilerError.kernelNotFound(sequenceGEMVKernelName)
                }
                selectedPipeline = sequencePipeline
                selectedKernelName = sequenceGEMVKernelName
            } else if useDirectQuantizedGEMM,
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
            if usesSequenceGEMVForStep {
                gridSize = MTLSize(
                    width: resolved.config.grid.width,
                    height: maximumSequenceLength,
                    depth: 1
                )
                threadgroupSize = resolved.config.threadgroup
            } else if usesMPPForStep && !useDirectQuantizedGEMM {
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
            let mppTileVariants: [PrefillTileVariant]
            if mode == .batch && usesMPPForStep && !useDirectQuantizedGEMM {
                mppTileVariants = makeMPPTileVariants(
                    baseKernelName: selectedKernelName,
                    gridWidth: gridSize.width,
                    maxSequenceLength: maximumSequenceLength,
                    threadgroupSize: threadgroupSize,
                    threadgroupMemoryLength: 0,
                    sync: .bufferBarrier)
            } else {
                mppTileVariants = []
            }
            let outputRowStride: Int
            if outputBuffer === buffers.hidden {
                outputRowStride = (buffers.hidden.length / max(maximumSequenceLength, 1)) / scratchElementSize
            } else if outputBuffer === buffers.logits {
                outputRowStride = projection.outputDimension
            } else {
                outputRowStride = slotDimension
            }
            return dequantSteps + [MetalPrefillStep(
                pipeline: selectedPipeline,
                gridSize: gridSize,
                threadgroupSize: threadgroupSize,
                bufferBindings: [
                    (0, inputBuffer, inputOffset),
                    (1, gemmWeightBuffer, gemmWeightOffset),
                    (2, outputBuffer, outputOffset),
                ],
                bytesBindings: {
                    var bindings: [(index: Int, value: [UInt8])] = [
                        uint32Binding(3, UInt32(projection.inputDimension)),
                        uint32Binding(4, UInt32(projection.outputDimension)),
                        uint32Binding(5, seqLenValue),
                        uint32Binding(6, UInt32(inputRowStride)),
                    ]
                    if usesSequenceGEMVForStep {
                        bindings.append(uint32Binding(7, UInt32(outputRowStride)))
                    }
                    return bindings
                }(),
                threadgroupMemoryLength: useDirectQuantizedGEMM
                    ? (directGEMM?.threadgroupMemoryLength ?? 0)
                    : ((usesMPPForStep || usesSequenceGEMVForStep) ? 0 : resolved.config.sharedMemoryBytes),
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
                ),
                tileVariants: mppTileVariants
            )]
        } else {
            let frag = entry.fragment
            // Projection-type fragments decompose in prefill
            if let batched = frag as? BatchedProjection {
                return try buildBatchedProjectionPrefillSteps(
                    batched, entry: entry, weightResolver: weightResolver
                )
            }
            if let batch = frag as? BatchedFragment {
                return try buildBatchedFragmentPrefillSteps(
                    batch, entry: entry
                )
            }
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
                            InternalLog.error("[Compiler] missing prefill kernel '\(name)'; related compiled kernels: \(relatedKernelNames)")
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
        }
    }

    // MARK: - Projection-type fragment prefill decomposition

    private mutating func buildBatchedProjectionPrefillSteps(
        _ batched: BatchedProjection,
        entry: DispatchEntry,
        weightResolver: WeightResolver
    ) throws -> [MetalPrefillStep] {
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
            : slotDimension
        let firstOutputSlot = firstNonAliasingScratchOutputSlot(
            inputBuffer: inputBuffer,
            inputOffset: inputOffset,
            scratchSlotSize: scratchSlotSize
        )

        // Try direct quantized GEMM: single dispatch for all projections
        let firstDescriptor = resolveProjectionWeightDescriptor(
            role: batched.projections[0].field, entry: entry
        )

        if let sequenceStep = try buildDecodeEquivalentBatchedSequenceGEMVStep(
            batched: batched,
            entry: entry,
            weightResolver: weightResolver,
            firstDescriptor: firstDescriptor,
            inputBuffer: inputBuffer,
            inputOffset: inputOffset,
            inputRowStride: inputRowStride,
            scratchSlotSize: scratchSlotSize
        ) {
            return annotate([sequenceStep], entryIndex: entry.index, layerIndex: entry.layerIndex)
        }

        // BF16 / FP16 / FP32 dense weights → batched MPP GEMM (matmul2d-based).
        // This path runs a single MPP kernel that processes all N projections
        // sharing the same input A, removing the barriers and dispatch-encode
        // cost that the per-projection fallback would incur.
        if let mppStep = try buildBatchedMPPGEMMStep(
            batched: batched,
            entry: entry,
            weightResolver: weightResolver,
            firstDescriptor: firstDescriptor,
            inputBuffer: inputBuffer,
            inputOffset: inputOffset,
            inputRowStride: inputRowStride,
            scratchSlotSize: scratchSlotSize
        ) {
            return annotate([mppStep], entryIndex: entry.index, layerIndex: entry.layerIndex)
        }

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

                let outputOffset = (firstOutputSlot + i) * scratchSlotSize
                lastOutputOffset = outputOffset
                bufferBindings.append((1 + count + i, buffers.scratch, outputOffset))

                totalOutputDim += projection.outputDimension
            }
            routingState.projectionIndex = firstOutputSlot + count - 1

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
        for (projectionIndex, projection) in batched.projections.enumerated() {
            let projInputRowStride = inputBuffer === buffers.hidden
                ? (buffers.hidden.length / max(maximumSequenceLength, 1)) / scratchElementSize
                : slotDimension
            let resolved = try resolveDispatch(
                DispatchEntry(
                    index: entry.index,
                    fragment: LinearFragment(
                        field: projection.field,
                        inputDimension: projection.inputDimension,
                        outputDimension: projection.outputDimension
                    ),
                    parameterBindings: entry.parameterBindings,
                    layerIndex: entry.layerIndex
                )
            )
            let (weightBuffer, weightOffset) = weightResolver.resolve(role: projection.field)
            let weightTensorName = entry.parameterBindings.first(where: { $0.role == projection.field })?.tensorName
            let quantizationDescriptor = resolveProjectionWeightDescriptor(role: projection.field, entry: entry)
            let outputOffset = (firstOutputSlot + projectionIndex) * scratchSlotSize
            lastOutputOffset = outputOffset
            routingState.projectionIndex = firstOutputSlot + projectionIndex

            // Prefer direct quantized GEMM (dequant in registers) when available.
            // Falls back to dequant→AMX when no direct kernel exists.
            let directGEMM = resolveDirectQuantizedGEMM(for: quantizationDescriptor.schemeIdentifier)
            let useDirectQuantizedGEMM = directGEMM.flatMap {
                planBuildContext.pipelineCache[$0.kernelName]
            } != nil
            let sequenceGEMVKernelName = decodeEquivalentSequenceGEMVKernelName(
                for: quantizationDescriptor
            )
            let usesSequenceGEMVForStep = sequenceGEMVKernelName != nil

            let canDequantForAMX = quantizationDescriptor.schemeIdentifier.isWeightQuantized
                && buffers.dequantScratch != nil
                && dequantKernelName(for: quantizationDescriptor.schemeIdentifier) != nil
            let usesMPPForStep = usesMPP
                && !usesSequenceGEMVForStep
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
            if let sequenceGEMVKernelName {
                guard let sequencePipeline = planBuildContext.pipelineCache[sequenceGEMVKernelName] else {
                    throw MetalCompilerError.kernelNotFound(sequenceGEMVKernelName)
                }
                selectedPipeline = sequencePipeline
                selectedKernelName = sequenceGEMVKernelName
            } else if useDirectQuantizedGEMM,
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
            if usesSequenceGEMVForStep {
                gridSize = MTLSize(
                    width: resolved.config.grid.width,
                    height: maximumSequenceLength,
                    depth: 1
                )
                threadgroupSize = resolved.config.threadgroup
            } else if usesMPPForStep && !useDirectQuantizedGEMM {
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
            let batchedMPPTileVariants: [PrefillTileVariant]
            if usesMPPForStep && !useDirectQuantizedGEMM {
                batchedMPPTileVariants = makeMPPTileVariants(
                    baseKernelName: selectedKernelName,
                    gridWidth: gridSize.width,
                    maxSequenceLength: maximumSequenceLength,
                    threadgroupSize: threadgroupSize,
                    threadgroupMemoryLength: 0,
                    sync: .bufferBarrier)
            } else {
                batchedMPPTileVariants = []
            }
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
                    bytesBindings: {
                        var bindings = [
                            uint32Binding(3, UInt32(projection.inputDimension)),
                            uint32Binding(4, UInt32(projection.outputDimension)),
                            uint32Binding(5, UInt32(maximumSequenceLength)),
                            uint32Binding(6, UInt32(projInputRowStride)),
                        ]
                        if usesSequenceGEMVForStep {
                            bindings.append(uint32Binding(7, UInt32(slotDimension)))
                        }
                        return bindings
                    }(),
                    threadgroupMemoryLength: useDirectQuantizedGEMM
                        ? (directGEMM?.threadgroupMemoryLength ?? 0)
                        : ((usesMPPForStep || usesSequenceGEMVForStep) ? 0 : resolved.config.sharedMemoryBytes),
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
                    ),
                    tileVariants: batchedMPPTileVariants
                )
            )
        }

        routingState.lastOutputIsHidden = false
        routingState.currentInputOffset = lastOutputOffset
        return annotate(steps, entryIndex: entry.index, layerIndex: entry.layerIndex)
    }

    private func firstNonAliasingScratchOutputSlot(
        inputBuffer: MTLBuffer,
        inputOffset: Int,
        scratchSlotSize: Int
    ) -> Int {
        let nextOutputSlot = routingState.projectionIndex + 1
        guard inputBuffer === buffers.scratch, scratchSlotSize > 0 else {
            return nextOutputSlot
        }
        return max(nextOutputSlot, inputOffset / scratchSlotSize + 1)
    }

    private mutating func buildDecodeEquivalentBatchedSequenceGEMVStep(
        batched: BatchedProjection,
        entry: DispatchEntry,
        weightResolver: WeightResolver,
        firstDescriptor: ProjectionWeightDescriptor,
        inputBuffer: MTLBuffer,
        inputOffset: Int,
        inputRowStride: Int,
        scratchSlotSize: Int
    ) throws -> MetalPrefillStep? {
        let count = batched.projections.count
        guard let kernelName = decodeEquivalentBatchedSequenceGEMVKernelName(
            for: firstDescriptor,
            count: count
        ) else {
            return nil
        }
        guard let pipeline = planBuildContext.pipelineCache[kernelName] else {
            throw MetalCompilerError.kernelNotFound(kernelName)
        }

        var bufferBindings: [(Int, MTLBuffer, Int)] = [(0, inputBuffer, inputOffset)]
        var totalOutputDim = 0
        var lastOutputOffset = routingState.currentInputOffset
        let firstOutputSlot = firstNonAliasingScratchOutputSlot(
            inputBuffer: inputBuffer,
            inputOffset: inputOffset,
            scratchSlotSize: scratchSlotSize
        )

        for (i, projection) in batched.projections.enumerated() {
            let (weightBuffer, weightOffset) = weightResolver.resolve(role: projection.field)
            bufferBindings.append((1 + i, weightBuffer, weightOffset))
        }

        for (i, projection) in batched.projections.enumerated() {
            let outputOffset = (firstOutputSlot + i) * scratchSlotSize
            lastOutputOffset = outputOffset
            bufferBindings.append((1 + count + i, buffers.scratch, outputOffset))
            totalOutputDim += projection.outputDimension
        }
        routingState.projectionIndex = firstOutputSlot + count - 1

        let dimBase = 1 + 2 * count
        var bytesBindings: [(index: Int, value: [UInt8])] = [
            uint32Binding(dimBase, UInt32(batched.inputDimension)),
        ]
        for (i, projection) in batched.projections.enumerated() {
            bytesBindings.append(uint32Binding(dimBase + 1 + i, UInt32(projection.outputDimension)))
        }
        let seqLenIndex = dimBase + 1 + count
        bytesBindings.append(uint32Binding(seqLenIndex, UInt32(maximumSequenceLength)))
        bytesBindings.append(uint32Binding(seqLenIndex + 1, UInt32(inputRowStride)))
        bytesBindings.append(uint32Binding(seqLenIndex + 2, UInt32(slotDimension)))

        let simdWidth = max(pipeline.threadExecutionWidth, 1)
        let rowsPerThreadgroup = 2
        let threads = min(simdWidth * rowsPerThreadgroup, pipeline.maxTotalThreadsPerThreadgroup)
        let gridSize = MTLSize(
            width: (totalOutputDim + rowsPerThreadgroup - 1) / rowsPerThreadgroup,
            height: maximumSequenceLength,
            depth: 1
        )
        let threadgroupSize = MTLSize(width: threads, height: 1, depth: 1)
        let readIndices = Set(0...count)
        let writeIndices = Set((count + 1)...(2 * count))

        for projection in batched.projections {
            let descriptor = resolveProjectionWeightDescriptor(role: projection.field, entry: entry)
            recordProjectionQuantization(
                entry: entry,
                descriptor: descriptor,
                mode: .batch,
                inputRowStride: inputRowStride,
                inputDimension: projection.inputDimension,
                selectedKernelName: kernelName,
                usesMPPForStep: false
            )
        }

        routingState.lastOutputIsHidden = false
        routingState.currentInputOffset = lastOutputOffset
        return MetalPrefillStep(
            pipeline: pipeline,
            gridSize: gridSize,
            threadgroupSize: threadgroupSize,
            bufferBindings: bufferBindings,
            bytesBindings: bytesBindings,
            threadgroupMemoryLength: 0,
            sync: .bufferBarrier,
            mode: .batch,
            sequenceLengthPolicy: .bindAndAdjustGridHeight(index: seqLenIndex),
            positionBufferIndex: nil,
            perPositionStrides: [:],
            metadata: .init(
                kernelName: kernelName,
                entryIndex: entry.index,
                weightTensorName: nil,
                bufferAccessPattern: .init(reads: readIndices, writes: writeIndices)
            )
        )
    }

    private mutating func buildBatchedFragmentPrefillSteps(
        _ batch: BatchedFragment,
        entry: DispatchEntry
    ) throws -> [MetalPrefillStep] {
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
                fragment: frag,
                parameterBindings: entry.parameterBindings,
                layerIndex: entry.layerIndex
            )
            let fragSteps = try buildSteps(for: singleEntry)
            steps.append(contentsOf: fragSteps)
        }
        return annotate(steps, entryIndex: entry.index, layerIndex: entry.layerIndex)
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
                fragment: CopyFragment(dimension: dimension),
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
        let kernelName = fallbackWeightFormat.isBFloat16
            ? "batched_qk_rms_norm_seq_bf16_f32"
            : "batched_qk_rms_norm_seq_f32"
        guard let pipeline = planBuildContext.pipelineCache[kernelName] else { return nil }

        let scratchSlotSize = slotDimension * scratchElementSize * maximumSequenceLength

        // Resolve Q and K weights
        let qWeightResolver = WeightResolver(
            entry: entry,
            stafWeightStore: stafWeightStore,
            executionPhase: .prefill,
            accessPolicyResolver: planBuildContext.compileContext.accessPolicyResolver
        )
        let (qWeightBuffer, qWeightOffset) = qWeightResolver.resolve(role: qNorm.weightRole)
        let (kWeightBuffer, kWeightOffset) = qWeightResolver.resolve(role: kNorm.weightRole)

        let totalHeadCount = qNorm.headCount + kNorm.headCount
        let threads = min(256, pipeline.maxTotalThreadsPerThreadgroup)

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

    /// Build tile-size variants for an MPP GEMM prefill step.
    ///
    /// Returns one `PrefillTileVariant` per emitted tile size (16/32/64) that
    /// is present in the pipeline cache. Returns an empty array when the
    /// variant kernels are not available (e.g. MPP library compile failed, or
    /// the kernel name is a direct-quantized GEMM that does not emit variants),
    /// in which case the caller uses the base descriptor unconditionally.
    ///
    /// Each variant's gridSize uses `(maxSeqLen + tileSize - 1) / tileSize`
    /// for the tile dimension; the runtime further narrows the grid height
    /// via `resolvedGridSize` using the actual sequence length.
    private func makeMPPTileVariants(
        baseKernelName: String,
        gridWidth: Int,
        maxSequenceLength: Int,
        threadgroupSize: MTLSize,
        threadgroupMemoryLength: Int,
        sync: SynchronizationKind
    ) -> [PrefillTileVariant] {
        var variants: [PrefillTileVariant] = []
        variants.reserveCapacity(MetalSourceGenerator.mppGEMMTileSizes.count)
        for tileSize in MetalSourceGenerator.mppGEMMTileSizes {
            let variantName = MetalSourceGenerator.mppGEMMVariantName(
                baseName: baseKernelName, tileSize: tileSize)
            guard let variantPipeline = planBuildContext.pipelineCache[variantName] else {
                continue
            }
            let paddedHeight = ((maxSequenceLength + tileSize - 1) / tileSize)
            let variantDescriptor = MetalDispatchDescriptor(
                pipeline: variantPipeline,
                gridSize: MTLSize(width: gridWidth, height: paddedHeight, depth: 1),
                threadgroupSize: threadgroupSize,
                threadgroupMemoryLength: threadgroupMemoryLength,
                barrierPolicy: MetalBarrierPolicy(sync))
            variants.append(PrefillTileVariant(
                tileHeight: tileSize,
                descriptor: variantDescriptor))
        }
        return variants
    }

    /// Build a single batched MPP GEMM step for BF16/FP16/FP32 dense weights.
    ///
    /// Returns nil when the pipeline is not available or the weight format is
    /// not dense (quantized weights go through `buildBatchedProjectionPrefillSteps`'s
    /// Q4 path or the per-projection fallback).
    private mutating func buildBatchedMPPGEMMStep(
        batched: BatchedProjection,
        entry: DispatchEntry,
        weightResolver: WeightResolver,
        firstDescriptor: ProjectionWeightDescriptor,
        inputBuffer: MTLBuffer,
        inputOffset: Int,
        inputRowStride: Int,
        scratchSlotSize: Int
    ) throws -> MetalPrefillStep? {
        let count = batched.projections.count
        guard count >= 2 else { return nil }

        // Only handle dense weight schemes here.
        let scheme = firstDescriptor.schemeIdentifier
        let kernelName: String
        switch scheme {
        case .bf16RowMajor:
            kernelName = "batched_gemm_bf16_f32s_\(count)"
        case .fp16RowMajor:
            kernelName = "batched_gemm_f16_f32s_\(count)"
        case .fp32RowMajor:
            kernelName = "batched_gemm_f32_f32s_\(count)"
        default:
            return nil
        }

        // Every projection's output dimension must be a multiple of N_TILE=32.
        // If any projection violates this, fall through to the per-projection
        // path so the edge handling stays correct.
        let nTile = 32
        for projection in batched.projections {
            if projection.outputDimension % nTile != 0 { return nil }
        }

        // All projections must share the same input dimension (shared A).
        let sharedInputDim = batched.projections[0].inputDimension
        for projection in batched.projections where projection.inputDimension != sharedInputDim {
            return nil
        }

        // Input row stride must match input dimension for MPP tensor_inline.
        guard inputRowStride == sharedInputDim else { return nil }

        guard let pipeline = planBuildContext.pipelineCache[kernelName] else {
            return nil
        }

        // Build buffer bindings: input(0), weight0..N-1(1..N), output0..N-1(N+1..2N)
        var bufferBindings: [(Int, MTLBuffer, Int)] = [(0, inputBuffer, inputOffset)]
        var lastOutputOffset = routingState.currentInputOffset
        var totalNTiles = 0
        let firstOutputSlot = firstNonAliasingScratchOutputSlot(
            inputBuffer: inputBuffer,
            inputOffset: inputOffset,
            scratchSlotSize: scratchSlotSize
        )

        for (i, projection) in batched.projections.enumerated() {
            let (weightBuffer, weightOffset) = weightResolver.resolve(role: projection.field)
            bufferBindings.append((1 + i, weightBuffer, weightOffset))

            let outputOffset = (firstOutputSlot + i) * scratchSlotSize
            lastOutputOffset = outputOffset
            bufferBindings.append((1 + count + i, buffers.scratch, outputOffset))

            totalNTiles += projection.outputDimension / nTile
        }
        routingState.projectionIndex = firstOutputSlot + count - 1

        // Bytes layout: inputDim(2N+1), outDim0..N-1(2N+2..3N+1), seqLen(3N+2), rowStride(3N+3)
        let dimBase = 1 + 2 * count
        var bytesBindings: [(index: Int, value: [UInt8])] = [
            uint32Binding(dimBase, UInt32(sharedInputDim)),
        ]
        for (i, projection) in batched.projections.enumerated() {
            bytesBindings.append(uint32Binding(dimBase + 1 + i, UInt32(projection.outputDimension)))
        }
        let seqLenIndex = dimBase + 1 + count
        bytesBindings.append(uint32Binding(seqLenIndex, UInt32(maximumSequenceLength)))
        bytesBindings.append(uint32Binding(seqLenIndex + 1, UInt32(inputRowStride)))

        // Grid:
        //   width  = total N-tiles across all projections (linear mapping)
        //   height = paddedSeqLen / M_TILE (set to maximumSequenceLength here;
        //            runtime will tile down via bindAndAdjustGridHeightTiled)
        //   depth  = 1
        let mTile = 64
        let paddedMaxSeqLen = ((maximumSequenceLength + mTile - 1) / mTile) * mTile
        let gridSize = MTLSize(
            width: totalNTiles,
            height: paddedMaxSeqLen / mTile,
            depth: 1
        )

        // Threadgroup: SIMD_WIDTH * 4 (execution_simdgroups<4>)
        let simdWidth = max(pipeline.threadExecutionWidth, 1)
        let threads = min(simdWidth * 4, pipeline.maxTotalThreadsPerThreadgroup)
        let threadgroupSize = MTLSize(width: threads, height: 1, depth: 1)

        // Buffer access pattern: reads input + all weights, writes all outputs.
        let readIndices = Set(0...count)
        let writeIndices = Set((count + 1)...(2 * count))

        let batchedTileVariants = makeMPPTileVariants(
            baseKernelName: kernelName,
            gridWidth: totalNTiles,
            maxSequenceLength: maximumSequenceLength,
            threadgroupSize: threadgroupSize,
            threadgroupMemoryLength: 0,
            sync: .bufferBarrier)

        let step = MetalPrefillStep(
            pipeline: pipeline,
            gridSize: gridSize,
            threadgroupSize: threadgroupSize,
            bufferBindings: bufferBindings,
            bytesBindings: bytesBindings,
            threadgroupMemoryLength: 0,
            sync: .bufferBarrier,
            mode: .batch,
            sequenceLengthPolicy: .bindAndAdjustGridHeightTiled(index: seqLenIndex, tileHeight: mTile),
            positionBufferIndex: nil,
            perPositionStrides: [:],
            metadata: .init(
                kernelName: kernelName,
                entryIndex: entry.index,
                weightTensorName: nil,
                bufferAccessPattern: .init(reads: readIndices, writes: writeIndices)
            ),
            tileVariants: batchedTileVariants
        )

        routingState.lastOutputIsHidden = false
        routingState.currentInputOffset = lastOutputOffset

        // Record quantization classification for each projection so the
        // observability plan reflects the MPP batched kernel choice.
        for projection in batched.projections {
            let descriptor = resolveProjectionWeightDescriptor(role: projection.field, entry: entry)
            recordProjectionQuantization(
                entry: entry,
                descriptor: descriptor,
                mode: .batch,
                inputRowStride: inputRowStride,
                inputDimension: projection.inputDimension,
                selectedKernelName: kernelName,
                usesMPPForStep: true
            )
        }

        return step
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
        fallbackWeightFormat.schemeIdentifier
    }
}

private struct ProjectionWeightDescriptor {
    let tensorName: String?
    let schemeIdentifier: QuantizationSchemeIdentifier
    let layout: STAFWeightLayout
    let usedFallback: Bool
    let fallbackReason: MetalQuantizationFallbackReason?
}

// MARK: - Quantized Dispatch Resolution
//
// Each capability (dequant kernel, direct GEMM, batched GEMM) is declared on
// `QuantizationFormat` itself. These private helpers just forward a scheme
// identifier to the corresponding format — no switch-over-format lives here.
//
// IMPORTANT: direct GEMM kernels MUST be multi-row GEMMs with `sequenceLength`
// and `inputRowStride` parameters. Decode-only GEMV kernels
// (`gemv_q4_g64` / `gemv_q4_g128` / `gemv_q8_g*`) must never be returned here
// — they ignore `gid.y` and silently corrupt non-first positions in prefill.

private func dequantKernelName(for scheme: QuantizationSchemeIdentifier) -> String? {
    QuantizationFormatRegistry.format(for: scheme)?.dequantToBFloatKernelName
}

private func resolveDirectQuantizedGEMM(
    for scheme: QuantizationSchemeIdentifier
) -> DirectQuantizedGEMM? {
    QuantizationFormatRegistry.format(for: scheme)?.directGEMMKernel()
}

private func resolveBatchedQuantizedGEMM(
    for scheme: QuantizationSchemeIdentifier,
    count: Int
) -> DirectQuantizedGEMM? {
    QuantizationFormatRegistry.format(for: scheme)?.batchedGEMMKernel(count: count)
}
