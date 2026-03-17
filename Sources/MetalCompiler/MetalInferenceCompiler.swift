import Metal
import LMIR

// MARK: - Compiler

/// Compiles a ModelGraph into a MetalDispatchPlan.
///
/// ## Phases
///
/// 1. **IR walk**: traverse the graph, read each MetalComponent's dispatchDeclarations
/// 2. **Fusion pass**: detect adjacent fusable operations via pattern matching
/// 3. **Compile**: build one MTLLibrary from compiler-owned kernel sources
/// 4. **Buffer routing**: assign concrete MTLBuffers and offsets to each dispatch
/// 5. **Dispatch plan**: compute grid/threadgroup, build MetalDispatchStep array
public struct MetalInferenceCompiler: Sendable {

    public init() {}

    public func compile(
        graph: ModelGraph,
        hiddenSize: Int,
        intermediateSize: Int = 0,
        vocabSize: Int = 0,
        stafWeightStore: STAFWeightStore? = nil,
        device: MTLDevice
    ) throws -> MetalDispatchPlan {

        // Phase 1: Walk IR → flat dispatch entry list
        var walkContext = WalkContext()
        walkRegion(
            graph.rootRegion,
            pathComponents: [],
            layerIndex: nil,
            hiddenSize: hiddenSize,
            context: &walkContext
        )
        let unfusedCount = walkContext.entries.count

        // Phase 2: Fusion pass
        let fusedEntries = fusionPass(entries: walkContext.entries)

        // Phase 3: Compile only the kernels needed by this model's dispatch entries
        // Decode uses F16 buffers (single token, no accumulation)
        let library = try compileLibrary(
            entries: fusedEntries, stafWeightStore: stafWeightStore,
            bufferPrecision: .float16, device: device)

        // Build pipeline cache: kernelName → MTLComputePipelineState
        var pipelineCache: [String: MTLComputePipelineState] = [:]
        for name in library.functionNames {
            if let function = library.makeFunction(name: name) {
                pipelineCache[name] = try device.makeComputePipelineState(function: function)
            }
        }

        // Phase 4: Allocate buffers
        //
        // GPU-only buffers use storageModePrivate + hazardTrackingModeUntracked:
        //   - No CPU cache coherency overhead
        //   - No automatic hazard tracking (we insert barriers manually)
        //
        // CPU-accessible buffers use storageModeShared:
        //   - tokenIn: CPU writes token ID each decode step
        //   - tokenOut: CPU reads output token
        //   - position: CPU writes position counter
        // Compute maximum non-output projection dimension for scratch slot sizing.
        // Output projections (o_proj, down_proj, lm_head) write to hidden/logits, not scratch.
        var maxProjectionOutput = 0
        for entry in fusedEntries {
            if case .projection(let proj, let isOutput) = entry.kind, !isOutput {
                maxProjectionOutput = max(maxProjectionOutput, proj.outputDimension)
            }
        }

        let elementSize = MemoryLayout<Float16>.size
        let resolvedIntermediateSize = max(intermediateSize, hiddenSize * 4)
        let resolvedVocabSize = max(vocabSize, 1)
        let decodeSlotDimension = max(hiddenSize, resolvedIntermediateSize, maxProjectionOutput)

        let scratchElementCount = max(decodeSlotDimension * 4, resolvedIntermediateSize * 4)

        // GPU-only buffers: private + untracked for optimal GPU access.
        // No CPU cache coherency overhead, no automatic hazard tracking.
        // We insert barriers manually via SynchronizationKind.bufferBarrier.
        let gpuOnlyOptions: MTLResourceOptions = [.storageModePrivate, .hazardTrackingModeUntracked]
        // CPU-accessible buffers: shared for CPU read/write each step.
        let cpuAccessOptions: MTLResourceOptions = [.storageModeShared]

        let hiddenBuffer = device.makeBuffer(length: hiddenSize * elementSize, options: gpuOnlyOptions)!
        let residualBuffer = device.makeBuffer(length: hiddenSize * elementSize, options: gpuOnlyOptions)!
        let scratchBuffer = device.makeBuffer(length: scratchElementCount * elementSize, options: gpuOnlyOptions)!
        let logitsBuffer = device.makeBuffer(length: resolvedVocabSize * elementSize, options: gpuOnlyOptions)!
        let positionBuffer = device.makeBuffer(length: 4, options: cpuAccessOptions)!
        let tokenInputBuffer = device.makeBuffer(length: 4, options: cpuAccessOptions)!
        let tokenOutputBuffer = device.makeBuffer(length: 4, options: cpuAccessOptions)!

        // Consolidated KV cache: single K buffer + single V buffer for all layers.
        let kvCache: MetalKVCache?
        if let firstSlot = walkContext.cacheSlots.first {
            kvCache = try MetalKVCache(
                device: device,
                specification: KVCacheSpecification(
                    layerCount: walkContext.cacheSlots.count,
                    kvHeadCount: firstSlot.kvHeadCount,
                    headDimension: firstSlot.headDimension))
        } else {
            kvCache = nil
        }

        let weightBuffers: [MTLBuffer]
        if let staf = stafWeightStore {
            weightBuffers = [staf.buffer]
        } else {
            weightBuffers = []
        }

        // Count conv layers for conv_state allocation
        var convLayerCount = 0
        var convDimension = 0
        var convKernelSize = 0
        for entry in fusedEntries {
            if case .fragment(let frag) = entry.kind, let convOp = frag as? Conv1dFragment {
                convLayerCount += 1
                convDimension = max(convDimension, convOp.dimension)
                convKernelSize = max(convKernelSize, convOp.kernelSize)
            }
        }
        let convStateBuffer: MTLBuffer?
        if convLayerCount > 0 {
            // [numConvLayers × kernelSize × convDimension] in Float16
            let convStateBytes = convLayerCount * convKernelSize * convDimension * elementSize
            convStateBuffer = device.makeBuffer(length: convStateBytes, options: cpuAccessOptions)
            // Zero-initialize
            if let buf = convStateBuffer { memset(buf.contents(), 0, buf.length) }
        } else {
            convStateBuffer = nil
        }

        let bufferSet = MetalBufferSet(
            hidden: hiddenBuffer, residual: residualBuffer, scratch: scratchBuffer,
            weights: weightBuffers,
            kvCache: kvCache,
            convState: convStateBuffer, convStateDimension: convDimension, convStateKernelSize: convKernelSize,
            logits: logitsBuffer, position: positionBuffer,
            tokenIn: tokenInputBuffer, tokenOut: tokenOutputBuffer
        )

        // Log STAF weight store contents
        if let staf = stafWeightStore {
            print("[Compiler] STAF has \(staf.entries.count) tensors")
            for (name, _) in staf.entries.prefix(5) {
                print("[Compiler]   tensor: \(name)")
            }
        } else {
            print("[Compiler] WARNING: no STAF weight store")
        }

        // Log parameterBindings from IR
        var totalBindings = 0
        for entry in fusedEntries {
            totalBindings += entry.parameterBindings.count
        }
        print("[Compiler] \(fusedEntries.count) dispatch entries, \(totalBindings) total parameterBindings")

        // Phase 5: Build dispatch steps with buffer routing
        var steps: [MetalDispatchStep] = []
        var kvCacheIndex = 0

        // Buffer routing state: tracks which buffer region each dispatch reads/writes
        var routingState = BufferRoutingState()

        for entry in fusedEntries {
            let resolvedKernelName = kernelName(for: entry.kind, entry: entry, stafWeightStore: stafWeightStore)
            guard let pipeline = pipelineCache[resolvedKernelName] else {
                throw MetalCompilerError.kernelNotFound(resolvedKernelName)
            }

            let dispatchDimension = self.dispatchDimension(for: entry.kind, hiddenSize: hiddenSize)
            let config = computeDispatchConfig(dimension: dispatchDimension, pipeline: pipeline)

            let bindings = buildBufferBindings(
                entry: entry,
                bufferSet: bufferSet,
                stafWeightStore: stafWeightStore,
                hiddenSize: hiddenSize,
                intermediateSize: resolvedIntermediateSize,
                slotDimension: decodeSlotDimension,
                vocabSize: resolvedVocabSize,
                kvCacheIndex: &kvCacheIndex,
                routingState: &routingState,
                device: device
            )

            steps.append(MetalDispatchStep(
                pipeline: pipeline,
                gridSize: config.grid,
                threadgroupSize: config.threadgroup,
                bufferBindings: bindings.buffers,
                bytesBindings: bindings.bytes,
                threadgroupMemoryLength: config.sharedMemoryBytes,
                sync: .bufferBarrier
            ))
        }

        return MetalDispatchPlan(
            steps: steps, buffers: bufferSet,
            unfusedEntryCount: unfusedCount, fusedEntryCount: fusedEntries.count)
    }

    // MARK: - Prefill Compilation

    /// Compile a sequence-aware prefill plan.
    ///
    /// The prefill plan is a **sequence graph**: step count is O(layers × ops_per_layer),
    /// NOT O(tokens × layers × ops_per_layer). Each kernel operates on [seqLen × dim]
    /// buffers. The GPU kernel itself iterates over the sequence dimension.
    ///
    /// - Projections: GEMM instead of GEMV ([seqLen × in] × [out × in]^T → [seqLen × out])
    /// - Embedding/Norm/Activation/Structural: batched variants with seqLen grid dimension
    /// - Attention: perPosition mode — runtime loops over positions for KV cache fill
    public func compilePrefill(
        graph: ModelGraph,
        hiddenSize: Int,
        intermediateSize: Int = 0,
        vocabSize: Int = 0,
        maximumSequenceLength: Int = 4096,
        stafWeightStore: STAFWeightStore? = nil,
        sharedKVCache: MetalKVCache? = nil,
        sharedConvState: MTLBuffer? = nil,
        sharedConvStateDimension: Int = 0,
        sharedConvStateKernelSize: Int = 0,
        device: MTLDevice
    ) throws -> MetalPrefillPlan {

        // Walk IR (same as decode)
        var walkContext = WalkContext()
        walkRegion(
            graph.rootRegion,
            pathComponents: [],
            layerIndex: nil,
            hiddenSize: hiddenSize,
            context: &walkContext
        )

        // Fusion pass (same as decode — fuses norm + structural)
        let fusedEntries = fusionPass(entries: walkContext.entries)

        // Compile only the kernels needed by this model's prefill dispatch entries
        let library = try compileLibrary(
            entries: fusedEntries, stafWeightStore: stafWeightStore,
            bufferPrecision: .float32, device: device)
        var pipelineCache: [String: MTLComputePipelineState] = [:]
        for name in library.functionNames {
            if let function = library.makeFunction(name: name) {
                let descriptor = MTLComputePipelineDescriptor()
                descriptor.computeFunction = function
                descriptor.label = name
                let pipeline = try device.makeComputePipelineState(descriptor: descriptor, options: [], reflection: nil)
                pipelineCache[name] = pipeline
            }
        }

        // Compute maximum non-output projection dimension for scratch slot sizing.
        // Only non-output projections write to scratch slots.
        // Output projections (o_proj, down_proj → hidden; lm_head → logits) are excluded.
        var maxProjectionOutputDimension = 0
        for entry in fusedEntries {
            if case .projection(let proj, let isOutput) = entry.kind, !isOutput {
                maxProjectionOutputDimension = max(maxProjectionOutputDimension, proj.outputDimension)
            }
        }

        // Allocate sequence-sized buffers
        // Prefill uses Float32 for hidden/residual/scratch to avoid Float16 precision loss
        // across 16+ layers. Decode stays Float16 (single token, no accumulation).
        let elementSize = MemoryLayout<Float16>.size
        let f32ElementSize = MemoryLayout<Float32>.size
        let resolvedIntermediateSize = max(intermediateSize, hiddenSize * 4)
        let resolvedVocabSize = max(vocabSize, 1)
        // Slot stride must accommodate the largest projection output
        let slotDimension = max(hiddenSize, resolvedIntermediateSize, maxProjectionOutputDimension)
        let maxSeq = maximumSequenceLength
        let scratchElementCount = max(slotDimension * 4, resolvedIntermediateSize * 4)
        // Prefill buffers: shared for CPU read during prefill→decode transfer.
        // (hidden F32→F16 conversion, KV cache memcpy, conv_state memcpy)
        let gpuOptions: MTLResourceOptions = [.storageModeShared]

        // Count conv layers for prefill conv_state allocation
        var prefillConvLayerCount = 0
        var prefillConvDimension = 0
        var prefillConvKernelSize = 0
        for entry in fusedEntries {
            if case .fragment(let frag) = entry.kind, let convOp = frag as? Conv1dFragment {
                prefillConvLayerCount += 1
                prefillConvDimension = max(prefillConvDimension, convOp.dimension)
                prefillConvKernelSize = max(prefillConvKernelSize, convOp.kernelSize)
            }
        }
        // Use shared conv_state buffer if provided, otherwise allocate a new one
        let prefillConvStateBuffer: MTLBuffer?
        let resolvedConvDimension: Int
        let resolvedConvKernelSize: Int
        if let shared = sharedConvState {
            prefillConvStateBuffer = shared
            resolvedConvDimension = sharedConvStateDimension
            resolvedConvKernelSize = sharedConvStateKernelSize
        } else if prefillConvLayerCount > 0 {
            let convStateBytes = prefillConvLayerCount * prefillConvKernelSize * prefillConvDimension * elementSize
            prefillConvStateBuffer = device.makeBuffer(length: convStateBytes, options: [.storageModeShared])
            if let buf = prefillConvStateBuffer { memset(buf.contents(), 0, buf.length) }
            resolvedConvDimension = prefillConvDimension
            resolvedConvKernelSize = prefillConvKernelSize
        } else {
            prefillConvStateBuffer = nil
            resolvedConvDimension = 0
            resolvedConvKernelSize = 0
        }

        // Use shared KV cache if provided, otherwise allocate a new one
        let prefillKVCache: MetalKVCache?
        if let shared = sharedKVCache {
            prefillKVCache = shared
        } else if !walkContext.cacheSlots.isEmpty {
            guard let firstSlot = walkContext.cacheSlots.first else {
                throw MetalCompilerError.deviceSetupFailed("No cache slots")
            }
            prefillKVCache = try MetalKVCache(
                device: device,
                specification: KVCacheSpecification(
                    layerCount: walkContext.cacheSlots.count,
                    kvHeadCount: firstSlot.kvHeadCount,
                    headDimension: firstSlot.headDimension))
        } else {
            prefillKVCache = nil
        }

        let prefillBuffers = PrefillBufferSet(
            hidden: device.makeBuffer(length: maxSeq * hiddenSize * f32ElementSize, options: gpuOptions)!,
            residual: device.makeBuffer(length: maxSeq * hiddenSize * f32ElementSize, options: gpuOptions)!,
            scratch: device.makeBuffer(length: maxSeq * scratchElementCount * f32ElementSize, options: gpuOptions)!,
            weights: stafWeightStore.map { [$0.buffer] } ?? [],
            kvCache: prefillKVCache,
            convState: prefillConvStateBuffer,
            convStateDimension: resolvedConvDimension,
            convStateKernelSize: resolvedConvKernelSize,
            logits: device.makeBuffer(length: resolvedVocabSize * f32ElementSize, options: gpuOptions)!,
            tokenIDs: device.makeBuffer(length: maxSeq * 4, options: [.storageModeShared])!,
            positions: device.makeBuffer(length: maxSeq * 4, options: [.storageModeShared])!,
            tokenOut: device.makeBuffer(length: 4, options: [.storageModeShared])!
        )

        // Build prefill steps — sequence-aware graph
        var steps: [MetalPrefillStep] = []
        var kvCacheIndex = 0
        var routingState = BufferRoutingState()

        for entry in fusedEntries {
            let prefillSteps = try buildPrefillSteps(
                entry: entry,
                buffers: prefillBuffers,
                stafWeightStore: stafWeightStore,
                hiddenSize: hiddenSize,
                intermediateSize: resolvedIntermediateSize,
                slotDimension: slotDimension,
                vocabSize: resolvedVocabSize,
                maximumSequenceLength: maxSeq,
                scratchElementSize: f32ElementSize,
                kvCacheIndex: &kvCacheIndex,
                routingState: &routingState,
                pipelineCache: pipelineCache,
                device: device
            )
            steps.append(contentsOf: prefillSteps)
        }

        print("[Compiler] prefill plan: \(steps.count) steps (sequence graph, token-independent)")

        return MetalPrefillPlan(
            steps: steps,
            buffers: prefillBuffers,
            maximumSequenceLength: maxSeq,
            stepCount: steps.count
        )
    }

    /// Build prefill step(s) for a single dispatch entry.
    /// Returns batch or perPosition steps depending on the operation type.
    private func buildPrefillSteps(
        entry: DispatchEntry,
        buffers: PrefillBufferSet,
        stafWeightStore: STAFWeightStore?,
        hiddenSize: Int,
        intermediateSize: Int,
        slotDimension: Int,
        vocabSize: Int,
        maximumSequenceLength: Int,
        scratchElementSize: Int,  // 4 for float32 scratch
        kvCacheIndex: inout Int,
        routingState: inout BufferRoutingState,
        pipelineCache: [String: MTLComputePipelineState],
        device: MTLDevice
    ) throws -> [MetalPrefillStep] {

        let elementSize = MemoryLayout<Float16>.size

        func resolveWeight(role: String) -> (MTLBuffer, Int) {
            if let binding = entry.parameterBindings.first(where: { $0.role == role }),
               let staf = stafWeightStore,
               let access = staf.bufferAccess(for: binding.tensorName) {
                return (access.buffer, access.offset)
            }
            return (buffers.hidden, 0)
        }

        func getPipeline(_ name: String) throws -> MTLComputePipelineState {
            guard let pipeline = pipelineCache[name] else {
                throw MetalCompilerError.kernelNotFound(name)
            }
            return pipeline
        }

        func seqLenBinding(_ index: Int) -> (index: Int, value: [UInt8]) {
            // Placeholder — runtime will override with actual seqLen
            uint32Binding(index, UInt32(maximumSequenceLength))
        }

        // Determine the sequence-aware kernel and buffer routing
        switch entry.kind {

        // MARK: Embedding (batch)
        case .fragment(let frag) where frag is GatherFragment:
            let embOp = frag as! GatherFragment
            let (weightBuffer, weightOffset) = resolveWeight(role: "embedding_table")
            // Determine kernel variant: F32 output for prefill precision
            var kernelName = "embedding_lookup_seq_f32"
            if let staf = stafWeightStore,
               let binding = entry.parameterBindings.first(where: { $0.role == "embedding_table" }),
               let info = staf.tensor(for: binding.tensorName),
               info.format.schemeIdentifier == .bf16RowMajor {
                kernelName = "embedding_lookup_seq_bf16_f32"
            }
            let pipeline = try getPipeline(kernelName)
            let dim = embOp.embeddingDimension
            let tgSize = min(256, pipeline.maxTotalThreadsPerThreadgroup)
            let gridX = (dim + tgSize - 1) / tgSize
            routingState.lastOutputIsHidden = true
            return [MetalPrefillStep(
                pipeline: pipeline,
                gridSize: MTLSize(width: gridX, height: maximumSequenceLength, depth: 1),
                threadgroupSize: MTLSize(width: tgSize, height: 1, depth: 1),
                bufferBindings: [
                    (0, buffers.tokenIDs, 0),
                    (1, weightBuffer, weightOffset),
                    (2, buffers.hidden, 0),
                ],
                bytesBindings: [
                    uint32Binding(3, UInt32(dim)),
                    seqLenBinding(4),
                ],
                threadgroupMemoryLength: 0,
                sync: .bufferBarrier,
                mode: .batch,
                sequenceLengthBindingIndex: 4,
                positionBufferIndex: nil,
                perPositionStrides: [:]
            )]

        // MARK: GEMM Projection (batch)
        case .projection(let projection, let isOutput):
            let (weightBuffer, weightOffset) = resolveWeight(role: projection.field)
            let isOutputHead = isOutput && projection.outputDimension > hiddenSize

            // Select GEMM kernel variant: F32 scratch I/O to prevent precision loss
            var kernelName = "gemm_f32s"
            if let binding = entry.parameterBindings.first(where: { $0.role == projection.field }),
               let staf = stafWeightStore,
               let tensorInfo = staf.tensor(for: binding.tensorName) {
                let baseGemv = tensorInfo.format.gemvKernelName
                if baseGemv == "gemv_bf16" {
                    kernelName = "gemm_bf16_f32s"
                } else if baseGemv == "gemv_q4_g64" {
                    kernelName = "gemm_q4_g64_f32s"
                } else if baseGemv == "gemv_q4_g128" {
                    kernelName = "gemm_q4_g128_f32s"
                } else {
                    kernelName = "gemm_f32s"
                }
                if pipelineCache[kernelName] == nil { kernelName = "gemm_f32s" }
            }
            let pipeline = try getPipeline(kernelName)

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

            if isOutputHead {
                outputBuffer = buffers.logits
                outputOffset = 0
                routingState.lastOutputIsHidden = false
            } else if isOutput {
                outputBuffer = buffers.hidden
                outputOffset = 0
                routingState.lastOutputIsHidden = true
            } else {
                let scratchSlot = routingState.projectionIndex + 1
                outputBuffer = buffers.scratch
                outputOffset = scratchSlot * slotDimension * scratchElementSize * maximumSequenceLength
                routingState.lastOutputIsHidden = false
            }
            routingState.projectionIndex += 1

            let simdWidth = pipeline.threadExecutionWidth
            let threads = min(2 * simdWidth, pipeline.maxTotalThreadsPerThreadgroup)
            let gridX = (projection.outputDimension + 1) / 2

            // Output head: only process last token (logits buffer is [vocabSize], not [seqLen × vocabSize]).
            // Uses the same GEMM kernel but with seqLen=1 (lastToken mode).
            // Runtime adjusts input offset to the last token's position.
            if isOutputHead {
                let inputStride = projection.inputDimension * scratchElementSize
                return [MetalPrefillStep(
                    pipeline: pipeline,
                    gridSize: MTLSize(width: gridX, height: 1, depth: 1),
                    threadgroupSize: MTLSize(width: threads, height: 1, depth: 1),
                    bufferBindings: [
                        (0, inputBuffer, inputOffset),
                        (1, weightBuffer, weightOffset),
                        (2, outputBuffer, outputOffset),
                    ],
                    bytesBindings: [
                        uint32Binding(3, UInt32(projection.inputDimension)),
                        uint32Binding(4, UInt32(projection.outputDimension)),
                        uint32Binding(5, UInt32(1)),  // seqLen=1 for last token
                    ],
                    threadgroupMemoryLength: 0,
                    sync: .bufferBarrier,
                    mode: .lastToken,
                    sequenceLengthBindingIndex: nil,
                    positionBufferIndex: nil,
                    perPositionStrides: [0: inputStride]
                )]
            }

            return [MetalPrefillStep(
                pipeline: pipeline,
                gridSize: MTLSize(width: gridX, height: maximumSequenceLength, depth: 1),
                threadgroupSize: MTLSize(width: threads, height: 1, depth: 1),
                bufferBindings: [
                    (0, inputBuffer, inputOffset),
                    (1, weightBuffer, weightOffset),
                    (2, outputBuffer, outputOffset),
                ],
                bytesBindings: [
                    uint32Binding(3, UInt32(projection.inputDimension)),
                    uint32Binding(4, UInt32(projection.outputDimension)),
                    seqLenBinding(5),
                ],
                threadgroupMemoryLength: 0,
                sync: .bufferBarrier,
                mode: .batch,
                sequenceLengthBindingIndex: 5,
                positionBufferIndex: nil,
                perPositionStrides: [:]
            )]

        // MARK: RMS Norm (batch)
        // Standalone RMSNorm writes in-place to hidden (Float32 in prefill).
        // embedding_norm and final_norm results stay in hidden for the next operation.
        case .fragment(let frag) where frag is Reduction:
            let normOp = frag as! Reduction
            let (weightBuffer, weightOffset) = resolveWeight(role: "scale")
            // F32 in-place norm: hidden(F32) → hidden(F32)
            var normKernelName = "rms_norm_seq_f32_inplace"
            if let staf = stafWeightStore,
               let binding = entry.parameterBindings.first(where: { $0.role == "scale" }),
               let info = staf.tensor(for: binding.tensorName),
               info.format.schemeIdentifier == .bf16RowMajor {
                normKernelName = "rms_norm_seq_bf16_f32_inplace"
            }
            let pipeline = try getPipeline(normKernelName)
            let threads = min(roundUp(min(max(normOp.dimension, 1), 1024), to: pipeline.threadExecutionWidth), pipeline.maxTotalThreadsPerThreadgroup)
            routingState.lastOutputIsHidden = true
            routingState.projectionIndex = 0
            return [MetalPrefillStep(
                pipeline: pipeline,
                gridSize: MTLSize(width: maximumSequenceLength, height: 1, depth: 1),
                threadgroupSize: MTLSize(width: threads, height: 1, depth: 1),
                bufferBindings: [
                    (0, buffers.hidden, 0),
                    (1, weightBuffer, weightOffset),
                    (2, buffers.hidden, 0),  // in-place: output = hidden
                ],
                bytesBindings: [
                    uint32Binding(3, UInt32(normOp.dimension)),
                    floatBinding(4, normOp.epsilon),
                    seqLenBinding(5),
                ],
                threadgroupMemoryLength: 0,
                sync: .bufferBarrier,
                mode: .batch,
                sequenceLengthBindingIndex: 5,
                positionBufferIndex: nil,
                perPositionStrides: [:]
            )]

        // MARK: Fused Copy+RMSNorm (batch) — treat as separate steps for prefill
        case .fusedCopyNorm(let fusedOp):
            
            let (weightBuffer, weightOffset) = resolveWeight(role: "scale")

            // F32 hidden → F32 scratch via F32 norm kernel
            var normKernelForFused = "rms_norm_seq_f32_inplace"
            if let staf = stafWeightStore,
               let binding = entry.parameterBindings.first(where: { $0.role == "scale" }),
               let info = staf.tensor(for: binding.tensorName),
               info.format.schemeIdentifier == .bf16RowMajor {
                normKernelForFused = "rms_norm_seq_bf16_f32_inplace"
            }
            let copyPipeline = try getPipeline("copy_buffer_seq_f32")
            let normPipeline = try getPipeline(normKernelForFused)

            let copyTgSize = min(256, copyPipeline.maxTotalThreadsPerThreadgroup)
            let copyGridX = (fusedOp.dimension + copyTgSize - 1) / copyTgSize
            let normThreads = min(roundUp(min(max(fusedOp.dimension, 1), 1024), to: normPipeline.threadExecutionWidth), normPipeline.maxTotalThreadsPerThreadgroup)

            routingState.lastOutputIsHidden = false
            routingState.projectionIndex = 0

            return [
                MetalPrefillStep(
                    pipeline: copyPipeline,
                    gridSize: MTLSize(width: copyGridX, height: maximumSequenceLength, depth: 1),
                    threadgroupSize: MTLSize(width: copyTgSize, height: 1, depth: 1),
                    bufferBindings: [
                        (0, buffers.hidden, 0),
                        (1, buffers.residual, 0),
                    ],
                    bytesBindings: [
                        uint32Binding(2, UInt32(fusedOp.dimension)),
                        seqLenBinding(3),
                    ],
                    threadgroupMemoryLength: 0,
                    sync: .bufferBarrier,
                    mode: .batch,
                    sequenceLengthBindingIndex: 3,
                    positionBufferIndex: nil,
                    perPositionStrides: [:]
                ),
                MetalPrefillStep(
                    pipeline: normPipeline,
                    gridSize: MTLSize(width: maximumSequenceLength, height: 1, depth: 1),
                    threadgroupSize: MTLSize(width: normThreads, height: 1, depth: 1),
                    bufferBindings: [
                        (0, buffers.hidden, 0),
                        (1, weightBuffer, weightOffset),
                        (2, buffers.scratch, 0),
                    ],
                    bytesBindings: [
                        uint32Binding(3, UInt32(fusedOp.dimension)),
                        floatBinding(4, fusedOp.epsilon),
                        seqLenBinding(5),
                    ],
                    threadgroupMemoryLength: 0,
                    sync: .bufferBarrier,
                    mode: .batch,
                    sequenceLengthBindingIndex: 5,
                    positionBufferIndex: nil,
                    perPositionStrides: [:]
                ),
            ]

        // MARK: Fused ResidualAdd+Copy+RMSNorm (batch) — decompose for prefill
        case .fusedResidualAddCopyNorm(let fusedOp):
            
            let (weightBuffer, weightOffset) = resolveWeight(role: "scale")

            // F32 variant for all sub-operations
            var normKernelForResidual = "rms_norm_seq_f32_inplace"
            if let staf = stafWeightStore,
               let binding = entry.parameterBindings.first(where: { $0.role == "scale" }),
               let info = staf.tensor(for: binding.tensorName),
               info.format.schemeIdentifier == .bf16RowMajor {
                normKernelForResidual = "rms_norm_seq_bf16_f32_inplace"
            }
            let addPipeline = try getPipeline("residual_add_seq_f32")
            let copyPipeline = try getPipeline("copy_buffer_seq_f32")
            let normPipeline = try getPipeline(normKernelForResidual)

            let tgSize = min(256, addPipeline.maxTotalThreadsPerThreadgroup)
            let gridX = (fusedOp.dimension + tgSize - 1) / tgSize
            let normThreads = min(roundUp(min(max(fusedOp.dimension, 1), 1024), to: normPipeline.threadExecutionWidth), normPipeline.maxTotalThreadsPerThreadgroup)

            routingState.lastOutputIsHidden = false
            routingState.projectionIndex = 0

            return [
                MetalPrefillStep(
                    pipeline: addPipeline,
                    gridSize: MTLSize(width: gridX, height: maximumSequenceLength, depth: 1),
                    threadgroupSize: MTLSize(width: tgSize, height: 1, depth: 1),
                    bufferBindings: [
                        (0, buffers.hidden, 0),
                        (1, buffers.residual, 0),
                        (2, buffers.hidden, 0),
                    ],
                    bytesBindings: [
                        uint32Binding(3, UInt32(fusedOp.dimension)),
                        seqLenBinding(4),
                    ],
                    threadgroupMemoryLength: 0,
                    sync: .bufferBarrier,
                    mode: .batch,
                    sequenceLengthBindingIndex: 4,
                    positionBufferIndex: nil,
                    perPositionStrides: [:]
                ),
                MetalPrefillStep(
                    pipeline: copyPipeline,
                    gridSize: MTLSize(width: gridX, height: maximumSequenceLength, depth: 1),
                    threadgroupSize: MTLSize(width: tgSize, height: 1, depth: 1),
                    bufferBindings: [
                        (0, buffers.hidden, 0),
                        (1, buffers.residual, 0),
                    ],
                    bytesBindings: [
                        uint32Binding(2, UInt32(fusedOp.dimension)),
                        seqLenBinding(3),
                    ],
                    threadgroupMemoryLength: 0,
                    sync: .bufferBarrier,
                    mode: .batch,
                    sequenceLengthBindingIndex: 3,
                    positionBufferIndex: nil,
                    perPositionStrides: [:]
                ),
                MetalPrefillStep(
                    pipeline: normPipeline,
                    gridSize: MTLSize(width: maximumSequenceLength, height: 1, depth: 1),
                    threadgroupSize: MTLSize(width: normThreads, height: 1, depth: 1),
                    bufferBindings: [
                        (0, buffers.hidden, 0),
                        (1, weightBuffer, weightOffset),
                        (2, buffers.scratch, 0),
                    ],
                    bytesBindings: [
                        uint32Binding(3, UInt32(fusedOp.dimension)),
                        floatBinding(4, fusedOp.epsilon),
                        seqLenBinding(5),
                    ],
                    threadgroupMemoryLength: 0,
                    sync: .bufferBarrier,
                    mode: .batch,
                    sequenceLengthBindingIndex: 5,
                    positionBufferIndex: nil,
                    perPositionStrides: [:]
                ),
            ]

        // MARK: Flash Attention (perPosition — KV cache is position-dependent)
        case .fragment(let frag) where frag is FlashAttentionFragment:
            let flashOp = frag as! FlashAttentionFragment
            let layerIndex = kvCacheIndex
            kvCacheIndex += 1
            let scale: Float = 1.0 / Float(flashOp.headDimension).squareRoot()

            routingState.lastOutputIsHidden = false
            routingState.projectionIndex = 0

            // Attention is per-position: the decode kernel processes 1 token at a time.
            // Runtime will loop over positions, adjusting offsets into seq buffers.
            guard let cache = buffers.kvCache else { return [] }
            let keyLayerOffset = cache.specification.layerOffset(
                layer: layerIndex, scheme: cache.specification.keyQuantizationScheme)
            let valueLayerOffset = cache.specification.layerOffset(
                layer: layerIndex, scheme: cache.specification.valueQuantizationScheme)
            let kHeadSlotBytes = cache.specification.bytesPerHeadSlot(
                scheme: cache.specification.keyQuantizationScheme)
            let vHeadSlotBytes = cache.specification.bytesPerHeadSlot(
                scheme: cache.specification.valueQuantizationScheme)

            let pipeline = try getPipeline("flash_attn_decode_f32")
            let threads = min(256, pipeline.maxTotalThreadsPerThreadgroup)
            let scratchSlotSize = slotDimension * scratchElementSize * maximumSequenceLength

            return [MetalPrefillStep(
                pipeline: pipeline,
                gridSize: MTLSize(width: flashOp.headCount, height: 1, depth: 1),
                threadgroupSize: MTLSize(width: threads, height: 1, depth: 1),
                bufferBindings: [
                    (0, buffers.scratch, 1 * scratchSlotSize),  // Q
                    (1, buffers.scratch, 2 * scratchSlotSize),  // K
                    (2, buffers.scratch, 3 * scratchSlotSize),  // V
                    (3, cache.keys, keyLayerOffset),
                    (4, cache.values, valueLayerOffset),
                    (5, buffers.scratch, 0),                     // output
                    (6, buffers.positions, 0),                   // position (placeholder)
                ],
                bytesBindings: [
                    uint32Binding(7, UInt32(flashOp.headCount)),
                    uint32Binding(8, UInt32(flashOp.kvHeadCount)),
                    uint32Binding(9, UInt32(flashOp.headDimension)),
                    floatBinding(10, scale),
                    uint32Binding(11, UInt32(cache.specification.layoutMode.rawValue)),
                    uint32Binding(12, UInt32(cache.specification.maximumSequenceLength)),
                    uint32Binding(13, UInt32(cache.specification.keyQuantizationScheme.rawValue)),
                    uint32Binding(14, UInt32(cache.specification.valueQuantizationScheme.rawValue)),
                    uint32Binding(15, UInt32(kHeadSlotBytes)),
                    uint32Binding(16, UInt32(vHeadSlotBytes)),
                ],
                threadgroupMemoryLength: 0,
                sync: .bufferBarrier,
                mode: .perPosition,
                sequenceLengthBindingIndex: nil,
                positionBufferIndex: 6,
                // Per-binding strides: Q, K, V, output have different dimensions per position.
                // Buffer index 0 (Q): headCount * headDim per position
                // Buffer index 1 (K): kvHeadCount * headDim per position
                // Buffer index 2 (V): kvHeadCount * headDim per position
                // Buffer index 3,4 (KV cache): NOT adjusted per position
                // Buffer index 5 (output): headCount * headDim per position (attention output)
                // Buffer index 6 (position): NOT adjusted
                perPositionStrides: [
                    0: flashOp.headCount * flashOp.headDimension * scratchElementSize,
                    1: flashOp.kvHeadCount * flashOp.headDimension * scratchElementSize,
                    2: flashOp.kvHeadCount * flashOp.headDimension * scratchElementSize,
                    5: flashOp.headCount * flashOp.headDimension * scratchElementSize,
                ]
            )]

        // MARK: SwiGLU (batch) — F32 scratch I/O
        case .fragment(let frag) where frag is ElementwiseFragment:
            let swigluOp = frag as! ElementwiseFragment
            let pipeline = try getPipeline("swiglu_seq_f32")
            let maxScratchSlot = slotDimension * scratchElementSize * maximumSequenceLength
            let tgSize = min(256, pipeline.maxTotalThreadsPerThreadgroup)
            let gridX = (swigluOp.count + tgSize - 1) / tgSize
            routingState.lastOutputIsHidden = false
            routingState.projectionIndex = 0
            return [MetalPrefillStep(
                pipeline: pipeline,
                gridSize: MTLSize(width: gridX, height: maximumSequenceLength, depth: 1),
                threadgroupSize: MTLSize(width: tgSize, height: 1, depth: 1),
                bufferBindings: [
                    (0, buffers.scratch, 1 * maxScratchSlot),
                    (1, buffers.scratch, 2 * maxScratchSlot),
                    (2, buffers.scratch, 0),
                ],
                bytesBindings: [
                    uint32Binding(3, UInt32(swigluOp.count)),
                    seqLenBinding(4),
                ],
                threadgroupMemoryLength: 0,
                sync: .bufferBarrier,
                mode: .batch,
                sequenceLengthBindingIndex: 4,
                positionBufferIndex: nil,
                perPositionStrides: [:]
            )]

        // MARK: Structural Copy (batch)
        case .structuralCopy(let dimension):
            let pipeline = try getPipeline("copy_buffer_seq_f32")
            let tgSize = min(256, pipeline.maxTotalThreadsPerThreadgroup)
            let gridX = (dimension + tgSize - 1) / tgSize
            routingState.projectionIndex = 0
            return [MetalPrefillStep(
                pipeline: pipeline,
                gridSize: MTLSize(width: gridX, height: maximumSequenceLength, depth: 1),
                threadgroupSize: MTLSize(width: tgSize, height: 1, depth: 1),
                bufferBindings: [
                    (0, buffers.hidden, 0),
                    (1, buffers.residual, 0),
                ],
                bytesBindings: [
                    uint32Binding(2, UInt32(dimension)),
                    seqLenBinding(3),
                ],
                threadgroupMemoryLength: 0,
                sync: .bufferBarrier,
                mode: .batch,
                sequenceLengthBindingIndex: 3,
                positionBufferIndex: nil,
                perPositionStrides: [:]
            )]

        // MARK: Structural Add (batch) — F32 buffers
        case .structuralAdd(let dimension):
            let pipeline = try getPipeline("residual_add_seq_f32")
            let tgSize = min(256, pipeline.maxTotalThreadsPerThreadgroup)
            let gridX = (dimension + tgSize - 1) / tgSize
            routingState.lastOutputIsHidden = true
            return [MetalPrefillStep(
                pipeline: pipeline,
                gridSize: MTLSize(width: gridX, height: maximumSequenceLength, depth: 1),
                threadgroupSize: MTLSize(width: tgSize, height: 1, depth: 1),
                bufferBindings: [
                    (0, buffers.hidden, 0),
                    (1, buffers.residual, 0),
                    (2, buffers.hidden, 0),
                ],
                bytesBindings: [
                    uint32Binding(3, UInt32(dimension)),
                    seqLenBinding(4),
                ],
                threadgroupMemoryLength: 0,
                sync: .bufferBarrier,
                mode: .batch,
                sequenceLengthBindingIndex: 4,
                positionBufferIndex: nil,
                perPositionStrides: [:]
            )]

        // MARK: RoPE (batch)
        case .fragment(let frag) where frag is RoPEFragment:
            let ropeOp = frag as! RoPEFragment
            let pipeline = try getPipeline("rope_seq_f32")
            let threads = min(32, pipeline.maxTotalThreadsPerThreadgroup)
            let totalHeads = ropeOp.headCount + ropeOp.kvHeadCount
            let scratchSlotSize = slotDimension * scratchElementSize * maximumSequenceLength
            return [MetalPrefillStep(
                pipeline: pipeline,
                gridSize: MTLSize(width: totalHeads, height: maximumSequenceLength, depth: 1),
                threadgroupSize: MTLSize(width: threads, height: 1, depth: 1),
                bufferBindings: [
                    (0, buffers.scratch, 1 * scratchSlotSize),
                    (1, buffers.scratch, 2 * scratchSlotSize),
                    (2, buffers.positions, 0),
                ],
                bytesBindings: [
                    uint32Binding(3, UInt32(ropeOp.headCount)),
                    uint32Binding(4, UInt32(ropeOp.kvHeadCount)),
                    uint32Binding(5, UInt32(ropeOp.headDimension)),
                    uint32Binding(6, UInt32(ropeOp.ropeDimension)),
                    floatBinding(7, ropeOp.base),
                    seqLenBinding(8),
                ],
                threadgroupMemoryLength: 0,
                sync: .bufferBarrier,
                mode: .batch,
                sequenceLengthBindingIndex: 8,
                positionBufferIndex: nil,
                perPositionStrides: [:]
            )]

        // MARK: QK Norm (batch)
        case .fragment(let frag) where frag is QKNormFragment:
            let qkOp = frag as! QKNormFragment
            let qkNormKernelName = "qk_rms_norm_seq_f32"
            let pipeline = try getPipeline(qkNormKernelName)
            let scratchSlotSize = slotDimension * scratchElementSize * maximumSequenceLength
            let scratchSlotIndex = qkOp.weightRole == "q_layernorm" ? 1 : 2
            let (weightBuffer, weightOffset) = resolveWeight(role: qkOp.weightRole)
            let totalDimension = qkOp.headCount * qkOp.headDimension
            let threads = min(32, pipeline.maxTotalThreadsPerThreadgroup)
            return [MetalPrefillStep(
                pipeline: pipeline,
                gridSize: MTLSize(width: qkOp.headCount, height: maximumSequenceLength, depth: 1),
                threadgroupSize: MTLSize(width: threads, height: 1, depth: 1),
                bufferBindings: [
                    (0, buffers.scratch, scratchSlotIndex * scratchSlotSize),
                    (1, weightBuffer, weightOffset),
                ],
                bytesBindings: [
                    uint32Binding(2, UInt32(qkOp.headCount)),
                    uint32Binding(3, UInt32(qkOp.headDimension)),
                    floatBinding(4, qkOp.epsilon),
                    seqLenBinding(5),
                    uint32Binding(6, UInt32(totalDimension)),
                ],
                threadgroupMemoryLength: 0,
                sync: .bufferBarrier,
                mode: .batch,
                sequenceLengthBindingIndex: 5,
                positionBufferIndex: nil,
                perPositionStrides: [:]
            )]

        // MARK: Argmax (single token — last position only)
        case .fragment(let frag) where frag is ArgmaxFragment:
            let argmaxOp = frag as! ArgmaxFragment
            let pipeline = try getPipeline("argmax_f32")
            let threads = min(roundUp(min(max(argmaxOp.vocabularySize, 1), 1024), to: pipeline.threadExecutionWidth), pipeline.maxTotalThreadsPerThreadgroup)
            return [MetalPrefillStep(
                pipeline: pipeline,
                gridSize: MTLSize(width: 1, height: 1, depth: 1),
                threadgroupSize: MTLSize(width: threads, height: 1, depth: 1),
                bufferBindings: [
                    (0, buffers.logits, 0),
                    (1, buffers.tokenOut, 0),
                ],
                bytesBindings: [
                    uint32Binding(2, UInt32(argmaxOp.vocabularySize)),
                ],
                threadgroupMemoryLength: 0,
                sync: .bufferBarrier,
                mode: .lastToken,
                sequenceLengthBindingIndex: nil,
                positionBufferIndex: nil,
                perPositionStrides: [:]
            )]

        // MARK: Conv1d (causal temporal conv across positions — batch mode)
        // in_proj GEMM writes [seqLen × inProjDim] to scratch slot 1.
        // conv1d_causal_seq reads first convDim elements across positions
        // with causal left-padding, writes [seqLen × convDim] to scratch slot 0.
        // Then extract_conv_state saves the last kernelSize positions' inputs
        // to conv_state for decode warm-start.
        case .fragment(let frag) where frag is Conv1dFragment:
            let convOp = frag as! Conv1dFragment
            let (weightBuffer, weightOffset) = resolveWeight(role: "conv_weight")
            let scratchSlotBytes = slotDimension * scratchElementSize * maximumSequenceLength
            // in_proj GEMM output dimension = 3 * convDim (for ShortConv: in_proj, gate, up)
            let inProjDim = convOp.dimension * 3
            routingState.lastOutputIsHidden = false
            routingState.projectionIndex = 0  // reset for out_proj

            // Step 1: Causal temporal conv1d across all positions (F32 I/O)
            let causalPipeline = try getPipeline("conv1d_causal_seq_f32")
            let causalTgSize = min(256, causalPipeline.maxTotalThreadsPerThreadgroup)
            let causalGridX = (convOp.dimension + causalTgSize - 1) / causalTgSize
            var steps: [MetalPrefillStep] = []
            steps.append(MetalPrefillStep(
                pipeline: causalPipeline,
                gridSize: MTLSize(width: causalGridX, height: maximumSequenceLength, depth: 1),
                threadgroupSize: MTLSize(width: causalTgSize, height: 1, depth: 1),
                bufferBindings: [
                    (0, buffers.scratch, 1 * scratchSlotBytes),  // in_proj output [seqLen × inProjDim]
                    (1, weightBuffer, weightOffset),             // conv weight [convDim × kernelSize]
                    (2, buffers.scratch, 0),                     // output [seqLen × convDim]
                ],
                bytesBindings: [
                    uint32Binding(3, UInt32(convOp.dimension)),
                    uint32Binding(4, UInt32(inProjDim)),
                    uint32Binding(5, UInt32(convOp.kernelSize)),
                    seqLenBinding(6),
                ],
                threadgroupMemoryLength: 0,
                sync: .bufferBarrier,
                mode: .batch,
                sequenceLengthBindingIndex: 6,
                positionBufferIndex: nil,
                perPositionStrides: [:]
            ))

            // Step 2: Extract conv_state from in_proj output (last kernelSize positions)
            if let convState = buffers.convState {
                let extractPipeline = try getPipeline("extract_conv_state_f32")
                let extractTgSize = min(256, extractPipeline.maxTotalThreadsPerThreadgroup)
                let extractGridX = (convOp.dimension + extractTgSize - 1) / extractTgSize
                let convLayerOffset = routingState.convLayerIndex
                    * buffers.convStateKernelSize * buffers.convStateDimension * elementSize
                routingState.convLayerIndex += 1
                steps.append(MetalPrefillStep(
                    pipeline: extractPipeline,
                    gridSize: MTLSize(width: extractGridX, height: convOp.kernelSize, depth: 1),
                    threadgroupSize: MTLSize(width: extractTgSize, height: 1, depth: 1),
                    bufferBindings: [
                        (0, buffers.scratch, 1 * scratchSlotBytes),  // in_proj output
                        (1, convState, convLayerOffset),             // conv_state for this layer
                    ],
                    bytesBindings: [
                        uint32Binding(2, UInt32(convOp.dimension)),
                        uint32Binding(3, UInt32(inProjDim)),
                        uint32Binding(4, UInt32(convOp.kernelSize)),
                        seqLenBinding(5),
                    ],
                    threadgroupMemoryLength: 0,
                    sync: .bufferBarrier,
                    mode: .batch,
                    sequenceLengthBindingIndex: 5,
                    positionBufferIndex: nil,
                    perPositionStrides: [:]
                ))
            }
            return steps

        // MARK: Default — skip unsupported ops in prefill
        default:
            return []
        }
    }

    private func detectPositionBufferIndex(
        entry: DispatchEntry,
        bindings: (buffers: [(index: Int, buffer: MTLBuffer, offset: Int)],
                   bytes: [(index: Int, value: [UInt8])]),
        bufferSet: MetalBufferSet
    ) -> Int? {
        // Position buffer is bound in RoPE (buffer index 2) and flash_attn_decode (buffer index 6)
        for (index, buffer, _) in bindings.buffers {
            if buffer === bufferSet.position {
                return index
            }
        }
        return nil
    }

    /// Detect the buffer binding index that carries the tokenIn value.
    /// Returns nil if this entry does not use tokenIn.
    private func detectTokenInputBufferIndex(
        entry: DispatchEntry,
        bindings: (buffers: [(index: Int, buffer: MTLBuffer, offset: Int)],
                   bytes: [(index: Int, value: [UInt8])]),
        bufferSet: MetalBufferSet
    ) -> Int? {
        // TokenIn buffer is bound in embedding lookup (buffer index 0)
        for (index, buffer, _) in bindings.buffers {
            if buffer === bufferSet.tokenIn {
                return index
            }
        }
        return nil
    }

    // MARK: - IR Walk

    struct WalkContext {
        var entries: [DispatchEntry] = []
        var cacheSlots: [CacheSlotInfo] = []
        var nextIndex: Int = 0

        mutating func emit(
            _ kind: DispatchKind,
            parameterBindings: [ParameterBinding] = [],
            layerIndex: Int? = nil
        ) {
            entries.append(DispatchEntry(
                index: nextIndex, kind: kind,
                parameterBindings: parameterBindings, layerIndex: layerIndex))
            nextIndex += 1
        }
    }

    struct CacheSlotInfo {
        let kvHeadCount: Int
        let headDimension: Int
    }

    private func walkRegion(
        _ region: Region,
        pathComponents: [StructuralPathComponent],
        layerIndex: Int?,
        hiddenSize: Int,
        context: inout WalkContext
    ) {
        for (operationIndex, operation) in region.operations.enumerated() {
            let operationPath = pathComponents + [.operation(operationIndex)]
            let _ = StructuralPath(components: operationPath)

            switch operation.kind {
            case .residual(_, let body):
                context.emit(.structuralCopy(dimension: hiddenSize), layerIndex: layerIndex)
                walkRegion(body, pathComponents: operationPath + [.regionBody],
                           layerIndex: layerIndex, hiddenSize: hiddenSize, context: &context)
                context.emit(.structuralAdd(dimension: hiddenSize), layerIndex: layerIndex)

            case .repeating(let count, let body):
                for iteration in 0..<count {
                    walkRegion(body,
                               pathComponents: operationPath + [.regionBody, .index(iteration)],
                               layerIndex: iteration, hiddenSize: hiddenSize, context: &context)
                }

            case .conditional(let condition, let thenBody, let elseBody):
                if let currentLayer = layerIndex, case .layerIndices(let indices) = condition {
                    let selectedBody = indices.contains(currentLayer) ? thenBody : elseBody
                    walkRegion(selectedBody, pathComponents: operationPath + [.regionBody],
                               layerIndex: currentLayer, hiddenSize: hiddenSize, context: &context)
                } else {
                    walkRegion(thenBody, pathComponents: operationPath + [.regionBody],
                               layerIndex: layerIndex, hiddenSize: hiddenSize, context: &context)
                }

            case .parallel(_, let branches):
                for (branchIndex, branch) in branches.enumerated() {
                    walkRegion(branch,
                               pathComponents: operationPath + [.regionBranch(branchIndex)],
                               layerIndex: layerIndex, hiddenSize: hiddenSize, context: &context)
                }

            case .primitive(let attributes):
                // Resolve layer index in parameterBindings
                let bindings: [ParameterBinding]
                if let currentLayerIndex = layerIndex {
                    bindings = operation.parameterBindings.map { binding in
                        let resolved = binding.tensorName.replacingOccurrences(
                            of: ".layers.0.", with: ".layers.\(currentLayerIndex).")
                        return ParameterBinding(role: binding.role, tensorName: resolved)
                    }
                } else {
                    bindings = operation.parameterBindings
                }

                // Fragment-driven path: walk the MetalKernelFragment tree
                guard let fragment = attributes as? (any MetalKernelFragment) else { continue }
                let startIndex = context.entries.count
                emitFragmentTree(fragment, bindings: bindings, layerIndex: layerIndex, context: &context)
                markLastProjectionAsOutput(entries: &context.entries, from: startIndex)
            }
        }
    }

    // MARK: - Fragment Tree Walk

    /// Recursively walk a MetalKernelFragment tree and emit dispatch entries.
    ///
    /// - Primitive fragments (Reduction, LinearFragment, etc.) → emit as .fragment or .projection
    /// - TupleFragment → walk each child in order
    /// - OptionalFragment → walk content if present
    /// - ConditionalFragment → walk the active branch
    private func emitFragmentTree(
        _ fragment: any MetalKernelFragment,
        bindings: [ParameterBinding],
        layerIndex: Int?,
        context: inout WalkContext
    ) {
        // Primitive fragment → emit dispatch entry
        if let primitive = fragment as? any PrimitiveMetalKernelFragment {
            // LinearFragment → .projection (for GEMV/GEMM dispatch)
            if let linear = primitive as? LinearFragment {
                let projection = MetalProjection(
                    field: linear.field,
                    inputDimension: linear.inputDimension,
                    outputDimension: linear.outputDimension)
                // isOutput is resolved later by marking the last projection
                context.emit(.projection(projection), parameterBindings: bindings, layerIndex: layerIndex)
            }
            // FlashAttentionFragment → register KV cache slot + emit .fragment
            else if let flash = primitive as? FlashAttentionFragment {
                context.cacheSlots.append(CacheSlotInfo(
                    kvHeadCount: flash.kvHeadCount,
                    headDimension: flash.headDimension))
                context.emit(.fragment(primitive), parameterBindings: bindings, layerIndex: layerIndex)
            }
            // Other primitives → emit .fragment
            else {
                context.emit(.fragment(primitive), parameterBindings: bindings, layerIndex: layerIndex)
            }
            return
        }

        // Walk composite fragments by type-casting
        walkCompositeFragment(fragment, bindings: bindings, layerIndex: layerIndex, context: &context)
    }

    /// Walk composite (non-primitive) fragment types.
    private func walkCompositeFragment(
        _ fragment: any MetalKernelFragment,
        bindings: [ParameterBinding],
        layerIndex: Int?,
        context: inout WalkContext
    ) {
        // TupleFragment — use parameter pack iteration
        if let _ = fragment as? any _TupleFragmentProtocol {
            (fragment as! any _TupleFragmentProtocol)._visitChildren { child in
                emitFragmentTree(child, bindings: bindings, layerIndex: layerIndex, context: &context)
            }
            return
        }

        // OptionalFragment
        if let opt = fragment as? any _OptionalFragmentProtocol {
            opt._visitContent { child in
                emitFragmentTree(child, bindings: bindings, layerIndex: layerIndex, context: &context)
            }
            return
        }

        // ConditionalFragment
        if let cond = fragment as? any _ConditionalFragmentProtocol {
            cond._visitActive { child in
                emitFragmentTree(child, bindings: bindings, layerIndex: layerIndex, context: &context)
            }
            return
        }

        // Generic composite: recurse into the fragment's body.
        // Uses type-erased access to avoid associated type constraints.
        if let bodyAccessor = fragment as? any _FragmentBodyAccessor {
            bodyAccessor._visitBody { child in
                emitFragmentTree(child, bindings: bindings, layerIndex: layerIndex, context: &context)
            }
        }
    }

    // MARK: - isOutput Resolution

    /// Mark the last projection in a range of entries as isOutput.
    private func markLastProjectionAsOutput(entries: inout [DispatchEntry], from startIndex: Int) {
        // Find the last projection in the range [startIndex..<entries.count]
        for i in stride(from: entries.count - 1, through: startIndex, by: -1) {
            if case .projection(let proj, _) = entries[i].kind {
                entries[i] = DispatchEntry(
                    index: entries[i].index,
                    kind: .projection(proj, isOutput: true),
                    parameterBindings: entries[i].parameterBindings,
                    layerIndex: entries[i].layerIndex)
                break
            }
        }
    }

    // MARK: - Fusion Pass

    func fusionPass(entries: [DispatchEntry]) -> [DispatchEntry] {
        var result = entries
        var changed = true

        while changed {
            changed = false
            var index = 0

            while index < result.count {
                // Extract Reduction from .fragment or .compute
                func extractReduction(at i: Int) -> Reduction? {
                    if case .fragment(let frag) = result[i].kind {
                        return frag as? Reduction
                    }
                    return nil
                }

                // Pattern 1: structuralAdd + structuralCopy + Reduction → fused (3 → 1)
                if index + 2 < result.count,
                   case .structuralAdd(let addDimension) = result[index].kind,
                   case .structuralCopy = result[index + 1].kind,
                   let reduction = extractReduction(at: index + 2) {
                    let fused = DispatchEntry(
                        index: result[index].index,
                        kind: .fusedResidualAddCopyNorm(FusedResidualAddCopyNorm(
                            dimension: addDimension, epsilon: reduction.epsilon)),
                        parameterBindings: result[index + 2].parameterBindings,
                        layerIndex: result[index].layerIndex)
                    result.replaceSubrange(index...index + 2, with: [fused])
                    changed = true
                    continue
                }

                // Pattern 2: structuralCopy + Reduction → fused (2 → 1)
                if index + 1 < result.count,
                   case .structuralCopy = result[index].kind,
                   let reduction = extractReduction(at: index + 1) {
                    let fused = DispatchEntry(
                        index: result[index].index,
                        kind: .fusedCopyNorm(FusedCopyNorm(
                            dimension: reduction.dimension, epsilon: reduction.epsilon)),
                        parameterBindings: result[index + 1].parameterBindings,
                        layerIndex: result[index].layerIndex)
                    result.replaceSubrange(index...index + 1, with: [fused])
                    changed = true
                    continue
                }

                index += 1
            }
        }

        return result
    }

    // MARK: - Kernel Name Resolution

    /// Map a DispatchKind to the MSL kernel function name.
    ///
    /// For projections, the kernel depends on the weight's quantization format.
    /// The STAF weight store is consulted to determine whether to use float16
    /// GEMV or a quantized variant (gemv_q4_g64, gemv_q8_g32, etc.).
    private func kernelName(
        for kind: DispatchKind,
        entry: DispatchEntry,
        stafWeightStore: STAFWeightStore?
    ) -> String {
        switch kind {
        case .projection(let projection, _):
            // Look up the weight's quantization format from STAF
            if let binding = entry.parameterBindings.first(where: { $0.role == projection.field }),
               let staf = stafWeightStore,
               let tensorInfo = staf.tensor(for: binding.tensorName) {
                return tensorInfo.format.gemvKernelName
            }
            return "gemv"  // fallback to float16
        case .fusedCopyNorm:
            let isBF16 = entry.parameterBindings.contains { binding in
                guard let staf = stafWeightStore,
                      let info = staf.tensor(for: binding.tensorName) else { return false }
                return info.format.schemeIdentifier == .bf16RowMajor
            }
            return isBF16 ? "fused_copy_rms_norm_bf16" : "fused_copy_rms_norm"
        case .fusedResidualAddCopyNorm:
            let isBF16 = entry.parameterBindings.contains { binding in
                guard let staf = stafWeightStore,
                      let info = staf.tensor(for: binding.tensorName) else { return false }
                return info.format.schemeIdentifier == .bf16RowMajor
            }
            return isBF16 ? "fused_residual_add_copy_rms_norm_bf16" : "fused_residual_add_copy_rms_norm"
        case .fragment(let frag):
            // Fragment-driven: kernel name derived from fragment type + weight format
            // TODO: implement dynamic name resolution from fragment parameters
            return resolveFragmentKernelName(frag, entry: entry, stafWeightStore: stafWeightStore)
        case .structuralCopy:
            return "copy_buffer"
        case .structuralAdd:
            return "residual_add"
        }
    }

    /// Derive kernel name from a primitive fragment and its context.
    private func resolveFragmentKernelName(
        _ fragment: any PrimitiveMetalKernelFragment,
        entry: DispatchEntry,
        stafWeightStore: STAFWeightStore?
    ) -> String {
        // Determine weight format suffix from STAF
        let isBF16 = entry.parameterBindings.contains { binding in
            guard let staf = stafWeightStore,
                  let info = staf.tensor(for: binding.tensorName) else { return false }
            return info.format.schemeIdentifier == .bf16RowMajor
        }
        let suffix = isBF16 ? "_bf16" : ""

        switch fragment {
        case is Reduction: return "rms_norm" + suffix
        case is ElementwiseFragment: return "swiglu"
        case is GatherFragment: return "embedding_lookup" + suffix
        case is ArgmaxFragment: return "argmax"
        case is FlashAttentionFragment: return "flash_attn_decode"
        case is RoPEFragment: return "rope"
        case is QKNormFragment: return "qk_rms_norm" + suffix
        case is Conv1dFragment: return "conv_state_update"
        case is SSMRecurrenceFragment: return "ssm_recurrence"
        case is SigmoidGateFragment: return "sigmoid_gate"
        default: return "unknown_fragment"
        }
    }

    /// Get the dispatch dimension for grid/threadgroup calculation.
    private func dispatchDimension(for kind: DispatchKind, hiddenSize: Int) -> MetalDispatchDimension {
        switch kind {
        case .projection(let projection, _):
            return .gemv(outputDimension: projection.outputDimension, inputDimension: projection.inputDimension)
        case .fusedCopyNorm(let fused):
            return .reduction(dimension: fused.dimension)
        case .fusedResidualAddCopyNorm(let fused):
            return .reduction(dimension: fused.dimension)
        case .fragment(let frag):
            return frag.dispatchDimension
        case .structuralCopy(let dimension):
            return .elementwise(count: dimension)
        case .structuralAdd(let dimension):
            return .elementwise(count: dimension)
        }
    }

    // MARK: - Buffer Routing

    /// Tracks the current data flow state through the dispatch sequence.
    struct BufferRoutingState {
        /// Which scratch sub-region the next GEMV should read from.
        var currentInputOffset: Int = 0
        /// Counter for parallel projections within one operation.
        var projectionIndex: Int = 0
        /// Whether the last dispatch wrote to hidden (vs scratch).
        var lastOutputIsHidden: Bool = true
        /// Counter for conv layers (for conv_state offset calculation).
        var convLayerIndex: Int = 0
    }

    /// Build buffer and bytes bindings for a single dispatch entry.
    private func buildBufferBindings(
        entry: DispatchEntry,
        bufferSet: MetalBufferSet,
        stafWeightStore: STAFWeightStore?,
        hiddenSize: Int,
        intermediateSize: Int,
        slotDimension: Int,
        vocabSize: Int,
        kvCacheIndex: inout Int,
        routingState: inout BufferRoutingState,
        device: MTLDevice
    ) -> (buffers: [(index: Int, buffer: MTLBuffer, offset: Int)],
          bytes: [(index: Int, value: [UInt8])]) {

        let elementSize = MemoryLayout<Float16>.size

        /// Resolve a weight tensor from the STAF weight store.
        /// Reads tensor name directly from IR parameterBindings — no mapping needed.
        func resolveWeight(role: String) -> (MTLBuffer, Int) {
            if let binding = entry.parameterBindings.first(where: { $0.role == role }),
               let staf = stafWeightStore,
               let access = staf.bufferAccess(for: binding.tensorName) {
                return (access.buffer, access.offset)
            }
            // Log weight miss
            let bindingName = entry.parameterBindings.first(where: { $0.role == role })?.tensorName ?? "(no binding)"
            print("[Compiler] WEIGHT MISS: role='\(role)' tensorName='\(bindingName)' bindings=\(entry.parameterBindings.map(\.role))")
            return (bufferSet.hidden, 0)  // fallback
        }

        switch entry.kind {

        // MARK: Embedding Lookup
        // MARK: RMS Norm
        // Standalone RMSNorm (not fused with structuralCopy) writes in-place to hidden.
        // This ensures embedding_norm and final_norm results stay in hidden for
        // the next operation (Residual's structuralCopy or OutputHead projection).
        // In-place is safe: the kernel reads all elements for RMS before any writes.
        // MARK: Fused Copy + RMS Norm
        case .fusedCopyNorm(let fusedOperation):
            
            let (weightBuffer, weightOffset) = resolveWeight(role: "scale")
            routingState.lastOutputIsHidden = false
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

        // MARK: Fused Residual Add + Copy + RMS Norm
        case .fusedResidualAddCopyNorm(let fusedOperation):
            
            let (weightBuffer, weightOffset) = resolveWeight(role: "scale")
            routingState.lastOutputIsHidden = false
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

        // MARK: GEMV Projection
        case .projection(let projection, let isOutput):
            let (weightBuffer, weightOffset) = resolveWeight(role: projection.field)

            // Input: scratch[0] after norm/compute, or hidden
            let inputBuffer: MTLBuffer
            let inputOffset: Int
            if routingState.lastOutputIsHidden {
                inputBuffer = bufferSet.hidden
                inputOffset = 0
            } else {
                inputBuffer = bufferSet.scratch
                inputOffset = 0
            }

            // Output routing:
            // - OutputHead (vocabSize > hiddenSize): write to logits buffer
            // - Other isOutput projections (o_proj, down_proj): write to hidden
            // - Non-output projections: write to scratch slots
            let outputBuffer: MTLBuffer
            let outputOffset: Int

            if isOutput && projection.outputDimension > hiddenSize {
                // OutputHead: output is vocabSize, too large for hidden buffer
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

        // MARK: Flash Attention Decode
        // MARK: SwiGLU
        // MARK: Argmax
        // MARK: Structural Copy
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

        // MARK: Structural Add
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

        // MARK: RoPE
        // MARK: QK Norm (per-head RMS norm on Q or K in scratch)
        // MARK: Conv1d (double-gated depthwise conv with conv_state)
        //
        // in_proj output [3 × hiddenSize] per token: [B | C | x].
        // conv_state_update kernel: Bx = B*x → shift + append to conv_state → conv → C*convOut.
        // conv_state is required — ShortConv always declares a .conv cache slot.
        // MARK: Fragment-driven buffer routing

        case .fragment(let fragment) where fragment is Reduction:
            let reduction = fragment as! Reduction
            let (weightBuffer, weightOffset) = resolveWeight(role: "scale")
            routingState.lastOutputIsHidden = true
            routingState.projectionIndex = 0
            return (
                buffers: [
                    (0, bufferSet.hidden, 0),
                    (1, weightBuffer, weightOffset),
                    (2, bufferSet.hidden, 0),
                ],
                bytes: [
                    uint32Binding(3, UInt32(reduction.dimension)),
                    floatBinding(4, reduction.epsilon),
                ]
            )

        case .fragment(let fragment) where fragment is ElementwiseFragment:
            let swiglu = fragment as! ElementwiseFragment
            let slotBytes = slotDimension * elementSize
            routingState.lastOutputIsHidden = false
            routingState.projectionIndex = 0
            return (
                buffers: [
                    (0, bufferSet.scratch, 1 * slotBytes),
                    (1, bufferSet.scratch, 2 * slotBytes),
                    (2, bufferSet.scratch, 0),
                ],
                bytes: [
                    uint32Binding(3, UInt32(swiglu.count)),
                ]
            )

        case .fragment(let fragment) where fragment is GatherFragment:
            let gather = fragment as! GatherFragment
            let (weightBuffer, weightOffset) = resolveWeight(role: "embedding_table")
            routingState.lastOutputIsHidden = true
            return (
                buffers: [
                    (0, bufferSet.tokenIn, 0),
                    (1, weightBuffer, weightOffset),
                    (2, bufferSet.hidden, 0),
                ],
                bytes: [
                    uint32Binding(3, UInt32(gather.embeddingDimension)),
                ]
            )

        case .fragment(let fragment) where fragment is ArgmaxFragment:
            let argmax = fragment as! ArgmaxFragment
            let threads = min(1024, argmax.vocabularySize)
            return (
                buffers: [
                    (0, bufferSet.logits, 0),
                    (1, bufferSet.tokenOut, 0),
                ],
                bytes: [
                    uint32Binding(2, UInt32(argmax.vocabularySize)),
                ]
            )

        case .fragment(let fragment) where fragment is FlashAttentionFragment:
            let flashAttention = fragment as! FlashAttentionFragment
            let layerIndex = kvCacheIndex
            kvCacheIndex += 1
            let scale: Float = 1.0 / Float(flashAttention.headDimension).squareRoot()

            routingState.lastOutputIsHidden = false
            routingState.projectionIndex = 0

            guard let cache = bufferSet.kvCache else {
                fatalError("[Compiler] FlashAttentionFragment requires KV cache")
            }
            let keyLayerOffset = cache.specification.layerOffset(
                layer: layerIndex, scheme: cache.specification.keyQuantizationScheme)
            let valueLayerOffset = cache.specification.layerOffset(
                layer: layerIndex, scheme: cache.specification.valueQuantizationScheme)
            let kHeadSlotBytes = cache.specification.bytesPerHeadSlot(
                scheme: cache.specification.keyQuantizationScheme)
            let vHeadSlotBytes = cache.specification.bytesPerHeadSlot(
                scheme: cache.specification.valueQuantizationScheme)

            return (
                buffers: [
                    (0, bufferSet.scratch, 1 * slotDimension * elementSize),
                    (1, bufferSet.scratch, 2 * slotDimension * elementSize),
                    (2, bufferSet.scratch, 3 * slotDimension * elementSize),
                    (3, cache.keys, keyLayerOffset),
                    (4, cache.values, valueLayerOffset),
                    (5, bufferSet.scratch, 0),
                    (6, bufferSet.position, 0),
                ],
                bytes: [
                    uint32Binding(7, UInt32(flashAttention.headCount)),
                    uint32Binding(8, UInt32(flashAttention.kvHeadCount)),
                    uint32Binding(9, UInt32(flashAttention.headDimension)),
                    floatBinding(10, scale),
                    uint32Binding(11, UInt32(cache.specification.layoutMode.rawValue)),
                    uint32Binding(12, UInt32(cache.specification.maximumSequenceLength)),
                    uint32Binding(13, UInt32(cache.specification.keyQuantizationScheme.rawValue)),
                    uint32Binding(14, UInt32(cache.specification.valueQuantizationScheme.rawValue)),
                    uint32Binding(15, UInt32(kHeadSlotBytes)),
                    uint32Binding(16, UInt32(vHeadSlotBytes)),
                ]
            )

        case .fragment(let fragment) where fragment is RoPEFragment:
            let ropeOp = fragment as! RoPEFragment
            let slotBytes = slotDimension * elementSize
            return (
                buffers: [
                    (0, bufferSet.scratch, 1 * slotBytes),
                    (1, bufferSet.scratch, 2 * slotBytes),
                    (2, bufferSet.position, 0),
                ],
                bytes: [
                    uint32Binding(3, UInt32(ropeOp.headCount)),
                    uint32Binding(4, UInt32(ropeOp.kvHeadCount)),
                    uint32Binding(5, UInt32(ropeOp.headDimension)),
                    uint32Binding(6, UInt32(ropeOp.ropeDimension)),
                    floatBinding(7, ropeOp.base),
                ]
            )

        case .fragment(let fragment) where fragment is QKNormFragment:
            let qkNorm = fragment as! QKNormFragment
            let slotBytes = slotDimension * elementSize
            let scratchSlotIndex = qkNorm.weightRole == "q_layernorm" ? 1 : 2
            let (weightBuffer, weightOffset) = resolveWeight(role: qkNorm.weightRole)
            return (
                buffers: [
                    (0, bufferSet.scratch, scratchSlotIndex * slotBytes),
                    (1, weightBuffer, weightOffset),
                ],
                bytes: [
                    uint32Binding(2, UInt32(qkNorm.headCount)),
                    uint32Binding(3, UInt32(qkNorm.headDimension)),
                    floatBinding(4, qkNorm.epsilon),
                ]
            )

        case .fragment(let fragment) where fragment is Conv1dFragment:
            let convOp = fragment as! Conv1dFragment
            let (weightBuffer, weightOffset) = resolveWeight(role: "conv_weight")
            let slotBytes = slotDimension * elementSize
            routingState.lastOutputIsHidden = false
            routingState.projectionIndex = 0

            guard let convState = bufferSet.convState else {
                fatalError("[Compiler] Conv1dFragment requires conv_state buffer")
            }
            let convLayerOffset = routingState.convLayerIndex
                * bufferSet.convStateKernelSize * bufferSet.convStateDimension * elementSize
            routingState.convLayerIndex += 1
            return (
                buffers: [
                    (0, convState, convLayerOffset),
                    (1, bufferSet.scratch, 1 * slotBytes),
                    (2, weightBuffer, weightOffset),
                    (3, bufferSet.scratch, 0),
                ],
                bytes: [
                    uint32Binding(4, UInt32(convOp.dimension)),
                    uint32Binding(5, UInt32(convOp.kernelSize)),
                ]
            )

        case .fragment(let fragment) where fragment is SigmoidGateFragment:
            let gate = fragment as! SigmoidGateFragment
            let slotBytes = slotDimension * elementSize
            routingState.lastOutputIsHidden = false
            return (
                buffers: [
                    (0, bufferSet.scratch, 0),
                    (1, bufferSet.scratch, 1 * slotBytes),
                    (2, bufferSet.scratch, 0),
                ],
                bytes: [
                    uint32Binding(3, UInt32(gate.dimension)),
                ]
            )

        case .fragment:
            print("[Compiler] WARNING: unhandled fragment in buffer routing")
            return (buffers: [], bytes: [])

        // MARK: Default
        default:
            print("[Compiler] WARNING: unhandled operation in buffer routing")
            return (buffers: [], bytes: [])
        }
    }

    // MARK: - Dispatch Config

    func computeDispatchConfig(
        dimension: MetalDispatchDimension,
        pipeline: MTLComputePipelineState
    ) -> (grid: MTLSize, threadgroup: MTLSize, sharedMemoryBytes: Int) {
        let maxThreads = pipeline.maxTotalThreadsPerThreadgroup
        let simdWidth = pipeline.threadExecutionWidth

        switch dimension {
        case .reduction(let dim):
            let threads = min(roundUp(min(max(dim, 1), 1024), to: simdWidth), maxThreads)
            return (MTLSize(width: 1, height: 1, depth: 1),
                    MTLSize(width: threads, height: 1, depth: 1),
                    32 * 4)

        case .elementwise(let count):
            let c = max(count, 1)
            let threadgroupSize = min(roundUp(min(c, 1024), to: simdWidth), maxThreads)
            let groupCount = (c + threadgroupSize - 1) / threadgroupSize
            return (MTLSize(width: groupCount, height: 1, depth: 1),
                    MTLSize(width: threadgroupSize, height: 1, depth: 1), 0)

        case .gemv(let outputDimension, _):
            let rowsPerThreadgroup = 2
            let simdgroupCount = 2
            let threads = min(simdgroupCount * simdWidth, maxThreads)
            let groupCount = (outputDimension + rowsPerThreadgroup - 1) / rowsPerThreadgroup
            return (MTLSize(width: groupCount, height: 1, depth: 1),
                    MTLSize(width: threads, height: 1, depth: 1),
                    simdgroupCount * rowsPerThreadgroup * 4)

        case .perHead(let headCount):
            let threads = min(256, maxThreads)
            return (MTLSize(width: headCount, height: 1, depth: 1),
                    MTLSize(width: threads, height: 1, depth: 1), 0)

        case .gather(let count):
            let c = max(count, 1)
            let threadgroupSize = min(256, maxThreads)
            let groupCount = (c + threadgroupSize - 1) / threadgroupSize
            return (MTLSize(width: groupCount, height: 1, depth: 1),
                    MTLSize(width: threadgroupSize, height: 1, depth: 1), 0)
        }
    }

    // MARK: - Helpers

    // MARK: - Metal Library Cache


    /// Compile a Metal library containing only the kernels needed by the given dispatch entries.
    ///
    /// Kernel source is generated on-demand from fragment parameters + STAF weight format.
    /// No hardcoded catalog — only the kernels actually used are compiled.
    private func compileLibrary(
        entries: [DispatchEntry],
        stafWeightStore: STAFWeightStore?,
        bufferPrecision: MetalSourceGenerator.BufferPrecision,
        device: MTLDevice
    ) throws -> MTLLibrary {
        var sources: [String] = [MetalSourceGenerator.commonHeader]
        var generatedNames: Set<String> = []

        // Collect flash_attn helper (shared by F16/F32 variants)
        var needsFlashAttnHelper = false

        for entry in entries {
            let name: String
            switch entry.kind {
            case .projection(let proj, _):
                // Determine weight format from STAF
                let weightFormat = resolveWeightFormat(role: proj.field, entry: entry, stafWeightStore: stafWeightStore)
                name = weightFormat == .bfloat16 ? "gemv_bf16" : "gemv"
                let isSeq = bufferPrecision == .float32
                if generatedNames.insert(isSeq ? name.replacingOccurrences(of: "gemv", with: "gemm") + (bufferPrecision == .float32 ? "_f32s" : "") : name).inserted {
                    if isSeq {
                        let gemmName = weightFormat == .bfloat16 ? "gemm_bf16_f32s" : "gemm_f32s"
                        sources.append(MetalSourceGenerator.generateGEMM(name: gemmName, bufferPrecision: bufferPrecision, weightFormat: weightFormat))
                    } else {
                        sources.append(MetalSourceGenerator.generateGEMV(name: name, bufferPrecision: bufferPrecision, weightFormat: weightFormat))
                    }
                }

            case .fragment(let frag):
                let weightFormat = resolveWeightFormat(forFragment: frag, entry: entry, stafWeightStore: stafWeightStore)
                name = resolveFragmentKernelName(frag, entry: entry, stafWeightStore: stafWeightStore)
                let kernelName = resolveKernelNameForPrecision(baseName: name, bufferPrecision: bufferPrecision)
                if generatedNames.insert(kernelName).inserted {
                    if let src = MetalSourceGenerator.generateForFragment(frag, name: kernelName, bufferPrecision: bufferPrecision, weightFormat: weightFormat) {
                        if frag is FlashAttentionFragment { needsFlashAttnHelper = true }
                        sources.append(src)
                    }
                }
                // Conv1d in prefill also needs extract_conv_state kernel
                if frag is Conv1dFragment && bufferPrecision == .float32 {
                    let extractName = "extract_conv_state_f32"
                    if generatedNames.insert(extractName).inserted {
                        sources.append(MetalSourceGenerator.generateExtractConvState(name: extractName, bufferPrecision: bufferPrecision))
                    }
                }

            case .fusedCopyNorm(_):
                let weightFormat = resolveWeightFormat(role: "scale", entry: entry, stafWeightStore: stafWeightStore)
                if bufferPrecision == .float16 {
                    let baseName = weightFormat == .bfloat16 ? "fused_copy_rms_norm_bf16" : "fused_copy_rms_norm"
                    if generatedNames.insert(baseName).inserted {
                        sources.append(MetalSourceGenerator.generateFusedCopyRMSNorm(name: baseName, bufferPrecision: bufferPrecision, weightFormat: weightFormat))
                    }
                } else {
                    let copyName = "copy_buffer_seq_f32"
                    let normName = weightFormat == .bfloat16 ? "rms_norm_seq_bf16_f32_inplace" : "rms_norm_seq_f32_inplace"
                    if generatedNames.insert(copyName).inserted {
                        sources.append(MetalSourceGenerator.generateCopy(name: copyName, bufferPrecision: bufferPrecision))
                    }
                    if generatedNames.insert(normName).inserted {
                        sources.append(MetalSourceGenerator.generateReduction(name: normName, dimension: 0, epsilon: 0, bufferPrecision: bufferPrecision, weightFormat: weightFormat))
                    }
                }

            case .fusedResidualAddCopyNorm(_):
                let weightFormat = resolveWeightFormat(role: "scale", entry: entry, stafWeightStore: stafWeightStore)
                if bufferPrecision == .float16 {
                    let baseName = weightFormat == .bfloat16 ? "fused_residual_add_copy_rms_norm_bf16" : "fused_residual_add_copy_rms_norm"
                    if generatedNames.insert(baseName).inserted {
                        sources.append(MetalSourceGenerator.generateFusedResidualAddCopyRMSNorm(name: baseName, bufferPrecision: bufferPrecision, weightFormat: weightFormat))
                    }
                } else {
                    let addName = "residual_add_seq_f32"
                    let copyName = "copy_buffer_seq_f32"
                    let normName = weightFormat == .bfloat16 ? "rms_norm_seq_bf16_f32_inplace" : "rms_norm_seq_f32_inplace"
                    if generatedNames.insert(addName).inserted {
                        sources.append(MetalSourceGenerator.generateResidualAdd(name: addName, bufferPrecision: bufferPrecision))
                    }
                    if generatedNames.insert(copyName).inserted {
                        sources.append(MetalSourceGenerator.generateCopy(name: copyName, bufferPrecision: bufferPrecision))
                    }
                    if generatedNames.insert(normName).inserted {
                        sources.append(MetalSourceGenerator.generateReduction(name: normName, dimension: 0, epsilon: 0, bufferPrecision: bufferPrecision, weightFormat: weightFormat))
                    }
                }


            case .structuralCopy(_):
                let kernelName = bufferPrecision == .float32 ? "copy_buffer_seq_f32" : "copy_buffer"
                if generatedNames.insert(kernelName).inserted {
                    sources.append(MetalSourceGenerator.generateCopy(name: kernelName, bufferPrecision: bufferPrecision, isSequence: bufferPrecision == .float32))
                }

            case .structuralAdd(_):
                let kernelName = bufferPrecision == .float32 ? "residual_add_seq_f32" : "residual_add"
                if generatedNames.insert(kernelName).inserted {
                    sources.append(MetalSourceGenerator.generateResidualAdd(name: kernelName, bufferPrecision: bufferPrecision, isSequence: bufferPrecision == .float32))
                }
            }
        }

        if needsFlashAttnHelper {
            sources.insert(MetalSourceGenerator.flashAttentionHelperSource, at: 1)
        }

        let compileOptions = MTLCompileOptions()
        compileOptions.fastMathEnabled = false
        compileOptions.languageVersion = .version3_0
        return try device.makeLibrary(source: sources.joined(separator: "\n\n"), options: compileOptions)
    }

    /// Determine weight format from STAF for a given role.
    private func resolveWeightFormat(role: String, entry: DispatchEntry, stafWeightStore: STAFWeightStore?) -> MetalSourceGenerator.WeightFormat {
        guard let staf = stafWeightStore,
              let binding = entry.parameterBindings.first(where: { $0.role == role }),
              let info = staf.tensor(for: binding.tensorName) else { return .float16 }
        return info.format.schemeIdentifier == .bf16RowMajor ? .bfloat16 : .float16
    }

    /// Determine weight format from STAF for a primitive fragment.
    private func resolveWeightFormat(forFragment frag: any PrimitiveMetalKernelFragment, entry: DispatchEntry, stafWeightStore: STAFWeightStore?) -> MetalSourceGenerator.WeightFormat {
        let roles = frag.weightSlots.compactMap(\.field) + ["scale", "embedding_table", "conv_weight"]
        for role in roles {
            let format = resolveWeightFormat(role: role, entry: entry, stafWeightStore: stafWeightStore)
            if format == .bfloat16 { return .bfloat16 }
        }
        return .float16
    }

    /// Derive prefill/decode kernel name from base name + precision.
    private func resolveKernelNameForPrecision(baseName: String, bufferPrecision: MetalSourceGenerator.BufferPrecision) -> String {
        guard bufferPrecision == .float32 else { return baseName }
        // Prefill F32 kernel naming convention
        switch baseName {
        case "rms_norm", "rms_norm_bf16": return baseName.replacingOccurrences(of: "rms_norm", with: "rms_norm_seq") + "_f32_inplace"
        case "swiglu": return "swiglu_seq_f32"
        case "embedding_lookup", "embedding_lookup_bf16": return baseName.replacingOccurrences(of: "embedding_lookup", with: "embedding_lookup_seq") + "_f32"
        case "argmax": return "argmax_f32"
        case "rope": return "rope_seq_f32"
        case "qk_rms_norm", "qk_rms_norm_bf16": return "qk_rms_norm_seq_f32"
        case "conv_state_update": return "conv1d_causal_seq_f32"
        case "flash_attn_decode": return "flash_attn_decode_f32"
        default: return baseName
        }
    }

    // MARK: - Helpers

    private func roundUp(_ value: Int, to multiple: Int) -> Int {
        guard multiple > 0 else { return max(value, 1) }
        return ((value + multiple - 1) / multiple) * multiple
    }

    private func uint32Binding(_ index: Int, _ value: UInt32) -> (index: Int, value: [UInt8]) {
        var v = value
        return (index, withUnsafeBytes(of: &v) { Array($0) })
    }

    private func floatBinding(_ index: Int, _ value: Float) -> (index: Int, value: [UInt8]) {
        var v = value
        return (index, withUnsafeBytes(of: &v) { Array($0) })
    }
}
