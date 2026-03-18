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

    /// The optimization strategy used for dispatch entry generation.
    public let optimizer: any DispatchOptimizer


    public init(optimizer: (any DispatchOptimizer)? = nil) {
        self.optimizer = optimizer ?? StandardOptimizer()
    }

    /// Dump optimized dispatch entries for diagnostic purposes.
    /// Returns a human-readable list of all dispatch entries after optimization.
    public func dumpDispatchEntries(
        graph: ModelGraph,
        hiddenSize: Int,
        stafWeightStore: STAFWeightStore? = nil
    ) -> String {
        var walkContext = WalkContext()
        walkRegion(
            graph.rootRegion,
            pathComponents: [],
            layerIndex: nil,
            hiddenSize: hiddenSize,
            context: &walkContext,
            kernelContext: KernelContext(
                bufferPrecision: .float16,
                weightFormat: resolveModelWeightFormat(stafWeightStore))
        )
        let optimized = optimizer.optimizeGraph(walkContext.entries)
        var lines: [String] = []
        lines.append("Dispatch Entries (\(optimized.count) total, \(walkContext.entries.count) unfused):")
        for entry in optimized {
            let layer = entry.layerIndex.map { "L\($0)" } ?? "--"
            let kind: String
            switch entry.kind {
            case .projection(let p, let isOut):
                kind = "projection(\(p.field), in=\(p.inputDimension), out=\(p.outputDimension), isOutput=\(isOut))"
            case .fragment(let f):
                kind = "fragment(\(type(of: f)), kernel=\(f.kernelName(context: KernelContext(bufferPrecision: .float16, weightFormat: resolveModelWeightFormat(stafWeightStore)))))"
            case .fusedCopyNorm(let f):
                kind = "fusedCopyNorm(dim=\(f.dimension), eps=\(f.epsilon))"
            case .fusedResidualAddCopyNorm(let f):
                kind = "fusedResidualAddCopyNorm(dim=\(f.dimension), eps=\(f.epsilon))"
            case .fusedResidualAddNorm(let f):
                kind = "fusedResidualAddNorm(dim=\(f.dimension), eps=\(f.epsilon))"
            case .batchedProjection(let b):
                kind = "batchedProjection(\(b.projections.map(\.field).joined(separator: ",")))"
            case .batchedFragment(let b):
                kind = "batchedFragment(\(b.fragments.count)x)"
            case .structuralCopy(let d):
                kind = "structuralCopy(dim=\(d))"
            case .structuralAdd(let d):
                kind = "structuralAdd(dim=\(d))"
            }
            lines.append("  [\(String(format: "%3d", entry.index))] \(layer) \(kind)")
        }
        return lines.joined(separator: "\n")
    }

    /// Analyze optimization without Metal compilation.
    /// Returns a report comparing unfused vs optimized dispatch counts.
    public func analyzeOptimization(
        graph: ModelGraph,
        hiddenSize: Int,
        stafWeightStore: STAFWeightStore? = nil
    ) -> OptimizationReport {
        var walkContext = WalkContext()
        walkRegion(
            graph.rootRegion,
            pathComponents: [],
            layerIndex: nil,
            hiddenSize: hiddenSize,
            context: &walkContext,
            kernelContext: KernelContext(
                bufferPrecision: .float16,
                weightFormat: resolveModelWeightFormat(stafWeightStore))
        )
        let unfusedCount = walkContext.entries.count
        let optimized = optimizer.optimizeGraph(walkContext.entries)

        // Count patterns
        var patterns: [String: (count: Int, saved: Int)] = [:]
        for entry in optimized {
            let name: String
            switch entry.kind {
            case .fusedCopyNorm: name = "fusedCopyNorm"
            case .fusedResidualAddCopyNorm: name = "fusedResidualAddCopyNorm"
            case .fusedResidualAddNorm: name = "fusedResidualAddNorm"
            case .batchedProjection(let b): name = "batchedProjection(\(b.projections.count)-way)"
            case .batchedFragment(let b): name = "batchedFragment(\(b.fragments.count)-way)"
            default: continue
            }
            patterns[name, default: (0, 0)].count += 1
        }

        return OptimizationReport(
            optimizerName: optimizer.name,
            unfusedCount: unfusedCount,
            optimizedCount: optimized.count,
            patterns: patterns.map { .init(name: $0.key, count: $0.value.count, savedDispatches: 0) }
        )
    }

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
        let kernelContext = KernelContext(
            bufferPrecision: .float16,
            weightFormat: resolveModelWeightFormat(stafWeightStore))
        walkRegion(
            graph.rootRegion,
            pathComponents: [],
            layerIndex: nil,
            hiddenSize: hiddenSize,
            context: &walkContext,
            kernelContext: kernelContext
        )
        let unfusedCount = walkContext.entries.count

        // Phase 2: Graph optimization (structural fusion)
        let fusedEntries = optimizer.optimizeGraph(walkContext.entries)

        // Phase 3: Compile only the kernels needed by this model's dispatch entries
        // Decode uses F16 buffers (single token, no accumulation)
        var (pipelineCache, _) = try compilePipelineCache(
            entries: fusedEntries, stafWeightStore: stafWeightStore,
            bufferPrecision: .float16, device: device)

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
            if case .fragment(let frag) = entry.kind,
               let convSlot = frag.cacheSlots.first(where: { $0.kind == .conv }),
               case .elementwise(let dim) = frag.dispatchDimension {
                convLayerCount += 1
                convDimension = max(convDimension, dim)
                convKernelSize = max(convKernelSize, convSlot.temporalSize)
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

        print("[Compiler] \(fusedEntries.count) dispatch entries (\(optimizer.name) optimizer)")

        // Phase 5: Build dispatch steps with buffer routing
        var steps: [MetalDispatchStep] = []
        var kvCacheIndex = 0

        // Buffer routing state: tracks which buffer region each dispatch reads/writes
        var routingState = BufferRoutingState()

        for entry in fusedEntries {
            let resolvedKernelName = kernelName(for: entry.kind, entry: entry, stafWeightStore: stafWeightStore, kernelContext: kernelContext)
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
            context: &walkContext,
            kernelContext: KernelContext(
                bufferPrecision: .float32,
                weightFormat: resolveModelWeightFormat(stafWeightStore))
        )

        // Graph optimization (same as decode — structural fusion)
        let fusedEntries = optimizer.optimizeGraph(walkContext.entries)

        // Compile only the kernels needed by this model's prefill dispatch entries
        // For prefill (F32), attempts Metal 4 MPP GEMM with fallback to naive GEMM.
        let (pipelineCache, prefillUsesMPP) = try compilePipelineCache(
            entries: fusedEntries, stafWeightStore: stafWeightStore,
            bufferPrecision: .float32, device: device)

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
            if case .fragment(let frag) = entry.kind,
               let convSlot = frag.cacheSlots.first(where: { $0.kind == .conv }),
               case .elementwise(let dim) = frag.dispatchDimension {
                prefillConvLayerCount += 1
                prefillConvDimension = max(prefillConvDimension, dim)
                prefillConvKernelSize = max(prefillConvKernelSize, convSlot.temporalSize)
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
                usesMPP: prefillUsesMPP,
                kvCacheIndex: &kvCacheIndex,
                routingState: &routingState,
                pipelineCache: pipelineCache,
                device: device
            )
            steps.append(contentsOf: prefillSteps)
        }

        // Prefill plan compiled silently — step count reported by ModelBundleLoader

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
        usesMPP: Bool = false,
        kvCacheIndex: inout Int,
        routingState: inout BufferRoutingState,
        pipelineCache: [String: MTLComputePipelineState],
        device: MTLDevice
    ) throws -> [MetalPrefillStep] {

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

        // MARK: GEMM Projection (compiler-owned — STAF-dependent kernel selection)
        case .projection(let projection, let isOutput):
            let (weightBuffer, weightOffset) = resolveWeight(role: projection.field)
            let isOutputHead = isOutput && projection.outputDimension > hiddenSize

            var kernelName = "gemm_f32s"
            if let binding = entry.parameterBindings.first(where: { $0.role == projection.field }),
               let staf = stafWeightStore,
               let tensorInfo = staf.tensor(for: binding.tensorName) {
                let baseGemv = tensorInfo.format.gemvKernelName
                if baseGemv == "gemv_bf16" { kernelName = "gemm_bf16_f32s" }
                else if baseGemv == "gemv_q4_g64" { kernelName = "gemm_q4_g64_f32s" }
                else if baseGemv == "gemv_q4_g128" { kernelName = "gemm_q4_g128_f32s" }
                if pipelineCache[kernelName] == nil { kernelName = "gemm_f32s" }
            }
            let pipeline = try getPipeline(kernelName)

            let inputBuffer = routingState.lastOutputIsHidden ? buffers.hidden : buffers.scratch
            let outputBuffer: MTLBuffer
            let outputOffset: Int
            if isOutputHead {
                outputBuffer = buffers.logits; outputOffset = 0
                routingState.lastOutputIsHidden = false
            } else if isOutput {
                outputBuffer = buffers.hidden; outputOffset = 0
                routingState.lastOutputIsHidden = true
            } else {
                let scratchSlot = routingState.projectionIndex + 1
                outputBuffer = buffers.scratch
                outputOffset = scratchSlot * slotDimension * scratchElementSize * maximumSequenceLength
                // Do not change lastOutputIsHidden here.
                // Parallel projections (gate+up in MLP, Q+K+V in attention) must all
                // read from the same input buffer. Only fragment and structural steps
                // update the routing state.
            }
            routingState.projectionIndex += 1

            let threads = min(2 * pipeline.threadExecutionWidth, pipeline.maxTotalThreadsPerThreadgroup)
            let gridX = (projection.outputDimension + 1) / 2

            if isOutputHead {
                let inputStride = projection.inputDimension * scratchElementSize
                return [MetalPrefillStep(
                    pipeline: pipeline,
                    gridSize: MTLSize(width: gridX, height: 1, depth: 1),
                    threadgroupSize: MTLSize(width: threads, height: 1, depth: 1),
                    bufferBindings: [(0, inputBuffer, 0), (1, weightBuffer, weightOffset), (2, outputBuffer, outputOffset)],
                    bytesBindings: [uint32Binding(3, UInt32(projection.inputDimension)),
                                    uint32Binding(4, UInt32(projection.outputDimension)),
                                    uint32Binding(5, UInt32(1))],
                    threadgroupMemoryLength: 0, sync: .bufferBarrier, mode: .lastToken,
                    sequenceLengthBindingIndex: nil, positionBufferIndex: nil,
                    perPositionStrides: [0: inputStride]
                )]
            }

            return [MetalPrefillStep(
                pipeline: pipeline,
                gridSize: MTLSize(width: gridX, height: maximumSequenceLength, depth: 1),
                threadgroupSize: MTLSize(width: threads, height: 1, depth: 1),
                bufferBindings: [(0, inputBuffer, 0), (1, weightBuffer, weightOffset), (2, outputBuffer, outputOffset)],
                bytesBindings: [uint32Binding(3, UInt32(projection.inputDimension)),
                                uint32Binding(4, UInt32(projection.outputDimension)),
                                seqLenBinding(5)],
                threadgroupMemoryLength: 0, sync: .bufferBarrier, mode: .batch,
                sequenceLengthBindingIndex: 5, positionBufferIndex: nil, perPositionStrides: [:]
            )]

        // MARK: Structural Copy (compiler-owned)
        case .structuralCopy(let dimension):
            let pipeline = try getPipeline("copy_buffer_seq_f32")
            let tgSize = min(256, pipeline.maxTotalThreadsPerThreadgroup)
            let gridX = (dimension + tgSize - 1) / tgSize
            routingState.projectionIndex = 0
            return [MetalPrefillStep(
                pipeline: pipeline,
                gridSize: MTLSize(width: gridX, height: maximumSequenceLength, depth: 1),
                threadgroupSize: MTLSize(width: tgSize, height: 1, depth: 1),
                bufferBindings: [(0, buffers.hidden, 0), (1, buffers.residual, 0)],
                bytesBindings: [uint32Binding(2, UInt32(dimension)), seqLenBinding(3)],
                threadgroupMemoryLength: 0, sync: .bufferBarrier, mode: .batch,
                sequenceLengthBindingIndex: 3, positionBufferIndex: nil, perPositionStrides: [:]
            )]

        // MARK: Structural Add (compiler-owned)
        case .structuralAdd(let dimension):
            let pipeline = try getPipeline("residual_add_seq_f32")
            let tgSize = min(256, pipeline.maxTotalThreadsPerThreadgroup)
            let gridX = (dimension + tgSize - 1) / tgSize
            routingState.lastOutputIsHidden = true
            return [MetalPrefillStep(
                pipeline: pipeline,
                gridSize: MTLSize(width: gridX, height: maximumSequenceLength, depth: 1),
                threadgroupSize: MTLSize(width: tgSize, height: 1, depth: 1),
                bufferBindings: [(0, buffers.hidden, 0), (1, buffers.residual, 0), (2, buffers.hidden, 0)],
                bytesBindings: [uint32Binding(3, UInt32(dimension)), seqLenBinding(4)],
                threadgroupMemoryLength: 0, sync: .bufferBarrier, mode: .batch,
                sequenceLengthBindingIndex: 4, positionBufferIndex: nil, perPositionStrides: [:]
            )]

        // MARK: Fragment-driven prefill steps (protocol dispatch — no type checks)
        case .fragment(let frag):
            let prefillContext = PrefillBindingContext(
                buffers: buffers, slotDimension: slotDimension,
                scratchElementSize: scratchElementSize,
                maximumSequenceLength: maximumSequenceLength,
                kvCacheIndex: kvCacheIndex,
                convLayerIndex: routingState.convLayerIndex,
                kernelContext: KernelContext(bufferPrecision: .float32,
                    weightFormat: resolveModelWeightFormat(stafWeightStore)),
                resolveWeight: resolveWeight,
                getPipeline: getPipeline)
            let result = try frag.prefillSteps(context: prefillContext)
            if result.resetsProjectionIndex { routingState.projectionIndex = 0 }
            if result.consumesKVCacheLayer { kvCacheIndex += 1 }
            if result.consumesConvLayer { routingState.convLayerIndex += 1 }
            routingState.lastOutputIsHidden = result.outputIsHidden
            return result.steps

        // MARK: Fused Copy + Norm → decompose into copy + norm steps for prefill
        case .fusedCopyNorm(let fusedOp):
            let copyEntry = DispatchEntry(index: entry.index, kind: .structuralCopy(dimension: fusedOp.dimension),
                                          parameterBindings: [], layerIndex: entry.layerIndex)
            let normEntry = DispatchEntry(index: entry.index + 1,
                                          kind: .fragment(Reduction(dimension: fusedOp.dimension, epsilon: fusedOp.epsilon)),
                                          parameterBindings: entry.parameterBindings, layerIndex: entry.layerIndex)
            var steps: [MetalPrefillStep] = []
            for decomposed in [copyEntry, normEntry] {
                let s = try buildPrefillSteps(entry: decomposed, buffers: buffers, stafWeightStore: stafWeightStore,
                    hiddenSize: hiddenSize, intermediateSize: intermediateSize, slotDimension: slotDimension,
                    vocabSize: vocabSize, maximumSequenceLength: maximumSequenceLength, scratchElementSize: scratchElementSize,
                    kvCacheIndex: &kvCacheIndex, routingState: &routingState, pipelineCache: pipelineCache, device: device)
                steps.append(contentsOf: s)
            }
            return steps

        // MARK: Fused Residual Add + Copy + Norm → decompose into add + copy + norm
        case .fusedResidualAddCopyNorm(let fusedOp):
            let addEntry = DispatchEntry(index: entry.index, kind: .structuralAdd(dimension: fusedOp.dimension),
                                         parameterBindings: [], layerIndex: entry.layerIndex)
            let copyEntry = DispatchEntry(index: entry.index + 1, kind: .structuralCopy(dimension: fusedOp.dimension),
                                          parameterBindings: [], layerIndex: entry.layerIndex)
            let normEntry = DispatchEntry(index: entry.index + 2,
                                          kind: .fragment(Reduction(dimension: fusedOp.dimension, epsilon: fusedOp.epsilon)),
                                          parameterBindings: entry.parameterBindings, layerIndex: entry.layerIndex)
            var steps: [MetalPrefillStep] = []
            for decomposed in [addEntry, copyEntry, normEntry] {
                let s = try buildPrefillSteps(entry: decomposed, buffers: buffers, stafWeightStore: stafWeightStore,
                    hiddenSize: hiddenSize, intermediateSize: intermediateSize, slotDimension: slotDimension,
                    vocabSize: vocabSize, maximumSequenceLength: maximumSequenceLength, scratchElementSize: scratchElementSize,
                    kvCacheIndex: &kvCacheIndex, routingState: &routingState, pipelineCache: pipelineCache, device: device)
                steps.append(contentsOf: s)
            }
            return steps

        // MARK: Batched Projection → decompose into individual GEMMs
        case .batchedProjection(let batched):
            var steps: [MetalPrefillStep] = []
            for (i, proj) in batched.projections.enumerated() {
                let singleProjection = MetalProjection(
                    field: proj.field,
                    inputDimension: proj.inputDimension,
                    outputDimension: proj.outputDimension)
                let singleEntry = DispatchEntry(
                    index: entry.index + i,
                    kind: .projection(singleProjection, isOutput: false),
                    parameterBindings: entry.parameterBindings,
                    layerIndex: entry.layerIndex)
                let projSteps = try buildPrefillSteps(
                    entry: singleEntry,
                    buffers: buffers,
                    stafWeightStore: stafWeightStore,
                    hiddenSize: hiddenSize,
                    intermediateSize: intermediateSize,
                    slotDimension: slotDimension,
                    vocabSize: vocabSize,
                    maximumSequenceLength: maximumSequenceLength,
                    scratchElementSize: scratchElementSize,
                    kvCacheIndex: &kvCacheIndex,
                    routingState: &routingState,
                    pipelineCache: pipelineCache,
                    device: device)
                steps.append(contentsOf: projSteps)
            }
            return steps

        // MARK: Batched Fragment → decompose into individual per-head dispatches
        case .batchedFragment(let batch):
            var steps: [MetalPrefillStep] = []
            for (i, frag) in batch.fragments.enumerated() {
                let singleEntry = DispatchEntry(
                    index: entry.index + i,
                    kind: .fragment(frag),
                    parameterBindings: entry.parameterBindings,
                    layerIndex: entry.layerIndex)
                let fragSteps = try buildPrefillSteps(
                    entry: singleEntry,
                    buffers: buffers,
                    stafWeightStore: stafWeightStore,
                    hiddenSize: hiddenSize,
                    intermediateSize: intermediateSize,
                    slotDimension: slotDimension,
                    vocabSize: vocabSize,
                    maximumSequenceLength: maximumSequenceLength,
                    scratchElementSize: scratchElementSize,
                    kvCacheIndex: &kvCacheIndex,
                    routingState: &routingState,
                    pipelineCache: pipelineCache,
                    device: device)
                steps.append(contentsOf: fragSteps)
            }
            return steps

        // MARK: Fused Residual Add + Norm → decompose into add + norm
        case .fusedResidualAddNorm(let fusedOp):
            // Decompose into structuralAdd + Reduction for prefill
            let addEntry = DispatchEntry(
                index: entry.index,
                kind: .structuralAdd(dimension: fusedOp.dimension),
                parameterBindings: [],
                layerIndex: entry.layerIndex)
            let normEntry = DispatchEntry(
                index: entry.index + 1,
                kind: .fragment(Reduction(dimension: fusedOp.dimension, epsilon: fusedOp.epsilon)),
                parameterBindings: entry.parameterBindings,
                layerIndex: entry.layerIndex)
            var steps: [MetalPrefillStep] = []
            for decomposed in [addEntry, normEntry] {
                let s = try buildPrefillSteps(
                    entry: decomposed,
                    buffers: buffers,
                    stafWeightStore: stafWeightStore,
                    hiddenSize: hiddenSize,
                    intermediateSize: intermediateSize,
                    slotDimension: slotDimension,
                    vocabSize: vocabSize,
                    maximumSequenceLength: maximumSequenceLength,
                    scratchElementSize: scratchElementSize,
                    kvCacheIndex: &kvCacheIndex,
                    routingState: &routingState,
                    pipelineCache: pipelineCache,
                    device: device)
                steps.append(contentsOf: s)
            }
            return steps

        // MARK: Fused Copy + Norm → copy hidden→residual, then norm hidden→scratch[0]
        //
        // Decode fused kernel: hidden → residual (copy) + hidden → scratch[0] (norm).
        // Prefill: decompose into separate steps with same buffer routing.
        // Norm output goes to scratch[0] (NOT hidden in-place) so that
        // parallel projections (gate_proj, up_proj) can both read from scratch[0].
        case .fusedCopyNorm(let fusedOp):
            var steps: [MetalPrefillStep] = []

            // Step 1: copy hidden → residual
            let copyEntry = DispatchEntry(
                index: entry.index,
                kind: .structuralCopy(dimension: fusedOp.dimension),
                parameterBindings: [],
                layerIndex: entry.layerIndex)
            let copySteps = try buildPrefillSteps(
                entry: copyEntry,
                buffers: buffers,
                stafWeightStore: stafWeightStore,
                hiddenSize: hiddenSize,
                intermediateSize: intermediateSize,
                slotDimension: slotDimension,
                vocabSize: vocabSize,
                maximumSequenceLength: maximumSequenceLength,
                scratchElementSize: scratchElementSize,
                kvCacheIndex: &kvCacheIndex,
                routingState: &routingState,
                pipelineCache: pipelineCache,
                device: device)
            steps.append(contentsOf: copySteps)

            // Step 2: norm hidden → scratch[0]
            let normSteps = try buildNormToScratchStep(
                dimension: fusedOp.dimension, epsilon: fusedOp.epsilon,
                entry: entry, buffers: buffers, stafWeightStore: stafWeightStore,
                slotDimension: slotDimension, maximumSequenceLength: maximumSequenceLength,
                scratchElementSize: scratchElementSize, pipelineCache: pipelineCache)
            steps.append(contentsOf: normSteps)

            routingState.lastOutputIsHidden = false
            routingState.projectionIndex = 0
            return steps

        // MARK: Fused Residual Add + Copy + Norm → add + copy + norm→scratch[0]
        case .fusedResidualAddCopyNorm(let fusedOp):
            var steps: [MetalPrefillStep] = []

            // Step 1: residual add → hidden
            let addEntry = DispatchEntry(
                index: entry.index,
                kind: .structuralAdd(dimension: fusedOp.dimension),
                parameterBindings: [],
                layerIndex: entry.layerIndex)
            let addSteps = try buildPrefillSteps(
                entry: addEntry,
                buffers: buffers,
                stafWeightStore: stafWeightStore,
                hiddenSize: hiddenSize,
                intermediateSize: intermediateSize,
                slotDimension: slotDimension,
                vocabSize: vocabSize,
                maximumSequenceLength: maximumSequenceLength,
                scratchElementSize: scratchElementSize,
                kvCacheIndex: &kvCacheIndex,
                routingState: &routingState,
                pipelineCache: pipelineCache,
                device: device)
            steps.append(contentsOf: addSteps)

            // Step 2: copy hidden → residual
            let copyEntry = DispatchEntry(
                index: entry.index + 1,
                kind: .structuralCopy(dimension: fusedOp.dimension),
                parameterBindings: [],
                layerIndex: entry.layerIndex)
            let copySteps = try buildPrefillSteps(
                entry: copyEntry,
                buffers: buffers,
                stafWeightStore: stafWeightStore,
                hiddenSize: hiddenSize,
                intermediateSize: intermediateSize,
                slotDimension: slotDimension,
                vocabSize: vocabSize,
                maximumSequenceLength: maximumSequenceLength,
                scratchElementSize: scratchElementSize,
                kvCacheIndex: &kvCacheIndex,
                routingState: &routingState,
                pipelineCache: pipelineCache,
                device: device)
            steps.append(contentsOf: copySteps)

            // Step 3: norm hidden → scratch[0]
            let normSteps = try buildNormToScratchStep(
                dimension: fusedOp.dimension, epsilon: fusedOp.epsilon,
                entry: entry, buffers: buffers, stafWeightStore: stafWeightStore,
                slotDimension: slotDimension, maximumSequenceLength: maximumSequenceLength,
                scratchElementSize: scratchElementSize, pipelineCache: pipelineCache)
            steps.append(contentsOf: normSteps)

            routingState.lastOutputIsHidden = false
            routingState.projectionIndex = 0
            return steps

        // MARK: Projection → GEMM (sequence matrix multiply)
        case .projection(let projection, let isOutput):
            let prefillKernelContext = KernelContext(
                bufferPrecision: .float32,
                weightFormat: resolveModelWeightFormat(stafWeightStore))
            let resolvedKernelName = kernelName(
                for: entry.kind, entry: entry,
                stafWeightStore: stafWeightStore, kernelContext: prefillKernelContext)
            let pipeline = try getPipeline(resolvedKernelName)
            let config = computeDispatchConfig(
                dimension: dispatchDimension(for: entry.kind, hiddenSize: hiddenSize),
                pipeline: pipeline)

            let (weightBuffer, weightOffset) = resolveWeight(role: projection.field)

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

            // Prefill scratch uses slot-major layout:
            // slot N offset = N * slotDimension * scratchElementSize * maxSeqLen
            let scratchSlotSize = slotDimension * scratchElementSize * maximumSequenceLength

            if isOutput && projection.outputDimension > hiddenSize {
                // OutputHead: logits buffer is [vocabSize], not seq-sized.
                // Compute only the last position.
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

            // MPP matmul2d uses tile 64(M)×32(N) with 4 simdgroups.
            // Grid: (outputDim/32, seqLen/64, 1). Kernel handles edge tiles internally.
            // Naive GEMM: (outputDim/2, seqLen, 1) with 2 simdgroups.
            let gridSize: MTLSize
            let threadgroupSize: MTLSize
            if usesMPP && mode == .batch {
                let simdWidth = pipeline.threadExecutionWidth
                gridSize = MTLSize(
                    width: (projection.outputDimension + 31) / 32,
                    height: (maximumSequenceLength + 63) / 64,
                    depth: 1)
                threadgroupSize = MTLSize(width: simdWidth * 4, height: 1, depth: 1)
            } else if mode == .lastToken {
                gridSize = MTLSize(width: config.grid.width, height: 1, depth: 1)
                threadgroupSize = config.threadgroup
            } else {
                gridSize = MTLSize(width: config.grid.width, height: maximumSequenceLength, depth: 1)
                threadgroupSize = config.threadgroup
            }

            return [MetalPrefillStep(
                pipeline: pipeline,
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
                threadgroupMemoryLength: (usesMPP && mode == .batch) ? 0 : config.sharedMemoryBytes,
                sync: .bufferBarrier,
                mode: mode,
                // MPP GEMM: grid is in tiles, not raw seqLen. Disable runtime grid adjustment.
                // The kernel reads seqLen from buffer(5) for tensor extents and handles edge tiles.
                sequenceLengthBindingIndex: (usesMPP && mode == .batch) ? nil : (mode == .batch ? 5 : nil),
                positionBufferIndex: nil,
                perPositionStrides: perPositionStrides
            )]

        // MARK: Structural Copy (hidden → residual)
        case .structuralCopy(let dimension):
            let prefillKernelContext = KernelContext(
                bufferPrecision: .float32,
                weightFormat: resolveModelWeightFormat(stafWeightStore))
            let resolvedKernelName = kernelName(
                for: entry.kind, entry: entry,
                stafWeightStore: stafWeightStore, kernelContext: prefillKernelContext)
            let pipeline = try getPipeline(resolvedKernelName)
            let config = computeDispatchConfig(
                dimension: dispatchDimension(for: entry.kind, hiddenSize: hiddenSize),
                pipeline: pipeline)

            routingState.projectionIndex = 0

            return [MetalPrefillStep(
                pipeline: pipeline,
                gridSize: MTLSize(width: config.grid.width, height: maximumSequenceLength, depth: 1),
                threadgroupSize: config.threadgroup,
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
                sequenceLengthBindingIndex: 3,
                positionBufferIndex: nil,
                perPositionStrides: [:]
            )]

        // MARK: Structural Add (hidden + residual → hidden)
        case .structuralAdd(let dimension):
            let prefillKernelContext = KernelContext(
                bufferPrecision: .float32,
                weightFormat: resolveModelWeightFormat(stafWeightStore))
            let resolvedKernelName = kernelName(
                for: entry.kind, entry: entry,
                stafWeightStore: stafWeightStore, kernelContext: prefillKernelContext)
            let pipeline = try getPipeline(resolvedKernelName)
            let config = computeDispatchConfig(
                dimension: dispatchDimension(for: entry.kind, hiddenSize: hiddenSize),
                pipeline: pipeline)

            routingState.lastOutputIsHidden = true

            return [MetalPrefillStep(
                pipeline: pipeline,
                gridSize: MTLSize(width: config.grid.width, height: maximumSequenceLength, depth: 1),
                threadgroupSize: config.threadgroup,
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
                sequenceLengthBindingIndex: 4,
                positionBufferIndex: nil,
                perPositionStrides: [:]
            )]
        }
    }

    /// Build a RMSNorm step that reads from hidden and writes to scratch[0].
    ///
    /// This matches the decode fusedCopyNorm/fusedResidualAddCopyNorm behavior where
    /// norm output goes to scratch[0], allowing parallel projections (gate+up, q+k+v)
    /// to all read from scratch[0] without interference.
    private func buildNormToScratchStep(
        dimension: Int,
        epsilon: Float,
        entry: DispatchEntry,
        buffers: PrefillBufferSet,
        stafWeightStore: STAFWeightStore?,
        slotDimension: Int,
        maximumSequenceLength: Int,
        scratchElementSize: Int,
        pipelineCache: [String: MTLComputePipelineState]
    ) throws -> [MetalPrefillStep] {
        func resolveWeight(role: String) -> (MTLBuffer, Int) {
            if let binding = entry.parameterBindings.first(where: { $0.role == role }),
               let staf = stafWeightStore,
               let access = staf.bufferAccess(for: binding.tensorName) {
                return (access.buffer, access.offset)
            }
            return (buffers.hidden, 0)
        }

        let weightFormat = resolveModelWeightFormat(stafWeightStore)
        let prefillKernelContext = KernelContext(bufferPrecision: .float32, weightFormat: weightFormat)
        let normKernelName = Reduction(dimension: dimension, epsilon: epsilon)
            .kernelName(context: prefillKernelContext)
        guard let pipeline = pipelineCache[normKernelName] else {
            throw MetalCompilerError.kernelNotFound(normKernelName)
        }
        let simdWidth = pipeline.threadExecutionWidth
        let clamped = min(max(dimension, 1), 1024)
        let rounded = ((clamped + simdWidth - 1) / simdWidth) * simdWidth
        let threads = min(rounded, pipeline.maxTotalThreadsPerThreadgroup)

        let (weightBuffer, weightOffset) = resolveWeight(role: "scale")

        return [MetalPrefillStep(
            pipeline: pipeline,
            gridSize: MTLSize(width: maximumSequenceLength, height: 1, depth: 1),
            threadgroupSize: MTLSize(width: threads, height: 1, depth: 1),
            bufferBindings: [
                (0, buffers.hidden, 0),      // input: hidden
                (1, weightBuffer, weightOffset),
                (2, buffers.scratch, 0),     // output: scratch[0]
            ],
            bytesBindings: [
                uint32Binding(3, UInt32(dimension)),
                floatBinding(4, epsilon),
                uint32Binding(5, UInt32(maximumSequenceLength)),
            ],
            threadgroupMemoryLength: 0,
            sync: .bufferBarrier,
            mode: .batch,
            sequenceLengthBindingIndex: 5,
            positionBufferIndex: nil,
            perPositionStrides: [:]
        )]
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

        /// Emit an optimizer result entry.
        mutating func emitOptimized(_ entry: OptimizedEntry) {
            switch entry {
            case .single(let p):
                // .gemv dispatch dimension → .projection, others → .fragment
                if case .gemv(let outputDim, let inputDim) = p.fragment.dispatchDimension {
                    let field = p.fragment.weightSlots.first?.field ?? "weight"
                    let projection = MetalProjection(
                        field: field,
                        inputDimension: inputDim,
                        outputDimension: outputDim)
                    emit(.projection(projection),
                         parameterBindings: p.parameterBindings,
                         layerIndex: p.layerIndex)
                } else {
                    emit(.fragment(p.fragment),
                         parameterBindings: p.parameterBindings,
                         layerIndex: p.layerIndex)
                }
            case .batchedProjection(let batched, let bindings, let layer):
                emit(.batchedProjection(batched),
                     parameterBindings: bindings,
                     layerIndex: layer)
            case .batchedFragment(let batched, let bindings, let layer):
                emit(.batchedFragment(batched),
                     parameterBindings: bindings,
                     layerIndex: layer)
            }
        }
    }

    struct CacheSlotInfo {
        let kvHeadCount: Int
        let headDimension: Int
    }

    /// Resolve the primary weight format from the STAF weight store.
    private func resolveModelWeightFormat(_ stafWeightStore: STAFWeightStore?) -> WeightFormat {
        guard let staf = stafWeightStore else { return .float16 }
        for name in staf.entries.keys {
            if let info = staf.tensor(for: name) {
                return info.format.schemeIdentifier == .bf16RowMajor ? .bfloat16 : .float16
            }
        }
        return .float16
    }

    private func walkRegion(
        _ region: Region,
        pathComponents: [StructuralPathComponent],
        layerIndex: Int?,
        hiddenSize: Int,
        context: inout WalkContext,
        kernelContext: KernelContext
    ) {
        for (operationIndex, operation) in region.operations.enumerated() {
            let operationPath = pathComponents + [.operation(operationIndex)]
            let _ = StructuralPath(components: operationPath)

            switch operation.kind {
            case .residual(_, let body):
                context.emit(.structuralCopy(dimension: hiddenSize), layerIndex: layerIndex)
                walkRegion(body, pathComponents: operationPath + [.regionBody],
                           layerIndex: layerIndex, hiddenSize: hiddenSize, context: &context,
                           kernelContext: kernelContext)
                context.emit(.structuralAdd(dimension: hiddenSize), layerIndex: layerIndex)

            case .repeating(let count, let body):
                for iteration in 0..<count {
                    walkRegion(body,
                               pathComponents: operationPath + [.regionBody, .index(iteration)],
                               layerIndex: iteration, hiddenSize: hiddenSize, context: &context,
                               kernelContext: kernelContext)
                }

            case .conditional(let condition, let thenBody, let elseBody):
                if let currentLayer = layerIndex, case .layerIndices(let indices) = condition {
                    let selectedBody = indices.contains(currentLayer) ? thenBody : elseBody
                    walkRegion(selectedBody, pathComponents: operationPath + [.regionBody],
                               layerIndex: currentLayer, hiddenSize: hiddenSize, context: &context,
                               kernelContext: kernelContext)
                } else {
                    walkRegion(thenBody, pathComponents: operationPath + [.regionBody],
                               layerIndex: layerIndex, hiddenSize: hiddenSize, context: &context,
                               kernelContext: kernelContext)
                }

            case .parallel(_, let branches):
                for (branchIndex, branch) in branches.enumerated() {
                    walkRegion(branch,
                               pathComponents: operationPath + [.regionBranch(branchIndex)],
                               layerIndex: layerIndex, hiddenSize: hiddenSize, context: &context,
                               kernelContext: kernelContext)
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

                // Fragment-driven path: collect → optimize → emit
                guard let fragment = attributes as? (any MetalKernelFragment) else { continue }
                var primitives: [CollectedPrimitive] = []
                collectPrimitives(fragment, bindings: bindings, layerIndex: layerIndex,
                                  primitives: &primitives, context: &context,
                                  kernelContext: kernelContext)
                let optimized = optimizer.optimizeFragment(primitives)
                let startIndex = context.entries.count
                for entry in optimized {
                    context.emitOptimized(entry)
                }
                markLastProjectionAsOutput(entries: &context.entries, from: startIndex)
            }
        }
    }

    // MARK: - Fragment Tree Walk

    // MARK: - Primitive Collection (for optimizer)

    /// Collect all primitives from a fragment tree without emitting.
    ///
    /// Similar to `emitFragmentTree()` but appends to an array instead of emitting.
    /// FlashAttentionFragment still registers KV cache slots in the context.
    private func collectPrimitives(
        _ fragment: any MetalKernelFragment,
        bindings: [ParameterBinding],
        layerIndex: Int?,
        primitives: inout [CollectedPrimitive],
        context: inout WalkContext,
        kernelContext: KernelContext
    ) {
        if let primitive = fragment as? any PrimitiveMetalKernelFragment {
            // Register KV cache slot from fragment's cache slot metadata
            for slot in primitive.cacheSlots where slot.kind == .kv {
                context.cacheSlots.append(CacheSlotInfo(
                    kvHeadCount: slot.kvHeadCount,
                    headDimension: slot.headDimension))
            }
            primitives.append(CollectedPrimitive(
                fragment: primitive,
                parameterBindings: bindings,
                layerIndex: layerIndex))
            return
        }

        // Walk composite fragments recursively
        if let tuple = fragment as? any _TupleFragmentProtocol {
            tuple._visitChildren { child in
                collectPrimitives(child, bindings: bindings, layerIndex: layerIndex,
                                  primitives: &primitives, context: &context,
                                  kernelContext: kernelContext)
            }
            return
        }
        if let opt = fragment as? any _OptionalFragmentProtocol {
            opt._visitContent { child in
                collectPrimitives(child, bindings: bindings, layerIndex: layerIndex,
                                  primitives: &primitives, context: &context,
                                  kernelContext: kernelContext)
            }
            return
        }
        if let cond = fragment as? any _ConditionalFragmentProtocol {
            cond._visitActive { child in
                collectPrimitives(child, bindings: bindings, layerIndex: layerIndex,
                                  primitives: &primitives, context: &context,
                                  kernelContext: kernelContext)
            }
            return
        }
        if let bodyAccessor = fragment as? any _FragmentBodyAccessor {
            bodyAccessor._visitBody(context: kernelContext) { child in
                collectPrimitives(child, bindings: bindings, layerIndex: layerIndex,
                                  primitives: &primitives, context: &context,
                                  kernelContext: kernelContext)
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

    // MARK: - Kernel Name Resolution

    /// Map a DispatchKind to the MSL kernel function name.
    /// Map a DispatchKind to the MSL kernel function name.
    ///
    /// Uses KernelContext for weight format resolution — no STAF lookups.
    /// Projection kernels still use STAF for per-tensor quantization format
    /// (e.g., gemv_q4_g64 for quantized models).
    private func kernelName(
        for kind: DispatchKind,
        entry: DispatchEntry,
        stafWeightStore: STAFWeightStore?,
        kernelContext: KernelContext
    ) -> String {
        let isBF16 = kernelContext.weightFormat == .bfloat16
        let bf16Suffix = isBF16 ? "_bf16" : ""

        let isPrefill = kernelContext.bufferPrecision == .float32

        switch kind {
        case .projection(let projection, _):
            // Projection uses per-tensor format from STAF (supports mixed quantization)
            if let binding = entry.parameterBindings.first(where: { $0.role == projection.field }),
               let staf = stafWeightStore,
               let tensorInfo = staf.tensor(for: binding.tensorName) {
                return isPrefill
                    ? tensorInfo.format.gemmKernelName(bufferPrecision: kernelContext.bufferPrecision)
                    : tensorInfo.format.gemvKernelName
            }
            return isPrefill ? (isBF16 ? "gemm_bf16_f32s" : "gemm_f32s") : "gemv"
        case .fragment(let frag):
            return frag.kernelName(context: kernelContext)
        case .fusedCopyNorm:
            return "fused_copy_rms_norm" + bf16Suffix
        case .fusedResidualAddCopyNorm:
            return "fused_residual_add_copy_rms_norm" + bf16Suffix
        case .fusedResidualAddNorm:
            return "fused_residual_add_rms_norm" + bf16Suffix
        case .batchedProjection(let batched):
            return "batched_gemv\(batched.projections.count)" + bf16Suffix
        case .batchedFragment(let batch):
            let baseName = batch.fragments[0].kernelName(context: kernelContext)
            return "batched_\(baseName)_\(batch.fragments.count)"
        case .structuralCopy:
            return isPrefill ? "copy_buffer_seq_f32" : "copy_buffer"
        case .structuralAdd:
            return isPrefill ? "residual_add_seq_f32" : "residual_add"
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
        case .fusedResidualAddNorm(let fused):
            return .reduction(dimension: fused.dimension)
        case .batchedProjection(let batched):
            return .gemv(outputDimension: batched.totalOutputDimension, inputDimension: batched.inputDimension)
        case .batchedFragment(let batch):
            return batch.dispatchDimension
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
        // MARK: Fragment-driven buffer routing (protocol dispatch — no type checks)
        case .fragment(let fragment):
            let bindingContext = BufferBindingContext(
                bufferSet: bufferSet, slotDimension: slotDimension,
                elementSize: elementSize, kvCacheIndex: kvCacheIndex,
                convLayerIndex: routingState.convLayerIndex,
                resolveWeight: resolveWeight)
            let bindings = fragment.decodeBindings(context: bindingContext)
            if bindings.resetsProjectionIndex { routingState.projectionIndex = 0 }
            if bindings.consumesKVCacheLayer { kvCacheIndex += 1 }
            if bindings.consumesConvLayer { routingState.convLayerIndex += 1 }
            routingState.lastOutputIsHidden = bindings.outputIsHidden
            return (buffers: bindings.buffers, bytes: bindings.bytes)

        // MARK: Fused Residual Add + RMS Norm (no copy)
        case .fusedResidualAddNorm(let fusedOperation):
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

        // MARK: Batched Projection
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

            // Bind weights: indices 1..<1+count
            for (i, proj) in batched.projections.enumerated() {
                let (weightBuf, weightOff) = resolveWeight(role: proj.field)
                bufferBindings.append((1 + i, weightBuf, weightOff))
            }

            // Bind outputs: indices 1+count..<1+2*count
            for i in 0..<count {
                let scratchSlot = routingState.projectionIndex + 1
                let outputOffset = scratchSlot * slotDimension * elementSize
                bufferBindings.append((1 + count + i, bufferSet.scratch, outputOffset))
                routingState.projectionIndex += 1
            }

            // Bytes: inputDimension, then each outputDimension
            let bytesStart = 1 + 2 * count
            bytesBindings.append(uint32Binding(bytesStart, UInt32(batched.inputDimension)))
            for (i, proj) in batched.projections.enumerated() {
                bytesBindings.append(uint32Binding(bytesStart + 1 + i, UInt32(proj.outputDimension)))
            }

            routingState.lastOutputIsHidden = false
            return (buffers: bufferBindings, bytes: bytesBindings)

        // MARK: Batched Fragment (per-head)
        case .batchedFragment(let batch):
            let slotBytes = slotDimension * elementSize
            var bufferBindings: [(index: Int, buffer: MTLBuffer, offset: Int)] = []
            var bytesBindings: [(index: Int, value: [UInt8])] = []

            // Data slots: in-place fragments operate on consecutive projection outputs.
            // First batched fragment → slot 1 (first projection output),
            // second → slot 2, etc.
            for i in 0..<batch.fragments.count {
                let scratchSlotIndex = 1 + i
                bufferBindings.append((i, bufferSet.scratch, scratchSlotIndex * slotBytes))
            }

            // Weight bindings: resolve from each fragment's weightSlots
            for (i, frag) in batch.fragments.enumerated() {
                if let weightSlot = frag.weightSlots.first {
                    let role = weightSlot.field ?? "weight"
                    let (weightBuffer, weightOffset) = resolveWeight(role: role)
                    bufferBindings.append((batch.fragments.count + i, weightBuffer, weightOffset))
                }
            }

            // Constants: head count per fragment, then shared headDimension and epsilon.
            // All derived from fragment properties — no type checks.
            let bytesStart = 2 * batch.fragments.count
            for (i, frag) in batch.fragments.enumerated() {
                if case .perHead(let headCount) = frag.dispatchDimension {
                    bytesBindings.append(uint32Binding(bytesStart + i, UInt32(headCount)))
                }
            }

            // Extract headDimension from dispatchDimension + weight slot count.
            // For per-head fragments: total elements / headCount = headDimension.
            if case .perHead(let totalHeads) = batch.dispatchDimension,
               let firstFrag = batch.fragments.first,
               case .perHead(let firstHeadCount) = firstFrag.dispatchDimension {
                // headDimension: infer from hiddenSize / headCount of first fragment.
                // For QK norm: q_proj output = headCount * headDimension.
                let headDimension = hiddenSize / firstHeadCount
                bytesBindings.append(uint32Binding(bytesStart + batch.fragments.count, UInt32(headDimension)))
                let epsilon = batch.fragments.first?.normEpsilon ?? 1e-6
                bytesBindings.append(floatBinding(bytesStart + batch.fragments.count + 1, epsilon))
            }

            return (buffers: bufferBindings, bytes: bytesBindings)

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


    /// Compile Metal libraries and build a pipeline cache for the given dispatch entries.
    ///
    /// Kernel source is generated on-demand from fragment parameters + STAF weight format.
    /// No hardcoded catalog — only the kernels actually used are compiled.
    /// For prefill (F32), attempts Metal 4 MPP GEMM with fallback to naive GEMM.
    private func compilePipelineCache(
        entries: [DispatchEntry],
        stafWeightStore: STAFWeightStore?,
        bufferPrecision: MetalSourceGenerator.BufferPrecision,
        device: MTLDevice
    ) throws -> (pipelines: [String: MTLComputePipelineState], usesMPP: Bool) {
        var detectedMPP = false
        var sources: [String] = [MetalSourceGenerator.commonHeader]
        var generatedNames: Set<String> = []

        // Metal 4 MPP GEMM: compiled separately with Metal 4 language version
        var mppGEMMNames: Set<String> = []
        var mppGEMMWeightFormat: MetalSourceGenerator.WeightFormat = .float16

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
                        // MPP GEMM compiled separately in Metal 4 library (see below)
                        mppGEMMNames.insert(gemmName)
                        mppGEMMWeightFormat = weightFormat
                        // Also generate fallback for non-Metal4
                        sources.append(MetalSourceGenerator.generateGEMM(name: gemmName, bufferPrecision: bufferPrecision, weightFormat: weightFormat))
                    } else {
                        sources.append(MetalSourceGenerator.generateGEMV(name: name, bufferPrecision: bufferPrecision, weightFormat: weightFormat))
                    }
                }

            case .fragment(let frag):
                let weightFormat = resolveWeightFormat(forFragment: frag, entry: entry, stafWeightStore: stafWeightStore)
                let fragCtx = KernelContext(bufferPrecision: bufferPrecision, weightFormat: weightFormat)
                let kernelName = frag.kernelName(context: fragCtx)
                if generatedNames.insert(kernelName).inserted {
                    // Fragment provides its own MSL source — no type switch needed
                    let src = frag.kernelSource(name: kernelName, bufferPrecision: bufferPrecision, weightFormat: weightFormat)
                    if !frag.cacheSlots.filter({ $0.kind == .kv }).isEmpty { needsFlashAttnHelper = true }
                    sources.append(src)
                }
                // Conv cache fragments in prefill also need extract_conv_state kernel
                if frag.cacheSlots.contains(where: { $0.kind == .conv }) && bufferPrecision == .float32 {
                    let extractName = "extract_conv_state_f32"
                    if generatedNames.insert(extractName).inserted {
                        sources.append(MetalSourceGenerator.generateExtractConvState(name: extractName, bufferPrecision: bufferPrecision))
                    }
                }
                // Prefill attention: batch KV fill + batch causal attention
                if frag.cacheSlots.contains(where: { $0.kind == .kv }) && bufferPrecision == .float32 {
                    for name in ["kv_cache_fill_seq_f32", "flash_attn_batch_f32"] {
                        if generatedNames.insert(name).inserted {
                            if name.contains("kv_cache_fill") {
                                sources.append(MetalSourceGenerator.generateKVCacheFillSeq(name: name, bufferPrecision: bufferPrecision))
                            } else {
                                sources.append(MetalSourceGenerator.generateBatchFlashAttention(name: name, bufferPrecision: bufferPrecision))
                            }
                        }
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

            case .fusedResidualAddNorm(_):
                let weightFormat = resolveWeightFormat(role: "scale", entry: entry, stafWeightStore: stafWeightStore)
                let kernelName = weightFormat == .bfloat16 ? "fused_residual_add_rms_norm_bf16" : "fused_residual_add_rms_norm"
                if bufferPrecision == .float16 {
                    if generatedNames.insert(kernelName).inserted {
                        sources.append(MetalSourceGenerator.generateFusedResidualAddRMSNorm(
                            name: kernelName, bufferPrecision: bufferPrecision, weightFormat: weightFormat))
                    }
                }
                // Prefill: decomposed into add + norm steps by buildPrefillSteps.
                // Individual kernels (residual_add_seq, rms_norm_seq) are generated
                // by their own dispatch entries — no additional generation needed here.

            case .batchedProjection(let batched):
                let count = batched.projections.count
                let weightFormat = resolveWeightFormat(
                    role: batched.projections[0].field, entry: entry, stafWeightStore: stafWeightStore)
                if bufferPrecision == .float16 {
                    let suffix = weightFormat == .bfloat16 ? "_bf16" : ""
                    let kernelName = "batched_gemv\(count)\(suffix)"
                    if generatedNames.insert(kernelName).inserted {
                        if count == 2 {
                            sources.append(MetalSourceGenerator.generateBatchedGEMV2(
                                name: kernelName, bufferPrecision: bufferPrecision, weightFormat: weightFormat))
                        } else {
                            sources.append(MetalSourceGenerator.generateBatchedGEMV3(
                                name: kernelName, bufferPrecision: bufferPrecision, weightFormat: weightFormat))
                        }
                    }
                } else {
                    // Prefill: decompose into individual GEMMs (handled by buildPrefillSteps)
                    let gemmName = weightFormat == .bfloat16 ? "gemm_bf16_f32s" : "gemm_f32s"
                    if generatedNames.insert(gemmName).inserted {
                        sources.append(MetalSourceGenerator.generateGEMM(
                            name: gemmName, bufferPrecision: bufferPrecision, weightFormat: weightFormat))
                    }
                }

            case .batchedFragment(let batch):
                if bufferPrecision == .float16 {
                    let ctx = KernelContext(bufferPrecision: bufferPrecision, weightFormat: resolveModelWeightFormat(stafWeightStore))
                    let kernelName = self.kernelName(for: entry.kind, entry: entry, stafWeightStore: stafWeightStore, kernelContext: ctx)
                    if generatedNames.insert(kernelName).inserted {
                        let weightFormat = resolveWeightFormat(role: "q_layernorm", entry: entry, stafWeightStore: stafWeightStore)
                        if batch.fragments.count == 2, case .perHead = batch.dispatchDimension {
                            sources.append(MetalSourceGenerator.generateBatchedPerHead2(
                                name: kernelName, bufferPrecision: bufferPrecision, weightFormat: weightFormat))
                        }
                    }
                } else {
                    // Prefill: decompose into individual QKNorm steps (handled by buildPrefillSteps)
                    // buildPrefillSteps uses "qk_rms_norm_seq_f32" regardless of weight format
                    let weightFormat = resolveWeightFormat(role: "q_layernorm", entry: entry, stafWeightStore: stafWeightStore)
                    let normName = "qk_rms_norm_seq_f32"
                    if generatedNames.insert(normName).inserted {
                        sources.append(MetalSourceGenerator.generateQKNormSeq(
                            name: normName, bufferPrecision: bufferPrecision, weightFormat: weightFormat))
                    }
                }
            }
        }

        if needsFlashAttnHelper {
            sources.insert(MetalSourceGenerator.flashAttentionHelperSource, at: 1)
        }

        let compileOptions = MTLCompileOptions()
        compileOptions.fastMathEnabled = false
        compileOptions.languageVersion = .version4_0
        let library = try device.makeLibrary(source: sources.joined(separator: "\n\n"), options: compileOptions)

        // Build base pipeline cache from Metal 3 library
        var pipelineCache: [String: MTLComputePipelineState] = [:]
        for name in library.functionNames {
            if let function = library.makeFunction(name: name) {
                let descriptor = MTLComputePipelineDescriptor()
                descriptor.computeFunction = function
                descriptor.label = name
                pipelineCache[name] = try device.makeComputePipelineState(
                    descriptor: descriptor, options: [], reflection: nil)
            }
        }

        // Compile Metal 4 MPP GEMM kernels as a separate library.
        // MPP requires Metal 4.0 language version and MetalPerformancePrimitives framework headers.
        // If Metal 4 compilation fails (older OS/GPU), the Metal 3 fallback GEMM is already in cache.
        if !mppGEMMNames.isEmpty {
            var mppSources: [String] = []
            for name in mppGEMMNames {
                mppSources.append(MetalSourceGenerator.generateMPPGEMM(
                    name: name, bufferPrecision: bufferPrecision, weightFormat: mppGEMMWeightFormat))
            }
            let mppOptions = MTLCompileOptions()
            mppOptions.languageVersion = .version4_0
            do {
                let mppLibrary = try device.makeLibrary(
                    source: mppSources.joined(separator: "\n\n"), options: mppOptions)
                detectedMPP = true
                for name in mppLibrary.functionNames {
                    if let function = mppLibrary.makeFunction(name: name) {
                        let descriptor = MTLComputePipelineDescriptor()
                        descriptor.computeFunction = function
                        descriptor.label = name
                        pipelineCache[name] = try device.makeComputePipelineState(
                            descriptor: descriptor, options: [], reflection: nil)
                    }
                }
            } catch { /* Metal 4 MPP unavailable — using Metal 3 fallback GEMM */ }
        }

        return (pipelineCache, detectedMPP)
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


    // MARK: - Helpers

    private func roundUp(_ value: Int, to multiple: Int) -> Int {
        guard multiple > 0 else { return max(value, 1) }
        return ((value + multiple - 1) / multiple) * multiple
    }

}
