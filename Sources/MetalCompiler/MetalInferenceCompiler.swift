import Metal
import LMIR

// MARK: - Dispatch Plan Types

public struct MetalDispatchStep: @unchecked Sendable {
    public let pipeline: MTLComputePipelineState
    public let gridSize: MTLSize
    public let threadgroupSize: MTLSize
    public let bufferBindings: [(index: Int, buffer: MTLBuffer, offset: Int)]
    public let bytesBindings: [(index: Int, value: [UInt8])]
    public let threadgroupMemoryLength: Int
    public let sync: SynchronizationKind
}

public struct MetalDispatchPlan: @unchecked Sendable {
    public let steps: [MetalDispatchStep]
    public let buffers: MetalBufferSet
    /// Number of entries before fusion (for diagnostics).
    public let unfusedEntryCount: Int
    /// Number of entries after fusion (for diagnostics).
    public let fusedEntryCount: Int
}

public struct MetalBufferSet: @unchecked Sendable {
    public let hidden: MTLBuffer
    public let residual: MTLBuffer
    public let scratch: MTLBuffer
    public let weights: [MTLBuffer]
    /// Consolidated KV cache (single K + single V buffer for all layers).
    public let kvCache: MetalKVCache?
    /// Conv state for ShortConv layers: [numConvLayers × kernelSize × convDimension] in Float16.
    /// Each conv layer has a temporal window of the last kernelSize tokens' conv inputs.
    public let convState: MTLBuffer?
    /// Conv state layout: dimension per conv layer slot.
    public let convStateDimension: Int
    /// Conv state layout: kernel size (temporal window).
    public let convStateKernelSize: Int
    public let logits: MTLBuffer
    public let position: MTLBuffer
    public let tokenIn: MTLBuffer
    public let tokenOut: MTLBuffer
}

// MARK: - Prefill Plan Types

/// Execution mode for a prefill step.
public enum PrefillStepMode: Sendable {
    /// Dispatched once for the entire sequence. Grid includes seqLen dimension.
    /// Kernel operates on [seqLen × dim] buffers.
    case batch
    /// Dispatched once per sequence position. Used for attention (KV cache is position-dependent).
    /// The runtime loops over positions and adjusts offsets per dispatch.
    case perPosition
    /// Dispatched once, for the last token only. Used for output head and argmax.
    /// Runtime adjusts input buffer offsets by (seqLen - 1) * stride using perPositionStrides.
    case lastToken
}

/// A single step in the prefill sequence graph.
///
/// `steps.count = O(layers × ops_per_layer)` — does NOT scale with token count.
/// Batch steps operate on [seqLen × dim] buffers. PerPosition steps are looped
/// by the runtime for the attention KV cache fill.
public struct MetalPrefillStep: @unchecked Sendable {
    public let pipeline: MTLComputePipelineState
    /// Grid size for batch steps. For perPosition steps, this is the per-token grid.
    public let gridSize: MTLSize
    public let threadgroupSize: MTLSize
    public let bufferBindings: [(index: Int, buffer: MTLBuffer, offset: Int)]
    public let bytesBindings: [(index: Int, value: [UInt8])]
    public let threadgroupMemoryLength: Int
    public let sync: SynchronizationKind
    public let mode: PrefillStepMode
    /// For batch steps with seqLen in grid.y: the seqLen bytes binding index.
    /// Runtime overwrites this with actual sequence length.
    public let sequenceLengthBindingIndex: Int?
    /// For perPosition steps: the buffer binding index for the position value.
    public let positionBufferIndex: Int?
    /// For perPosition steps: per-buffer-binding stride (bytes per position).
    /// Key = buffer binding index, Value = stride in bytes.
    /// Buffers not in this map are not adjusted per position (e.g., KV cache, weights).
    public let perPositionStrides: [Int: Int]
}

/// Sequence-aware execution plan for prefill.
///
/// `steps.count` is constant regardless of token count.
/// The runtime dispatches batch steps once and loops perPosition steps.
///
/// Buffer layout: all intermediate buffers are [maxSeqLen × dim].
public struct MetalPrefillPlan: @unchecked Sendable {
    public let steps: [MetalPrefillStep]
    public let buffers: PrefillBufferSet
    /// Maximum sequence length supported by the allocated buffers.
    public let maximumSequenceLength: Int
    /// Step count (should be constant, not proportional to token count).
    public let stepCount: Int
}

/// Prefill-specific buffer set with [maxSeqLen × dim] layout.
public struct PrefillBufferSet: @unchecked Sendable {
    /// Hidden states: [maxSeqLen × hiddenSize]
    public let hidden: MTLBuffer
    /// Residual: [maxSeqLen × hiddenSize]
    public let residual: MTLBuffer
    /// Scratch workspace: [maxSeqLen × maxDim]
    public let scratch: MTLBuffer
    /// Weight buffers (shared with decode — same STAF data)
    public let weights: [MTLBuffer]
    /// KV cache (shared with decode)
    public let kvCache: MetalKVCache?
    /// Logits: [vocabSize] — only last token needed
    public let logits: MTLBuffer
    /// Token IDs input: [maxSeqLen]
    public let tokenIDs: MTLBuffer
    /// Position array: [maxSeqLen]
    public let positions: MTLBuffer
    /// Token output: [1]
    public let tokenOut: MTLBuffer
}

// MARK: - Dispatch Entry (Intermediate Representation)

/// A single kernel dispatch in the plan.
/// Produced by the IR walk, transformed by the fusion pass.
public struct DispatchEntry: Sendable {
    public let index: Int
    public let kind: DispatchKind
    /// Parameter bindings from the IR operation (weight tensor names).
    public let parameterBindings: [ParameterBinding]
    public let layerIndex: Int?

    public init(index: Int, kind: DispatchKind, parameterBindings: [ParameterBinding] = [], layerIndex: Int? = nil) {
        self.index = index
        self.kind = kind
        self.parameterBindings = parameterBindings
        self.layerIndex = layerIndex
    }
}

public enum DispatchKind: Sendable {
    /// GEMV projection. `isOutput` marks the last projection in an operation
    /// (writes to hidden buffer instead of scratch).
    case projection(MetalProjection, isOutput: Bool = false)
    case compute(any MetalComputeOperation)
    case structuralCopy(dimension: Int)
    case structuralAdd(dimension: Int)
}

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

        // Phase 3: Compile all kernel source → single MTLLibrary
        // Use MTLBinaryArchive to cache compiled pipelines on disk.
        // First load avoids JIT compilation on subsequent launches.
        let library = try Self.getOrCompileLibrary(device: device)

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

        // DEBUG: all buffers shared for CPU inspection. Change back to private for production.
        let gpuOnlyOptions: MTLResourceOptions = [.storageModeShared]
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
            if case .compute(let op) = entry.kind, let convOp = op as? Conv1dOperation {
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
            // Conv1d with conv_state uses a different kernel
            var resolvedKernelName = kernelName(for: entry.kind, entry: entry, stafWeightStore: stafWeightStore)
            if case .compute(let op) = entry.kind, op is Conv1dOperation, bufferSet.convState != nil {
                resolvedKernelName = "conv_state_update"
            }
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

        // Get compiled library
        let library = try Self.getOrCompileLibrary(device: device)
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
        let elementSize = MemoryLayout<Float16>.size
        let resolvedIntermediateSize = max(intermediateSize, hiddenSize * 4)
        let resolvedVocabSize = max(vocabSize, 1)
        // Slot stride must accommodate the largest projection output
        let slotDimension = max(hiddenSize, resolvedIntermediateSize, maxProjectionOutputDimension)
        let maxSeq = maximumSequenceLength
        let scratchElementCount = max(slotDimension * 4, resolvedIntermediateSize * 4)
        print("[Compiler] prefill: resolvedIntermediate=\(resolvedIntermediateSize) maxProjOutput=\(maxProjectionOutputDimension) slotDim=\(slotDimension) scratchElements=\(scratchElementCount) scratch=\(maxSeq * scratchElementCount * elementSize)")
        let gpuOptions: MTLResourceOptions = [.storageModeShared]

        let f32ElementSize = MemoryLayout<Float>.size  // 4 bytes — all intermediate buffers are float32
        let prefillBuffers = PrefillBufferSet(
            hidden: device.makeBuffer(length: maxSeq * hiddenSize * f32ElementSize, options: gpuOptions)!,
            residual: device.makeBuffer(length: maxSeq * hiddenSize * f32ElementSize, options: gpuOptions)!,
            scratch: device.makeBuffer(length: maxSeq * scratchElementCount * f32ElementSize, options: gpuOptions)!,
            weights: stafWeightStore.map { [$0.buffer] } ?? [],
            kvCache: walkContext.cacheSlots.isEmpty ? nil : try {
                guard let firstSlot = walkContext.cacheSlots.first else { throw MetalCompilerError.deviceSetupFailed("No cache slots") }
                return try MetalKVCache(
                    device: device,
                    specification: KVCacheSpecification(
                        layerCount: walkContext.cacheSlots.count,
                        kvHeadCount: firstSlot.kvHeadCount,
                        headDimension: firstSlot.headDimension))
            }(),
            logits: device.makeBuffer(length: resolvedVocabSize * elementSize, options: gpuOptions)!,
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
        case .compute(let op) where op is EmbeddingLookupOperation:
            let embOp = op as! EmbeddingLookupOperation
            let (weightBuffer, weightOffset) = resolveWeight(role: "embedding_table")
            // Determine kernel variant (BF16 or FP16)
            var kernelName = "embedding_lookup_seq_f32"  // FP16 default with float32 output
            if let staf = stafWeightStore {
                if let binding = entry.parameterBindings.first(where: { $0.role == "embedding_table" }) {
                    if let info = staf.tensor(for: binding.tensorName) {
                        print("[Prefill compiler] embedding '\(binding.tensorName)' scheme=\(info.format.schemeIdentifier) kernel→\(info.format.schemeIdentifier == .bf16RowMajor ? "embedding_lookup_seq_bf16" : "embedding_lookup_seq")")
                        if info.format.schemeIdentifier == .bf16RowMajor {
                            kernelName = "embedding_lookup_seq_bf16_f32"
                        }
                    } else {
                        print("[Prefill compiler] WARNING: embedding '\(binding.tensorName)' NOT FOUND in STAF")
                    }
                } else {
                    print("[Prefill compiler] WARNING: no 'embedding_table' binding in parameterBindings: \(entry.parameterBindings.map(\.role))")
                }
            } else {
                print("[Prefill compiler] WARNING: no STAF weight store")
            }
            print("[Prefill compiler] embedding kernel selected: \(kernelName)")
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

            // Select GEMM kernel variant based on weight format
            var kernelName = "gemm_f32s"
            if let binding = entry.parameterBindings.first(where: { $0.role == projection.field }),
               let staf = stafWeightStore,
               let tensorInfo = staf.tensor(for: binding.tensorName) {
                // Select float32 scratch GEMM variant based on weight format
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
            // All projections use f32s (float32 I/O) — hidden is also float32 now
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

            // Log projection details for debugging
            let bindingName = entry.parameterBindings.first(where: { $0.role == projection.field })?.tensorName ?? "?"
            let inputSrc = routingState.lastOutputIsHidden ? "hidden" : "scratch"
            let outputDst = isOutput ? (projection.outputDimension > hiddenSize ? "logits" : "hidden") : "scratch[\(routingState.projectionIndex + 1)]"
            print("[Prefill proj] \(bindingName) in=\(projection.inputDimension) out=\(projection.outputDimension) \(inputSrc)→\(outputDst) isOutput=\(isOutput)")

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
            // Uses GEMV-style single-token dispatch. Runtime adjusts input offset to last token.
            if isOutputHead {
                // Use float32 input → half output kernel for output head (hidden is float32, logits is float16)
                let gemvPipeline = try getPipeline("gemm_bf16_f32_to_half")
                let gemvThreads = min(2 * gemvPipeline.threadExecutionWidth, gemvPipeline.maxTotalThreadsPerThreadgroup)
                let inputStride = projection.inputDimension * scratchElementSize
                return [MetalPrefillStep(
                    pipeline: gemvPipeline,
                    gridSize: MTLSize(width: gridX, height: 1, depth: 1),
                    threadgroupSize: MTLSize(width: gemvThreads, height: 1, depth: 1),
                    bufferBindings: [
                        (0, inputBuffer, inputOffset),
                        (1, weightBuffer, weightOffset),
                        (2, outputBuffer, outputOffset),
                    ],
                    bytesBindings: [
                        uint32Binding(3, UInt32(projection.inputDimension)),
                        uint32Binding(4, UInt32(projection.outputDimension)),
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
        // Standalone RMSNorm writes in-place to hidden (Float16).
        // embedding_norm and final_norm results stay in hidden for the next operation.
        // Uses Float16 kernel (NOT f32s) because output is hidden, not scratch.
        case .compute(let op) where op is RMSNormOperation:
            let normOp = op as! RMSNormOperation
            let (weightBuffer, weightOffset) = resolveWeight(role: "scale")
            // Detect BF16 norm weights — still Float16 output (hidden is Float16)
            var normKernelName = "rms_norm_seq_f32_inplace"
            if let staf = stafWeightStore,
               let binding = entry.parameterBindings.first(where: { $0.role == "scale" }),
               let info = staf.tensor(for: binding.tensorName),
               info.format.schemeIdentifier == .bf16RowMajor {
                normKernelName = "rms_norm_seq_bf16_f32_inplace"
                print("[Prefill compiler] norm '\(binding.tensorName)' scheme=bf16 → \(normKernelName)")
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
        case .compute(let op) where op is CopyRMSNormOperation:
            let fusedOp = op as! CopyRMSNormOperation
            let (weightBuffer, weightOffset) = resolveWeight(role: "scale")

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
        case .compute(let op) where op is ResidualAddCopyRMSNormOperation:
            let fusedOp = op as! ResidualAddCopyRMSNormOperation
            let (weightBuffer, weightOffset) = resolveWeight(role: "scale")

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
        case .compute(let op) where op is FlashAttentionDecodeOperation:
            let flashOp = op as! FlashAttentionDecodeOperation
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

        // MARK: SwiGLU (batch)
        case .compute(let op) where op is SwiGLUOperation:
            let swigluOp = op as! SwiGLUOperation
            let pipeline = try getPipeline("swiglu_seq_f32")
            let maxScratchSlot = slotDimension * scratchElementSize * maximumSequenceLength
            let tgSize = min(256, pipeline.maxTotalThreadsPerThreadgroup)
            let gridX = (swigluOp.dimension + tgSize - 1) / tgSize
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
                    uint32Binding(3, UInt32(swigluOp.dimension)),
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

        // MARK: Structural Add (batch)
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
        case .compute(let op) where op is RoPEOperation:
            let ropeOp = op as! RoPEOperation
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
        case .compute(let op) where op is QKNormOperation:
            let qkOp = op as! QKNormOperation
            var qkNormKernelName = "qk_rms_norm_seq_f32"
            if let staf = stafWeightStore,
               let binding = entry.parameterBindings.first(where: { $0.role == qkOp.weightRole }),
               let info = staf.tensor(for: binding.tensorName),
               info.format.schemeIdentifier == .bf16RowMajor {
                qkNormKernelName = "qk_rms_norm_seq_f32"
            }
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
        case .compute(let op) where op is ArgmaxOperation:
            let argmaxOp = op as! ArgmaxOperation
            let pipeline = try getPipeline("argmax")
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

        // MARK: Conv1d (gated depthwise conv for ShortConv — perPosition due to conv state)
        case .compute(let op) where op is Conv1dOperation:
            let convOp = op as! Conv1dOperation
            let (weightBuffer, weightOffset) = resolveWeight(role: "conv_weight")
            let pipeline = try getPipeline("conv1d_f32")
            let tgSize = min(256, pipeline.maxTotalThreadsPerThreadgroup)
            let gridX = (convOp.dimension + tgSize - 1) / tgSize
            // in_proj wrote to scratch slot 1. Conv1d reads from slot 1, writes to slot 0.
            // The GEMM output stride per position is outputDimension (= convOp.dimension * convOp.kernelSize)
            let scratchSlotBytes = slotDimension * scratchElementSize * maximumSequenceLength
            let inProjOutputDimension = convOp.dimension * convOp.kernelSize
            let perPosInputStride = inProjOutputDimension * scratchElementSize
            let perPosOutputStride = convOp.dimension * scratchElementSize
            routingState.lastOutputIsHidden = false
            routingState.projectionIndex = 0  // reset for out_proj
            return [MetalPrefillStep(
                pipeline: pipeline,
                gridSize: MTLSize(width: gridX, height: 1, depth: 1),
                threadgroupSize: MTLSize(width: tgSize, height: 1, depth: 1),
                bufferBindings: [
                    (0, buffers.scratch, 1 * scratchSlotBytes),  // input from in_proj (slot 1)
                    (1, weightBuffer, weightOffset),
                    (2, buffers.scratch, 0),                      // output to slot 0
                ],
                bytesBindings: [
                    uint32Binding(3, UInt32(convOp.dimension)),
                    uint32Binding(4, UInt32(convOp.kernelSize)),
                ],
                threadgroupMemoryLength: 0,
                sync: .bufferBarrier,
                mode: .perPosition,
                sequenceLengthBindingIndex: nil,
                positionBufferIndex: nil,
                perPositionStrides: [
                    0: perPosInputStride,   // in_proj output layout per position
                    2: perPosOutputStride,   // conv output layout per position
                ]
            )]

        // MARK: Default — skip unsupported ops in prefill
        default:
            print("[Prefill compiler] WARNING: unhandled operation skipped: \(entry.kind)")
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
            let path = StructuralPath(components: operationPath)

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
                guard let component = attributes as? (any MetalComponent) else { continue }
                // Resolve layer index in parameterBindings:
                // ParameterResolver uses layer 0 as template in repeating bodies.
                // When compiler unrolls, substitute the actual layer index.
                let bindings: [ParameterBinding]
                if let currentLayerIndex = layerIndex {
                    bindings = operation.parameterBindings.map { binding in
                        let resolved = binding.tensorName.replacingOccurrences(
                            of: ".layers.0.", with: ".layers.\(currentLayerIndex).")
                        if resolved != binding.tensorName {
                            print("[Compiler] layer substitution: \(binding.tensorName) → \(resolved) (layer=\(currentLayerIndex))")
                        }
                        return ParameterBinding(role: binding.role, tensorName: resolved)
                    }
                } else {
                    // LayerStack: parameterBindings already have correct layer-specific names.
                    // No substitution needed.
                    bindings = operation.parameterBindings
                }

                for cacheSlot in component.cacheSlots where cacheSlot.kind == .kv {
                    if let flashAttention = component.dispatchDeclarations
                        .compactMap({ decl -> FlashAttentionDecodeOperation? in
                            if case .compute(let op) = decl {
                                return op as? FlashAttentionDecodeOperation
                            }
                            return nil
                        }).first {
                        context.cacheSlots.append(CacheSlotInfo(
                            kvHeadCount: flashAttention.kvHeadCount,
                            headDimension: flashAttention.headDimension))
                    }
                }

                // Find the last projection index to mark it as output
                let declarations = component.dispatchDeclarations
                let lastProjectionIndex = declarations.lastIndex(where: {
                    if case .projection = $0 { return true }
                    return false
                })

                for (declarationIndex, declaration) in declarations.enumerated() {
                    switch declaration {
                    case .projection(let projection):
                        let isOutput = (declarationIndex == lastProjectionIndex)
                        context.emit(.projection(projection, isOutput: isOutput),
                                     parameterBindings: bindings, layerIndex: layerIndex)
                    case .compute(let computeOperation):
                        context.emit(.compute(computeOperation),
                                     parameterBindings: bindings, layerIndex: layerIndex)
                    }
                }
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
                // Pattern 1: structuralAdd + structuralCopy + rmsNorm → fused (3 → 1)
                if index + 2 < result.count,
                   case .structuralAdd(let addDimension) = result[index].kind,
                   case .structuralCopy = result[index + 1].kind,
                   case .compute(let operation) = result[index + 2].kind,
                   let normOperation = operation as? RMSNormOperation {
                    let fused = DispatchEntry(
                        index: result[index].index,
                        kind: .compute(ResidualAddCopyRMSNormOperation(
                            dimension: addDimension, epsilon: normOperation.epsilon)),
                        parameterBindings: result[index + 2].parameterBindings,
                        layerIndex: result[index].layerIndex)
                    result.replaceSubrange(index...index + 2, with: [fused])
                    changed = true
                    continue
                }

                // Pattern 2: structuralCopy + rmsNorm → fused (2 → 1)
                if index + 1 < result.count,
                   case .structuralCopy = result[index].kind,
                   case .compute(let operation) = result[index + 1].kind,
                   let normOperation = operation as? RMSNormOperation {
                    let fused = DispatchEntry(
                        index: result[index].index,
                        kind: .compute(CopyRMSNormOperation(
                            dimension: normOperation.dimension, epsilon: normOperation.epsilon)),
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
        case .compute(let operation):
            // Check if this operation's weight uses BF16 — select BF16 variant kernel
            let baseName = operation.kernelName
            if let staf = stafWeightStore {
                // QKNormOperation: check the specific weight role for BF16
                if let qkNormOperation = operation as? QKNormOperation {
                    if let binding = entry.parameterBindings.first(where: { $0.role == qkNormOperation.weightRole }),
                       let tensorInfo = staf.tensor(for: binding.tensorName),
                       tensorInfo.format.schemeIdentifier == .bf16RowMajor {
                        return baseName + "_bf16"
                    }
                    return baseName
                }
                let weightRoles = ["scale", "embedding_table"]
                for role in weightRoles {
                    if let binding = entry.parameterBindings.first(where: { $0.role == role }),
                       let tensorInfo = staf.tensor(for: binding.tensorName),
                       tensorInfo.format.schemeIdentifier == .bf16RowMajor {
                        let bf16Name = baseName + "_bf16"
                        return bf16Name
                    }
                }
            }
            return baseName
        case .structuralCopy:
            return "copy_buffer"
        case .structuralAdd:
            return "residual_add"
        }
    }

    /// Get the dispatch dimension for grid/threadgroup calculation.
    private func dispatchDimension(for kind: DispatchKind, hiddenSize: Int) -> MetalDispatchDimension {
        switch kind {
        case .projection(let projection, _):
            return .gemv(outputDimension: projection.outputDimension, inputDimension: projection.inputDimension)
        case .compute(let operation):
            return operation.dispatchDimension
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
        case .compute(let operation) where operation is EmbeddingLookupOperation:
            let embeddingOperation = operation as! EmbeddingLookupOperation
            let (weightBuffer, weightOffset) = resolveWeight(role: "embedding_table")
            routingState.lastOutputIsHidden = true
            return (
                buffers: [
                    (0, bufferSet.tokenIn, 0),
                    (1, weightBuffer, weightOffset),
                    (2, bufferSet.hidden, 0),
                ],
                bytes: [
                    uint32Binding(3, UInt32(embeddingOperation.embeddingDimension)),
                ]
            )

        // MARK: RMS Norm
        // Standalone RMSNorm (not fused with structuralCopy) writes in-place to hidden.
        // This ensures embedding_norm and final_norm results stay in hidden for
        // the next operation (Residual's structuralCopy or OutputHead projection).
        // In-place is safe: the kernel reads all elements for RMS before any writes.
        case .compute(let operation) where operation is RMSNormOperation:
            let normOperation = operation as! RMSNormOperation
            let (weightBuffer, weightOffset) = resolveWeight(role: "scale")
            routingState.lastOutputIsHidden = true
            routingState.projectionIndex = 0
            return (
                buffers: [
                    (0, bufferSet.hidden, 0),
                    (1, weightBuffer, weightOffset),
                    (2, bufferSet.hidden, 0),  // in-place: output = hidden
                ],
                bytes: [
                    uint32Binding(3, UInt32(normOperation.dimension)),
                    floatBinding(4, normOperation.epsilon),
                ]
            )

        // MARK: Fused Copy + RMS Norm
        case .compute(let operation) where operation is CopyRMSNormOperation:
            let fusedOperation = operation as! CopyRMSNormOperation
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
        case .compute(let operation) where operation is ResidualAddCopyRMSNormOperation:
            let fusedOperation = operation as! ResidualAddCopyRMSNormOperation
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
        case .compute(let operation) where operation is FlashAttentionDecodeOperation:
            let flashAttention = operation as! FlashAttentionDecodeOperation
            let layerIndex = kvCacheIndex
            kvCacheIndex += 1
            let scale: Float = 1.0 / Float(flashAttention.headDimension).squareRoot()

            routingState.lastOutputIsHidden = false
            routingState.projectionIndex = 0

            guard let cache = bufferSet.kvCache else {
                return (buffers: [], bytes: [])
            }
            let keyLayerOffset = cache.specification.layerOffset(
                layer: layerIndex, scheme: cache.specification.keyQuantizationScheme)
            let valueLayerOffset = cache.specification.layerOffset(
                layer: layerIndex, scheme: cache.specification.valueQuantizationScheme)

            // Scratch layout after Q/K/V projections:
            //   scratch[1 * slot]: Q (headCount * headDim)
            //   scratch[2 * slot]: K (kvHeadCount * headDim)
            //   scratch[3 * slot]: V (kvHeadCount * headDim)
            let scratchSlotSize = slotDimension * elementSize
            let kHeadSlotBytes = cache.specification.bytesPerHeadSlot(
                scheme: cache.specification.keyQuantizationScheme)
            let vHeadSlotBytes = cache.specification.bytesPerHeadSlot(
                scheme: cache.specification.valueQuantizationScheme)
            return (
                buffers: [
                    (0, bufferSet.scratch, 1 * scratchSlotSize),  // Q (after RoPE)
                    (1, bufferSet.scratch, 2 * scratchSlotSize),  // new K (after RoPE)
                    (2, bufferSet.scratch, 3 * scratchSlotSize),  // new V
                    (3, cache.keys, keyLayerOffset),               // K cache (read+write)
                    (4, cache.values, valueLayerOffset),           // V cache (read+write)
                    (5, bufferSet.scratch, 0),                     // output
                    (6, bufferSet.position, 0),                    // position (runtime)
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

        // MARK: SwiGLU
        case .compute(let operation) where operation is SwiGLUOperation:
            let swigluOperation = operation as! SwiGLUOperation
            let maxScratchSlot = slotDimension * elementSize
            routingState.lastOutputIsHidden = false
            routingState.projectionIndex = 0
            return (
                buffers: [
                    (0, bufferSet.scratch, 1 * maxScratchSlot),  // gate
                    (1, bufferSet.scratch, 2 * maxScratchSlot),  // up
                    (2, bufferSet.scratch, 0),                    // output
                ],
                bytes: [
                    uint32Binding(3, UInt32(swigluOperation.dimension)),
                ]
            )

        // MARK: Argmax
        case .compute(let operation) where operation is ArgmaxOperation:
            let argmaxOperation = operation as! ArgmaxOperation
            return (
                buffers: [
                    (0, bufferSet.logits, 0),
                    (1, bufferSet.tokenOut, 0),
                ],
                bytes: [
                    uint32Binding(2, UInt32(argmaxOperation.vocabularySize)),
                ]
            )

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
        case .compute(let operation) where operation is RoPEOperation:
            let ropeOperation = operation as! RoPEOperation
            // RoPE is applied in-place to Q (scratch[1]) and K (scratch[2])
            // after their respective GEMV projections.
            let scratchSlotSize = slotDimension * elementSize
            return (
                buffers: [
                    (0, bufferSet.scratch, 1 * scratchSlotSize),  // Q (in-place)
                    (1, bufferSet.scratch, 2 * scratchSlotSize),  // K (in-place)
                    (2, bufferSet.position, 0),                    // position (runtime)
                ],
                bytes: [
                    uint32Binding(3, UInt32(ropeOperation.headCount)),
                    uint32Binding(4, UInt32(ropeOperation.kvHeadCount)),
                    uint32Binding(5, UInt32(ropeOperation.headDimension)),
                    uint32Binding(6, UInt32(ropeOperation.ropeDimension)),
                    floatBinding(7, ropeOperation.base),
                ]
            )

        // MARK: QK Norm (per-head RMS norm on Q or K in scratch)
        case .compute(let operation) where operation is QKNormOperation:
            let qkNormOperation = operation as! QKNormOperation
            let scratchSlotSize = slotDimension * elementSize
            // Q lives in scratch[1], K lives in scratch[2].
            // weightRole tells us which one: "q_layernorm" → scratch[1], "k_layernorm" → scratch[2].
            let scratchSlotIndex = qkNormOperation.weightRole == "q_layernorm" ? 1 : 2
            let (weightBuffer, weightOffset) = resolveWeight(role: qkNormOperation.weightRole)
            return (
                buffers: [
                    (0, bufferSet.scratch, scratchSlotIndex * scratchSlotSize),  // in-place Q or K
                    (1, weightBuffer, weightOffset),                             // norm weight [headDim]
                ],
                bytes: [
                    uint32Binding(2, UInt32(qkNormOperation.headCount)),
                    uint32Binding(3, UInt32(qkNormOperation.headDimension)),
                    floatBinding(4, qkNormOperation.epsilon),
                ]
            )

        // MARK: Conv1d (temporal depthwise conv with conv_state)
        // Uses conv_state_update kernel: shifts state, appends in_proj output, convolves.
        // conv_state persists across decode steps for causal temporal convolution.
        case .compute(let operation) where operation is Conv1dOperation:
            let convOp = operation as! Conv1dOperation
            let (weightBuffer, weightOffset) = resolveWeight(role: "conv_weight")
            let slotBytes = slotDimension * elementSize
            routingState.lastOutputIsHidden = false
            routingState.projectionIndex = 0

            if let convState = bufferSet.convState {
                // Use conv_state_update: temporal conv with state
                let convLayerOffset = kvCacheIndex > 0 ? 0 : routingState.convLayerIndex * bufferSet.convStateKernelSize * bufferSet.convStateDimension * elementSize
                routingState.convLayerIndex += 1
                return (
                    buffers: [
                        (0, convState, convLayerOffset),           // conv state (in-place update)
                        (1, bufferSet.scratch, 1 * slotBytes),    // new input (first `dimension` of in_proj)
                        (2, weightBuffer, weightOffset),           // conv weight
                        (3, bufferSet.scratch, 0),                 // conv output
                    ],
                    bytes: [
                        uint32Binding(4, UInt32(convOp.dimension)),
                        uint32Binding(5, UInt32(convOp.kernelSize)),
                    ]
                )
            } else {
                // Fallback: no conv state (original behavior)
                return (
                    buffers: [
                        (0, bufferSet.scratch, 1 * slotBytes),
                        (1, weightBuffer, weightOffset),
                        (2, bufferSet.scratch, 0),
                    ],
                    bytes: [
                        uint32Binding(3, UInt32(convOp.dimension)),
                        uint32Binding(4, UInt32(convOp.kernelSize)),
                    ]
                )
            }

        // MARK: Default (unsupported compute operations)
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

    /// Cached MTLLibrary to avoid recompiling MSL on every load.
    private static let libraryCache = LibraryCache()

    private final class LibraryCache: @unchecked Sendable {
        private var cached: MTLLibrary?
        private let lock = NSLock()

        func get() -> MTLLibrary? {
            lock.lock()
            defer { lock.unlock() }
            return cached
        }

        func set(_ library: MTLLibrary) {
            lock.lock()
            defer { lock.unlock() }
            cached = library
        }
    }

    /// Get or compile the Metal library. Cached in memory after first compilation.
    private static func getOrCompileLibrary(device: MTLDevice) throws -> MTLLibrary {
        // Disable library cache to ensure fresh compilation with current options.
        // TODO: re-enable cache after fastMath investigation is complete.
        // if let cached = libraryCache.get() {
        //     return cached
        // }

        let compileOptions = MTLCompileOptions()
        compileOptions.fastMathEnabled = false
        compileOptions.languageVersion = .version3_0
        let library = try device.makeLibrary(
            source: MetalKernelSource.allKernelSource, options: compileOptions)
        libraryCache.set(library)
        return library
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
