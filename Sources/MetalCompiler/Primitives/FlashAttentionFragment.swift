import Metal
import LMIR

/// Single-token attention against KV cache.
///
/// When `ropeDimension > 0`, this fragment includes inline RoPE rotation,
/// eliminating the need for a separate RoPE dispatch and its barrier.
///
/// When `directScratchMode` is true, the prefill path reads K/V directly from
/// scratch buffers instead of filling and reading from KV cache. This is used for
/// embedding-only models (no OutputHead) where KV cache persistence is unnecessary.
public struct FlashAttentionFragment: PrimitiveMetalKernelFragment {
    public let headCount: Int
    public let kvHeadCount: Int
    public let headDimension: Int
    public let attentionScale: Float?
    public let ropeDimension: Int
    public let ropeBase: Float
    public let ropeScaling: RoPEScaling?
    public let mropeAxes: MRoPEAxes?
    public let querySlotIndex: Int
    public let causal: Bool
    public let windowLeft: Int?
    public let windowRight: Int?
    public let sharedKVSourceLayerIndex: Int?

    /// When true, prefill reads K/V from scratch buffers instead of KV cache.
    /// Set by the optimizer when the model graph has no OutputHead.
    public let directScratchMode: Bool

    /// When true, the prefill RoPE dispatch is skipped — RoPE has already been
    /// applied to Q/K scratch upstream (e.g. by BatchedQKNormRoPEFragment).
    /// The decode path is unaffected: inline RoPE in `rope_flash_attn_decode`
    /// always runs when `hasInlineRoPE` is true.
    public let suppressPrefillRoPE: Bool

    public init(headCount: Int, kvHeadCount: Int, headDimension: Int,
                attentionScale: Float? = nil,
                ropeDimension: Int = 0, ropeBase: Float = 0,
                ropeScaling: RoPEScaling? = nil,
                mropeAxes: MRoPEAxes? = nil,
                querySlotIndex: Int = 1,
                causal: Bool = true,
                windowLeft: Int? = nil,
                windowRight: Int? = nil,
                sharedKVSourceLayerIndex: Int? = nil,
                directScratchMode: Bool = false,
                suppressPrefillRoPE: Bool = false) {
        self.headCount = headCount
        self.kvHeadCount = kvHeadCount
        self.headDimension = headDimension
        self.attentionScale = attentionScale
        self.ropeDimension = ropeDimension
        self.ropeBase = ropeBase
        self.ropeScaling = ropeScaling
        self.mropeAxes = mropeAxes
        self.querySlotIndex = querySlotIndex
        self.causal = causal
        self.windowLeft = windowLeft
        self.windowRight = windowRight
        self.sharedKVSourceLayerIndex = sharedKVSourceLayerIndex
        self.directScratchMode = directScratchMode
        self.suppressPrefillRoPE = suppressPrefillRoPE
    }

    /// Whether this fragment includes inline RoPE computation.
    public var hasInlineRoPE: Bool { ropeDimension > 0 }

    public var isFusable: Bool { false }
    public func kernelName(context: KernelContext) -> String {
        if hasInlineRoPE {
            return context.bufferPrecision == .float32 ? "rope_flash_attn_decode_f32" : "rope_flash_attn_decode"
        }
        return context.bufferPrecision == .float32 ? "flash_attn_decode_f32" : "flash_attn_decode"
    }
    public var dispatchDimension: MetalDispatchDimension {
        .perHead(headCount: headCount)
    }
    public var cacheSlots: [MetalCacheSlot] {
        sharedKVSourceLayerIndex == nil
            ? [MetalCacheSlot(name: "kv_cache", kind: .kv, kvHeadCount: kvHeadCount, headDimension: headDimension)]
            : []
    }
    public var kvCacheIndexOverride: Int? { sharedKVSourceLayerIndex }

    public func kernelSource(name: String, bufferPrecision: BufferPrecision, weightFormat: WeightFormat) -> String {
        MetalSourceGenerator.generateFlashAttentionKernel(name: name, bufferPrecision: bufferPrecision, inlineRoPE: hasInlineRoPE)
    }

    public func decodeBindings(context: BufferBindingContext) -> FragmentBindings {
        let scale: Float = attentionScale ?? (1.0 / Float(headDimension).squareRoot())
        let slotBytes = context.slotDimension * context.elementSize

        guard let cache = context.bufferSet.kvCache else {
            fatalError("[Compiler] FlashAttentionFragment requires KV cache")
        }
        let keyLayerOffset = cache.specification.layerOffset(
            layer: context.kvCacheIndex,
            scheme: cache.specification.keyQuantizationScheme)
        let valueLayerOffset = cache.specification.layerOffset(
            layer: context.kvCacheIndex,
            scheme: cache.specification.valueQuantizationScheme)
        let kHeadSlotBytes = cache.specification.bytesPerHeadSlot(
            scheme: cache.specification.keyQuantizationScheme)
        let vHeadSlotBytes = cache.specification.bytesPerHeadSlot(
            scheme: cache.specification.valueQuantizationScheme)

        // RotorQuant / QJL buffers — kernel checks is_rotor_scheme() before access,
        // so a placeholder buffer is safe for non-rotor schemes.
        let placeholder = cache.keys
        let rotorParamsBuffer = cache.rotorParameters ?? placeholder
        let rotorParamsOffset = cache.rotorParameters != nil
            ? cache.rotorParameterOffset(layer: context.kvCacheIndex) : 0
        let qjlMatrixBuffer = cache.qjlMatrix ?? placeholder
        let qjlResidualBuffer = cache.qjlResidualK ?? placeholder
        let qjlResidualOffset = cache.qjlResidualK != nil
            ? cache.qjlResidualOffset(layer: context.kvCacheIndex) : 0

        var buffers: [(Int, MTLBuffer, Int)] = [
            (0, context.bufferSet.scratch, querySlotIndex * slotBytes),
            (1, context.bufferSet.scratch, 2 * slotBytes),
            (2, context.bufferSet.scratch, 3 * slotBytes),
            (3, cache.keys, keyLayerOffset),
            (4, cache.values, valueLayerOffset),
            (5, context.bufferSet.scratch, 0),
            (6, context.bufferSet.position, 0),
            (17, rotorParamsBuffer, rotorParamsOffset),
            (18, qjlMatrixBuffer, 0),
            (19, qjlResidualBuffer, qjlResidualOffset),
        ]

        var bytes: [(Int, [UInt8])] = [
            uint32Binding(7, UInt32(headCount)),
            uint32Binding(8, UInt32(kvHeadCount)),
            uint32Binding(9, UInt32(headDimension)),
            floatBinding(10, scale),
            uint32Binding(11, UInt32(cache.specification.layoutMode.rawValue)),
            uint32Binding(12, UInt32(cache.specification.maximumSequenceLength)),
            uint32Binding(13, UInt32(cache.specification.keyQuantizationScheme.rawValue)),
            uint32Binding(14, UInt32(cache.specification.valueQuantizationScheme.rawValue)),
            uint32Binding(15, UInt32(kHeadSlotBytes)),
            uint32Binding(16, UInt32(vHeadSlotBytes)),
            uint32Binding(20, UInt32(cache.numRotorGroups)),
            uint32Binding(21, UInt32(cache.qjlDimension)),
            uint32Binding(22, windowLeft.map(UInt32.init) ?? UInt32.max),
            uint32Binding(
                29,
                UInt32((sharedKVSourceLayerIndex == nil ? 0 : 1) | (usesProportionalRoPE ? 0b10 : 0))
            ),
        ]

        if hasInlineRoPE {
            buffers.append((22, context.bufferSet.ropePositionAxes, 0))
            bytes.append(contentsOf: [
                uint32Binding(23, UInt32(ropeDimension)),
                floatBinding(24, ropeBase),
                uint32Binding(25, UInt32(mropeSectionCount(at: 0))),
                uint32Binding(26, UInt32(mropeSectionCount(at: 1))),
                uint32Binding(27, UInt32(mropeSectionCount(at: 2))),
                uint32Binding(28, UInt32(mropeAxes?.interleaved == true ? 1 : 0)),
            ])
        }

        return FragmentBindings(
            buffers: buffers,
            bytes: bytes,
            outputIsHidden: false,
            resetsProjectionIndex: true,
            consumesKVCacheLayer: sharedKVSourceLayerIndex == nil,
            writeBufferIndices: sharedKVSourceLayerIndex == nil
                ? Set<Int>([3, 4, 5, 19])
                : Set<Int>([5])
        )
    }

    private func mropeSectionCount(at index: Int) -> Int {
        guard let mropeAxes, index < mropeAxes.sections.count else { return 0 }
        return mropeAxes.sections[index]
    }

    public func prefillSteps(context: PrefillBindingContext) throws -> FragmentPrefillSteps {
        let scale: Float = attentionScale ?? (1.0 / Float(headDimension).squareRoot())
        let scratchSlotSize = context.slotDimension * context.scratchElementSize * context.maximumSequenceLength

        // Direct-scratch mode: read K/V from scratch instead of KV cache.
        // Skips KV cache fill entirely. Used for embedding-only models.
        if directScratchMode {
            return try prefillStepsDirectScratch(
                context: context, scale: scale, scratchSlotSize: scratchSlotSize)
        }

        guard let cache = context.buffers.kvCache else {
            return FragmentPrefillSteps(
                steps: [],
                outputIsHidden: false,
                resetsProjectionIndex: true,
                consumesKVCacheLayer: sharedKVSourceLayerIndex == nil
            )
        }
        let keyLayerOffset = cache.specification.layerOffset(
            layer: context.kvCacheIndex,
            scheme: cache.specification.keyQuantizationScheme)
        let valueLayerOffset = cache.specification.layerOffset(
            layer: context.kvCacheIndex,
            scheme: cache.specification.valueQuantizationScheme)
        let kHeadSlotBytes = cache.specification.bytesPerHeadSlot(
            scheme: cache.specification.keyQuantizationScheme)
        let vHeadSlotBytes = cache.specification.bytesPerHeadSlot(
            scheme: cache.specification.valueQuantizationScheme)

        var steps: [MetalPrefillStep] = []

        // RotorQuant / QJL buffers for prefill — placeholder when nil
        let placeholder = cache.keys
        let rotorParamsBuffer = cache.rotorParameters ?? placeholder
        let rotorParamsOffset = cache.rotorParameters != nil
            ? cache.rotorParameterOffset(layer: context.kvCacheIndex) : 0
        let qjlMatrixBuffer = cache.qjlMatrix ?? placeholder
        let qjlResidualBuffer = cache.qjlResidualK ?? placeholder
        let qjlResidualOffset = cache.qjlResidualK != nil
            ? cache.qjlResidualOffset(layer: context.kvCacheIndex) : 0

        // Step 0 (inline RoPE only): Apply RoPE to Q and K in scratch buffers.
        // This replaces the separate RoPEFragment prefill step.
        // Skipped when RoPE was already fused upstream (BatchedQKNormRoPEFragment).
        if hasInlineRoPE && !suppressPrefillRoPE {
            steps.append(try makeRoPEStep(context: context, scratchSlotSize: scratchSlotSize))
        }

        if sharedKVSourceLayerIndex == nil {
            // Step 1: Fill KV cache for all positions in one batch dispatch.
            // Each threadgroup handles one position; threads within handle headDim elements.
            // Loops over all kvHeads internally.
            let fillPipeline = try context.getPipeline("kv_cache_fill_seq_f32")
            let fillTgSize = min(headDimension, fillPipeline.maxTotalThreadsPerThreadgroup)
            steps.append(MetalPrefillStep(
                pipeline: fillPipeline,
                gridSize: MTLSize(width: context.maximumSequenceLength, height: 1, depth: 1),
                threadgroupSize: MTLSize(width: fillTgSize, height: 1, depth: 1),
                bufferBindings: [
                    (0, context.buffers.scratch, 2 * scratchSlotSize),
                    (1, context.buffers.scratch, 3 * scratchSlotSize),
                    (2, cache.keys, keyLayerOffset),
                    (3, cache.values, valueLayerOffset),
                    (13, rotorParamsBuffer, rotorParamsOffset),
                    (14, qjlMatrixBuffer, 0),
                    (15, qjlResidualBuffer, qjlResidualOffset),
                ],
                bytesBindings: [
                    uint32Binding(4, UInt32(kvHeadCount)),
                    uint32Binding(5, UInt32(headDimension)),
                    uint32Binding(6, UInt32(cache.specification.maximumSequenceLength)),
                    uint32Binding(7, UInt32(context.maximumSequenceLength)),
                    uint32Binding(8, UInt32(cache.specification.layoutMode.rawValue)),
                    uint32Binding(9, UInt32(cache.specification.keyQuantizationScheme.rawValue)),
                    uint32Binding(10, UInt32(cache.specification.valueQuantizationScheme.rawValue)),
                    uint32Binding(11, UInt32(kHeadSlotBytes)),
                    uint32Binding(12, UInt32(vHeadSlotBytes)),
                    uint32Binding(16, UInt32(cache.numRotorGroups)),
                    uint32Binding(17, UInt32(cache.qjlDimension)),
                ],
                threadgroupMemoryLength: 0,
                sync: .bufferBarrier,
                mode: .batch,
                sequenceLengthPolicy: .bind(index: 7),
                positionBufferIndex: nil,
                perPositionStrides: [:]
            ))
        }

        // Step 2: Batch causal attention — one threadgroup per (head, position) pair.
        // 1D grid: flatGroupId = posId * headCount + headIndex.
        // Causal masking: each position attends to positions [0..posId].
        let attnPipeline = try context.getPipeline("flash_attn_batch_f32")
        let simdWidth = 32
        let minimumThreads = max(headDimension, simdWidth)
        let roundedThreads = ((minimumThreads + simdWidth - 1) / simdWidth) * simdWidth
        let threads = min(roundedThreads, attnPipeline.maxTotalThreadsPerThreadgroup)
        let attnGridSize = headCount * context.maximumSequenceLength
        steps.append(MetalPrefillStep(
            pipeline: attnPipeline,
            gridSize: MTLSize(width: attnGridSize, height: 1, depth: 1),
            threadgroupSize: MTLSize(width: threads, height: 1, depth: 1),
            bufferBindings: [
                (0, context.buffers.scratch, querySlotIndex * scratchSlotSize),
                (1, cache.keys, keyLayerOffset),
                (2, cache.values, valueLayerOffset),
                (3, context.buffers.scratch, 0),
                (15, rotorParamsBuffer, rotorParamsOffset),
                (16, qjlMatrixBuffer, 0),
                (17, qjlResidualBuffer, qjlResidualOffset),
            ],
            bytesBindings: [
                uint32Binding(4, UInt32(headCount)),
                uint32Binding(5, UInt32(kvHeadCount)),
                uint32Binding(6, UInt32(headDimension)),
                floatBinding(7, scale),
                uint32Binding(8, UInt32(cache.specification.layoutMode.rawValue)),
                uint32Binding(9, UInt32(cache.specification.maximumSequenceLength)),
                uint32Binding(10, UInt32(context.maximumSequenceLength)),
                uint32Binding(11, UInt32(cache.specification.keyQuantizationScheme.rawValue)),
                uint32Binding(12, UInt32(cache.specification.valueQuantizationScheme.rawValue)),
                uint32Binding(13, UInt32(kHeadSlotBytes)),
                uint32Binding(14, UInt32(vHeadSlotBytes)),
                uint32Binding(18, UInt32(cache.numRotorGroups)),
                uint32Binding(19, UInt32(cache.qjlDimension)),
                uint32Binding(20, causal ? 1 : 0),
                uint32Binding(21, windowLeft.map(UInt32.init) ?? UInt32.max),
                uint32Binding(22, windowRight.map(UInt32.init) ?? UInt32.max),
            ],
            threadgroupMemoryLength: 0,
            sync: .bufferBarrier,
            mode: .batch,
            sequenceLengthPolicy: .bind(index: 10),
            positionBufferIndex: nil,
            perPositionStrides: [:]
        ))

        return FragmentPrefillSteps(
            steps: steps,
            outputIsHidden: false,
            resetsProjectionIndex: true,
            consumesKVCacheLayer: sharedKVSourceLayerIndex == nil
        )
    }

    // MARK: - Direct-Scratch Prefill

    /// Prefill steps for direct-scratch mode: RoPE (if needed) + scratch-based attention.
    /// No KV cache fill, no KV cache read. K/V come from scratch buffers.
    private func prefillStepsDirectScratch(
        context: PrefillBindingContext,
        scale: Float,
        scratchSlotSize: Int
    ) throws -> FragmentPrefillSteps {
        var steps: [MetalPrefillStep] = []

        // Step 0 (inline RoPE): same as cache path — applies to Q and K in scratch.
        // Skipped when RoPE was already fused upstream (BatchedQKNormRoPEFragment).
        if hasInlineRoPE && !suppressPrefillRoPE {
            steps.append(try makeRoPEStep(context: context, scratchSlotSize: scratchSlotSize))
        }

        // Step 1: Direct-scratch attention — reads K/V from scratch instead of KV cache.
        let totalQDimension = headCount * headDimension
        let totalKDimension = kvHeadCount * headDimension

        let attnPipeline = try context.getPipeline("flash_attn_batch_scratch_f32")
        let simdWidth = 32
        let minimumThreads = max(headDimension, simdWidth)
        let roundedThreads = ((minimumThreads + simdWidth - 1) / simdWidth) * simdWidth
        let threads = min(roundedThreads, attnPipeline.maxTotalThreadsPerThreadgroup)

        // 2D grid: width = headCount, height = maximumSequenceLength (adjusted to actual
        // sequenceLength at dispatch via .bindAndAdjustGridHeight). This avoids dispatching
        // ~(maxSeqLen/actualSeqLen) empty threadgroups that only read bindings and return.
        steps.append(MetalPrefillStep(
            pipeline: attnPipeline,
            gridSize: MTLSize(width: headCount, height: context.maximumSequenceLength, depth: 1),
            threadgroupSize: MTLSize(width: threads, height: 1, depth: 1),
            bufferBindings: [
                (0, context.buffers.scratch, querySlotIndex * scratchSlotSize),
                (1, context.buffers.scratch, 2 * scratchSlotSize),
                (2, context.buffers.scratch, 3 * scratchSlotSize),
                (3, context.buffers.scratch, 0),
            ],
            bytesBindings: [
                uint32Binding(4, UInt32(headCount)),
                uint32Binding(5, UInt32(kvHeadCount)),
                uint32Binding(6, UInt32(headDimension)),
                floatBinding(7, scale),
                uint32Binding(8, UInt32(context.maximumSequenceLength)),
                uint32Binding(9, UInt32(totalQDimension)),
                uint32Binding(10, UInt32(totalKDimension)),
                uint32Binding(11, causal ? 1 : 0),
                uint32Binding(12, windowLeft.map(UInt32.init) ?? UInt32.max),
                uint32Binding(13, windowRight.map(UInt32.init) ?? UInt32.max),
            ],
            threadgroupMemoryLength: 0,
            sync: .bufferBarrier,
            mode: .batch,
            sequenceLengthPolicy: .bindAndAdjustGridHeight(index: 8),
            positionBufferIndex: nil,
            perPositionStrides: [:]
        ))

        return FragmentPrefillSteps(
            steps: steps,
            outputIsHidden: false,
            resetsProjectionIndex: true,
            consumesKVCacheLayer: false
        )
    }

    // MARK: - Shared Prefill Helpers

    /// Generate the inline RoPE prefill step (shared between cache and direct-scratch paths).
    private func makeRoPEStep(
        context: PrefillBindingContext,
        scratchSlotSize: Int
    ) throws -> MetalPrefillStep {
        let ropeKernelName = context.kernelContext.bufferPrecision == .float32 ? "rope_seq_f32" : "rope"
        let ropePipeline = try context.getPipeline(ropeKernelName)
        let ropeThreads = min(32, ropePipeline.maxTotalThreadsPerThreadgroup)
        let totalHeads = headCount + kvHeadCount
        return MetalPrefillStep(
            pipeline: ropePipeline,
            gridSize: MTLSize(width: totalHeads, height: context.maximumSequenceLength, depth: 1),
            threadgroupSize: MTLSize(width: ropeThreads, height: 1, depth: 1),
            bufferBindings: [
                (0, context.buffers.scratch, querySlotIndex * scratchSlotSize),
                (1, context.buffers.scratch, 2 * scratchSlotSize),
                (2, context.buffers.ropePositionAxes, 0),
            ],
            bytesBindings: [
                uint32Binding(3, UInt32(headCount)),
                uint32Binding(4, UInt32(kvHeadCount)),
                uint32Binding(5, UInt32(headDimension)),
                uint32Binding(6, UInt32(ropeDimension)),
                floatBinding(7, ropeBase),
                uint32Binding(8, UInt32(mropeSectionCount(at: 0))),
                uint32Binding(9, UInt32(mropeSectionCount(at: 1))),
                uint32Binding(10, UInt32(mropeSectionCount(at: 2))),
                uint32Binding(11, UInt32(mropeAxes?.interleaved == true ? 1 : 0)),
                uint32Binding(12, UInt32(context.maximumSequenceLength)),
                uint32Binding(13, UInt32(usesProportionalRoPE ? 1 : 0)),
            ],
            threadgroupMemoryLength: 0,
            sync: .bufferBarrier,
            mode: .batch,
            sequenceLengthPolicy: .bindAndAdjustGridHeight(index: 12),
            positionBufferIndex: nil,
            perPositionStrides: [:]
        )
    }

    private var usesProportionalRoPE: Bool {
        ropeScaling?.kind == .custom("proportional")
    }
}
