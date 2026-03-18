import Metal

/// Single-token attention against KV cache.
public struct FlashAttentionFragment: PrimitiveMetalKernelFragment {
    public let headCount: Int
    public let kvHeadCount: Int
    public let headDimension: Int
    public let ropeDimension: Int
    public let ropeBase: Float

    public init(headCount: Int, kvHeadCount: Int, headDimension: Int,
                ropeDimension: Int = 0, ropeBase: Float = 0) {
        self.headCount = headCount
        self.kvHeadCount = kvHeadCount
        self.headDimension = headDimension
        self.ropeDimension = ropeDimension
        self.ropeBase = ropeBase
    }

    public var isFusable: Bool { false }
    public func kernelName(context: KernelContext) -> String {
        context.bufferPrecision == .float32 ? "flash_attn_decode_f32" : "flash_attn_decode"
    }
    public var dispatchDimension: MetalDispatchDimension {
        .perHead(headCount: headCount)
    }
    public var cacheSlots: [MetalCacheSlot] {
        [MetalCacheSlot(name: "kv_cache", kind: .kv, kvHeadCount: kvHeadCount, headDimension: headDimension)]
    }

    public func kernelSource(name: String, bufferPrecision: BufferPrecision, weightFormat: WeightFormat) -> String {
        MetalSourceGenerator.generateFlashAttentionKernel(name: name, bufferPrecision: bufferPrecision)
    }

    public func decodeBindings(context: BufferBindingContext) -> FragmentBindings {
        let scale: Float = 1.0 / Float(headDimension).squareRoot()
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

        return FragmentBindings(
            buffers: [
                (0, context.bufferSet.scratch, 1 * slotBytes),
                (1, context.bufferSet.scratch, 2 * slotBytes),
                (2, context.bufferSet.scratch, 3 * slotBytes),
                (3, cache.keys, keyLayerOffset),
                (4, cache.values, valueLayerOffset),
                (5, context.bufferSet.scratch, 0),
                (6, context.bufferSet.position, 0),
            ],
            bytes: [
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
            ],
            outputIsHidden: false,
            resetsProjectionIndex: true,
            consumesKVCacheLayer: true
        )
    }

    public func prefillSteps(context: PrefillBindingContext) throws -> FragmentPrefillSteps {
        let scale: Float = 1.0 / Float(headDimension).squareRoot()

        guard let cache = context.buffers.kvCache else { return FragmentPrefillSteps(steps: [], outputIsHidden: false, resetsProjectionIndex: true, consumesKVCacheLayer: true) }
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

        let scratchSlotSize = context.slotDimension * context.scratchElementSize * context.maximumSequenceLength

        var steps: [MetalPrefillStep] = []

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
            ],
            bytesBindings: [
                uint32Binding(4, UInt32(kvHeadCount)),
                uint32Binding(5, UInt32(headDimension)),
                uint32Binding(6, UInt32(cache.specification.maximumSequenceLength)),
                uint32Binding(7, UInt32(context.maximumSequenceLength)),
                uint32Binding(8, UInt32(cache.specification.layoutMode.rawValue)),
                uint32Binding(9, UInt32(kHeadSlotBytes)),
                uint32Binding(10, UInt32(vHeadSlotBytes)),
            ],
            threadgroupMemoryLength: 0,
            sync: .bufferBarrier,
            mode: .batch,
            sequenceLengthBindingIndex: 7,
            positionBufferIndex: nil,
            perPositionStrides: [:]
        ))

        // Step 2: Batch causal attention — one threadgroup per (head, position) pair.
        // 1D grid: flatGroupId = posId * headCount + headIndex.
        // Causal masking: each position attends to positions [0..posId].
        let attnPipeline = try context.getPipeline("flash_attn_batch_f32")
        let threads = min(256, attnPipeline.maxTotalThreadsPerThreadgroup)
        let attnGridSize = headCount * context.maximumSequenceLength
        steps.append(MetalPrefillStep(
            pipeline: attnPipeline,
            gridSize: MTLSize(width: attnGridSize, height: 1, depth: 1),
            threadgroupSize: MTLSize(width: threads, height: 1, depth: 1),
            bufferBindings: [
                (0, context.buffers.scratch, 1 * scratchSlotSize),
                (1, cache.keys, keyLayerOffset),
                (2, cache.values, valueLayerOffset),
                (3, context.buffers.scratch, 0),
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
            ],
            threadgroupMemoryLength: 0,
            sync: .bufferBarrier,
            mode: .batch,
            sequenceLengthBindingIndex: 10,
            positionBufferIndex: nil,
            perPositionStrides: [:]
        ))

        return FragmentPrefillSteps(
            steps: steps,
            outputIsHidden: false,
            resetsProjectionIndex: true,
            consumesKVCacheLayer: true
        )
    }
}
