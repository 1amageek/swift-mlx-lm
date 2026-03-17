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
    public var cacheSlots: [MetalCacheSlot] { [MetalCacheSlot(name: "kv_cache", kind: .kv)] }

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
}
