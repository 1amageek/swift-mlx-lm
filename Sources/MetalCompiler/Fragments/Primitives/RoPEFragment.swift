import Metal

/// Rotary position embedding (in-place on Q and K).
public struct RoPEFragment: PrimitiveMetalKernelFragment {
    public let headCount: Int
    public let kvHeadCount: Int
    public let headDimension: Int
    public let ropeDimension: Int
    public let base: Float

    public init(headCount: Int, kvHeadCount: Int, headDimension: Int,
                ropeDimension: Int, base: Float) {
        self.headCount = headCount
        self.kvHeadCount = kvHeadCount
        self.headDimension = headDimension
        self.ropeDimension = ropeDimension
        self.base = base
    }

    public var isFusable: Bool { false }
    public var isInPlace: Bool { true }
    public func kernelName(context: KernelContext) -> String {
        context.bufferPrecision == .float32 ? "rope_seq_f32" : "rope"
    }
    public var dispatchDimension: MetalDispatchDimension {
        .perHead(headCount: max(headCount, kvHeadCount))
    }

    public func decodeBindings(context: BufferBindingContext) -> FragmentBindings {
        let slotBytes = context.slotDimension * context.elementSize
        return FragmentBindings(
            buffers: [
                (0, context.bufferSet.scratch, 1 * slotBytes),
                (1, context.bufferSet.scratch, 2 * slotBytes),
                (2, context.bufferSet.position, 0),
            ],
            bytes: [
                uint32Binding(3, UInt32(headCount)),
                uint32Binding(4, UInt32(kvHeadCount)),
                uint32Binding(5, UInt32(headDimension)),
                uint32Binding(6, UInt32(ropeDimension)),
                floatBinding(7, base),
            ],
            outputIsHidden: false
        )
    }
}
