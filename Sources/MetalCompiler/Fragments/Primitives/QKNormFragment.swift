import Metal

/// Per-head RMS normalization for Q or K projections.
public struct QKNormFragment: PrimitiveMetalKernelFragment {
    public let headCount: Int
    public let headDimension: Int
    public let epsilon: Float
    public let weightRole: String  // "q_layernorm" or "k_layernorm"

    public init(headCount: Int, headDimension: Int, epsilon: Float, weightRole: String) {
        self.headCount = headCount
        self.headDimension = headDimension
        self.epsilon = epsilon
        self.weightRole = weightRole
    }

    public var isFusable: Bool { true }
    public var isInPlace: Bool { true }
    public var normEpsilon: Float? { epsilon }
    public func kernelName(context: KernelContext) -> String {
        if context.bufferPrecision == .float32 { return "qk_rms_norm_seq_f32" }
        return context.weightFormat == .bfloat16 ? "qk_rms_norm_bf16" : "qk_rms_norm"
    }
    public var dispatchDimension: MetalDispatchDimension { .perHead(headCount: headCount) }
    public var weightSlots: [MetalWeightSlot] { [MetalWeightSlot(field: weightRole, role: .weight)] }

    public func decodeBindings(context: BufferBindingContext) -> FragmentBindings {
        let slotBytes = context.slotDimension * context.elementSize
        let scratchSlotIndex = weightRole == "q_layernorm" ? 1 : 2
        let (weightBuffer, weightOffset) = context.resolveWeight(weightRole)
        return FragmentBindings(
            buffers: [
                (0, context.bufferSet.scratch, scratchSlotIndex * slotBytes),
                (1, weightBuffer, weightOffset),
            ],
            bytes: [
                uint32Binding(2, UInt32(headCount)),
                uint32Binding(3, UInt32(headDimension)),
                floatBinding(4, epsilon),
            ],
            outputIsHidden: false
        )
    }
}
