import Metal

/// Reduction: all threads cooperate to reduce across a dimension.
/// Used by: RMSNorm, LayerNorm.
public struct Reduction: PrimitiveMetalKernelFragment {
    public let dimension: Int
    public let epsilon: Float

    public init(dimension: Int, epsilon: Float = 0) {
        self.dimension = dimension
        self.epsilon = epsilon
    }

    public var isFusable: Bool { true }
    public var normEpsilon: Float? { epsilon }
    public func kernelName(context: KernelContext) -> String {
        let bf16 = context.weightFormat == .bfloat16
        if context.bufferPrecision == .float32 {
            return bf16 ? "rms_norm_seq_bf16_f32_inplace" : "rms_norm_seq_f32_inplace"
        }
        return bf16 ? "rms_norm_bf16" : "rms_norm"
    }
    public var dispatchDimension: MetalDispatchDimension { .reduction(dimension: dimension) }
    public var weightSlots: [MetalWeightSlot] { [MetalWeightSlot(field: nil, role: .weight)] }

    public func decodeBindings(context: BufferBindingContext) -> FragmentBindings {
        let (weightBuffer, weightOffset) = context.resolveWeight("scale")
        return FragmentBindings(
            buffers: [
                (0, context.bufferSet.hidden, 0),
                (1, weightBuffer, weightOffset),
                (2, context.bufferSet.hidden, 0),
            ],
            bytes: [
                uint32Binding(3, UInt32(dimension)),
                floatBinding(4, epsilon),
            ],
            outputIsHidden: true,
            resetsProjectionIndex: true
        )
    }
}
