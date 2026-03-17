import Metal

/// Depthwise temporal convolution with double gating (decode: state update).
public struct Conv1dFragment: PrimitiveMetalKernelFragment {
    public let dimension: Int
    public let kernelSize: Int

    public init(dimension: Int, kernelSize: Int) {
        self.dimension = dimension
        self.kernelSize = kernelSize
    }

    public var isFusable: Bool { false }
    public func kernelName(context: KernelContext) -> String {
        if context.bufferPrecision == .float32 { return "conv1d_causal_seq_f32" }
        return context.weightFormat == .bfloat16 ? "conv_state_update_bf16" : "conv_state_update"
    }
    public var dispatchDimension: MetalDispatchDimension { .elementwise(count: dimension) }
    public var weightSlots: [MetalWeightSlot] { [MetalWeightSlot(field: "conv_weight", role: .weight)] }
    public var cacheSlots: [MetalCacheSlot] { [MetalCacheSlot(name: "conv_cache", kind: .conv)] }

    public func decodeBindings(context: BufferBindingContext) -> FragmentBindings {
        let (weightBuffer, weightOffset) = context.resolveWeight("conv_weight")
        let slotBytes = context.slotDimension * context.elementSize

        guard let convState = context.bufferSet.convState else {
            fatalError("[Compiler] Conv1dFragment requires conv_state buffer")
        }
        let convLayerOffset = context.convLayerIndex
            * context.bufferSet.convStateKernelSize * context.bufferSet.convStateDimension * context.elementSize

        return FragmentBindings(
            buffers: [
                (0, convState, convLayerOffset),
                (1, context.bufferSet.scratch, 1 * slotBytes),
                (2, weightBuffer, weightOffset),
                (3, context.bufferSet.scratch, 0),
            ],
            bytes: [
                uint32Binding(4, UInt32(dimension)),
                uint32Binding(5, UInt32(kernelSize)),
            ],
            outputIsHidden: false,
            resetsProjectionIndex: true,
            consumesConvLayer: true
        )
    }
}
