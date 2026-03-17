import Metal

/// Sigmoid-gated element-wise operation.
public struct SigmoidGateFragment: PrimitiveMetalKernelFragment {
    public let dimension: Int

    public init(dimension: Int) {
        self.dimension = dimension
    }

    public var isFusable: Bool { true }
    public func kernelName(context: KernelContext) -> String { "sigmoid_gate" }
    public var dispatchDimension: MetalDispatchDimension { .elementwise(count: dimension) }

    public func decodeBindings(context: BufferBindingContext) -> FragmentBindings {
        let slotBytes = context.slotDimension * context.elementSize
        return FragmentBindings(
            buffers: [
                (0, context.bufferSet.scratch, 0),
                (1, context.bufferSet.scratch, 1 * slotBytes),
                (2, context.bufferSet.scratch, 0),
            ],
            bytes: [
                uint32Binding(3, UInt32(dimension)),
            ],
            outputIsHidden: false
        )
    }
}
