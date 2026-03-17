import Metal

/// Elementwise: one thread per element, trivially parallel.
/// Used by: SwiGLU, SigmoidGate.
public struct ElementwiseFragment: PrimitiveMetalKernelFragment {
    public let count: Int
    public let kind: ElementwiseKind

    public enum ElementwiseKind: Sendable {
        case swiglu
        case sigmoidGate
    }

    public init(count: Int, kind: ElementwiseKind = .swiglu) {
        self.count = count
        self.kind = kind
    }

    public var isFusable: Bool { true }
    public func kernelName(context: KernelContext) -> String {
        switch kind {
        case .swiglu:
            return context.bufferPrecision == .float32 ? "swiglu_seq_f32" : "swiglu"
        case .sigmoidGate:
            return "sigmoid_gate"
        }
    }
    public var dispatchDimension: MetalDispatchDimension { .elementwise(count: count) }

    public func decodeBindings(context: BufferBindingContext) -> FragmentBindings {
        let slotBytes = context.slotDimension * context.elementSize
        return FragmentBindings(
            buffers: [
                (0, context.bufferSet.scratch, 1 * slotBytes),
                (1, context.bufferSet.scratch, 2 * slotBytes),
                (2, context.bufferSet.scratch, 0),
            ],
            bytes: [
                uint32Binding(3, UInt32(count)),
            ],
            outputIsHidden: false,
            resetsProjectionIndex: true
        )
    }
}
