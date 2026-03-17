import Metal

/// Argmax over vocabulary: logits → token ID.
public struct ArgmaxFragment: PrimitiveMetalKernelFragment {
    public let vocabularySize: Int

    public init(vocabularySize: Int) {
        self.vocabularySize = vocabularySize
    }

    public var isFusable: Bool { false }
    public func kernelName(context: KernelContext) -> String {
        context.bufferPrecision == .float32 ? "argmax_f32" : "argmax"
    }
    public var dispatchDimension: MetalDispatchDimension { .reduction(dimension: vocabularySize) }

    public func decodeBindings(context: BufferBindingContext) -> FragmentBindings {
        return FragmentBindings(
            buffers: [
                (0, context.bufferSet.logits, 0),
                (1, context.bufferSet.tokenOut, 0),
            ],
            bytes: [
                uint32Binding(2, UInt32(vocabularySize)),
            ],
            outputIsHidden: false
        )
    }
}
