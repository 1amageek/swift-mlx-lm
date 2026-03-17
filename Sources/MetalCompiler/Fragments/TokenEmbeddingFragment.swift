import LMIR

extension TokenEmbeddingAttributes: MetalKernelFragment, _FragmentBodyAccessor {
    public func fragment(context: KernelContext) -> GatherFragment {
        GatherFragment(vocabularySize: vocabSize, embeddingDimension: embeddingSize)
    }
    public var isFusable: Bool { false }
    public func _visitBody(context: KernelContext, _ visitor: (any MetalKernelFragment) -> Void) { visitor(fragment(context: context)) }
}
