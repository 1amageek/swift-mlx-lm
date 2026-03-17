import LMIR

extension TokenEmbeddingAttributes: MetalKernelFragment, _FragmentBodyAccessor {
    public var fragment: GatherFragment {
        GatherFragment(vocabularySize: vocabSize, embeddingDimension: embeddingSize)
    }
    public var isFusable: Bool { false }
    public func _visitBody(_ visitor: (any MetalKernelFragment) -> Void) { visitor(fragment) }
}
