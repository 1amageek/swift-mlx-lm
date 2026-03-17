import LMIR

extension OutputHeadAttributes: MetalKernelFragment, _FragmentBodyAccessor {
    @MetalKernelFragmentBuilder
    public var fragment: some MetalKernelFragment {
        LinearFragment(field: "weight", inputDimension: inputSize, outputDimension: vocabSize)
        ArgmaxFragment(vocabularySize: vocabSize)
    }
    public var isFusable: Bool { false }
    public func _visitBody(_ visitor: (any MetalKernelFragment) -> Void) { visitor(fragment) }
}
