import LMIR

extension ShortConvAttributes: MetalKernelFragment, _FragmentBodyAccessor {
    @MetalKernelFragmentBuilder
    public var fragment: some MetalKernelFragment {
        LinearFragment(field: "in_proj", inputDimension: hiddenSize, outputDimension: hiddenSize * 3)
        Conv1dFragment(dimension: hiddenSize, kernelSize: kernelSize)
        LinearFragment(field: "out_proj", inputDimension: hiddenSize, outputDimension: hiddenSize)
    }
    public var isFusable: Bool { false }
    public func _visitBody(_ visitor: (any MetalKernelFragment) -> Void) { visitor(fragment) }
}
