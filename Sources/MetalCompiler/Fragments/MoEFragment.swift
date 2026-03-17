import LMIR

extension MoEAttributes: MetalKernelFragment, _FragmentBodyAccessor {
    @MetalKernelFragmentBuilder
    public var fragment: some MetalKernelFragment {
        LinearFragment(field: "router", inputDimension: expertMLP.inputSize, outputDimension: expertCount)
    }
    public var isFusable: Bool { false }
    public func _visitBody(_ visitor: (any MetalKernelFragment) -> Void) { visitor(fragment) }
}
