import LMIR

extension LinearAttributes: MetalKernelFragment, _FragmentBodyAccessor {
    public var fragment: LinearFragment {
        LinearFragment(field: "weight", inputDimension: inputSize, outputDimension: outputSize)
    }
    public var isFusable: Bool { false }
    public func _visitBody(_ visitor: (any MetalKernelFragment) -> Void) { visitor(fragment) }
}
