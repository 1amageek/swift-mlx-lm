import LMIR

extension LinearAttributes: MetalKernelFragment, _FragmentBodyAccessor {
    public func fragment(context: KernelContext) -> LinearFragment {
        LinearFragment(field: "weight", inputDimension: inputSize, outputDimension: outputSize)
    }
    public var isFusable: Bool { false }
    public func _visitBody(context: KernelContext, _ visitor: (any MetalKernelFragment) -> Void) { visitor(fragment(context: context)) }
}
