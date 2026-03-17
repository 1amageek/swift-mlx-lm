import LMIR

extension MLPAttributes: MetalKernelFragment, _FragmentBodyAccessor {
    @MetalKernelFragmentBuilder
    public func fragment(context: KernelContext) -> some MetalKernelFragment {
        LinearFragment(field: "gate_proj", inputDimension: inputSize, outputDimension: intermediateSize)
        LinearFragment(field: "up_proj", inputDimension: inputSize, outputDimension: intermediateSize)
        ElementwiseFragment(count: intermediateSize, kind: .swiglu)
        LinearFragment(field: "down_proj", inputDimension: intermediateSize, outputDimension: outputSize)
    }
    public var isFusable: Bool { false }
    public func _visitBody(context: KernelContext, _ visitor: (any MetalKernelFragment) -> Void) { visitor(fragment(context: context)) }
}
