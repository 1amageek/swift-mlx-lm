import LMIR

extension LayerScaleAttributes: MetalKernelFragment, _FragmentBodyAccessor {
    @MetalKernelFragmentBuilder
    public func fragment(context: KernelContext) -> some MetalKernelFragment {
        ScalarMultiplyFragment(
            count: dimension,
            weightRole: "layer_scalar"
        )
    }

    public var isFusable: Bool { false }

    public func _visitBody(
        context: KernelContext,
        _ visitor: (any MetalKernelFragment) -> Void
    ) {
        visitor(fragment(context: context))
    }
}
