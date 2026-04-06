import LMIR

extension PerLayerInputAttributes: MetalKernelFragment, _FragmentBodyAccessor {
    @MetalKernelFragmentBuilder
    public func fragment(context: KernelContext) -> some MetalKernelFragment {
        LinearFragment(
            field: "per_layer_input_gate",
            inputDimension: hiddenSize,
            outputDimension: perLayerInputSize
        )
        PerLayerInputModulationFragment(
            dimension: perLayerInputSize,
            activation: activation
        )
        LinearFragment(
            field: "per_layer_projection",
            inputDimension: perLayerInputSize,
            outputDimension: hiddenSize
        )
        Reduction(
            dimension: hiddenSize,
            epsilon: 1e-6,
            weightRole: "post_per_layer_input_norm"
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
