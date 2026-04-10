import LMIR

extension RMSNormAttributes: MetalKernelFragment, _FragmentBodyAccessor {
    public func fragment(context: KernelContext) -> Reduction {
        Reduction(dimension: dimension, epsilon: epsilon, weightBias: weightBias)
    }
    public var isFusable: Bool { true }
    public func _visitBody(context: KernelContext, _ visitor: (any MetalKernelFragment) -> Void) { visitor(fragment(context: context)) }
}

extension LayerNormAttributes: MetalKernelFragment, _FragmentBodyAccessor {
    public func fragment(context: KernelContext) -> Reduction {
        Reduction(dimension: dimension, epsilon: epsilon)
    }
    public var isFusable: Bool { true }
    public func _visitBody(context: KernelContext, _ visitor: (any MetalKernelFragment) -> Void) { visitor(fragment(context: context)) }
}
