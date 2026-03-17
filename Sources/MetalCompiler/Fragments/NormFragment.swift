import LMIR

extension RMSNormAttributes: MetalKernelFragment, _FragmentBodyAccessor {
    public var fragment: Reduction {
        Reduction(dimension: dimension, epsilon: epsilon)
    }
    public var isFusable: Bool { true }
    public func _visitBody(_ visitor: (any MetalKernelFragment) -> Void) { visitor(fragment) }
}

extension LayerNormAttributes: MetalKernelFragment, _FragmentBodyAccessor {
    public var fragment: Reduction {
        Reduction(dimension: dimension, epsilon: epsilon)
    }
    public var isFusable: Bool { true }
    public func _visitBody(_ visitor: (any MetalKernelFragment) -> Void) { visitor(fragment) }
}
