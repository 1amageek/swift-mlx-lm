import LMIR

extension StateSpaceAttributes: MetalKernelFragment, _FragmentBodyAccessor {
    public func fragment(context: KernelContext) -> SSMRecurrenceFragment {
        SSMRecurrenceFragment(headCount: numHeads, keyHeadDimension: keyHeadDim, valueHeadDimension: valueHeadDim)
    }
    public var isFusable: Bool { false }
    public func _visitBody(context: KernelContext, _ visitor: (any MetalKernelFragment) -> Void) { visitor(fragment(context: context)) }
}
