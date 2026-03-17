import LMIR

extension StateSpaceAttributes: MetalKernelFragment, _FragmentBodyAccessor {
    public var fragment: SSMRecurrenceFragment {
        SSMRecurrenceFragment(headCount: numHeads, keyHeadDimension: keyHeadDim, valueHeadDimension: valueHeadDim)
    }
    public var isFusable: Bool { false }
    public func _visitBody(_ visitor: (any MetalKernelFragment) -> Void) { visitor(fragment) }
}
