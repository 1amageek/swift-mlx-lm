import LMIR

extension StateSpaceAttributes: MetalKernelFragment, _FragmentBodyAccessor {
    @MetalKernelFragmentBuilder
    public func fragment(context: KernelContext) -> some MetalKernelFragment {
        let projectedStateDimension = 2 * groupCount * keyHeadDim + numHeads * valueHeadDim
        let outputDimension = numHeads * valueHeadDim

        LinearFragment(
            field: "in_proj_qkv",
            inputDimension: hiddenSize,
            outputDimension: projectedStateDimension
        )
        LinearFragment(
            field: "in_proj_z",
            inputDimension: hiddenSize,
            outputDimension: outputDimension
        )
        LinearFragment(
            field: "in_proj_b",
            inputDimension: hiddenSize,
            outputDimension: numHeads
        )
        LinearFragment(
            field: "in_proj_a",
            inputDimension: hiddenSize,
            outputDimension: numHeads
        )
        SSMRecurrenceFragment(
            headCount: numHeads,
            groupCount: groupCount,
            keyHeadDimension: keyHeadDim,
            valueHeadDimension: valueHeadDim,
            convKernelSize: convKernelSize
        )
        LinearFragment(
            field: "out_proj",
            inputDimension: outputDimension,
            outputDimension: hiddenSize
        )
    }
    public var isFusable: Bool { false }
    public func _visitBody(context: KernelContext, _ visitor: (any MetalKernelFragment) -> Void) { visitor(fragment(context: context)) }
}
