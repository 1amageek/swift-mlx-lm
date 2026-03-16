import LMIR

extension StateSpaceAttributes: MetalComponent {

    public var dispatchDeclarations: [MetalDispatchDeclaration] {
        let qkvDimension = numHeads * (keyHeadDim + keyHeadDim + valueHeadDim)
        let zDimension = numHeads * valueHeadDim
        let outputDimension = numHeads * valueHeadDim
        return [
            .projection(MetalProjection(field: "in_proj_qkv", inputDimension: hiddenSize, outputDimension: qkvDimension)),
            .projection(MetalProjection(field: "in_proj_z", inputDimension: hiddenSize, outputDimension: zDimension)),
            .projection(MetalProjection(field: "in_proj_b", inputDimension: hiddenSize, outputDimension: numHeads)),
            .projection(MetalProjection(field: "in_proj_a", inputDimension: hiddenSize, outputDimension: numHeads)),
            .compute(SSMRecurrenceOperation(headCount: numHeads, keyHeadDimension: keyHeadDim, valueHeadDimension: valueHeadDim)),
            .projection(MetalProjection(field: "out_proj", inputDimension: outputDimension, outputDimension: hiddenSize)),
        ]
    }

    public var weightSlots: [MetalWeightSlot] {
        [MetalWeightSlot(field: "in_proj_qkv", role: .weight),
         MetalWeightSlot(field: "in_proj_z", role: .weight),
         MetalWeightSlot(field: "in_proj_b", role: .weight),
         MetalWeightSlot(field: "in_proj_a", role: .weight),
         MetalWeightSlot(field: "norm", role: .scale),
         MetalWeightSlot(field: "out_proj", role: .weight)]
    }

    public var cacheSlots: [MetalCacheSlot] {
        [MetalCacheSlot(name: "ssm_state", kind: .recurrent)]
    }
}
