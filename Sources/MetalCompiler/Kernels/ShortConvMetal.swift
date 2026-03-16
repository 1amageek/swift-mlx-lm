import LMIR

extension ShortConvAttributes: MetalComponent {

    public var dispatchDeclarations: [MetalDispatchDeclaration] {
        [.projection(MetalProjection(field: "in_proj", inputDimension: hiddenSize, outputDimension: hiddenSize * 3)),
         .compute(Conv1dOperation(dimension: hiddenSize, kernelSize: kernelSize)),
         .projection(MetalProjection(field: "out_proj", inputDimension: hiddenSize, outputDimension: hiddenSize))]
    }

    public var weightSlots: [MetalWeightSlot] {
        [MetalWeightSlot(field: "in_proj", role: .weight),
         MetalWeightSlot(field: "conv_weight", role: .weight),
         MetalWeightSlot(field: "out_proj", role: .weight)]
    }

    public var cacheSlots: [MetalCacheSlot] {
        [MetalCacheSlot(name: "conv_cache", kind: .conv)]
    }
}
