import LMIR

extension OutputHeadAttributes: MetalComponent {

    public var dispatchDeclarations: [MetalDispatchDeclaration] {
        [.projection(MetalProjection(field: "weight", inputDimension: inputSize, outputDimension: vocabSize)),
         .compute(ArgmaxOperation(vocabularySize: vocabSize))]
    }

    public var weightSlots: [MetalWeightSlot] {
        [MetalWeightSlot(role: .weight)]
    }
}
