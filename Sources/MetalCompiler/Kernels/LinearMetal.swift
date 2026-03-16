import LMIR

extension LinearAttributes: MetalComponent {

    public var dispatchDeclarations: [MetalDispatchDeclaration] {
        [.projection(MetalProjection(field: "weight", inputDimension: inputSize, outputDimension: outputSize))]
    }

    public var weightSlots: [MetalWeightSlot] {
        [MetalWeightSlot(role: .weight)]
    }
}
