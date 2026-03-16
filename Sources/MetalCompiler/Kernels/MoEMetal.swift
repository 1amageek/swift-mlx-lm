import LMIR

extension MoEAttributes: MetalComponent {

    public var dispatchDeclarations: [MetalDispatchDeclaration] {
        [.projection(MetalProjection(field: "router", inputDimension: expertMLP.inputSize, outputDimension: expertCount))]
    }

    public var weightSlots: [MetalWeightSlot] {
        [MetalWeightSlot(field: "router", role: .weight),
         MetalWeightSlot(field: "expert_gates", role: .weight),
         MetalWeightSlot(field: "expert_ups", role: .weight),
         MetalWeightSlot(field: "expert_downs", role: .weight)]
    }
}
