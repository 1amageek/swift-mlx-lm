import LMIR

extension LayerNormAttributes: MetalComponent {

    public var dispatchDeclarations: [MetalDispatchDeclaration] {
        [.compute(LayerNormOperation(dimension: dimension, epsilon: epsilon, affine: affine))]
    }

    public var weightSlots: [MetalWeightSlot] {
        var slots = [MetalWeightSlot(role: .scale)]
        if affine { slots.append(MetalWeightSlot(field: "bias", role: .scale)) }
        return slots
    }
}
