import LMIR

extension RMSNormAttributes: MetalComponent {

    public var dispatchDeclarations: [MetalDispatchDeclaration] {
        [.compute(RMSNormOperation(dimension: dimension, epsilon: epsilon))]
    }

    public var weightSlots: [MetalWeightSlot] {
        [MetalWeightSlot(role: .scale)]
    }
}
