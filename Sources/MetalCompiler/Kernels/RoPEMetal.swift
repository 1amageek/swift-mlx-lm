import LMIR

extension RoPEAttributes: MetalComponent {

    public var dispatchDeclarations: [MetalDispatchDeclaration] {
        [.compute(RoPEOperation(headCount: 0, kvHeadCount: 0, headDimension: dimension, ropeDimension: dimension, base: base))]
    }

    public var weightSlots: [MetalWeightSlot] { [] }
}
