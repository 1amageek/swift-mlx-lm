import LMIR

extension MLPAttributes: MetalComponent {

    public var dispatchDeclarations: [MetalDispatchDeclaration] {
        var declarations: [MetalDispatchDeclaration] = [
            .projection(MetalProjection(field: "gate_proj", inputDimension: inputSize, outputDimension: intermediateSize)),
        ]
        if gating != .none {
            declarations.append(.projection(MetalProjection(field: "up_proj", inputDimension: inputSize, outputDimension: intermediateSize)))
            declarations.append(.compute(SwiGLUOperation(dimension: intermediateSize)))
        }
        declarations.append(.projection(MetalProjection(field: "down_proj", inputDimension: intermediateSize, outputDimension: outputSize)))
        return declarations
    }

    public var weightSlots: [MetalWeightSlot] {
        var slots = [MetalWeightSlot(field: "gate_proj", role: .weight)]
        if gating != .none {
            slots.append(MetalWeightSlot(field: "up_proj", role: .weight))
        }
        slots.append(MetalWeightSlot(field: "down_proj", role: .weight))
        return slots
    }
}
