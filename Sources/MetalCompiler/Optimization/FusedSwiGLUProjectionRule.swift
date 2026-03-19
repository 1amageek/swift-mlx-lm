import LMIR

struct FusedSwiGLUProjectionRule {
    static func match(
        at index: Int,
        primitives: [CollectedPrimitive]
    ) -> OptimizedEntry? {
        guard index + 2 < primitives.count,
              let firstProjection = projectionInfo(primitives[index]),
              let secondProjection = projectionInfo(primitives[index + 1]),
              let swiglu = primitives[index + 2].fragment as? ElementwiseFragment,
              swiglu.kind == .swiglu,
              firstProjection.inputDimension == secondProjection.inputDimension,
              firstProjection.outputDimension == secondProjection.outputDimension,
              swiglu.count == firstProjection.outputDimension else {
            return nil
        }

        let projection: FusedSwiGLUProjection
        switch (firstProjection.field, secondProjection.field) {
        case ("gate_proj", "up_proj"):
            projection = FusedSwiGLUProjection(
                inputDimension: firstProjection.inputDimension,
                outputDimension: firstProjection.outputDimension,
                gateField: firstProjection.field,
                upField: secondProjection.field)
        case ("up_proj", "gate_proj"):
            projection = FusedSwiGLUProjection(
                inputDimension: firstProjection.inputDimension,
                outputDimension: firstProjection.outputDimension,
                gateField: secondProjection.field,
                upField: firstProjection.field)
        default:
            return nil
        }

        let bindings = primitives[index...index + 2].flatMap(\.parameterBindings)
        return .fusedSwiGLUProjection(
            projection,
            parameterBindings: bindings,
            layerIndex: primitives[index].layerIndex)
    }

    private static func projectionInfo(
        _ primitive: CollectedPrimitive
    ) -> (field: String, inputDimension: Int, outputDimension: Int)? {
        guard case .gemv(let outputDimension, let inputDimension) = primitive.fragment.dispatchDimension,
              let field = primitive.fragment.weightSlots.first?.field else {
            return nil
        }
        return (field, inputDimension, outputDimension)
    }
}
