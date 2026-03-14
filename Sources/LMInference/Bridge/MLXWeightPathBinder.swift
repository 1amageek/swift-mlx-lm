import LMCompiler
import SwiftLM

/// Weight binding error for the inference compiler path.
enum WeightBindingError: Error, CustomStringConvertible {

    /// A tensor's MLX path could not be mapped to any ParameterSlot.
    case unmappedTensor(mlxPath: String)

    /// A required parameter slot has no matching tensor.
    case missingRequiredSlot(slot: ParameterSlot, mlxPath: String)

    var description: String {
        switch self {
        case .unmappedTensor(let path):
            return "No ParameterSlot found for MLX weight path: \(path)"
        case .missingRequiredSlot(let slot, let mlxPath):
            return "Missing tensor for required slot \(slot.role) at \(mlxPath)"
        }
    }
}

/// Binds MLX-path-keyed weight tensors to semantic ParameterSlots.
///
/// Accepts `RawWeights` keyed by MLX weight paths (e.g., "model.layers.0.self_attn.q_proj.weight")
/// and maps them to `ParameterSlot`s using `ModelGraphSlotEnumerator`.
///
/// Usage:
/// ```swift
/// let bound = try MLXWeightPathBinder().bind(rawWeights, to: graph)
/// let store = try InferenceWeightStore(boundWeights: bound)
/// ```
package struct MLXWeightPathBinder: WeightBinder {

    let naming: WeightNamingConvention

    package init(naming: WeightNamingConvention = .llamaFamily) {
        self.naming = naming
    }

    package func bind(_ raw: RawWeights, to graph: ModelGraph) throws -> BoundWeights {
        // Enumerate all expected slots from the graph
        let enumerator = ModelGraphSlotEnumerator()
        let manifest = enumerator.enumerate(graph, naming: naming)

        // Build MLX path → slot lookup
        let slotByPath: [String: ParameterSlot] = Dictionary(
            manifest.map { ($0.mlxWeightPath, $0.slot) },
            uniquingKeysWith: { first, _ in first }
        )

        // Map each raw tensor to its slot
        var result: [ParameterSlot: TensorData] = [:]
        for (mlxPath, tensor) in raw.tensors {
            guard let slot = slotByPath[mlxPath] else {
                // Skip unmapped tensors (e.g., LoRA weights, quantization metadata)
                // The caller can validate completeness separately
                continue
            }
            result[slot] = tensor
        }

        return BoundWeights(tensors: result)
    }
}
