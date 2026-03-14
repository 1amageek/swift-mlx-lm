@preconcurrency import MLX
import MLXNN
import SwiftLM

/// Feed-forward MLP module compiled from ModelGraph.
///
/// All linear projections use `@ModuleInfo`-annotated `Linear` layers,
/// enabling automatic quantization via `MLXNN.quantize()`.
/// Supports SwiGLU, GeGLU, GLU gating and SiLU/GELU/ReLU activations.
final class GraphMLP: Module, UnaryLayer {

    @ModuleInfo(key: "gate_proj") var gateProj: MLXNN.Linear
    @ModuleInfo(key: "down_proj") var downProj: MLXNN.Linear
    @ModuleInfo(key: "up_proj") var upProj: MLXNN.Linear

    let activation: ActivationKind
    let hasGating: Bool

    init(attrs: MLPAttributes, store: MLXWeightStore, path: StructuralPath) throws {
        self.activation = attrs.activation

        self._gateProj.wrappedValue = MLXNN.Linear(
            weight: try store.require(ParameterSlot(path: path.appending(.field("gate_proj")), role: .weight)),
            bias: store.get(ParameterSlot(path: path.appending(.field("gate_proj")), role: .bias))
        )
        self._downProj.wrappedValue = MLXNN.Linear(
            weight: try store.require(ParameterSlot(path: path.appending(.field("down_proj")), role: .weight)),
            bias: store.get(ParameterSlot(path: path.appending(.field("down_proj")), role: .bias))
        )

        switch attrs.gating {
        case .swiglu, .geglu, .glu, .custom:
            self._upProj.wrappedValue = MLXNN.Linear(
                weight: try store.require(ParameterSlot(path: path.appending(.field("up_proj")), role: .weight)),
                bias: store.get(ParameterSlot(path: path.appending(.field("up_proj")), role: .bias))
            )
            self.hasGating = true
        case .none:
            // Non-gated MLP: up_proj is unused but must be initialized for @ModuleInfo
            self._upProj.wrappedValue = MLXNN.Linear(
                weight: try store.require(ParameterSlot(path: path.appending(.field("gate_proj")), role: .weight)),
                bias: nil
            )
            self.hasGating = false
        }
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let gate: MLXArray
        switch activation {
        case .silu, .swish:
            gate = silu(gateProj(x))
        case .gelu:
            gate = gelu(gateProj(x))
        case .relu:
            gate = relu(gateProj(x))
        case .custom:
            gate = silu(gateProj(x))
        }

        if hasGating {
            return downProj(gate * upProj(x))
        }
        return downProj(gate)
    }
}
