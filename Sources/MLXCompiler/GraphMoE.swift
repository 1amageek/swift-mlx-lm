@preconcurrency import MLX
import MLXNN
import SwiftLM

/// Mixture-of-experts module compiled from ModelGraph.
///
/// Each expert is a `GraphExpertMLP` with its own quantizable `Linear` layers.
/// The router gate is also a `Linear` layer subject to quantization.
final class GraphMoE: Module, UnaryLayer {

    @ModuleInfo(key: "router") var router: MLXNN.Linear
    @ModuleInfo(key: "experts") var experts: [GraphExpertMLP]

    let expertsPerToken: Int

    init(attrs: MoEAttributes, store: MLXWeightStore, path: StructuralPath) throws {
        self.expertsPerToken = attrs.expertsPerToken

        self._router.wrappedValue = MLXNN.Linear(
            weight: try store.require(ParameterSlot(path: path.appending(.field("router")), role: .weight)),
            bias: store.get(ParameterSlot(path: path.appending(.field("router")), role: .bias))
        )

        var builtExperts: [GraphExpertMLP] = []
        builtExperts.reserveCapacity(attrs.expertCount)
        for i in 0..<attrs.expertCount {
            let expertPath = path.appending(.field("experts")).appending(.index(i))
            let expert = try GraphExpertMLP(
                attrs: attrs.expertMLP, store: store, path: expertPath
            )
            builtExperts.append(expert)
        }
        self._experts.wrappedValue = builtExperts
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (B, L, D) = (x.dim(0), x.dim(1), x.dim(2))
        let flat = x.reshaped(-1, D)

        let gateLogits = router(flat)

        let topKIndices = MLX.argSort(gateLogits, axis: -1)[
            0..., (gateLogits.dim(-1) - expertsPerToken)...]
        let topKGateLogits = MLX.takeAlong(gateLogits, topKIndices, axis: -1)
        let gateWeights = softmax(topKGateLogits, axis: -1)

        var output = MLXArray.zeros(like: flat)

        for (expertIdx, expert) in experts.enumerated() {
            // Accumulate combined weight across all topK slots (pure MLX, no item() sync)
            var expertWeight = MLXArray.zeros([flat.dim(0), 1])
            for k in 0..<expertsPerToken {
                let kMask = topKIndices[0..., k..<(k + 1)] .== MLXArray(Int32(expertIdx))
                let kMaskFloat = kMask.asType(.float32)
                expertWeight = expertWeight + gateWeights[0..., k..<(k + 1)] * kMaskFloat
            }

            output = output + expert(flat) * expertWeight
        }

        return output.reshaped(B, L, D)
    }
}

// MARK: - Expert MLP

/// Single expert MLP within a MoE layer.
///
/// Each expert has its own `gate_proj`, `up_proj`, `down_proj` Linear layers,
/// all individually quantizable.
final class GraphExpertMLP: Module, UnaryLayer {

    @ModuleInfo(key: "gate_proj") var gateProj: MLXNN.Linear
    @ModuleInfo(key: "up_proj") var upProj: MLXNN.Linear
    @ModuleInfo(key: "down_proj") var downProj: MLXNN.Linear

    let activation: ActivationKind

    init(attrs: MLPAttributes, store: MLXWeightStore, path: StructuralPath) throws {
        self.activation = attrs.activation

        self._gateProj.wrappedValue = MLXNN.Linear(
            weight: try store.require(ParameterSlot(path: path.appending(.field("gate_proj")), role: .weight)),
            bias: store.get(ParameterSlot(path: path.appending(.field("gate_proj")), role: .bias))
        )
        self._upProj.wrappedValue = MLXNN.Linear(
            weight: try store.require(ParameterSlot(path: path.appending(.field("up_proj")), role: .weight)),
            bias: store.get(ParameterSlot(path: path.appending(.field("up_proj")), role: .bias))
        )
        self._downProj.wrappedValue = MLXNN.Linear(
            weight: try store.require(ParameterSlot(path: path.appending(.field("down_proj")), role: .weight)),
            bias: store.get(ParameterSlot(path: path.appending(.field("down_proj")), role: .bias))
        )
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let gate: MLXArray
        switch activation {
        case .gelu:
            gate = gelu(gateProj(x))
        default:
            gate = silu(gateProj(x))
        }
        return downProj(gate * upProj(x))
    }
}
