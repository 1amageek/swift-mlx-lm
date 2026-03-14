@preconcurrency import MLX
import MLXNN
import SwiftLM

/// Lowered single expert MLP within a MoE layer.
///
/// Each expert has its own gate, up, and down projections with
/// independently resolved kernels.
public struct LoweredExpertMLP: @unchecked Sendable {

    public let gateProj: LoweredProjection
    public let upProj: LoweredProjection
    public let downProj: LoweredProjection
    public let activation: ActivationKind

    public init(
        gateProj: LoweredProjection,
        upProj: LoweredProjection,
        downProj: LoweredProjection,
        activation: ActivationKind
    ) {
        self.gateProj = gateProj
        self.upProj = upProj
        self.downProj = downProj
        self.activation = activation
    }

    /// Apply expert MLP: `downProj(activation(gateProj(x)) * upProj(x))`.
    public func apply(_ x: MLXArray) -> MLXArray {
        let gate: MLXArray
        switch activation {
        case .silu, .swish:
            gate = silu(gateProj.apply(x))
        case .gelu:
            gate = gelu(gateProj.apply(x))
        case .relu:
            gate = relu(gateProj.apply(x))
        case .custom:
            gate = silu(gateProj.apply(x))
        }
        return downProj.apply(gate * upProj.apply(x))
    }
}

/// Lowered Mixture-of-Experts with compile-time kernel selection.
///
/// Router gate and all experts have independently resolved projection kernels.
///
/// Reference: `MLXExecutor.executeMoE()` in `MLXExecutor.swift:584-635`.
public struct LoweredMoE: @unchecked Sendable {

    /// Router gate projection.
    public let router: LoweredProjection

    /// Expert MLPs — each with its own compile-time resolved kernels.
    public let experts: [LoweredExpertMLP]

    /// Number of experts selected per token.
    public let expertsPerToken: Int

    public init(
        router: LoweredProjection,
        experts: [LoweredExpertMLP],
        expertsPerToken: Int
    ) {
        self.router = router
        self.experts = experts
        self.expertsPerToken = expertsPerToken
    }

    /// Apply MoE routing and expert computation.
    public func apply(_ x: MLXArray) -> MLXArray {
        let (B, L, D) = (x.dim(0), x.dim(1), x.dim(2))
        let flat = x.reshaped(-1, D)

        let gateLogits = router.apply(flat)

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

            output = output + expert.apply(flat) * expertWeight
        }

        return output.reshaped(B, L, D)
    }
}
