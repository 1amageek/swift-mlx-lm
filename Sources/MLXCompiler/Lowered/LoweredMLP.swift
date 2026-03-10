@preconcurrency import MLX
import MLXNN
import SwiftLM

/// Lowered MLP (feed-forward network) with compile-time kernel selection.
///
/// Supports both gated (SwiGLU, GeGLU, GLU) and ungated architectures.
/// When ungated, `upProj` is nil — no dummy weight, no wasted memory.
public struct LoweredMLP: @unchecked Sendable {

    /// Gate projection: `gate = activation(gateProj(x))`.
    public let gateProj: LoweredProjection

    /// Down projection: `output = downProj(gated)`.
    public let downProj: LoweredProjection

    /// Up projection for gated variants: `up = upProj(x)`.
    /// Nil when gating is `.none`.
    public let upProj: LoweredProjection?

    /// Activation function to apply to the gate projection output.
    public let activation: ActivationKind

    public init(
        gateProj: LoweredProjection,
        downProj: LoweredProjection,
        upProj: LoweredProjection?,
        activation: ActivationKind
    ) {
        self.gateProj = gateProj
        self.downProj = downProj
        self.upProj = upProj
        self.activation = activation
    }

    /// Apply the MLP: `output = downProj(activation(gateProj(x)) * upProj(x))`.
    public func apply(_ x: MLXArray) -> MLXArray {
        let gate = gateProj.apply(x)

        let activated: MLXArray
        switch activation {
        case .silu, .swish:
            activated = silu(gate)
        case .gelu:
            activated = gelu(gate)
        case .relu:
            activated = relu(gate)
        case .custom:
            activated = silu(gate)
        }

        let gated: MLXArray
        if let upProj {
            gated = activated * upProj.apply(x)
        } else {
            gated = activated
        }

        return downProj.apply(gated)
    }
}
