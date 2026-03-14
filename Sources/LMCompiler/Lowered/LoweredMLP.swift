@preconcurrency import MLX
import MLXNN
import SwiftLM

/// Lowered MLP (feed-forward network) with compile-time kernel selection.
///
/// Supports both gated (SwiGLU, GeGLU, GLU) and ungated architectures.
/// When gated and both projections share the same kernel variant,
/// gate+up are packed into a single matmul + split.
public struct LoweredMLP: @unchecked Sendable {

    /// Packed gate+up projection (single matmul + split).
    /// Non-nil when gating is enabled and packing succeeded at compile time.
    public let gateUpPacked: PackedProjection?

    /// Gate projection (fallback when packing is not possible or ungated).
    public let gateProj: LoweredProjection?

    /// Up projection for gated variants (fallback when packing is not possible).
    /// Nil when gating is `.none` or when packed.
    public let upProj: LoweredProjection?

    /// Down projection: `output = downProj(gated)`.
    public let downProj: LoweredProjection

    /// Activation function to apply to the gate projection output.
    public let activation: ActivationKind

    /// Initialize with packed gate+up projection.
    public init(
        gateUpPacked: PackedProjection,
        downProj: LoweredProjection,
        activation: ActivationKind
    ) {
        self.gateUpPacked = gateUpPacked
        self.gateProj = nil
        self.upProj = nil
        self.downProj = downProj
        self.activation = activation
    }

    /// Initialize with individual projections.
    public init(
        gateProj: LoweredProjection,
        downProj: LoweredProjection,
        upProj: LoweredProjection?,
        activation: ActivationKind
    ) {
        self.gateUpPacked = nil
        self.gateProj = gateProj
        self.upProj = upProj
        self.downProj = downProj
        self.activation = activation
    }

    /// Apply the MLP: `output = downProj(activation(gateProj(x)) * upProj(x))`.
    public func apply(_ x: MLXArray) -> MLXArray {
        let gate: MLXArray
        let up: MLXArray?

        if let gateUpPacked {
            let parts = gateUpPacked.apply(x)
            gate = parts[0]
            up = parts[1]
        } else {
            gate = gateProj!.apply(x)
            up = upProj?.apply(x)
        }

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
        if let up {
            gated = activated * up
        } else {
            gated = activated
        }

        return downProj.apply(gated)
    }
}
