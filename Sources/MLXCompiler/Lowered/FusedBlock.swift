@preconcurrency import MLX
import SwiftLM

/// Fused sub-layer: pre-norm + operation + residual add in a single execution step.
///
/// Eliminates per-step FlatStep switch overhead and residual stack push/pop
/// by inlining the common transformer pattern `x + op(norm(x))`.
///
/// Pattern detection occurs during `flattenSteps()` — when a `.residual(body:)`
/// contains exactly `[.op(.norm(...)), .op(attention/mlp/deltaNet/moe)]`,
/// it is replaced with a single `.fusedSubLayer(...)` step.
public enum FusedSubLayer: @unchecked Sendable {

    /// Pre-norm + attention + residual add.
    case attention(norm: LoweredNorm, attention: LoweredAttention)

    /// Pre-norm + MLP + residual add.
    case mlp(norm: LoweredNorm, mlp: LoweredMLP)

    /// Pre-norm + DeltaNet + residual add.
    case deltaNet(norm: LoweredNorm, deltaNet: LoweredDeltaNet)

    /// Pre-norm + MoE + residual add.
    case moe(norm: LoweredNorm, moe: LoweredMoE)

    /// Apply the fused sub-layer: `output = x + op(norm(x))`.
    ///
    /// Residual addition is performed inline — no external stack needed.
    public func apply(_ x: MLXArray, state: inout InferenceState) -> MLXArray {
        switch self {
        case .attention(let norm, let attn):
            return x + attn.apply(norm.apply(x), caches: &state.caches)
        case .mlp(let norm, let mlp):
            return x + mlp.apply(norm.apply(x))
        case .deltaNet(let norm, let dn):
            return x + dn.apply(norm.apply(x), caches: &state.caches)
        case .moe(let norm, let moe):
            return x + moe.apply(norm.apply(x))
        }
    }
}

// MARK: - Pattern Detection

/// Attempt to fuse a residual body into a single `FusedSubLayer`.
///
/// Recognizes the common transformer pattern:
///   `.residual(body: [.op(.norm(...)), .op(attention/mlp/deltaNet/moe)])`
///
/// Returns `nil` if the body does not match any fuseable pattern.
func tryFuseResidual(_ body: [LoweredStep]) -> FusedSubLayer? {
    guard body.count == 2 else { return nil }

    // First step must be a norm operation
    guard case .op(.norm(let norm)) = body[0] else { return nil }

    // Second step must be a fuseable operation
    switch body[1] {
    case .op(.attention(let attn)):
        return .attention(norm: norm, attention: attn)
    case .op(.mlp(let mlp)):
        return .mlp(norm: norm, mlp: mlp)
    case .op(.deltaNet(let dn)):
        return .deltaNet(norm: norm, deltaNet: dn)
    case .op(.moe(let moe)):
        return .moe(norm: norm, moe: moe)
    default:
        return nil
    }
}
