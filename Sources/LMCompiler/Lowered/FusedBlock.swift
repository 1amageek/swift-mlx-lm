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

    /// Pre-norm + ShortConv + residual add.
    case shortConv(norm: LoweredNorm, shortConv: LoweredShortConv)

    /// Pre-norm + MoE + residual add.
    case moe(norm: LoweredNorm, moe: LoweredMoE)

    /// Apply the fused sub-layer: `output = x + op(norm(x))`.
    ///
    /// The residual `x` is passed to each op's apply so it can be fused
    /// into the Metal kernel (avoiding a separate dispatch for `+`).
    /// Ops that support residual fusion accept it and add inline.
    /// Ops that don't return the raw result, and we add here.
    public func apply(
        _ x: MLXArray, state: inout InferenceState,
        options: ExecutionOptions = .default
    ) -> MLXArray {
        let normed = switch self {
        case .attention(let norm, _), .mlp(let norm, _),
             .deltaNet(let norm, _), .shortConv(let norm, _),
             .moe(let norm, _):
            norm.apply(x)
        }

        switch self {
        case .attention(_, let attn):
            return x + attn.apply(
                normed, caches: &state.caches,
                positionIds: options.positionIds)
        case .mlp(_, let mlp):
            return mlp.applyWithResidual(normed, residual: x)
        case .deltaNet(_, let dn):
            return x + dn.apply(normed, caches: &state.caches)
        case .shortConv(_, let sc):
            return sc.applyWithResidual(normed, residual: x, caches: &state.caches)
        case .moe(_, let moe):
            return x + moe.apply(normed)
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
    case .op(.shortConv(let sc)):
        return .shortConv(norm: norm, shortConv: sc)
    case .op(.moe(let moe)):
        return .moe(norm: norm, moe: moe)
    default:
        return nil
    }
}
