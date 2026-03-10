@preconcurrency import MLX
import MLXFast
import SwiftLM

/// Lowered inference operation — a leaf computation node.
///
/// Each case wraps a concrete lowered primitive with compile-time
/// resolved kernels. No MLXNN Module hierarchy, no `@ModuleInfo`.
public enum LoweredInferenceOp: @unchecked Sendable {
    case tokenEmbedding(LoweredEmbedding)
    case attention(LoweredAttention)
    case mlp(LoweredMLP)
    case moe(LoweredMoE)
    case norm(LoweredNorm)
    case outputHead(LoweredOutputHead)
    case deltaNet(LoweredDeltaNet)
    case rope(LoweredRoPE)
    case positionalEmbedding(LoweredPositionalEmbedding)
    case linear(LoweredProjection)
}

/// Execution steps with structural annotations.
///
/// `.repeating` is NOT a case — repeating blocks are fully unrolled at
/// compile time, giving each layer its own lowered ops and cache index.
public enum LoweredStep: @unchecked Sendable {

    /// A leaf operation.
    case op(LoweredInferenceOp)

    /// Residual connection: `x + executeSteps(body, input: x)`.
    case residual(body: [LoweredStep])

    /// Parallel branches executed on the same input, merged by strategy.
    case parallel(merge: ParallelMergeStrategy, branches: [[LoweredStep]])
}

// MARK: - Standalone RoPE

/// Lowered standalone RoPE module.
///
/// Uses `state.nextPosition` as offset for position-dependent rotation.
/// In practice, RoPE is usually embedded within `LoweredAttention`.
public struct LoweredRoPE: @unchecked Sendable {

    public let attrs: RoPEAttributes

    public init(attrs: RoPEAttributes) {
        self.attrs = attrs
    }

    public func apply(_ x: MLXArray, offset: Int) -> MLXArray {
        let scale: Float
        switch attrs.scaling?.kind {
        case .linear:
            scale = 1.0 / attrs.scaling!.factor
        default:
            scale = 1.0
        }
        return MLXFast.RoPE(
            x, dimensions: attrs.dimension, traditional: false,
            base: attrs.base, scale: scale, offset: offset
        )
    }
}

// MARK: - Standalone Positional Embedding

/// Lowered positional embedding: `x + table[0..<seqLen]`.
public struct LoweredPositionalEmbedding: @unchecked Sendable {

    public let table: MLXArray

    public init(table: MLXArray) {
        self.table = table
    }

    public func apply(_ x: MLXArray, offset: Int) -> MLXArray {
        let seqLen = x.dim(1)
        let positions = MLXArray(offset..<(offset + seqLen))
        return x + table[positions]
    }
}
