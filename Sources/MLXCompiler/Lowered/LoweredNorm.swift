@preconcurrency import MLX
import MLXFast

/// Lowered normalization operation.
///
/// Supports RMSNorm and LayerNorm. Norm weights are always dense (not quantized).
public enum LoweredNorm: @unchecked Sendable {

    /// RMS normalization: `rmsNorm(x, weight: weight, eps: epsilon)`.
    case rms(weight: MLXArray, epsilon: Float)

    /// Layer normalization with optional bias.
    case layer(weight: MLXArray, bias: MLXArray?, epsilon: Float)

    /// Apply normalization to the input tensor.
    public func apply(_ x: MLXArray) -> MLXArray {
        switch self {
        case .rms(let weight, let epsilon):
            return MLXFast.rmsNorm(x, weight: weight, eps: epsilon)
        case .layer(let weight, let bias, let epsilon):
            return layerNormOp(x, weight: weight, bias: bias, eps: epsilon)
        }
    }
}

/// Manual layer normalization implementation.
///
/// Shared across `LoweredNorm`, `LoweredAttention`, `GraphAttention`, and `MLXExecutor`.
/// Normalizes along the last axis: `(x - mean) / sqrt(var + eps) * weight + bias`.
func layerNormOp(
    _ x: MLXArray, weight: MLXArray, bias: MLXArray?, eps: Float = 1e-5
) -> MLXArray {
    let mean = x.mean(axis: -1, keepDims: true)
    let variance = (x - mean).square().mean(axis: -1, keepDims: true)
    let normalized = (x - mean) / MLX.sqrt(variance + MLXArray(eps))
    var result = normalized * weight
    if let bias {
        result = result + bias
    }
    return result
}
