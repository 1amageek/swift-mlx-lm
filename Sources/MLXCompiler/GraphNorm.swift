@preconcurrency import MLX
import MLXFast
import MLXNN
import SwiftLM

// MARK: - GraphRMSNorm

/// RMSNorm module compiled from ModelGraph.
///
/// Wraps a scale parameter and applies `MLXFast.rmsNorm`.
/// Passes weight directly to `MLXFast.rmsNorm` (mlx-swift convention: weight * x_normalized).
final class GraphRMSNorm: Module, UnaryLayer {

    let weight: MLXArray
    let epsilon: Float

    init(attrs: RMSNormAttributes, store: MLXWeightStore, path: StructuralPath) throws {
        self.weight = try store.require(ParameterSlot(path: path, role: .scale))
        self.epsilon = attrs.epsilon
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        MLXFast.rmsNorm(x, weight: weight, eps: epsilon)
    }
}

// MARK: - GraphLayerNorm

/// LayerNorm module compiled from ModelGraph.
///
/// Manual implementation matching MLXExecutor's behavior.
final class GraphLayerNorm: Module, UnaryLayer {

    let weight: MLXArray
    let bias: MLXArray?
    let epsilon: Float

    init(attrs: LayerNormAttributes, store: MLXWeightStore, path: StructuralPath) throws {
        self.weight = try store.require(ParameterSlot(path: path, role: .scale))
        self.bias = attrs.affine ? store.get(ParameterSlot(path: path, role: .bias)) : nil
        self.epsilon = attrs.epsilon
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let mean = x.mean(axis: -1, keepDims: true)
        let variance = x.variance(axis: -1, keepDims: true)
        var normalized = (x - mean) / (variance + MLXArray(epsilon)).sqrt()
        normalized = normalized * weight
        if let bias { normalized = normalized + bias }
        return normalized
    }
}
