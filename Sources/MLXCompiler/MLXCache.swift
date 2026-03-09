import MLX
import MLXFast

// MARK: - KV Cache Protocol

/// Cache protocol for attention layers in the MLXCompiler executor.
public protocol MLXKVCache: AnyObject {

    /// Number of tokens currently cached.
    var offset: Int { get }

    /// Append new keys/values, return full cached key/value sequences.
    func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray)

    /// Create an attention mask for the current cache state.
    func makeMask(queryLength: Int) -> MLXFast.ScaledDotProductAttentionMaskMode
}

// MARK: - Simple KV Cache

/// Simple append-only KV cache with pre-allocated growth.
///
/// Matches the behavior of MLXLM's `KVCacheSimple` — grows by `step` tokens
/// at a time and returns sliced views for attention.
public final class MLXKVCacheSimple: MLXKVCache {

    private var keys: MLXArray?
    private var values: MLXArray?
    private let step: Int
    public private(set) var offset: Int = 0

    public init(step: Int = 256) {
        self.step = step
    }

    public func update(keys newKeys: MLXArray, values newValues: MLXArray) -> (MLXArray, MLXArray) {
        let B = newKeys.dim(0)
        let heads = newKeys.dim(1)
        let headDim = newKeys.dim(3)
        let nTokens = newKeys.dim(2)

        if self.keys == nil {
            let capacity = alignUp(nTokens + step, to: step)
            self.keys = MLXArray.zeros([B, heads, capacity, headDim], dtype: newKeys.dtype)
            self.values = MLXArray.zeros([B, heads, capacity, headDim], dtype: newValues.dtype)
            self.offset = 0
        }

        let currentCapacity = self.keys!.dim(2)
        let needed = offset + nTokens
        if needed > currentCapacity {
            let newCapacity = alignUp(needed + step, to: step)
            let extShape = [B, heads, newCapacity - currentCapacity, headDim]
            self.keys = concatenated(
                [self.keys!, MLXArray.zeros(extShape, dtype: newKeys.dtype)], axis: 2)
            self.values = concatenated(
                [self.values!, MLXArray.zeros(extShape, dtype: newValues.dtype)], axis: 2)
        }

        self.keys![0..., 0..., offset..<(offset + nTokens), 0...] = newKeys
        self.values![0..., 0..., offset..<(offset + nTokens), 0...] = newValues
        offset += nTokens

        return (
            self.keys![0..., 0..., 0..<offset, 0...],
            self.values![0..., 0..., 0..<offset, 0...]
        )
    }

    public func makeMask(queryLength n: Int) -> MLXFast.ScaledDotProductAttentionMaskMode {
        n > 1 ? .causal : .none
    }

    private func alignUp(_ value: Int, to alignment: Int) -> Int {
        ((value + alignment - 1) / alignment) * alignment
    }
}

// MARK: - Recurrent State Cache

/// Recurrent state cache for DeltaNet and similar state-space models.
///
/// Stores a Conv1D sliding window buffer and a fixed-size recurrent state matrix,
/// unlike the growing KV cache used by standard attention.
public final class MLXRecurrentCache {

    /// Conv1D sliding window state [B, K, C].
    public var convState: MLXArray?

    /// Recurrent state matrix [B, H, d_k, d_v].
    public var recurrentState: MLXArray?

    /// Number of tokens processed.
    public private(set) var offset: Int = 0

    public init() {}

    /// Increment the offset by the number of new tokens processed.
    public func incrementOffset(by n: Int) {
        offset += n
    }
}
