@preconcurrency import MLX
import MLXFast

/// External cache state for lowered inference.
///
/// Value type — passed in/out of inference steps rather than hidden in modules.
/// This enables functional-style inference and `MLX.compile()` tracing.
public enum LoweredCacheState: @unchecked Sendable {

    /// KV cache for standard attention.
    case kv(LoweredKVCache)

    /// Recurrent state cache for DeltaNet and similar state-space models.
    case recurrent(LoweredRecurrentCache)

    /// Uninitialized cache slot.
    case empty
}

// MARK: - KV Cache

/// Value-type KV cache replicating `MLXKVCacheSimple` behavior.
///
/// Append-only cache with pre-allocated growth. Returns sliced views for attention.
///
/// Reference: `MLXKVCacheSimple` in `MLXCache.swift:25-77`.
public struct LoweredKVCache: @unchecked Sendable {

    /// Cached key tensor of shape `[B, H, capacity, D]`.
    public var keys: MLXArray?

    /// Cached value tensor of shape `[B, H, capacity, D]`.
    public var values: MLXArray?

    /// Number of tokens currently cached.
    public var offset: Int

    /// Growth step for pre-allocation.
    public let step: Int

    public init(step: Int = 256) {
        self.keys = nil
        self.values = nil
        self.offset = 0
        self.step = step
    }

    /// Append new keys/values and return the full cached sequences.
    public mutating func update(
        newKeys: MLXArray, newValues: MLXArray
    ) -> (MLXArray, MLXArray) {
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

    /// Create an attention mask for the current cache state.
    public func makeMask(queryLength n: Int) -> MLXFast.ScaledDotProductAttentionMaskMode {
        n > 1 ? .causal : .none
    }

    private func alignUp(_ value: Int, to alignment: Int) -> Int {
        ((value + alignment - 1) / alignment) * alignment
    }
}

// MARK: - Recurrent Cache

/// Value-type recurrent state cache for DeltaNet.
///
/// Reference: `MLXRecurrentCache` in `MLXCache.swift:85-102`.
public struct LoweredRecurrentCache: @unchecked Sendable {

    /// Conv1D sliding window state `[B, K, C]`.
    public var convState: MLXArray?

    /// Recurrent state matrix `[B, H, d_k, d_v]`.
    public var recurrentState: MLXArray?

    /// Number of tokens processed.
    public var offset: Int

    public init(
        convState: MLXArray? = nil,
        recurrentState: MLXArray? = nil,
        offset: Int = 0
    ) {
        self.convState = convState
        self.recurrentState = recurrentState
        self.offset = offset
    }
}
