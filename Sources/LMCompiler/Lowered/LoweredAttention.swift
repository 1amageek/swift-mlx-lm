@preconcurrency import MLX
import MLXFast
import SwiftLM

/// Lowered attention operation with compile-time kernel selection.
///
/// Uses packed QKV projection (single matmul + split) when all three projections
/// share the same kernel variant. Falls back to individual projections otherwise.
///
/// Reference: `MLXExecutor.executeAttention()` in `MLXExecutor.swift:410-527`.
public struct LoweredAttention: @unchecked Sendable {

    /// Packed Q/K/V projection (single matmul + split).
    /// Non-nil when packing succeeded at compile time.
    public let qkvPacked: PackedProjection?

    /// Individual Q/K/V projections (fallback when packing is not possible).
    public let qProj: LoweredProjection?
    public let kProj: LoweredProjection?
    public let vProj: LoweredProjection?

    /// Output projection (never packed — different input than QKV).
    public let oProj: LoweredProjection

    /// Attention attributes (head count, KV head count, head dimension, etc.).
    public let attrs: AttentionAttributes

    /// Optional QK normalization weights.
    public let qNormWeight: MLXArray?
    public let kNormWeight: MLXArray?
    /// Optional QK normalization bias (for layerNorm).
    public let qNormBias: MLXArray?
    public let kNormBias: MLXArray?

    /// Compile-time resolved cache slot index.
    public let cacheSlotIndex: Int

    /// Initialize with packed QKV projection.
    public init(
        qkvPacked: PackedProjection,
        oProj: LoweredProjection,
        attrs: AttentionAttributes,
        qNormWeight: MLXArray?,
        kNormWeight: MLXArray?,
        qNormBias: MLXArray?,
        kNormBias: MLXArray?,
        cacheSlotIndex: Int
    ) {
        self.qkvPacked = qkvPacked
        self.qProj = nil
        self.kProj = nil
        self.vProj = nil
        self.oProj = oProj
        self.attrs = attrs
        self.qNormWeight = qNormWeight
        self.kNormWeight = kNormWeight
        self.qNormBias = qNormBias
        self.kNormBias = kNormBias
        self.cacheSlotIndex = cacheSlotIndex
    }

    /// Initialize with individual Q/K/V projections (fallback).
    public init(
        qProj: LoweredProjection,
        kProj: LoweredProjection,
        vProj: LoweredProjection,
        oProj: LoweredProjection,
        attrs: AttentionAttributes,
        qNormWeight: MLXArray?,
        kNormWeight: MLXArray?,
        qNormBias: MLXArray?,
        kNormBias: MLXArray?,
        cacheSlotIndex: Int
    ) {
        self.qkvPacked = nil
        self.qProj = qProj
        self.kProj = kProj
        self.vProj = vProj
        self.oProj = oProj
        self.attrs = attrs
        self.qNormWeight = qNormWeight
        self.kNormWeight = kNormWeight
        self.qNormBias = qNormBias
        self.kNormBias = kNormBias
        self.cacheSlotIndex = cacheSlotIndex
    }

    /// Apply attention with external cache state.
    ///
    /// When `positionIds` is provided (shape `[3, B, S]` for M-RoPE), per-token
    /// multi-axis rotation is applied instead of a scalar cache offset.
    public func apply(
        _ x: MLXArray, caches: inout [LoweredCacheState],
        positionIds: MLXArray? = nil
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))
        let headDim = attrs.headDimension
        let scale = 1.0 / Float(headDim).squareRoot()

        // Project Q, K, V — packed or individual
        var queries: MLXArray
        var keys: MLXArray
        var values: MLXArray

        if let qkvPacked {
            let qkv = qkvPacked.apply(x)
            queries = qkv[0]
            keys = qkv[1]
            values = qkv[2]
        } else {
            queries = qProj!.apply(x)
            keys = kProj!.apply(x)
            values = vProj!.apply(x)
        }

        // Output gate: extract gate from packed Q projection
        var gateValues: MLXArray? = nil
        if let outputGate = attrs.outputGate {
            switch outputGate {
            case .sigmoidPackedInQProj:
                // Q projection output: [B, L, headCount * 2 * headDim]
                // Per-head layout: [q_dims, gate_dims] interleaved per head.
                // Must reshape per-head then split within each head (not flat split).
                let headCount = attrs.headCount
                let perHeadDim = queries.dim(-1) / headCount  // 2 * headDim
                let hd = perHeadDim / 2  // headDim
                let perHead = queries.reshaped(B, L, headCount, perHeadDim)
                gateValues = perHead[0..., 0..., 0..., hd...].reshaped(B, L, -1)
                queries = perHead[0..., 0..., 0..., 0..<hd].reshaped(B, L, -1)
            }
        }

        // Reshape to head layout [B, H, L, D]
        queries = queries.reshaped(B, L, attrs.headCount, -1).transposed(0, 2, 1, 3)
        keys = keys.reshaped(B, L, attrs.kvHeadCount, -1).transposed(0, 2, 1, 3)
        values = values.reshaped(B, L, attrs.kvHeadCount, -1).transposed(0, 2, 1, 3)

        // QK normalization
        if let qkNorm = attrs.qkNorm {
            switch qkNorm {
            case .rmsNorm:
                if let qnw = qNormWeight {
                    queries = MLXFast.rmsNorm(queries, weight: qnw, eps: 1e-6)
                }
                if let knw = kNormWeight {
                    keys = MLXFast.rmsNorm(keys, weight: knw, eps: 1e-6)
                }
            case .layerNorm:
                if let qnw = qNormWeight {
                    queries = layerNormOp(queries, weight: qnw, bias: qNormBias)
                }
                if let knw = kNormWeight {
                    keys = layerNormOp(keys, weight: knw, bias: kNormBias)
                }
            case .none, .custom:
                break
            }
        }

        // Resolve cache offset for RoPE
        let cacheOffset: Int
        switch caches[cacheSlotIndex] {
        case .kv(let cache):
            cacheOffset = cache.offset
        default:
            cacheOffset = 0
        }

        // RoPE
        if let ropeAttrs = attrs.rope {
            if let mropeAxes = ropeAttrs.mropeAxes, let posIds = positionIds {
                // M-RoPE: per-token 3D position encoding for VLM
                let ropeDim = ropeAttrs.dimension
                if mropeAxes.interleaved {
                    queries = applyInterleavedMRoPE(
                        queries, positionIds: posIds, ropeDim: ropeDim,
                        ropeBase: ropeAttrs.base, sections: mropeAxes.sections,
                        headDim: headDim)
                    keys = applyInterleavedMRoPE(
                        keys, positionIds: posIds, ropeDim: ropeDim,
                        ropeBase: ropeAttrs.base, sections: mropeAxes.sections,
                        headDim: headDim)
                } else {
                    queries = applyContiguousMRoPE(
                        queries, positionIds: posIds,
                        ropeBase: ropeAttrs.base, sections: mropeAxes.sections,
                        headDim: headDim)
                    keys = applyContiguousMRoPE(
                        keys, positionIds: posIds,
                        ropeBase: ropeAttrs.base, sections: mropeAxes.sections,
                        headDim: headDim)
                }
            } else {
                // Standard RoPE with scalar offset
                let ropeScale: Float = {
                    if let scaling = ropeAttrs.scaling, scaling.kind == .linear {
                        return 1.0 / scaling.factor
                    }
                    return 1.0
                }()

                let ropeDim = ropeAttrs.dimension
                if ropeDim < headDim {
                    let qRot = MLXFast.RoPE(
                        queries[0..., 0..., 0..., 0..<ropeDim],
                        dimensions: ropeDim, traditional: false,
                        base: ropeAttrs.base, scale: ropeScale, offset: cacheOffset
                    )
                    queries = concatenated(
                        [qRot, queries[0..., 0..., 0..., ropeDim...]], axis: -1)

                    let kRot = MLXFast.RoPE(
                        keys[0..., 0..., 0..., 0..<ropeDim],
                        dimensions: ropeDim, traditional: false,
                        base: ropeAttrs.base, scale: ropeScale, offset: cacheOffset
                    )
                    keys = concatenated(
                        [kRot, keys[0..., 0..., 0..., ropeDim...]], axis: -1)
                } else {
                    queries = MLXFast.RoPE(
                        queries, dimensions: ropeDim, traditional: false,
                        base: ropeAttrs.base, scale: ropeScale, offset: cacheOffset
                    )
                    keys = MLXFast.RoPE(
                        keys, dimensions: ropeDim, traditional: false,
                        base: ropeAttrs.base, scale: ropeScale, offset: cacheOffset
                    )
                }
            }
        }

        // KV cache update
        let mask: MLXFast.ScaledDotProductAttentionMaskMode
        switch caches[cacheSlotIndex] {
        case .kv(var cache):
            let (cachedKeys, cachedValues) = cache.update(newKeys: keys, newValues: values)
            mask = cache.makeMask(queryLength: L)
            keys = cachedKeys
            values = cachedValues
            caches[cacheSlotIndex] = .kv(cache)
        default:
            mask = L > 1 ? .causal : .none
        }

        // Scaled dot-product attention
        let attnOutput = MLXFast.scaledDotProductAttention(
            queries: queries, keys: keys, values: values,
            scale: scale, mask: mask
        )

        // Qwen3.5-style output gates modulate the concatenated head output
        // before the final output projection.
        var output = attnOutput.transposed(0, 2, 1, 3).reshaped(B, L, -1)
        if let gate = gateValues {
            output = sigmoid(gate) * output
        }

        return oProj.apply(output)
    }
}

// layerNormOp is shared from LoweredNorm.swift
