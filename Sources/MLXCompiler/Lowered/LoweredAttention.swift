@preconcurrency import MLX
import MLXFast
import SwiftLM

/// Lowered attention operation with compile-time kernel selection.
///
/// Contains 4 lowered projections (Q, K, V, O), optional QK normalization,
/// optional RoPE, and SDPA. Cache slot is resolved at compile time.
///
/// Reference: `MLXExecutor.executeAttention()` in `MLXExecutor.swift:410-527`.
public struct LoweredAttention: @unchecked Sendable {

    /// Query projection.
    public let qProj: LoweredProjection
    /// Key projection.
    public let kProj: LoweredProjection
    /// Value projection.
    public let vProj: LoweredProjection
    /// Output projection.
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
    public func apply(_ x: MLXArray, caches: inout [LoweredCacheState]) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))
        let headDim = attrs.headDimension
        let scale = 1.0 / Float(headDim).squareRoot()

        // Project Q, K, V
        var queries = qProj.apply(x)
        var keys = kProj.apply(x)
        var values = vProj.apply(x)

        // Output gate: extract gate from packed Q projection
        var gateValues: MLXArray? = nil
        if let outputGate = attrs.outputGate {
            switch outputGate {
            case .sigmoidPackedInQProj:
                // Q projection output is [B, L, 2 * headCount * headDim]
                // Split into queries and gate along last axis
                let qDim = queries.dim(-1) / 2
                gateValues = queries[0..., 0..., qDim...]
                queries = queries[0..., 0..., 0..<qDim]
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
                    queries = MLXFast.rmsNorm(queries, weight: 1 + qnw, eps: 1e-6)
                }
                if let knw = kNormWeight {
                    keys = MLXFast.rmsNorm(keys, weight: 1 + knw, eps: 1e-6)
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
            let ropeScale: Float = {
                if let scaling = ropeAttrs.scaling, scaling.kind == .linear {
                    return 1.0 / scaling.factor
                }
                return 1.0
            }()

            let ropeDim = ropeAttrs.dimension
            if ropeDim < headDim {
                // Partial RoPE (e.g. Qwen 3.5: 64 of 256)
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
