@preconcurrency import MLX
import MLXFast
import MLXNN
import SwiftLM

/// Multi-head attention module compiled from ModelGraph.
///
/// All linear projections (Q, K, V, O) use `@ModuleInfo`-annotated `Linear`
/// layers, enabling automatic quantization via `MLXNN.quantize()`.
/// When quantized, each projection uses `quantizedMM` instead of `matmul`.
final class GraphAttention: Module, UnaryLayer {

    @ModuleInfo(key: "q_proj") var qProj: MLXNN.Linear
    @ModuleInfo(key: "k_proj") var kProj: MLXNN.Linear
    @ModuleInfo(key: "v_proj") var vProj: MLXNN.Linear
    @ModuleInfo(key: "o_proj") var oProj: MLXNN.Linear

    // QK normalization weights (raw arrays, not quantizable modules)
    let qNormWeight: MLXArray?
    let kNormWeight: MLXArray?
    let qNormBias: MLXArray?
    let kNormBias: MLXArray?

    let attrs: AttentionAttributes
    var kvCache: MLXKVCache

    init(
        attrs: AttentionAttributes,
        store: MLXWeightStore,
        path: StructuralPath,
        cache: MLXKVCache
    ) throws {
        self.attrs = attrs
        self.kvCache = cache

        // Load projection weights
        self._qProj.wrappedValue = MLXNN.Linear(
            weight: try store.require(ParameterSlot(path: path.appending(.field("q_proj")), role: .weight)),
            bias: store.get(ParameterSlot(path: path.appending(.field("q_proj")), role: .bias))
        )
        self._kProj.wrappedValue = MLXNN.Linear(
            weight: try store.require(ParameterSlot(path: path.appending(.field("k_proj")), role: .weight)),
            bias: store.get(ParameterSlot(path: path.appending(.field("k_proj")), role: .bias))
        )
        self._vProj.wrappedValue = MLXNN.Linear(
            weight: try store.require(ParameterSlot(path: path.appending(.field("v_proj")), role: .weight)),
            bias: store.get(ParameterSlot(path: path.appending(.field("v_proj")), role: .bias))
        )
        self._oProj.wrappedValue = MLXNN.Linear(
            weight: try store.require(ParameterSlot(path: path.appending(.field("o_proj")), role: .weight)),
            bias: store.get(ParameterSlot(path: path.appending(.field("o_proj")), role: .bias))
        )

        // QK normalization weights
        self.qNormWeight = store.get(ParameterSlot(path: path.appending(.field("q_norm")), role: .scale))
        self.kNormWeight = store.get(ParameterSlot(path: path.appending(.field("k_norm")), role: .scale))
        self.qNormBias = store.get(ParameterSlot(path: path.appending(.field("q_norm")), role: .bias))
        self.kNormBias = store.get(ParameterSlot(path: path.appending(.field("k_norm")), role: .bias))
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))
        let headDim = attrs.headDimension
        let scale = 1.0 / Float(headDim).squareRoot()

        // Project Q, K, V through Linear modules (quantizedMM-compatible)
        var queries = qProj(x)
        var keys = kProj(x)
        var values = vProj(x)

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

        // RoPE
        let offset = kvCache.offset
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
                    base: ropeAttrs.base, scale: ropeScale, offset: offset
                )
                queries = concatenated(
                    [qRot, queries[0..., 0..., 0..., ropeDim...]], axis: -1)

                let kRot = MLXFast.RoPE(
                    keys[0..., 0..., 0..., 0..<ropeDim],
                    dimensions: ropeDim, traditional: false,
                    base: ropeAttrs.base, scale: ropeScale, offset: offset
                )
                keys = concatenated(
                    [kRot, keys[0..., 0..., 0..., ropeDim...]], axis: -1)
            } else {
                queries = MLXFast.RoPE(
                    queries, dimensions: ropeDim, traditional: false,
                    base: ropeAttrs.base, scale: ropeScale, offset: offset
                )
                keys = MLXFast.RoPE(
                    keys, dimensions: ropeDim, traditional: false,
                    base: ropeAttrs.base, scale: ropeScale, offset: offset
                )
            }
        }

        // KV cache update
        let (cachedKeys, cachedValues) = kvCache.update(keys: keys, values: values)
        let mask = kvCache.makeMask(queryLength: L)

        // Scaled dot-product attention
        let attnOutput = MLXFast.scaledDotProductAttention(
            queries: queries, keys: cachedKeys, values: cachedValues,
            scale: scale, mask: mask
        )

        // Output projection through Linear module (quantizedMM-compatible)
        return oProj(attnOutput.transposed(0, 2, 1, 3).reshaped(B, L, -1))
    }

    // MARK: - Utility

    // layerNormOp is shared from LoweredNorm.swift
}
