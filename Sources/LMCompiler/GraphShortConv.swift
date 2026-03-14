@preconcurrency import MLX
import MLXNN
import SwiftLM

/// Short convolution module for LFM2 family.
///
/// Implements double-gated LIV depthwise convolution:
///   in_proj(D → 3D) → chunk(B, C, x) → B*x → depthwise_conv1d → C*conv_out → out_proj
///
/// Reference: LiquidAI LFM2 (Lfm2ShortConv in HuggingFace transformers)
final class GraphShortConv: Module, UnaryLayer {

    @ModuleInfo(key: "in_proj") var inProj: MLXNN.Linear
    @ModuleInfo(key: "out_proj") var outProj: MLXNN.Linear

    /// Depthwise conv1d weight: [D, 1, K] where K = kernelSize.
    let convWeight: MLXArray

    let attrs: ShortConvAttributes
    var cache: MLXRecurrentCache

    init(
        attrs: ShortConvAttributes,
        store: MLXWeightStore,
        path: StructuralPath,
        cache: MLXRecurrentCache
    ) throws {
        self.attrs = attrs
        self.cache = cache

        // Linear projections (quantization-compatible)
        self._inProj.wrappedValue = MLXNN.Linear(
            weight: try store.require(ParameterSlot(path: path.appending(.field("in_proj")), role: .weight)),
            bias: store.get(ParameterSlot(path: path.appending(.field("in_proj")), role: .bias))
        )
        self._outProj.wrappedValue = MLXNN.Linear(
            weight: try store.require(ParameterSlot(path: path.appending(.field("out_proj")), role: .weight)),
            bias: store.get(ParameterSlot(path: path.appending(.field("out_proj")), role: .bias))
        )

        // Depthwise conv1d kernel (raw array, not quantized)
        self.convWeight = try store.require(
            ParameterSlot(path: path.appending(.field("conv")), role: .weight))
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let D = attrs.hiddenSize
        let K = attrs.kernelSize
        let B = x.dim(0)
        let T = x.dim(1)

        // in_proj: [B, T, D] → [B, T, 3D]
        let bcx = inProj(x)

        // Chunk into B-gate, C-gate, x-input along last dimension
        let bGate = bcx[0..., 0..., ..<D]
        let cGate = bcx[0..., 0..., D..<(2 * D)]
        let xInput = bcx[0..., 0..., (2 * D)...]

        // First gate: element-wise multiply
        let bx = bGate * xInput  // [B, T, D]

        // Depthwise causal conv1d
        let convOut: MLXArray

        // Transpose to [B, D, T] for conv operations
        let bxT = bx.transposed(0, 2, 1)

        if let existingState = cache.convState {
            // Token-by-token decode: update rolling cache
            let prefix = existingState.asType(bxT.dtype)
            let combined = concatenated([prefix, bxT], axis: 2)  // [B, D, prevK + T]

            // Store last K timesteps as new cache
            let totalLen = combined.dim(2)
            cache.convState = combined[0..., 0..., (totalLen - K)...]

            // Compute depthwise conv1d via dot product with weight
            // convWeight: [D, 1, K] → [D, K]
            let w = convWeight.squeezed(axis: 1)

            if T == 1 {
                // Single token: direct dot product
                let state = cache.convState!  // [B, D, K]
                convOut = sum(state.asType(bxT.dtype) * w, axis: 2)
                    .expandedDimensions(axis: 1)  // [B, 1, D]
            } else {
                // Multi-token with cache: full conv over combined input
                let convWeight3d = convWeight.ndim == 2
                    ? convWeight.expandedDimensions(axis: -1)
                    : convWeight
                let raw = conv1d(combined, convWeight3d, stride: 1, padding: 0, groups: D)
                // raw: [B, totalLen - K + 1, D], take last T elements
                let rawLen = raw.dim(1)
                convOut = raw[0..., (rawLen - T)..., 0...]  // [B, T, D]
            }
        } else {
            // Prefill: full causal conv1d
            // Prepend zeros for causal padding: [B, D, K-1] + [B, D, T] → [B, D, T+K-1]
            let prefix = MLXArray.zeros([B, D, K - 1])
            let padded = concatenated([prefix, bxT], axis: 2)
            let convWeight3d = convWeight.ndim == 2
                ? convWeight.expandedDimensions(axis: -1)
                : convWeight
            // conv1d: [B, D, T+K-1] × [D, 1, K] → [B, T, D] (groups=D for depthwise)
            let raw = conv1d(padded, convWeight3d, stride: 1, padding: 0, groups: D)
            convOut = raw  // [B, T, D]

            // Initialize cache with last K timesteps of bx
            cache.convState = bxT[0..., 0..., (T - min(K, T))...]
        }

        // Second gate: element-wise multiply
        let y: MLXArray
        if convOut.dim(1) == T && convOut.dim(2) == D {
            // convOut already in [B, T, D] from conv1d output
            y = cGate * convOut
        } else {
            // convOut in [B, 1, D] from single-token path
            y = cGate * convOut
        }

        // out_proj: [B, T, D] → [B, T, D]
        return outProj(y)
    }
}
