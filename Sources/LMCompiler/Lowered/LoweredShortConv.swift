@preconcurrency import MLX
import SwiftLM

/// Lowered short convolution module for LFM2 family with compile-time kernel selection.
///
/// Implements double-gated LIV depthwise convolution:
///   in_proj(D -> 3D) -> chunk(B, C, x) -> B*x -> depthwise_conv1d -> C*conv_out -> out_proj
///
/// ## Design
///
/// ShortConv is significantly simpler than DeltaNet:
/// - 2 projections (in_proj, out_proj) vs 5 (QKV, Z, B, A, out)
/// - No recurrent state — conv cache only
/// - Plain element-wise gating (no sigmoid/softplus)
/// - No dtype boundary management (no recurrence accumulation)
///
/// ## Compile-time optimizations
///
/// - `inProj` and `outProj` use `LoweredProjection` for compile-time
///   kernel selection (matmul vs quantizedMatmul)
/// - Cache state is externalized as `LoweredCacheState.recurrent`
///   for functional-style inference
///
/// Reference: `Lfm2ShortConv` in HuggingFace transformers.
public struct LoweredShortConv: @unchecked Sendable {

    /// Input projection (D -> 3D) with compile-time resolved kernel.
    public let inProj: LoweredProjection

    /// Output projection (D -> D) with compile-time resolved kernel.
    public let outProj: LoweredProjection

    /// Depthwise conv1d weight in MLX layout: [D, K, 1] (C_out, kernel, C_in/groups).
    public let convWeight: MLXArray


    /// Short convolution attributes (hiddenSize, kernelSize).
    public let attrs: ShortConvAttributes

    /// Compile-time resolved cache slot index.
    public let cacheSlotIndex: Int

    public init(
        inProj: LoweredProjection,
        outProj: LoweredProjection,
        convWeight rawWeight: MLXArray,
        attrs: ShortConvAttributes,
        cacheSlotIndex: Int
    ) {
        self.inProj = inProj
        self.outProj = outProj
        // Pre-compute weight layout at compile time (not per-token).
        // Raw weight from safetensors: [D, 1, K]
        // MLX conv1d (channels-last) expects: [C_out, K, C_in/groups] = [D, K, 1]
        if rawWeight.ndim == 3 {
            self.convWeight = rawWeight.transposed(0, 2, 1)  // [D, K, 1]
        } else {
            self.convWeight = rawWeight.expandedDimensions(axis: -1)  // [D, K, 1]
        }
        self.attrs = attrs
        self.cacheSlotIndex = cacheSlotIndex
    }

    /// Apply short convolution with external cache state.
    public func apply(_ x: MLXArray, caches: inout [LoweredCacheState]) -> MLXArray {
        let D = attrs.hiddenSize
        let K = attrs.kernelSize
        let B = x.dim(0)

        // in_proj: [B, T, D] -> [B, T, 3D] — compile-time kernel selection
        let bcx = inProj.apply(x)

        // Split into B-gate, C-gate, x-input (single op, zero-copy)
        let parts = bcx.split(parts: 3, axis: -1)
        let bGate = parts[0]
        let cGate = parts[1]
        let xInput = parts[2]

        // First gate: element-wise multiply (plain, not sigmoid)
        var bx = bGate * xInput  // [B, T, D]

        // Extract cache state or initialize with zeros
        var state: MLXArray?
        var cacheOffset: Int = 0
        switch caches[cacheSlotIndex] {
        case .recurrent(let cache):
            state = cache.convState
            cacheOffset = cache.offset
        default:
            break
        }
        if state == nil {
            state = MLXArray.zeros([B, K - 1, D], dtype: bx.dtype)
        }

        // Concatenate cache + input (zero-copy views)
        bx = concatenated([state!, bx], axis: -2)

        // Update cache: last K-1 timesteps
        // Cache stores K-1 elements. After concat with T new tokens, total length = (K-1) + T.
        // We need the last K-1 from that, so start = T.
        // For decode (T=1): start = 1, for prefill: start = T.
        let newConvState = bx[0..., x.dim(1)..., 0...]

        // Depthwise causal conv1d
        let convOut = conv1d(bx, convWeight, stride: 1, padding: 0, groups: D)

        // Store updated cache
        caches[cacheSlotIndex] = .recurrent(LoweredRecurrentCache(
            convState: newConvState,
            recurrentState: nil,
            offset: cacheOffset + x.dim(1)
        ))

        // Second gate: element-wise multiply (plain, not sigmoid)
        let y = cGate * convOut

        // out_proj: [B, T, D] -> [B, T, D] — compile-time kernel selection
        return outProj.apply(y)
    }
}
