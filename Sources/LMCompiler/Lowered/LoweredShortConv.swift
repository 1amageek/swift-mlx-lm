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

    /// Depthwise conv1d weight: [D, 1, K] where K = kernelSize.
    public let convWeight: MLXArray

    /// Short convolution attributes (hiddenSize, kernelSize).
    public let attrs: ShortConvAttributes

    /// Compile-time resolved cache slot index.
    public let cacheSlotIndex: Int

    public init(
        inProj: LoweredProjection,
        outProj: LoweredProjection,
        convWeight: MLXArray,
        attrs: ShortConvAttributes,
        cacheSlotIndex: Int
    ) {
        self.inProj = inProj
        self.outProj = outProj
        self.convWeight = convWeight
        self.attrs = attrs
        self.cacheSlotIndex = cacheSlotIndex
    }

    /// Apply short convolution with external cache state.
    public func apply(_ x: MLXArray, caches: inout [LoweredCacheState]) -> MLXArray {
        let D = attrs.hiddenSize
        let K = attrs.kernelSize
        let B = x.dim(0)
        let T = x.dim(1)

        // in_proj: [B, T, D] -> [B, T, 3D] — compile-time kernel selection
        let bcx = inProj.apply(x)

        // Chunk into B-gate, C-gate, x-input along last dimension
        let bGate = bcx[0..., 0..., ..<D]
        let cGate = bcx[0..., 0..., D..<(2 * D)]
        let xInput = bcx[0..., 0..., (2 * D)...]

        // First gate: element-wise multiply (plain, not sigmoid)
        let bx = bGate * xInput  // [B, T, D]

        // Transpose to [B, D, T] for conv operations
        let bxT = bx.transposed(0, 2, 1)

        // Extract cache
        var convState: MLXArray?
        var cacheOffset: Int = 0
        switch caches[cacheSlotIndex] {
        case .recurrent(let cache):
            convState = cache.convState
            cacheOffset = cache.offset
        default:
            break
        }

        // Depthwise causal conv1d
        let convOut: MLXArray
        let newConvState: MLXArray

        if let existingState = convState {
            // Decode path: use rolling cache
            let prefix = existingState.asType(bxT.dtype)
            let combined = concatenated([prefix, bxT], axis: 2)  // [B, D, prevK + T]

            // Store last K timesteps as new cache
            let totalLen = combined.dim(2)
            newConvState = combined[0..., 0..., (totalLen - K)...]

            if T == 1 {
                // Single token: direct dot product for maximum efficiency
                let w = convWeight.squeezed(axis: 1)  // [D, K]
                let state = newConvState  // [B, D, K]
                convOut = sum(state.asType(bxT.dtype) * w, axis: 2)
                    .expandedDimensions(axis: 1)  // [B, 1, D]
            } else {
                // Multi-token with cache: full conv over combined input
                let w3d = convWeight.ndim == 2
                    ? convWeight.expandedDimensions(axis: -1)
                    : convWeight
                let raw = conv1d(combined, w3d, stride: 1, padding: 0, groups: D)
                let rawLen = raw.dim(1)
                convOut = raw[0..., (rawLen - T)..., 0...]  // [B, T, D]
            }
        } else {
            // Prefill: causal conv1d with zero padding
            let prefix = MLXArray.zeros([B, D, K - 1])
            let padded = concatenated([prefix, bxT], axis: 2)
            let w3d = convWeight.ndim == 2
                ? convWeight.expandedDimensions(axis: -1)
                : convWeight
            convOut = conv1d(padded, w3d, stride: 1, padding: 0, groups: D)

            // Initialize cache with last K timesteps
            newConvState = bxT[0..., 0..., (T - min(K, T))...]
        }

        // Update cache (conv-only, no recurrent state)
        caches[cacheSlotIndex] = .recurrent(LoweredRecurrentCache(
            convState: newConvState,
            recurrentState: nil,
            offset: cacheOffset + T
        ))

        // Second gate: element-wise multiply (plain, not sigmoid)
        let y = cGate * convOut

        // out_proj: [B, T, D] -> [B, T, D] — compile-time kernel selection
        return outProj.apply(y)
    }
}
