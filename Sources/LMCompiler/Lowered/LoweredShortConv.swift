@preconcurrency import MLX
import MLXFast
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

    /// Depthwise conv1d weight: [D, K] for SSM conv kernel (squeezed from [D, 1, K]).
    public let convWeight: MLXArray

    /// SSM conv Metal kernel — specialized dot product for short convolution.
    /// Much faster than generic conv1d for small kernel sizes (K=3..4).
    private let ssmConvKernel: MLXFast.MLXFastKernel

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

        // Pre-compute weight layout at compile time.
        // Raw weight from safetensors: [D, 1, K]
        // SSM conv kernel uses [D, K] — squeezed for direct dot product.
        if rawWeight.ndim == 3 {
            self.convWeight = rawWeight.squeezed(axis: 1)  // [D, K]
        } else {
            self.convWeight = rawWeight  // already [D, K]
        }

        // Create SSM conv Metal kernel — llama.cpp-style dot product per channel.
        //
        // Auto-generated signature by MLX (from source string analysis):
        //   const device T* input [[buffer(0)]],
        //   const constant int* input_shape [[buffer(1)]],      ← auto-added (source contains "input_shape")
        //   device T* output [[buffer(2)]],
        //   uint3 thread_position_in_grid [[thread_position_in_grid]]
        //
        // Input: padded [B, L_in, D] row-contiguous (channels-last). L_in = K-1 + T.
        // Weight: [D, K] row-contiguous.
        // Output: [B, T, D] where output[b,t,d] = sum_{k=0}^{K-1} input[b,t+k,d] * weight[d,k]
        //
        // Template params: CONV_K (kernel size), HIDDEN_D (hidden dimension) — compile-time constants.
        // Grid: (D, T, B). Each thread computes one output element.
        let K = attrs.kernelSize
        let D = attrs.hiddenSize
        self.ssmConvKernel = MLXFast.metalKernel(
            name: "ssm_conv_k\(K)_d\(D)",
            inputNames: ["input", "weight"],
            outputNames: ["output"],
            source: """
                // grid: (D, T, B). Each thread computes one output[b, t, d].
                uint d = thread_position_in_grid.x;
                uint t = thread_position_in_grid.y;
                uint b = thread_position_in_grid.z;

                // input: [B, L_in, D] row-contiguous. L_in = input_shape[1].
                uint L_in = (uint)input_shape[1];
                uint in_base = b * L_in * HIDDEN_D;

                // Accumulate in T (matches input/output dtype: bfloat16 or float32).
                // DType template param ensures correct type throughout.
                DType sum = DType(0);
                for (uint k = 0; k < (uint)CONV_K; k++) {
                    sum += DType(input[in_base + (t + k) * HIDDEN_D + d])
                         * DType(weight[d * CONV_K + k]);
                }

                // output: [B, T, D] row-contiguous. T = L_in - CONV_K + 1.
                uint out_T = L_in - CONV_K + 1;
                output[b * out_T * HIDDEN_D + t * HIDDEN_D + d] = sum;
            """,
            ensureRowContiguous: true
        )

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

        let convOut: MLXArray
        let newConvState: MLXArray

        if T == 1 && B == 1 {
            // ── Fused decode path ──
            // Single Metal kernel: split + B-gate + conv1d + C-gate + cache update
            // Eliminates 5+ intermediate MLX dispatches
            if state == nil {
                state = MLXArray.zeros([1, K - 1, D], dtype: bcx.dtype)
            }
            let (fusedOut, fusedState) = FusedShortConvKernel.call(
                bcx: bcx,
                state: state!,
                weight: convWeight,
                hiddenSize: D,
                kernelSize: K,
                dtype: bcx.dtype)
            convOut = fusedOut
            newConvState = fusedState
        } else {
            // ── Prefill / multi-token path ──
            // Use SSM conv Metal kernel (handles variable sequence length)
            let parts = bcx.split(parts: 3, axis: -1)
            let bGate = parts[0]
            let cGate = parts[1]
            let xInput = parts[2]

            var bx = bGate * xInput
            if state == nil {
                state = MLXArray.zeros([B, K - 1, D], dtype: bx.dtype)
            }
            bx = concatenated([state!, bx], axis: -2)
            newConvState = bx[0..., T..., 0...]

            let ssmOut = ssmConvKernel(
                [bx, convWeight],
                template: [("CONV_K", K), ("HIDDEN_D", D), ("DType", bx.dtype)],
                grid: (D, T, B),
                threadGroup: (min(D, 256), 1, 1),
                outputShapes: [[B, T, D]],
                outputDTypes: [bx.dtype],
                verbose: false
            )[0]

            convOut = cGate * ssmOut
        }

        // Store updated cache
        caches[cacheSlotIndex] = .recurrent(LoweredRecurrentCache(
            convState: newConvState,
            recurrentState: nil,
            offset: cacheOffset + T
        ))

        // out_proj: [B, T, D] -> [B, T, D]
        return outProj.apply(convOut)
    }
}
