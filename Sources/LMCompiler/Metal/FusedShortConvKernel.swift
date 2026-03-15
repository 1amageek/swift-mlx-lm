@preconcurrency import MLX
import MLXFast

/// Fused Metal kernel for ShortConv decode step (T=1).
///
/// Combines split + B-gate + depthwise conv1d + C-gate + cache update
/// into a single Metal kernel dispatch. Eliminates 5+ intermediate
/// dispatches per conv layer during single-token decode.
///
/// ## Operations fused
///
/// 1. Split inProj output into B-gate, C-gate, x-input
/// 2. Compute Bx = B-gate * x-input
/// 3. Depthwise conv1d over [cache, Bx] window
/// 4. Compute output = C-gate * conv_out
/// 5. Update cache: shift left, append Bx
///
/// ## Grid layout
///
/// - `thread_position_in_grid.x` = d (channel index, 0..<D)
/// - Each thread processes one channel independently
///
/// ## Template parameters
///
/// - `T` (DType): float or half
/// - `CONV_K` (Int): kernel size (typically 3)
/// - `HIDDEN_D` (Int): hidden dimension
enum FusedShortConvKernel {

    // MARK: - Kernel factory

    private static let kernel: MLXFastKernel = MLXFast.metalKernel(
        name: "fused_short_conv_decode",
        inputNames: ["bcx", "state", "weight"],
        outputNames: ["out", "new_state"],
        source: metalSource
    )

    // MARK: - Public API

    /// Execute fused ShortConv decode kernel.
    ///
    /// - Parameters:
    ///   - bcx: inProj output, flattened [3*D] (B-gate, C-gate, x concatenated)
    ///   - state: conv cache [K-1, D] (previous timesteps)
    ///   - weight: conv weight [D, K] (squeezed from [D, 1, K])
    ///   - hiddenSize: D dimension
    ///   - kernelSize: K (conv kernel size)
    ///   - dtype: compute dtype
    /// - Returns: (output [D], newState [(K-1)*D])
    static func call(
        bcx: MLXArray,
        state: MLXArray,
        weight: MLXArray,
        hiddenSize D: Int,
        kernelSize K: Int,
        dtype: DType
    ) -> (output: MLXArray, newState: MLXArray) {
        let results = kernel(
            [bcx.reshaped(-1), state.reshaped(-1), weight.reshaped(D, K)],
            template: [
                ("T", dtype),
                ("CONV_K", K),
                ("HIDDEN_D", D),
            ],
            grid: (D, 1, 1),
            threadGroup: (min(D, 1024), 1, 1),
            outputShapes: [
                [1, 1, D],
                [1, K - 1, D],
            ],
            outputDTypes: [dtype, dtype]
        )

        return (results[0], results[1])
    }

    // MARK: - Metal source

    private static let metalSource = """
    // Grid: (D, 1, 1) — one thread per channel
    uint d = thread_position_in_grid.x;

    if (d >= HIDDEN_D) return;

    // Split: bcx is [B_gate | C_gate | x_input], each of size D
    T b_gate = bcx[d];
    T c_gate = bcx[HIDDEN_D + d];
    T x_val  = bcx[2 * HIDDEN_D + d];

    // First gate: B * x
    T bx = b_gate * x_val;

    // Depthwise conv1d over window [state[0..K-2, d], bx]
    // state layout: [K-1, D] row-major → state[k * D + d]
    // weight layout: [D, K] row-major → weight[d * CONV_K + k]
    T conv_sum = 0;
    for (int k = 0; k < CONV_K - 1; k++) {
        conv_sum += state[k * HIDDEN_D + d] * weight[d * CONV_K + k];
    }
    conv_sum += bx * weight[d * CONV_K + (CONV_K - 1)];

    // Second gate: C * conv_out
    out[d] = c_gate * conv_sum;

    // Cache update: shift left by 1, append bx at the end
    for (int k = 0; k < CONV_K - 2; k++) {
        new_state[k * HIDDEN_D + d] = state[(k + 1) * HIDDEN_D + d];
    }
    new_state[(CONV_K - 2) * HIDDEN_D + d] = bx;
    """
}
