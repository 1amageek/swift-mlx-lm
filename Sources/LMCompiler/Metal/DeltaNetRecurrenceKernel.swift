@preconcurrency import MLX
import MLXFast

/// Fused Metal kernel for DeltaNet state-space recurrence.
///
/// Replaces the per-timestep Swift loop with a single Metal kernel launch.
/// Each thread owns one column S[:, j] of the state matrix in thread-local
/// registers and processes all T timesteps sequentially.
///
/// ## Performance gains
///
/// - Eliminates T × 7 kernel launches per recurrence call
/// - Removes `eval(S)` hack for graph explosion prevention
/// - Removes per-timestep MLXArray allocations (split, squeeze, expand)
/// - State stays in registers across timesteps (no device memory round-trips)
///
/// ## Grid layout
///
/// - `thread_position_in_grid.x` = j (column index in dv, 0..<DV)
/// - `thread_position_in_grid.y` = bh (batch × heads, 0..<B*NH)
/// - Each thread is fully independent — no shared memory or synchronization
///
/// ## Template parameters (compile-time constants)
///
/// - `T` (DType): float or half — matches IR `StateSpaceAttributes.computeDType`
/// - `DK` (Int): key dimension per head (e.g. 128 for Qwen3.5)
/// - `DV` (Int): value dimension per head (e.g. 128 for Qwen3.5)
/// - `NH` (Int): number of heads (e.g. 8 for Qwen3.5-0.8B)
/// - `SEQ` (Int): sequence length T (1 for decode, prompt length for prefill)
enum DeltaNetRecurrenceKernel {

    // MARK: - Kernel factory

    /// Cached kernel factory — compiled once, handles all template param combinations.
    private static let kernel: MLXFastKernel = MLXFast.metalKernel(
        name: "deltanet_recurrence",
        inputNames: ["q", "k", "v", "decay_arr", "beta_arr", "s_in"],
        outputNames: ["out", "s_out"],
        source: metalSource
    )

    // MARK: - Public API

    /// Execute fused DeltaNet recurrence on Metal.
    ///
    /// Pre-conditions (caller's responsibility):
    /// - `query` and `key` are already l2-normalized
    /// - `query` is already scaled by `1/√dk`
    /// - All inputs are in `dtype` precision
    ///
    /// - Parameters:
    ///   - query: L2-normalized, scaled queries [B, T, H, dk]
    ///   - key: L2-normalized keys [B, T, H, dk]
    ///   - value: Values [B, T, H, dv]
    ///   - decay: Decay factors exp(g) [B, T, H]
    ///   - beta: Beta gates [B, T, H]
    ///   - state: Previous state [B, H, dk, dv], or zeros for initial
    ///   - dtype: Compute dtype (.float32 or .float16)
    /// - Returns: Tuple of (output [B, T, H, dv], newState [B, H, dk, dv])
    static func call(
        query: MLXArray,
        key: MLXArray,
        value: MLXArray,
        decay: MLXArray,
        beta: MLXArray,
        state: MLXArray,
        dtype: DType
    ) -> (output: MLXArray, newState: MLXArray) {
        let B = query.dim(0)
        let T = query.dim(1)
        let H = query.dim(2)
        let dk = query.dim(3)
        let dv = value.dim(3)

        let results = kernel(
            [query, key, value, decay, beta, state],
            template: [
                ("T", dtype),
                ("DK", dk),
                ("DV", dv),
                ("NH", H),
                ("SEQ", T),
            ],
            grid: (dv, B * H, 1),
            threadGroup: (min(dv, 256), 1, 1),
            outputShapes: [
                [B, T, H, dv],
                [B, H, dk, dv],
            ],
            outputDTypes: [dtype, dtype]
        )

        return (results[0], results[1])
    }

    // MARK: - Metal source

    /// Metal kernel body for DeltaNet recurrence.
    ///
    /// Computes for each (batch, head, column j):
    ///   for t in 0..<SEQ:
    ///     S[:, j] *= decay[t]                        // Phase 1: Decay
    ///     kvMem = dot(S[:, j], k[t, :])              // Phase 2: Memory readout
    ///     delta = beta[t] * (v[t, j] - kvMem)        // Phase 3: Delta rule
    ///     S[:, j] += k[t, :] * delta                 // Phase 4: State update
    ///     out[t, j] = dot(S[:, j], q[t, :])          // Phase 5: Output readout
    private static let metalSource = """
    // Grid: (DV, B*NH, 1)
    // Thread (j, bh) owns state column S[:, j] for batch-head pair bh
    uint j  = thread_position_in_grid.x;
    uint bh = thread_position_in_grid.y;
    uint b  = bh / NH;
    uint h  = bh % NH;

    // Load state column S[:, j] into thread-local registers
    T S_col[DK];
    uint s_base = bh * DK * DV;
    for (int d = 0; d < DK; d++) {
        S_col[d] = s_in[s_base + d * DV + j];
    }

    // Sequential recurrence over all timesteps
    for (int t = 0; t < SEQ; t++) {
        uint qk_off = ((b * SEQ + t) * NH + h) * DK;
        uint v_off  = ((b * SEQ + t) * NH + h) * DV;
        uint g_off  = (b * SEQ + t) * NH + h;

        T alpha = decay_arr[g_off];
        T bt    = beta_arr[g_off];

        // Phase 1: Decay state
        for (int d = 0; d < DK; d++) {
            S_col[d] *= alpha;
        }

        // Phase 2: Memory readout — kvMem = S[:, j] · k[t, :]
        T kvMem = 0;
        for (int d = 0; d < DK; d++) {
            kvMem += S_col[d] * k[qk_off + d];
        }

        // Phase 3: Delta rule
        T delta = bt * (v[v_off + j] - kvMem);

        // Phase 4: State update — S[d, j] += k[d] * delta
        for (int d = 0; d < DK; d++) {
            S_col[d] += k[qk_off + d] * delta;
        }

        // Phase 5: Output readout — out[j] = S[:, j] · q[t, :]
        T o_val = 0;
        for (int d = 0; d < DK; d++) {
            o_val += S_col[d] * q[qk_off + d];
        }

        out[v_off + j] = o_val;
    }

    // Write final state back to device memory
    for (int d = 0; d < DK; d++) {
        s_out[s_base + d * DV + j] = S_col[d];
    }
    """
}
