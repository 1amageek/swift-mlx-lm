@preconcurrency import MLX

/// Chunked Gated DeltaNet recurrence using the WY factorization.
///
/// For prefill (T > 1), decomposes the sequence into chunks of `chunkSize` tokens.
/// Within each chunk, uses matrix operations (matmul) instead of sequential
/// token-by-token updates. This reduces numerical error accumulation from O(T)
/// sequential floating-point operations to O(T/C) inter-chunk state propagations.
///
/// Reference: flash-linear-attention (FLA) `naive_chunk_gated_delta_rule`.
///
/// ## WY factorization and (I+A)^{-1}
///
/// The adjacency matrix A is strictly lower triangular with **positive** entries:
///   `A[i,j] = beta[i] * (k[i] · k[j]) * L[i,j]` for `i > j`
///
/// where `L[i,j] = exp(gCumsum[i] - gCumsum[j])` is the inter-position decay.
///
/// `(I+A)` is unit lower triangular, so `triInv` (forward substitution, O(m²))
/// is exact and numerically stable. This replaces the Neumann series approach
/// which overflows float32 for chunk sizes ≥ 64.
///
/// The FLA reference computes `(I - A_neg)^{-1}` via Neumann expansion where
/// `A_neg` has negative entries. This is equivalent to `(I + A_pos)^{-1}` with
/// our positive A, since `A_pos = -A_neg`.
///
/// - Parameters:
///   - query: L2-normalized, scaled queries `[B, T, H, dk]`
///   - key: L2-normalized keys `[B, T, H, dk]`
///   - value: Values `[B, T, H, dv]`
///   - gateLog: Log-space gate values (g, negative) `[B, T, H]`
///   - beta: Sigmoid-gated learning rate `[B, T, H]`
///   - state: Previous recurrent state `[B, H, dk, dv]`
///   - chunkSize: Number of tokens per chunk (default 64, matching FLA/llama.cpp)
///   - dtype: Computation dtype for internal operations
/// - Returns: `(output: [B, T, H, dv], newState: [B, H, dk, dv])`
public func chunkedGatedDeltaNetRecurrence(
    query: MLXArray,
    key: MLXArray,
    value: MLXArray,
    gateLog: MLXArray,
    beta: MLXArray,
    state: MLXArray,
    chunkSize: Int = 64,
    dtype: DType
) -> (output: MLXArray, newState: MLXArray) {
    let T = query.dim(1)
    let C = chunkSize

    // Transpose to [B, H, T, D] for batched matmul
    let q = query.transposed(0, 2, 1, 3)
    let k = key.transposed(0, 2, 1, 3)
    let v = value.transposed(0, 2, 1, 3)
    let g = gateLog.transposed(0, 2, 1)
    let bt = beta.transposed(0, 2, 1)

    var S = state  // [B, H, dk, dv]
    let numChunks = (T + C - 1) / C
    var chunkOutputs = [MLXArray]()
    chunkOutputs.reserveCapacity(numChunks)

    for c in 0..<numChunks {
        let start = c * C
        let end = min(start + C, T)
        let m = end - start

        let bq = q[0..., 0..., start..<end, 0...]  // [B, H, m, dk]
        let bk = k[0..., 0..., start..<end, 0...]
        let bv = v[0..., 0..., start..<end, 0...]  // [B, H, m, dv]
        let bg = g[0..., 0..., start..<end]         // [B, H, m]
        let bb = bt[0..., 0..., start..<end]        // [B, H, m]

        // --- Step 1: Cumulative gating within chunk (inclusive) ---
        let gCumsum = bg.cumsum(axis: -1)  // [B, H, m]

        // --- Step 2: Decay mask L (lower triangular including diagonal) ---
        // L[i,j] = exp(gCumsum[i] - gCumsum[j]) for i >= j, 0 otherwise
        // Matches FLA: L_mask = ((decay[:, None] - decay[None, :]).tril().exp()).tril()
        let gDiff = gCumsum.expandedDimensions(axis: -1) - gCumsum.expandedDimensions(axis: -2)
        let triMask = tri(m, dtype: dtype)  // [m, m] lower triangular ones
        let L = MLX.exp(gDiff * triMask) * triMask

        // --- Step 3: WY adjacency matrix A (positive, strict lower tri) ---
        // A[i,j] = beta[i] * (k[i] · k[j]) * L[i,j] for i > j
        // FLA uses negative A then (I-A)^{-1}; we use positive A then (I+A)^{-1} — equivalent.
        let kkT = MLX.matmul(bk, bk.transposed(0, 1, 3, 2))  // [B, H, m, m]
        let strictTri = tri(m, k: -1, dtype: dtype)  // [m, m] strict lower tri
        let A = (bb.expandedDimensions(axis: -1) * kkT * L) * strictTri

        // --- Step 4: Compute (I+A)^{-1} via triangular inverse ---
        // triInv on CPU (GPU not yet supported for linalg::inv in mlx)
        let eye = MLXArray.eye(m, dtype: dtype)
        let inv = MLXLinalg.triInv(eye + A, stream: .cpu)  // [B, H, m, m]

        // --- Step 5: WY factors w and u ---
        // Matches FLA: k_cumdecay = attn @ (k_beta * decay_exp)
        //              k_cumsum   = attn @ v_beta
        let gCumsumExp = MLX.exp(gCumsum).expandedDimensions(axis: -1)  // [B, H, m, 1]
        let kBetaDecay = bk * bb.expandedDimensions(axis: -1) * gCumsumExp  // [B, H, m, dk]
        let vBeta = bv * bb.expandedDimensions(axis: -1)                     // [B, H, m, dv]
        let w = MLX.matmul(inv, kBetaDecay)  // [B, H, m, dk]
        let u = MLX.matmul(inv, vBeta)        // [B, H, m, dv]

        // --- Step 6: Inter-chunk corrected values ---
        // v_new = k_cumsum - k_cumdecay @ S_prev
        let vNew = u - MLX.matmul(w, S)  // [B, H, m, dv]

        // --- Step 7: Output ---
        // o_inter = (q * exp(gCumsum)) @ S_prev
        let qDecayed = bq * gCumsumExp
        let oInter = MLX.matmul(qDecayed, S)

        // o_intra = (q @ k^T * L) @ v_new  (L serves as causal decay mask)
        let qkT = MLX.matmul(bq, bk.transposed(0, 1, 3, 2))  // [B, H, m, m]
        let oIntra = MLX.matmul(qkT * L, vNew)

        chunkOutputs.append(oInter + oIntra)  // [B, H, m, dv]

        // --- Step 8: State update ---
        // S_new = S_prev * exp(total_gate) + k_decayed^T @ v_new
        let gLast = gCumsum[0..., 0..., (m - 1)..<m]  // [B, H, 1]
        let decayInter = MLX.exp(gLast - gCumsum).expandedDimensions(axis: -1)  // [B, H, m, 1]
        let kDecayed = bk * decayInter  // [B, H, m, dk]
        S = S * MLX.exp(gLast).expandedDimensions(axis: -1) + MLX.matmul(kDecayed.transposed(0, 1, 3, 2), vNew)

        // Materialize state between chunks to bound computation graph
        eval(S)
    }

    // Concatenate chunk outputs and transpose back to [B, T, H, dv]
    let output = concatenated(chunkOutputs, axis: 2)  // [B, H, T, dv]
    return (output.transposed(0, 2, 1, 3), S)
}
