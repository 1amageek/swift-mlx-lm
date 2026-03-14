@preconcurrency import MLX
import MLXFast
import MLXNN
import SwiftLM

/// Lowered DeltaNet state-space module with compile-time kernel and dtype selection.
///
/// Contains 5 lowered projections (inProjQKV, inProjZ, inProjB, inProjA, outProj)
/// plus raw parameter arrays (convWeight, normWeight, dtBias, aLog).
///
/// ## Compile-time dtype resolution
///
/// `StateSpaceAttributes.computeDType` declares the required computation
/// precision for the recurrence loop. The compiler resolves this at construction
/// time — no runtime dtype decisions are made.
///
/// Dtype boundaries:
/// - **Projection → Recurrence**: projection output (Float16 from quantizedMM)
///   is cast to `computeDType` before entering the recurrence.
/// - **Recurrence → Output projection**: recurrence output is cast back to
///   the activation dtype (Float16) before the output projection.
/// - **State cache**: stored in `computeDType` to avoid precision loss across
///   decode steps.
///
/// Reference: `MLXExecutor.executeDeltaNet()` in `MLXExecutor.swift:649-735`.
public struct LoweredDeltaNet: @unchecked Sendable {

    /// Input projections — each with compile-time resolved kernel.
    public let inProjQKV: LoweredProjection
    public let inProjZ: LoweredProjection
    public let inProjB: LoweredProjection
    public let inProjA: LoweredProjection
    public let outProj: LoweredProjection

    /// Raw (non-quantizable) parameters.
    public let convWeight: MLXArray
    public let normWeight: MLXArray
    public let dtBias: MLXArray
    public let aLog: MLXArray

    /// State-space attributes (stateSize, projectionSize, variant, computeDType).
    public let attrs: StateSpaceAttributes

    /// Compile-time resolved MLX dtype for the recurrence loop.
    /// Derived from `attrs.computeDType` at construction time.
    public let recurrenceDType: DType

    /// Compile-time resolved cache slot index.
    public let cacheSlotIndex: Int

    public init(
        inProjQKV: LoweredProjection,
        inProjZ: LoweredProjection,
        inProjB: LoweredProjection,
        inProjA: LoweredProjection,
        outProj: LoweredProjection,
        convWeight: MLXArray,
        normWeight: MLXArray,
        dtBias: MLXArray,
        aLog: MLXArray,
        attrs: StateSpaceAttributes,
        cacheSlotIndex: Int
    ) {
        self.inProjQKV = inProjQKV
        self.inProjZ = inProjZ
        self.inProjB = inProjB
        self.inProjA = inProjA
        self.outProj = outProj
        self.convWeight = convWeight
        self.normWeight = normWeight
        self.dtBias = dtBias
        self.aLog = aLog
        self.attrs = attrs
        self.cacheSlotIndex = cacheSlotIndex

        // Resolve IR compute dtype declaration to MLX DType at compile time
        switch attrs.computeDType {
        case .float32: self.recurrenceDType = .float32
        case .float16: self.recurrenceDType = .float16
        }
    }

    /// Apply DeltaNet with external cache state.
    public func apply(_ x: MLXArray, caches: inout [LoweredCacheState]) -> MLXArray {
        let inputDType = x.dtype
        let B = x.dim(0)
        let T = x.dim(1)

        // Dimensions from IR attributes (resolved from GGUF metadata at build time)
        let numHeads = attrs.numHeads
        let dk = attrs.keyHeadDim
        let dv = attrs.valueHeadDim
        let keyHeadCount = attrs.groupCount
        let keyDim = keyHeadCount * dk
        let valueDim = numHeads * dv
        let convDim = 2 * keyDim + valueDim
        let convKernelSize = convWeight.dim(1)

        // Projections through compile-time resolved kernels (output: Float16 from quantizedMM)
        let mixedQKV = inProjQKV.apply(x)
        let z = inProjZ.apply(x)
        let b = inProjB.apply(x)
        let a = inProjA.apply(x)

        // Extract recurrent cache
        var convState: MLXArray?
        var recurrentState: MLXArray?
        var cacheOffset: Int = 0
        switch caches[cacheSlotIndex] {
        case .recurrent(let cache):
            convState = cache.convState
            recurrentState = cache.recurrentState
            cacheOffset = cache.offset
        default:
            break
        }

        // Causal Conv1D (feedforward — stays in projection dtype)
        let prefix: MLXArray
        if let existing = convState {
            prefix = existing.asType(mixedQKV.dtype)
        } else {
            prefix = MLXArray.zeros([B, convKernelSize, convDim])
        }
        let convInput = concatenated([prefix, mixedQKV], axis: 1)
        let newConvState = convInput[0..., (convInput.dim(1) - convKernelSize)..., 0...]

        // Depthwise conv1d — convWeight is guaranteed to be 3D after sanitizeCompiledWeights()
        let rawConv = conv1d(convInput, convWeight, stride: 1, padding: 0, groups: convDim)
        let activated = silu(rawConv[0..., 1..., 0...])

        // === Dtype boundary: projection output → recurrence dtype ===
        // Cast activations to the IR-declared compute precision before recurrence.
        // This is resolved at compile time from StateSpaceAttributes.computeDType.

        // Split Q, K, V and cast to recurrence dtype
        let parts = activated.asType(recurrenceDType).split(indices: [keyDim, 2 * keyDim], axis: -1)

        // Gates (computed in recurrence dtype to prevent exp/softplus overflow)
        let betaRaw = sigmoid(b.asType(recurrenceDType))
        let g = -MLX.exp(aLog.asType(recurrenceDType)) * softplus(a.asType(recurrenceDType) + dtBias.asType(recurrenceDType))
        let decayRaw = MLX.exp(g)

        // Prepare Q, K, V with head repetition for asymmetric models
        let repeatFactor = numHeads / keyHeadCount
        let scale = 1.0 / Float(dk).squareRoot()
        let query: MLXArray
        let key: MLXArray
        if repeatFactor > 1 {
            // GGUF stores V heads in tiled order (convert_hf_to_gguf.py _reorder_v_heads).
            // Must tile Q/K to match: [K0,...,K15] → [K0,...,K15, K0,...,K15].
            // repeated() interleaves [K0,K0,...] which mismatches the tiled V layout.
            query = tiled(parts[0].reshaped(B, T, keyHeadCount, dk), repetitions: [1, 1, repeatFactor, 1])
            key = tiled(parts[1].reshaped(B, T, keyHeadCount, dk), repetitions: [1, 1, repeatFactor, 1])
        } else {
            query = parts[0].reshaped(B, T, numHeads, dk)
            key = parts[1].reshaped(B, T, numHeads, dk)
        }
        let value = parts[2].reshaped(B, T, numHeads, dv)
        let betaGate = betaRaw.reshaped(B, T, numHeads)
        let S = recurrentState ?? MLXArray.zeros(
            [B, numHeads, dk, dv],
            dtype: recurrenceDType
        )

        let attnOut: MLXArray
        let newState: MLXArray

        if T == 1 {
            // Single-token decode: use fast token-by-token recurrence
            if keyHeadCount == numHeads {
                // Symmetric: use fused Metal kernel for single token
                let qN = l2Norm(query) * MLXArray(scale).asType(recurrenceDType)
                let kN = l2Norm(key)
                (attnOut, newState) = DeltaNetRecurrenceKernel.call(
                    query: qN, key: kN, value: value,
                    decay: decayRaw, beta: betaRaw, state: S,
                    dtype: recurrenceDType
                )
            } else {
                (attnOut, newState) = singleTokenRecurrence(
                    query: query, key: key, value: value,
                    decay: decayRaw.reshaped(B, T, numHeads),
                    beta: betaGate, state: S, scale: scale
                )
            }
        } else {
            // Multi-token prefill: use chunked WY algorithm for numerical stability
            let qN = l2Norm(query) * MLXArray(scale).asType(recurrenceDType)
            let kN = l2Norm(key)
            (attnOut, newState) = chunkedGatedDeltaNetRecurrence(
                query: qN, key: kN, value: value,
                gateLog: g, beta: betaGate, state: S,
                dtype: recurrenceDType
            )
        }

        // Update cache (state stored in recurrence dtype to preserve precision across decode steps)
        caches[cacheSlotIndex] = .recurrent(LoweredRecurrentCache(
            convState: newConvState,
            recurrentState: newState,
            offset: cacheOffset + T
        ))

        // === Dtype boundary: recurrence output → activation dtype ===
        // Cast back to input dtype for gated output norm and output projection.

        // Gated output norm
        let flat = attnOut.asType(inputDType).reshaped(B * T * numHeads, dv)
        let zFlat = z.reshaped(B, T, numHeads, dv).reshaped(B * T * numHeads, dv)
        let normed = MLXFast.rmsNorm(flat, weight: normWeight, eps: 1e-6) * silu(zFlat)
        let gated = normed.reshaped(B, T, valueDim)

        return outProj.apply(gated)
    }

    // MARK: - Token-by-token recurrence (decode, T == 1)

    /// Single-token DeltaNet recurrence for autoregressive decode.
    private func singleTokenRecurrence(
        query: MLXArray,
        key: MLXArray,
        value: MLXArray,
        decay: MLXArray,
        beta: MLXArray,
        state: MLXArray,
        scale: Float
    ) -> (MLXArray, MLXArray) {
        let B = query.dim(0)
        let H = query.dim(2)
        let dv = value.dim(3)

        let qN = l2Norm(query) * MLXArray(scale).asType(recurrenceDType)
        let kN = l2Norm(key)

        let qt = qN.squeezed(axis: 1)  // [B, H, dk]
        let kt = kN.squeezed(axis: 1)
        let vt = value.squeezed(axis: 1)
        let gt = decay.squeezed(axis: 1)  // [B, H]
        let bt = beta.squeezed(axis: 1)

        let gE = gt.expandedDimensions(axis: -1).expandedDimensions(axis: -1)
        var S = state * gE

        let kE = kt.expandedDimensions(axis: -1)
        let kvMem = (S * kE).sum(axis: -2)
        let delta = bt.expandedDimensions(axis: -1) * (vt - kvMem)
        S = S + kE * delta.expandedDimensions(axis: -2)

        let qE = qt.expandedDimensions(axis: -1)
        let ot = (S * qE).sum(axis: -2)

        return (ot.expandedDimensions(axis: 1).reshaped(B, 1, H, dv), S)
    }

    // MARK: - Utility

    private func l2Norm(_ x: MLXArray, eps: Float = 1e-6) -> MLXArray {
        x / MLX.sqrt((x * x).sum(axis: -1, keepDims: true) + MLXArray(eps))
    }

    private func softplus(_ x: MLXArray) -> MLXArray {
        let zero = MLXArray(0.0, dtype: x.dtype)
        let maxPart = maximum(x, zero)
        return maxPart + log1p(exp(-abs(x)))
    }
}
