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

        // Infer dimensions from weight shapes
        let totalQKV: Int
        switch inProjQKV.kernel {
        case .dense(let w): totalQKV = w.dim(0)
        case .affineQuantized(let q): totalQKV = q.logicalShape[0]
        case .dequantizeMatmul(let q): totalQKV = q.logicalShape[0]
        }
        let valueDim: Int
        switch inProjZ.kernel {
        case .dense(let w): valueDim = w.dim(0)
        case .affineQuantized(let q): valueDim = q.logicalShape[0]
        case .dequantizeMatmul(let q): valueDim = q.logicalShape[0]
        }

        let keyDim = (totalQKV - valueDim) / 2
        let linearKeyHeadDim = attrs.stateSize
        let linearKeyHeads = keyDim / linearKeyHeadDim
        let linearValueHeadDim = normWeight.dim(0)
        let linearValueHeads = valueDim / linearValueHeadDim
        let convDim = totalQKV
        let convKernelSize = convWeight.dim(1)
        let scale = 1.0 / Float(linearKeyHeadDim).squareRoot()

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
        let query = parts[0].reshaped(B, T, linearKeyHeads, linearKeyHeadDim)
        let key = parts[1].reshaped(B, T, linearKeyHeads, linearKeyHeadDim)
        let value = parts[2].reshaped(B, T, linearValueHeads, linearValueHeadDim)

        // Gates (computed in recurrence dtype to prevent exp/softplus overflow)
        let beta = sigmoid(b.asType(recurrenceDType))
        let g = -MLX.exp(aLog.asType(recurrenceDType)) * softplus(a.asType(recurrenceDType) + dtBias.asType(recurrenceDType))
        let decay = MLX.exp(g)

        // Delta rule recurrence (entirely in recurrence dtype)
        let (attnOut, newState) = deltaNetRecurrence(
            query: query, key: key, value: value,
            decay: decay, beta: beta, state: recurrentState,
            scale: scale
        )

        // Update cache (state stored in recurrence dtype to preserve precision across decode steps)
        caches[cacheSlotIndex] = .recurrent(LoweredRecurrentCache(
            convState: newConvState,
            recurrentState: newState,
            offset: cacheOffset + T
        ))

        // === Dtype boundary: recurrence output → activation dtype ===
        // Cast back to input dtype for gated output norm and output projection.

        // Gated output norm
        let dv = linearValueHeadDim
        let numHeads = linearValueHeads
        let flat = attnOut.asType(inputDType).reshaped(B * T * numHeads, dv)
        let zFlat = z.reshaped(B, T, numHeads, dv).reshaped(B * T * numHeads, dv)
        let normed = MLXFast.rmsNorm(flat, weight: 1 + normWeight, eps: 1e-6) * silu(zFlat)
        let gated = normed.reshaped(B, T, valueDim)

        return outProj.apply(gated)
    }

    // MARK: - Recurrence

    /// Per-token DeltaNet state update (runs entirely in `recurrenceDType`).
    ///
    /// S_t = exp(g) * S_{t-1} + k_t ⊗ [β(v_t − exp(g)·S^T·k_t)]
    /// o_t = S_t^T · (q_t / √d_k)
    private func deltaNetRecurrence(
        query: MLXArray, key: MLXArray, value: MLXArray,
        decay: MLXArray, beta: MLXArray, state: MLXArray?,
        scale: Float
    ) -> (MLXArray, MLXArray) {
        let B = query.dim(0), T = query.dim(1), H = query.dim(2)
        let dk = query.dim(3), dv = value.dim(3)

        let qN = l2Norm(query) * MLXArray(scale).asType(recurrenceDType)
        let kN = l2Norm(key)

        let qSlices = qN.split(parts: T, axis: 1)
        let kSlices = kN.split(parts: T, axis: 1)
        let vSlices = value.split(parts: T, axis: 1)
        let gSlices = decay.split(parts: T, axis: 1)
        let bSlices = beta.split(parts: T, axis: 1)

        var S = state ?? MLXArray.zeros([B, H, dk, dv], dtype: recurrenceDType)
        var outputs = [MLXArray]()
        outputs.reserveCapacity(T)

        for t in 0..<T {
            let qt = qSlices[t].squeezed(axis: 1)
            let kt = kSlices[t].squeezed(axis: 1)
            let vt = vSlices[t].squeezed(axis: 1)
            let gt = gSlices[t].squeezed(axis: 1)
            let bt = bSlices[t].squeezed(axis: 1)

            let gE = gt.expandedDimensions(axes: [-1, -2])
            S = S * gE

            let kE = kt.expandedDimensions(axis: -1)
            let kvMem = (S * kE).sum(axis: -2)

            let delta = bt.expandedDimensions(axis: -1) * (vt - kvMem)
            S = S + kE * delta.expandedDimensions(axis: -2)

            let qE = qt.expandedDimensions(axis: -1)
            let ot = (S * qE).sum(axis: -2)
            outputs.append(ot.expandedDimensions(axis: 1))

            // Periodically evaluate to prevent graph explosion during long prefills
            if T > 1 && (t + 1) % 64 == 0 {
                eval(S)
            }
        }

        return (concatenated(outputs, axis: 1), S)
    }

    // MARK: - Utility

    private func l2Norm(_ x: MLXArray, eps: Float = 1e-6) -> MLXArray {
        x / MLX.sqrt((x * x).sum(axis: -1, keepDims: true) + MLXArray(eps))
    }

    private func softplus(_ x: MLXArray) -> MLXArray {
        MLX.log(1 + MLX.exp(x))
    }
}
