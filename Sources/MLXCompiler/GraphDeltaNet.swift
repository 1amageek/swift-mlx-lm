@preconcurrency import MLX
import MLXFast
import MLXNN
import SwiftLM

/// DeltaNet state-space module compiled from ModelGraph.
///
/// Linear projections (in_proj_qkv, in_proj_z, in_proj_b, in_proj_a, out_proj)
/// use `@ModuleInfo`-annotated `Linear` layers for quantization support.
/// Conv1d weight and other scalar parameters remain as raw `MLXArray`.
///
/// Computation precision is resolved from `StateSpaceAttributes.computeDType`.
final class GraphDeltaNet: Module, UnaryLayer {

    @ModuleInfo(key: "in_proj_qkv") var inProjQKV: MLXNN.Linear
    @ModuleInfo(key: "in_proj_z") var inProjZ: MLXNN.Linear
    @ModuleInfo(key: "in_proj_b") var inProjB: MLXNN.Linear
    @ModuleInfo(key: "in_proj_a") var inProjA: MLXNN.Linear
    @ModuleInfo(key: "out_proj") var outProj: MLXNN.Linear

    // Non-linear parameters (raw arrays)
    let convWeight: MLXArray
    let normWeight: MLXArray
    let dtBias: MLXArray
    let aLog: MLXArray

    let attrs: StateSpaceAttributes
    let recurrenceDType: DType
    var cache: MLXRecurrentCache

    init(
        attrs: StateSpaceAttributes,
        store: MLXWeightStore,
        path: StructuralPath,
        cache: MLXRecurrentCache
    ) throws {
        self.attrs = attrs
        switch attrs.computeDType {
        case .float32: self.recurrenceDType = .float32
        case .float16: self.recurrenceDType = .float16
        }
        self.cache = cache

        // Linear projections
        self._inProjQKV.wrappedValue = MLXNN.Linear(
            weight: try store.require(ParameterSlot(path: path.appending(.field("in_proj_qkv")), role: .weight)),
            bias: store.get(ParameterSlot(path: path.appending(.field("in_proj_qkv")), role: .bias))
        )
        self._inProjZ.wrappedValue = MLXNN.Linear(
            weight: try store.require(ParameterSlot(path: path.appending(.field("in_proj_z")), role: .weight)),
            bias: store.get(ParameterSlot(path: path.appending(.field("in_proj_z")), role: .bias))
        )
        self._inProjB.wrappedValue = MLXNN.Linear(
            weight: try store.require(ParameterSlot(path: path.appending(.field("in_proj_b")), role: .weight)),
            bias: store.get(ParameterSlot(path: path.appending(.field("in_proj_b")), role: .bias))
        )
        self._inProjA.wrappedValue = MLXNN.Linear(
            weight: try store.require(ParameterSlot(path: path.appending(.field("in_proj_a")), role: .weight)),
            bias: store.get(ParameterSlot(path: path.appending(.field("in_proj_a")), role: .bias))
        )
        self._outProj.wrappedValue = MLXNN.Linear(
            weight: try store.require(ParameterSlot(path: path.appending(.field("out_proj")), role: .weight)),
            bias: store.get(ParameterSlot(path: path.appending(.field("out_proj")), role: .bias))
        )

        // Raw parameters
        self.convWeight = try store.require(ParameterSlot(path: path.appending(.field("conv1d")), role: .weight))
        self.normWeight = try store.require(ParameterSlot(path: path.appending(.field("norm")), role: .scale))
        self.dtBias = try store.require(ParameterSlot(path: path.appending(.field("dt_bias")), role: .bias))
        self.aLog = try store.require(ParameterSlot(path: path.appending(.field("A_log")), role: .weight))
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let inputDType = x.dtype
        let B = x.dim(0)
        let T = x.dim(1)

        // Infer dimensions from weight shapes
        let qkvWeight = inProjQKV.weight
        let totalQKV = qkvWeight.dim(0)
        let valueDim = inProjZ.weight.dim(0)
        let keyDim = (totalQKV - valueDim) / 2
        let linearKeyHeadDim = attrs.stateSize
        let linearKeyHeads = keyDim / linearKeyHeadDim
        let linearValueHeadDim = normWeight.dim(0)
        let linearValueHeads = valueDim / linearValueHeadDim
        let convDim = totalQKV
        let convKernelSize = convWeight.dim(1)
        let scale = 1.0 / Float(linearKeyHeadDim).squareRoot()

        // Projections through Linear modules (quantizedMM-compatible)
        let mixedQKV = inProjQKV(x)
        let z = inProjZ(x)
        let b = inProjB(x)
        let a = inProjA(x)

        // Causal Conv1D (feedforward — stays in projection dtype)
        let prefix: MLXArray
        if let existing = cache.convState {
            prefix = existing.asType(mixedQKV.dtype)
        } else {
            prefix = MLXArray.zeros([B, convKernelSize, convDim])
        }
        let convInput = concatenated([prefix, mixedQKV], axis: 1)
        cache.convState = convInput[0..., (convInput.dim(1) - convKernelSize)..., 0...]

        // Depthwise conv1d
        let convWeight3d = convWeight.ndim == 2
            ? convWeight.expandedDimensions(axis: -1)
            : convWeight
        let rawConv = conv1d(convInput, convWeight3d, stride: 1, padding: 0, groups: convDim)
        let activated = silu(rawConv[0..., 1..., 0...])

        // === Dtype boundary: projection output → recurrence dtype ===
        let parts = activated.asType(recurrenceDType).split(indices: [keyDim, 2 * keyDim], axis: -1)
        let query = parts[0].reshaped(B, T, linearKeyHeads, linearKeyHeadDim)
        let key = parts[1].reshaped(B, T, linearKeyHeads, linearKeyHeadDim)
        let value = parts[2].reshaped(B, T, linearValueHeads, linearValueHeadDim)

        // Gates (computed in recurrence dtype)
        let beta = sigmoid(b.asType(recurrenceDType))
        let g = -MLX.exp(aLog.asType(recurrenceDType)) * softplus(a.asType(recurrenceDType) + dtBias.asType(recurrenceDType))
        let decay = MLX.exp(g)

        // Delta rule recurrence (entirely in recurrence dtype)
        let (attnOut, newState) = deltaNetRecurrence(
            query: query, key: key, value: value,
            decay: decay, beta: beta, state: cache.recurrentState,
            scale: scale
        )
        cache.recurrentState = newState
        cache.incrementOffset(by: T)

        // === Dtype boundary: recurrence output → activation dtype ===
        let dv = linearValueHeadDim
        let numHeads = linearValueHeads
        let flat = attnOut.asType(inputDType).reshaped(B * T * numHeads, dv)
        let zFlat = z.reshaped(B, T, numHeads, dv).reshaped(B * T * numHeads, dv)
        let normed = MLXFast.rmsNorm(flat, weight: normWeight, eps: 1e-6) * silu(zFlat)
        let gated = normed.reshaped(B, T, valueDim)

        return outProj(gated)
    }

    // MARK: - Recurrence

    /// Per-token DeltaNet state update.
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
