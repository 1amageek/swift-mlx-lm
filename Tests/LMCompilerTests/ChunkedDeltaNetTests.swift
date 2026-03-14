import Testing
@preconcurrency import MLX
@testable import LMCompiler

// MARK: - Reference: token-by-token recurrence

/// Token-by-token DeltaNet recurrence (known-correct for short sequences).
/// Used as ground truth to verify the chunked algorithm.
private func tokenByTokenDeltaNetRecurrence(
    query: MLXArray, key: MLXArray, value: MLXArray,
    gateLog: MLXArray, beta: MLXArray, state: MLXArray,
    dtype: DType
) -> (output: MLXArray, newState: MLXArray) {
    let B = query.dim(0), T = query.dim(1), H = query.dim(2)
    let dk = query.dim(3), dv = value.dim(3)

    var S = state
    var outputs = [MLXArray]()
    outputs.reserveCapacity(T)

    for t in 0..<T {
        let qt = query[0..., t..<(t+1), 0..., 0...].squeezed(axis: 1)
        let kt = key[0..., t..<(t+1), 0..., 0...].squeezed(axis: 1)
        let vt = value[0..., t..<(t+1), 0..., 0...].squeezed(axis: 1)
        let gt = gateLog[0..., t..<(t+1), 0...].squeezed(axis: 1)
        let bt = beta[0..., t..<(t+1), 0...].squeezed(axis: 1)

        // Decay state: S = exp(g) * S
        let decay = MLX.exp(gt)  // [B, H]
        let gE = decay.expandedDimensions(axis: -1).expandedDimensions(axis: -1)
        S = S * gE

        // Memory readout: S^T @ k
        let kE = kt.expandedDimensions(axis: -1)  // [B, H, dk, 1]
        let kvMem = (S * kE).sum(axis: -2)  // [B, H, dv]

        // Delta update: S += k ⊗ beta * (v - S^T @ k)
        let delta = bt.expandedDimensions(axis: -1) * (vt - kvMem)
        S = S + kE * delta.expandedDimensions(axis: -2)

        // Output: o = S^T @ q
        let qE = qt.expandedDimensions(axis: -1)
        let ot = (S * qE).sum(axis: -2)
        outputs.append(ot.expandedDimensions(axis: 1))
    }

    return (concatenated(outputs, axis: 1).reshaped(B, T, H, dv), S)
}

/// L2-normalize along last dimension.
private func l2Norm(_ x: MLXArray, eps: Float = 1e-6) -> MLXArray {
    x / MLX.sqrt((x * x).sum(axis: -1, keepDims: true) + MLXArray(eps))
}

// MARK: - Tests

@Suite("Chunked DeltaNet Recurrence", .tags(.unit))
struct ChunkedDeltaNetTests {

    @Test("chunked matches token-by-token for T=4, single chunk")
    func singleChunkMatchesTokenByToken() {
        let B = 1, T = 4, H = 2, dk = 8, dv = 8

        MLX.GPU.set(cacheLimit: 0)

        let query = l2Norm(MLXRandom.normal([B, T, H, dk]).asType(.float32))
        let key = l2Norm(MLXRandom.normal([B, T, H, dk]).asType(.float32))
        let value = MLXRandom.normal([B, T, H, dv]).asType(.float32) * 0.1
        let gateLog = -MLX.abs(MLXRandom.normal([B, T, H]).asType(.float32))  // negative
        let beta = sigmoid(MLXRandom.normal([B, T, H]).asType(.float32))
        let state = MLXArray.zeros([B, H, dk, dv], dtype: .float32)

        let (refOut, refState) = tokenByTokenDeltaNetRecurrence(
            query: query, key: key, value: value,
            gateLog: gateLog, beta: beta, state: state, dtype: .float32
        )

        let (chunkOut, chunkState) = chunkedGatedDeltaNetRecurrence(
            query: query, key: key, value: value,
            gateLog: gateLog, beta: beta, state: state,
            chunkSize: 64, dtype: .float32
        )

        eval(refOut, refState, chunkOut, chunkState)

        // Check no NaN
        #expect(!MLX.isNaN(chunkOut).any().item(Bool.self), "chunked output has NaN")
        #expect(!MLX.isNaN(chunkState).any().item(Bool.self), "chunked state has NaN")

        // Check numerical agreement
        let outDiff = MLX.abs(refOut - chunkOut).max().item(Float.self)
        let stateDiff = MLX.abs(refState - chunkState).max().item(Float.self)
        #expect(outDiff < 1e-4, "output max diff \(outDiff) exceeds tolerance")
        #expect(stateDiff < 1e-4, "state max diff \(stateDiff) exceeds tolerance")
    }

    @Test("chunked matches token-by-token for T=128, multiple chunks")
    func multiChunkMatchesTokenByToken() {
        let B = 1, T = 128, H = 2, dk = 8, dv = 8

        MLX.GPU.set(cacheLimit: 0)

        let query = l2Norm(MLXRandom.normal([B, T, H, dk]).asType(.float32))
        let key = l2Norm(MLXRandom.normal([B, T, H, dk]).asType(.float32))
        let value = MLXRandom.normal([B, T, H, dv]).asType(.float32) * 0.1
        let gateLog = -MLX.abs(MLXRandom.normal([B, T, H]).asType(.float32))
        let beta = sigmoid(MLXRandom.normal([B, T, H]).asType(.float32))
        let state = MLXArray.zeros([B, H, dk, dv], dtype: .float32)

        let (refOut, refState) = tokenByTokenDeltaNetRecurrence(
            query: query, key: key, value: value,
            gateLog: gateLog, beta: beta, state: state, dtype: .float32
        )

        let (chunkOut, chunkState) = chunkedGatedDeltaNetRecurrence(
            query: query, key: key, value: value,
            gateLog: gateLog, beta: beta, state: state,
            chunkSize: 64, dtype: .float32
        )

        eval(refOut, refState, chunkOut, chunkState)

        #expect(!MLX.isNaN(chunkOut).any().item(Bool.self), "chunked output has NaN")
        #expect(!MLX.isNaN(chunkState).any().item(Bool.self), "chunked state has NaN")

        let outDiff = MLX.abs(refOut - chunkOut).max().item(Float.self)
        let stateDiff = MLX.abs(refState - chunkState).max().item(Float.self)
        #expect(outDiff < 1e-3, "output max diff \(outDiff) exceeds tolerance")
        #expect(stateDiff < 1e-3, "state max diff \(stateDiff) exceeds tolerance")
    }

    @Test("chunked produces no NaN for T=64 exact chunk boundary")
    func exactChunkBoundaryNoNaN() {
        let B = 1, T = 64, H = 4, dk = 16, dv = 16

        MLX.GPU.set(cacheLimit: 0)

        let query = l2Norm(MLXRandom.normal([B, T, H, dk]).asType(.float32))
        let key = l2Norm(MLXRandom.normal([B, T, H, dk]).asType(.float32))
        let value = MLXRandom.normal([B, T, H, dv]).asType(.float32) * 0.1
        let gateLog = -MLX.abs(MLXRandom.normal([B, T, H]).asType(.float32))
        let beta = sigmoid(MLXRandom.normal([B, T, H]).asType(.float32))
        let state = MLXArray.zeros([B, H, dk, dv], dtype: .float32)

        let (chunkOut, chunkState) = chunkedGatedDeltaNetRecurrence(
            query: query, key: key, value: value,
            gateLog: gateLog, beta: beta, state: state,
            chunkSize: 64, dtype: .float32
        )

        eval(chunkOut, chunkState)

        #expect(!MLX.isNaN(chunkOut).any().item(Bool.self), "chunked output has NaN")
        #expect(!MLX.isNaN(chunkState).any().item(Bool.self), "chunked state has NaN")
    }

    @Test("chunked handles non-zero initial state")
    func nonZeroInitialState() {
        let B = 1, T = 16, H = 2, dk = 8, dv = 8

        MLX.GPU.set(cacheLimit: 0)

        let query = l2Norm(MLXRandom.normal([B, T, H, dk]).asType(.float32))
        let key = l2Norm(MLXRandom.normal([B, T, H, dk]).asType(.float32))
        let value = MLXRandom.normal([B, T, H, dv]).asType(.float32) * 0.1
        let gateLog = -MLX.abs(MLXRandom.normal([B, T, H]).asType(.float32))
        let beta = sigmoid(MLXRandom.normal([B, T, H]).asType(.float32))
        let state = MLXRandom.normal([B, H, dk, dv]).asType(.float32) * 0.01

        let (refOut, refState) = tokenByTokenDeltaNetRecurrence(
            query: query, key: key, value: value,
            gateLog: gateLog, beta: beta, state: state, dtype: .float32
        )

        let (chunkOut, chunkState) = chunkedGatedDeltaNetRecurrence(
            query: query, key: key, value: value,
            gateLog: gateLog, beta: beta, state: state,
            chunkSize: 64, dtype: .float32
        )

        eval(refOut, refState, chunkOut, chunkState)

        #expect(!MLX.isNaN(chunkOut).any().item(Bool.self), "chunked output has NaN")

        let outDiff = MLX.abs(refOut - chunkOut).max().item(Float.self)
        let stateDiff = MLX.abs(refState - chunkState).max().item(Float.self)
        #expect(outDiff < 1e-4, "output max diff \(outDiff) exceeds tolerance")
        #expect(stateDiff < 1e-4, "state max diff \(stateDiff) exceeds tolerance")
    }

    @Test("chunked handles realistic head dimensions (dk=128, dv=128)")
    func realisticDimensions() {
        let B = 1, T = 128, H = 16, dk = 128, dv = 128

        MLX.GPU.set(cacheLimit: 0)

        let query = l2Norm(MLXRandom.normal([B, T, H, dk]).asType(.float32))
        let key = l2Norm(MLXRandom.normal([B, T, H, dk]).asType(.float32))
        let value = MLXRandom.normal([B, T, H, dv]).asType(.float32) * 0.1
        let gateLog = -MLX.abs(MLXRandom.normal([B, T, H]).asType(.float32))
        let beta = sigmoid(MLXRandom.normal([B, T, H]).asType(.float32))
        let state = MLXArray.zeros([B, H, dk, dv], dtype: .float32)

        let (chunkOut, chunkState) = chunkedGatedDeltaNetRecurrence(
            query: query, key: key, value: value,
            gateLog: gateLog, beta: beta, state: state,
            chunkSize: 64, dtype: .float32
        )

        eval(chunkOut, chunkState)

        #expect(!MLX.isNaN(chunkOut).any().item(Bool.self), "chunked output has NaN")
        #expect(!MLX.isNaN(chunkState).any().item(Bool.self), "chunked state has NaN")

        // Also verify shapes
        #expect(chunkOut.shape == [B, T, H, dv])
        #expect(chunkState.shape == [B, H, dk, dv])
    }
}
