import Testing
import TestHeartbeat
@preconcurrency import MLX
import MLXFast
import MLXNN
import Metal
import MetalPerformanceShadersGraph
@testable import SwiftLM
@testable import MLXCompiler

/// Diagnostic tests to isolate the MPSGraph MLIR crash.
/// Progressively builds larger graphs to find the exact failure point.
private let dev = MTLCreateSystemDefaultDevice()!
private let q = dev.makeCommandQueue()!

private func tryRun(
    _ graph: MPSGraph, feeds: [MPSGraphTensor: MPSGraphTensorData],
    target: MPSGraphTensor
) -> Bool {
    let results = graph.run(with: q, feeds: feeds, targetTensors: [target], targetOperations: nil)
    let r = results[target]!
    let count = r.shape.reduce(1) { $0 * $1.intValue }
    var buf = [Float16](repeating: 0, count: count)
    r.mpsndarray().readBytes(&buf, strideBytes: nil)
    return buf.allSatisfy { $0.isFinite }
}

private func f16Var(_ g: MPSGraph, _ data: [Float16], _ shape: [Int], _ name: String) -> MPSGraphTensor {
    g.variable(with: data.withUnsafeBytes { Data($0) },
               shape: shape.map { $0 as NSNumber }, dataType: .float16, name: name)
}

private func randomF16(_ count: Int) -> [Float16] {
    (0..<count).map { _ in Float16(Float.random(in: -0.02...0.02)) }
}

private func tokenInput(_ g: MPSGraph, _ T: Int) -> (MPSGraphTensor, MPSGraphTensorData) {
    let ph = g.placeholder(shape: [1, T as NSNumber], dataType: .int32, name: "tokens")
    let data = (0..<T).map { Int32($0 % 4) }
    let td = data.withUnsafeBytes { ptr in
        MPSGraphTensorData(device: MPSGraphDevice(mtlDevice: dev), data: Data(ptr),
                            shape: [1, T as NSNumber], dataType: .int32)
    }
    return (ph, td)
}

@Suite("MPSGraphMLIRDiagnostic", .serialized, .heartbeat)
struct MPSGraphMLIRDiagnosticTests {

    // Level 1: Just embedding
    @Test("L1: embedding only")
    func embeddingOnly() {
        let g = MPSGraph()
        let (tokPh, tokData) = tokenInput(g, 1)
        let emb = f16Var(g, randomF16(4 * 8), [8, 4], "emb") // vocab=8, D=4
        let h = g.gather(withUpdatesTensor: emb, indicesTensor: tokPh, axis: 0, batchDimensions: 0, name: "embed")
        #expect(tryRun(g, feeds: [tokPh: tokData], target: h))
    }

    // Level 2: Embedding + RMSNorm
    @Test("L2: embedding + rmsNorm")
    func embeddingNorm() {
        let g = MPSGraph()
        let (tokPh, tokData) = tokenInput(g, 1)
        let emb = f16Var(g, randomF16(4 * 8), [8, 4], "emb")
        var h = g.gather(withUpdatesTensor: emb, indicesTensor: tokPh, axis: 0, batchDimensions: 0, name: "embed")
        let nw = f16Var(g, [Float16](repeating: 1.0, count: 4), [4], "nw")
        h = MPSGraphOps.rmsNorm(g, input: h, weight: nw, epsilon: 1e-5, name: "norm")
        #expect(tryRun(g, feeds: [tokPh: tokData], target: h))
    }

    // Level 3: Embedding + linear projection
    @Test("L3: embedding + linear")
    func embeddingLinear() {
        let g = MPSGraph()
        let (tokPh, tokData) = tokenInput(g, 1)
        let emb = f16Var(g, randomF16(4 * 8), [8, 4], "emb")
        var h = g.gather(withUpdatesTensor: emb, indicesTensor: tokPh, axis: 0, batchDimensions: 0, name: "embed")
        let w = f16Var(g, randomF16(4 * 4), [4, 4], "w")
        h = MPSGraphOps.linear(g, input: h, weight: w, name: "proj")
        #expect(tryRun(g, feeds: [tokPh: tokData], target: h))
    }

    // Level 4: Embedding + norm + QKV projections
    @Test("L4: embedding + norm + QKV linear")
    func embeddingNormQKV() {
        let g = MPSGraph()
        let D = 4; let H = 2; let hd = 2
        let (tokPh, tokData) = tokenInput(g, 1)
        let emb = f16Var(g, randomF16(D * 8), [8, D], "emb")
        var h = g.gather(withUpdatesTensor: emb, indicesTensor: tokPh, axis: 0, batchDimensions: 0, name: "embed")
        let nw = f16Var(g, [Float16](repeating: 1.0, count: D), [D], "nw")
        h = MPSGraphOps.rmsNorm(g, input: h, weight: nw, epsilon: 1e-5, name: "norm")
        let qw = f16Var(g, randomF16(H*hd * D), [H*hd, D], "qw")
        let kw = f16Var(g, randomF16(H*hd * D), [H*hd, D], "kw")
        let vw = f16Var(g, randomF16(H*hd * D), [H*hd, D], "vw")
        let q = MPSGraphOps.linear(g, input: h, weight: qw, name: "q")
        let k = MPSGraphOps.linear(g, input: h, weight: kw, name: "k")
        let v = MPSGraphOps.linear(g, input: h, weight: vw, name: "v")
        // Just sum to get a single output
        let sum = g.addition(g.addition(q, k, name: nil), v, name: "sum")
        #expect(tryRun(g, feeds: [tokPh: tokData], target: sum))
    }

    // Level 5: QKV + reshape to heads + SDPA (no mask)
    @Test("L5: QKV + heads reshape + SDPA no mask")
    func qkvHeadsSDPA() {
        let g = MPSGraph()
        let D = 4; let H = 2; let hd = 2
        let (tokPh, tokData) = tokenInput(g, 1)
        let emb = f16Var(g, randomF16(D * 8), [8, D], "emb")
        let h = g.gather(withUpdatesTensor: emb, indicesTensor: tokPh, axis: 0, batchDimensions: 0, name: "embed")
        let qw = f16Var(g, randomF16(H*hd * D), [H*hd, D], "qw")
        let kw = f16Var(g, randomF16(H*hd * D), [H*hd, D], "kw")
        let vw = f16Var(g, randomF16(H*hd * D), [H*hd, D], "vw")
        let q = MPSGraphOps.toHeads(g, input: MPSGraphOps.linear(g, input: h, weight: qw, name: "q"), heads: H, headDim: hd, name: "q")
        let k = MPSGraphOps.toHeads(g, input: MPSGraphOps.linear(g, input: h, weight: kw, name: "k"), heads: H, headDim: hd, name: "k")
        let v = MPSGraphOps.toHeads(g, input: MPSGraphOps.linear(g, input: h, weight: vw, name: "v"), heads: H, headDim: hd, name: "v")
        let attn = MPSGraphOps.scaledDotProductAttention(g, query: q, key: k, value: v, mask: nil, headDim: hd, name: "attn")
        let flat = MPSGraphOps.fromHeads(g, input: attn, totalDim: D, name: "out")
        #expect(tryRun(g, feeds: [tokPh: tokData], target: flat))
    }

    // Level 6: Attention + O projection + residual
    @Test("L6: full attention sublayer + residual")
    func fullAttentionResidual() {
        let g = MPSGraph()
        let D = 4; let H = 2; let hd = 2
        let (tokPh, tokData) = tokenInput(g, 1)
        let emb = f16Var(g, randomF16(D * 8), [8, D], "emb")
        let input = g.gather(withUpdatesTensor: emb, indicesTensor: tokPh, axis: 0, batchDimensions: 0, name: "embed")
        let nw = f16Var(g, [Float16](repeating: 1.0, count: D), [D], "nw")
        let normed = MPSGraphOps.rmsNorm(g, input: input, weight: nw, epsilon: 1e-5, name: "norm")
        let qw = f16Var(g, randomF16(H*hd * D), [H*hd, D], "qw")
        let kw = f16Var(g, randomF16(H*hd * D), [H*hd, D], "kw")
        let vw = f16Var(g, randomF16(H*hd * D), [H*hd, D], "vw")
        let ow = f16Var(g, randomF16(D * H*hd), [D, H*hd], "ow")
        let q = MPSGraphOps.toHeads(g, input: MPSGraphOps.linear(g, input: normed, weight: qw, name: "q"), heads: H, headDim: hd, name: "q")
        let k = MPSGraphOps.toHeads(g, input: MPSGraphOps.linear(g, input: normed, weight: kw, name: "k"), heads: H, headDim: hd, name: "k")
        let v = MPSGraphOps.toHeads(g, input: MPSGraphOps.linear(g, input: normed, weight: vw, name: "v"), heads: H, headDim: hd, name: "v")
        let attn = MPSGraphOps.scaledDotProductAttention(g, query: q, key: k, value: v, mask: nil, headDim: hd, name: "attn")
        let flat = MPSGraphOps.fromHeads(g, input: attn, totalDim: D, name: "out")
        let proj = MPSGraphOps.linear(g, input: flat, weight: ow, name: "oproj")
        let h = g.addition(input, proj, name: "res")
        #expect(tryRun(g, feeds: [tokPh: tokData], target: h))
    }

    // Level 7: Attention + MLP (1 full layer)
    @Test("L7: 1 full layer (attention + MLP)")
    func oneFullLayer() {
        let g = MPSGraph()
        let D = 4; let H = 2; let hd = 2; let I = 8
        let (tokPh, tokData) = tokenInput(g, 1)
        let emb = f16Var(g, randomF16(D * 8), [8, D], "emb")
        var h = g.gather(withUpdatesTensor: emb, indicesTensor: tokPh, axis: 0, batchDimensions: 0, name: "embed")

        // Attention sublayer
        let n1 = f16Var(g, [Float16](repeating: 1.0, count: D), [D], "n1")
        let normed = MPSGraphOps.rmsNorm(g, input: h, weight: n1, epsilon: 1e-5, name: "an")
        let qw = f16Var(g, randomF16(H*hd*D), [H*hd,D], "qw")
        let kw = f16Var(g, randomF16(H*hd*D), [H*hd,D], "kw")
        let vw = f16Var(g, randomF16(H*hd*D), [H*hd,D], "vw")
        let ow = f16Var(g, randomF16(D*H*hd), [D,H*hd], "ow")
        let q = MPSGraphOps.toHeads(g, input: MPSGraphOps.linear(g, input: normed, weight: qw, name: "q"), heads: H, headDim: hd, name: "q")
        let k = MPSGraphOps.toHeads(g, input: MPSGraphOps.linear(g, input: normed, weight: kw, name: "k"), heads: H, headDim: hd, name: "k")
        let v = MPSGraphOps.toHeads(g, input: MPSGraphOps.linear(g, input: normed, weight: vw, name: "v"), heads: H, headDim: hd, name: "v")
        let attn = MPSGraphOps.scaledDotProductAttention(g, query: q, key: k, value: v, mask: nil, headDim: hd, name: "attn")
        let flat = MPSGraphOps.fromHeads(g, input: attn, totalDim: D, name: "aflat")
        let proj = MPSGraphOps.linear(g, input: flat, weight: ow, name: "oproj")
        h = g.addition(h, proj, name: "ares")

        // MLP sublayer
        let n2 = f16Var(g, [Float16](repeating: 1.0, count: D), [D], "n2")
        let norm2 = MPSGraphOps.rmsNorm(g, input: h, weight: n2, epsilon: 1e-5, name: "mn")
        let gw = f16Var(g, randomF16(I*D), [I,D], "gw")
        let uw = f16Var(g, randomF16(I*D), [I,D], "uw")
        let dw = f16Var(g, randomF16(D*I), [D,I], "dw")
        let gate = MPSGraphOps.linear(g, input: norm2, weight: gw, name: "gate")
        let up = MPSGraphOps.linear(g, input: norm2, weight: uw, name: "up")
        let act = MPSGraphOps.siluGate(g, gate: gate, up: up, name: "act")
        let down = MPSGraphOps.linear(g, input: act, weight: dw, name: "down")
        h = g.addition(h, down, name: "mres")

        #expect(tryRun(g, feeds: [tokPh: tokData], target: h))
    }

    // Level 8: 1 full layer + final norm + lm_head
    @Test("L8: 1 layer + norm + lm_head")
    func oneLayerWithHead() {
        let g = MPSGraph()
        let D = 4; let H = 2; let hd = 2; let I = 8; let V = 8
        let (tokPh, tokData) = tokenInput(g, 1)
        let emb = f16Var(g, randomF16(D * V), [V, D], "emb")
        var h = g.gather(withUpdatesTensor: emb, indicesTensor: tokPh, axis: 0, batchDimensions: 0, name: "embed")

        // Attention
        let n1 = f16Var(g, [Float16](repeating: 1.0, count: D), [D], "n1")
        let normed = MPSGraphOps.rmsNorm(g, input: h, weight: n1, epsilon: 1e-5, name: "an")
        let qw = f16Var(g, randomF16(H*hd*D), [H*hd,D], "qw")
        let kw = f16Var(g, randomF16(H*hd*D), [H*hd,D], "kw")
        let vw = f16Var(g, randomF16(H*hd*D), [H*hd,D], "vw")
        let ow = f16Var(g, randomF16(D*H*hd), [D,H*hd], "ow")
        let q2 = MPSGraphOps.toHeads(g, input: MPSGraphOps.linear(g, input: normed, weight: qw, name: "q"), heads: H, headDim: hd, name: "q")
        let k2 = MPSGraphOps.toHeads(g, input: MPSGraphOps.linear(g, input: normed, weight: kw, name: "k"), heads: H, headDim: hd, name: "k")
        let v2 = MPSGraphOps.toHeads(g, input: MPSGraphOps.linear(g, input: normed, weight: vw, name: "v"), heads: H, headDim: hd, name: "v")
        let attn = MPSGraphOps.scaledDotProductAttention(g, query: q2, key: k2, value: v2, mask: nil, headDim: hd, name: "attn")
        let proj = MPSGraphOps.linear(g, input: MPSGraphOps.fromHeads(g, input: attn, totalDim: D, name: "af"), weight: ow, name: "op")
        h = g.addition(h, proj, name: "ar")

        // MLP
        let n2 = f16Var(g, [Float16](repeating: 1.0, count: D), [D], "n2")
        let norm2 = MPSGraphOps.rmsNorm(g, input: h, weight: n2, epsilon: 1e-5, name: "mn")
        let gw = f16Var(g, randomF16(I*D), [I,D], "gw")
        let uw = f16Var(g, randomF16(I*D), [I,D], "uw")
        let dw = f16Var(g, randomF16(D*I), [D,I], "dw")
        let act = MPSGraphOps.siluGate(g, gate: MPSGraphOps.linear(g, input: norm2, weight: gw, name: "gate"),
                                        up: MPSGraphOps.linear(g, input: norm2, weight: uw, name: "up"), name: "act")
        h = g.addition(h, MPSGraphOps.linear(g, input: act, weight: dw, name: "down"), name: "mr")

        // Final norm + LM head
        let fn = f16Var(g, [Float16](repeating: 1.0, count: D), [D], "fn")
        h = MPSGraphOps.rmsNorm(g, input: h, weight: fn, epsilon: 1e-5, name: "final")
        let logits = MPSGraphOps.linear(g, input: h, weight: emb, name: "lm_head")

        #expect(tryRun(g, feeds: [tokPh: tokData], target: logits))
    }

    // Level 9: Same as L8 but with causal mask placeholder
    @Test("L9: 1 layer + mask placeholder")
    func oneLayerWithMask() {
        let g = MPSGraph()
        let D = 4; let H = 2; let hd = 2; let I = 8; let V = 8
        let (tokPh, tokData) = tokenInput(g, 1)
        let maskPh = g.placeholder(shape: [1, 1, 1, 1], dataType: .float16, name: "mask")
        let maskData = [Float16](repeating: 0, count: 1).withUnsafeBytes { ptr in
            MPSGraphTensorData(device: MPSGraphDevice(mtlDevice: dev), data: Data(ptr),
                                shape: [1, 1, 1, 1], dataType: .float16)
        }

        let emb = f16Var(g, randomF16(D * V), [V, D], "emb")
        var h = g.gather(withUpdatesTensor: emb, indicesTensor: tokPh, axis: 0, batchDimensions: 0, name: "embed")

        let n1 = f16Var(g, [Float16](repeating: 1.0, count: D), [D], "n1")
        let normed = MPSGraphOps.rmsNorm(g, input: h, weight: n1, epsilon: 1e-5, name: "an")
        let qw = f16Var(g, randomF16(H*hd*D), [H*hd,D], "qw")
        let kw = f16Var(g, randomF16(H*hd*D), [H*hd,D], "kw")
        let vw = f16Var(g, randomF16(H*hd*D), [H*hd,D], "vw")
        let ow = f16Var(g, randomF16(D*H*hd), [D,H*hd], "ow")
        let q2 = MPSGraphOps.toHeads(g, input: MPSGraphOps.linear(g, input: normed, weight: qw, name: "q"), heads: H, headDim: hd, name: "q")
        let k2 = MPSGraphOps.toHeads(g, input: MPSGraphOps.linear(g, input: normed, weight: kw, name: "k"), heads: H, headDim: hd, name: "k")
        let v2 = MPSGraphOps.toHeads(g, input: MPSGraphOps.linear(g, input: normed, weight: vw, name: "v"), heads: H, headDim: hd, name: "v")
        // With mask
        let attn = MPSGraphOps.scaledDotProductAttention(g, query: q2, key: k2, value: v2, mask: maskPh, headDim: hd, name: "attn")
        let proj = MPSGraphOps.linear(g, input: MPSGraphOps.fromHeads(g, input: attn, totalDim: D, name: "af"), weight: ow, name: "op")
        h = g.addition(h, proj, name: "ar")

        let n2 = f16Var(g, [Float16](repeating: 1.0, count: D), [D], "n2")
        let norm2 = MPSGraphOps.rmsNorm(g, input: h, weight: n2, epsilon: 1e-5, name: "mn")
        let gw = f16Var(g, randomF16(I*D), [I,D], "gw")
        let uw = f16Var(g, randomF16(I*D), [I,D], "uw")
        let dw = f16Var(g, randomF16(D*I), [D,I], "dw")
        let act = MPSGraphOps.siluGate(g, gate: MPSGraphOps.linear(g, input: norm2, weight: gw, name: "gate"),
                                        up: MPSGraphOps.linear(g, input: norm2, weight: uw, name: "up"), name: "act")
        h = g.addition(h, MPSGraphOps.linear(g, input: act, weight: dw, name: "down"), name: "mr")

        let fn = f16Var(g, [Float16](repeating: 1.0, count: D), [D], "fn")
        h = MPSGraphOps.rmsNorm(g, input: h, weight: fn, epsilon: 1e-5, name: "final")
        let logits = MPSGraphOps.linear(g, input: h, weight: emb, name: "lm_head")

        #expect(tryRun(g, feeds: [tokPh: tokData, maskPh: maskData], target: logits))
    }

    // Level 10: Same as L9 but with RoPE cos/sin placeholders
    @Test("L10: 1 layer + mask + RoPE placeholders")
    func oneLayerWithMaskAndRoPE() {
        let g = MPSGraph()
        let D = 4; let H = 2; let hd = 2; let I = 8; let V = 8
        let (tokPh, tokData) = tokenInput(g, 1)
        let maskPh = g.placeholder(shape: [1, 1, 1, 1], dataType: .float16, name: "mask")
        let cosPh = g.placeholder(shape: [1, 1, 1, (hd/2) as NSNumber], dataType: .float16, name: "cos")
        let sinPh = g.placeholder(shape: [1, 1, 1, (hd/2) as NSNumber], dataType: .float16, name: "sin")

        let maskData = [Float16](repeating: 0, count: 1).withUnsafeBytes { ptr in
            MPSGraphTensorData(device: MPSGraphDevice(mtlDevice: dev), data: Data(ptr), shape: [1,1,1,1], dataType: .float16)
        }
        let (cosBytes, sinBytes) = MPSGraphOps.buildRoPETables(seqLen: 1, headDim: hd, theta: 10000)
        let cosData = MPSGraphTensorData(device: MPSGraphDevice(mtlDevice: dev), data: cosBytes, shape: [1,1,1,(hd/2) as NSNumber], dataType: .float16)
        let sinData = MPSGraphTensorData(device: MPSGraphDevice(mtlDevice: dev), data: sinBytes, shape: [1,1,1,(hd/2) as NSNumber], dataType: .float16)

        let emb = f16Var(g, randomF16(D * V), [V, D], "emb")
        var h = g.gather(withUpdatesTensor: emb, indicesTensor: tokPh, axis: 0, batchDimensions: 0, name: "embed")

        let n1 = f16Var(g, [Float16](repeating: 1.0, count: D), [D], "n1")
        let normed = MPSGraphOps.rmsNorm(g, input: h, weight: n1, epsilon: 1e-5, name: "an")
        let qw = f16Var(g, randomF16(H*hd*D), [H*hd,D], "qw")
        let kw = f16Var(g, randomF16(H*hd*D), [H*hd,D], "kw")
        let vw = f16Var(g, randomF16(H*hd*D), [H*hd,D], "vw")
        let ow = f16Var(g, randomF16(D*H*hd), [D,H*hd], "ow")
        var q2 = MPSGraphOps.toHeads(g, input: MPSGraphOps.linear(g, input: normed, weight: qw, name: "q"), heads: H, headDim: hd, name: "q")
        var k2 = MPSGraphOps.toHeads(g, input: MPSGraphOps.linear(g, input: normed, weight: kw, name: "k"), heads: H, headDim: hd, name: "k")
        let v2 = MPSGraphOps.toHeads(g, input: MPSGraphOps.linear(g, input: normed, weight: vw, name: "v"), heads: H, headDim: hd, name: "v")

        // RoPE
        q2 = MPSGraphOps.applyRoPE(g, input: q2, cosTable: cosPh, sinTable: sinPh, headDim: hd, name: "q")
        k2 = MPSGraphOps.applyRoPE(g, input: k2, cosTable: cosPh, sinTable: sinPh, headDim: hd, name: "k")

        let attn = MPSGraphOps.scaledDotProductAttention(g, query: q2, key: k2, value: v2, mask: maskPh, headDim: hd, name: "attn")
        let proj = MPSGraphOps.linear(g, input: MPSGraphOps.fromHeads(g, input: attn, totalDim: D, name: "af"), weight: ow, name: "op")
        h = g.addition(h, proj, name: "ar")

        let n2 = f16Var(g, [Float16](repeating: 1.0, count: D), [D], "n2")
        let norm2 = MPSGraphOps.rmsNorm(g, input: h, weight: n2, epsilon: 1e-5, name: "mn")
        let gw = f16Var(g, randomF16(I*D), [I,D], "gw")
        let uw = f16Var(g, randomF16(I*D), [I,D], "uw")
        let dw = f16Var(g, randomF16(D*I), [D,I], "dw")
        let act = MPSGraphOps.siluGate(g, gate: MPSGraphOps.linear(g, input: norm2, weight: gw, name: "gate"),
                                        up: MPSGraphOps.linear(g, input: norm2, weight: uw, name: "up"), name: "act")
        h = g.addition(h, MPSGraphOps.linear(g, input: act, weight: dw, name: "down"), name: "mr")

        let fn = f16Var(g, [Float16](repeating: 1.0, count: D), [D], "fn")
        h = MPSGraphOps.rmsNorm(g, input: h, weight: fn, epsilon: 1e-5, name: "final")
        let logits = MPSGraphOps.linear(g, input: h, weight: emb, name: "lm_head")

        #expect(tryRun(g, feeds: [tokPh: tokData, maskPh: maskData, cosPh: cosData, sinPh: sinData], target: logits))
    }
}
