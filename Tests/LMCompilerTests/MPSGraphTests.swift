import Testing
import TestHeartbeat
@preconcurrency import MLX
import MLXFast
import MLXNN
import Metal
import MetalPerformanceShadersGraph
@testable import SwiftLM
@testable import LMCompiler

// MARK: - Shared Helpers

private func slot(
    _ c: [StructuralPathComponent], role: ParameterRole
) -> ParameterSlot {
    ParameterSlot(path: StructuralPath(components: c), role: role)
}

private func bind(_ pairs: [ParameterSlot: MLXArray]) -> BoundWeights {
    var tensors: [ParameterSlot: TensorData] = [:]
    for (s, a) in pairs {
        tensors[s] = TensorData(shape: a.shape.map { $0 }, dtype: .float16, storage: a)
    }
    return BoundWeights(tensors: tensors)
}

private let device = MTLCreateSystemDefaultDevice()!
private let queue = device.makeCommandQueue()!

private func makeGraph() -> MPSGraph { MPSGraph() }

private func runGraph(
    _ graph: MPSGraph, feeds: [MPSGraphTensor: MPSGraphTensorData],
    target: MPSGraphTensor
) -> [Float16] {
    let results = graph.run(with: queue, feeds: feeds, targetTensors: [target], targetOperations: nil)
    let r = results[target]!
    let count = r.shape.reduce(1) { $0 * $1.intValue }
    var buf = [Float16](repeating: 0, count: count)
    r.mpsndarray().readBytes(&buf, strideBytes: nil)
    return buf
}

private func mpsInput(_ data: [Float16], shape: [Int]) -> (MPSGraphTensor, MPSGraphTensorData, MPSGraph) {
    let g = makeGraph()
    let ph = g.placeholder(shape: shape.map { $0 as NSNumber }, dataType: .float16, name: "x")
    let td = data.withUnsafeBytes { ptr in
        MPSGraphTensorData(device: MPSGraphDevice(mtlDevice: device), data: Data(ptr),
                            shape: shape.map { $0 as NSNumber }, dataType: .float16)
    }
    return (ph, td, g)
}

private func mpsVar(_ g: MPSGraph, _ data: [Float16], _ shape: [Int], _ name: String) -> MPSGraphTensor {
    g.variable(with: data.withUnsafeBytes { Data($0) },
               shape: shape.map { $0 as NSNumber }, dataType: .float16, name: name)
}

// MARK: - Tiny Model

private struct TinyTransformer: ModelComponent {
    let L: Int; let V: Int; let D: Int; let H: Int; let KVH: Int; let hd: Int; let I: Int
    @ModelComponentBuilder var body: some ModelComponent {
        TokenEmbedding(vocabSize: V, embeddingSize: D)
        Repeat(count: L) {
            Residual { RMSNorm(dimension: D); Attention(hiddenSize: D, headCount: H, kvHeadCount: KVH, headDimension: hd) }
            Residual { RMSNorm(dimension: D); MLP(inputSize: D, intermediateSize: I) }
        }
        RMSNorm(dimension: D)
        OutputHead(inputSize: D, vocabSize: V, tiedToEmbedding: true)
    }
}

private func tinyWeights(L: Int, D: Int = 4, H: Int = 2, KVH: Int = 2, hd: Int = 2, I: Int = 8, V: Int = 8) -> BoundWeights {
    MLXRandom.seed(42)
    var dict: [ParameterSlot: MLXArray] = [:]
    dict[slot([.operation(0)], role: .embeddingTable)] = (MLXRandom.normal([V, D]) * 0.1).asType(.float16)
    for i in 0..<L {
        let lp: [StructuralPathComponent] = [.operation(1), .regionBody, .index(i)]
        dict[slot(lp + [.operation(0), .regionBody, .operation(0)], role: .scale)] = MLXArray.ones([D]).asType(.float16)
        let ap = lp + [.operation(0), .regionBody, .operation(1)]
        dict[slot(ap + [.field("q_proj")], role: .weight)] = (MLXRandom.normal([H*hd, D]) * 0.1).asType(.float16)
        dict[slot(ap + [.field("k_proj")], role: .weight)] = (MLXRandom.normal([KVH*hd, D]) * 0.1).asType(.float16)
        dict[slot(ap + [.field("v_proj")], role: .weight)] = (MLXRandom.normal([KVH*hd, D]) * 0.1).asType(.float16)
        dict[slot(ap + [.field("o_proj")], role: .weight)] = (MLXRandom.normal([D, H*hd]) * 0.1).asType(.float16)
        dict[slot(lp + [.operation(1), .regionBody, .operation(0)], role: .scale)] = MLXArray.ones([D]).asType(.float16)
        let mp = lp + [.operation(1), .regionBody, .operation(1)]
        dict[slot(mp + [.field("gate_proj")], role: .weight)] = (MLXRandom.normal([I, D]) * 0.1).asType(.float16)
        dict[slot(mp + [.field("up_proj")], role: .weight)] = (MLXRandom.normal([I, D]) * 0.1).asType(.float16)
        dict[slot(mp + [.field("down_proj")], role: .weight)] = (MLXRandom.normal([D, I]) * 0.1).asType(.float16)
    }
    dict[slot([.operation(2)], role: .scale)] = MLXArray.ones([D]).asType(.float16)
    eval(Array(dict.values))
    return bind(dict)
}

// ============================================================================
// MARK: - Level 1: Op Primitives
// ============================================================================

@Suite("MPSGraphOps", .tags(.unit), .heartbeat)
struct MPSGraphOpsTests {

    @Test("rmsNorm matches MLXFast.rmsNorm")
    func rmsNormParity() {
        let D = 8
        let x = (MLXRandom.normal([1, 4, D]) * 0.5).asType(.float16)
        let w = MLXArray.ones([D]).asType(.float16)
        eval(x, w)

        let mlxOut = MLXFast.rmsNorm(x, weight: w, eps: 1e-5)
        eval(mlxOut)

        let g = makeGraph()
        let xPh = g.placeholder(shape: [1, 4, D as NSNumber], dataType: .float16, name: "x")
        let wV = mpsVar(g, w.asArray(Float16.self), [D], "w")
        let out = MPSGraphOps.rmsNorm(g, input: xPh, weight: wV, epsilon: 1e-5, name: "test")

        let xData = x.asArray(Float16.self).withUnsafeBytes { ptr in
            MPSGraphTensorData(device: MPSGraphDevice(mtlDevice: device), data: Data(ptr),
                                shape: [1, 4, D as NSNumber], dataType: .float16)
        }
        let mpsOut = runGraph(g, feeds: [xPh: xData], target: out)

        let mlxFlat: [Float16] = mlxOut.asArray(Float16.self)
        let maxDiff = zip(mlxFlat, mpsOut).map { abs(Float($0) - Float($1)) }.max() ?? 0
        #expect(maxDiff < 0.01, "rmsNorm diff \(maxDiff)")
    }

    @Test("linear matches matmul(x, w.T)")
    func linearParity() {
        let x = (MLXRandom.normal([1, 1, 4]) * 0.5).asType(.float16)
        let w = (MLXRandom.normal([8, 4]) * 0.5).asType(.float16)
        eval(x, w)

        let mlxOut = matmul(x, w.T)
        eval(mlxOut)

        let g = makeGraph()
        let xPh = g.placeholder(shape: [1, 1, 4], dataType: .float16, name: "x")
        let wV = mpsVar(g, w.asArray(Float16.self), [8, 4], "w")
        let out = MPSGraphOps.linear(g, input: xPh, weight: wV, name: "test")

        let xData = x.asArray(Float16.self).withUnsafeBytes { ptr in
            MPSGraphTensorData(device: MPSGraphDevice(mtlDevice: device), data: Data(ptr),
                                shape: [1, 1, 4], dataType: .float16)
        }
        let mpsOut = runGraph(g, feeds: [xPh: xData], target: out)
        let mlxFlat: [Float16] = mlxOut.asArray(Float16.self)
        let maxDiff = zip(mlxFlat, mpsOut).map { abs(Float($0) - Float($1)) }.max() ?? 0
        #expect(maxDiff < 0.01, "linear diff \(maxDiff)")
    }

    @Test("toHeads/fromHeads round-trip preserves shape")
    func headsRoundTrip() {
        let g = makeGraph()
        let x = g.placeholder(shape: [1, 3, 8], dataType: .float16, name: "x")
        let heads = MPSGraphOps.toHeads(g, input: x, heads: 4, headDim: 2, name: "to")
        let back = MPSGraphOps.fromHeads(g, input: heads, totalDim: 8, name: "from")

        let data = [Float16](repeating: 1.0, count: 24).withUnsafeBytes { ptr in
            MPSGraphTensorData(device: MPSGraphDevice(mtlDevice: device), data: Data(ptr),
                                shape: [1, 3, 8], dataType: .float16)
        }
        let results = graph_run(g, feeds: [x: data], targets: [heads, back])
        #expect(results[heads]!.shape == [1, 4, 3, 2] as [NSNumber])
        #expect(results[back]!.shape == [1, 3, 8] as [NSNumber])
    }

    @Test("SDPA T=1 without mask produces valid output")
    func sdpaT1() {
        let g = makeGraph()
        let q = g.placeholder(shape: [1, 2, 1, 4], dataType: .float16, name: "q")
        let k = g.placeholder(shape: [1, 2, 1, 4], dataType: .float16, name: "k")
        let v = g.placeholder(shape: [1, 2, 1, 4], dataType: .float16, name: "v")
        let out = MPSGraphOps.scaledDotProductAttention(
            g, query: q, key: k, value: v, mask: nil, headDim: 4, name: "t")

        let ones = [Float16](repeating: 0.5, count: 8).withUnsafeBytes { ptr in
            MPSGraphTensorData(device: MPSGraphDevice(mtlDevice: device), data: Data(ptr),
                                shape: [1, 2, 1, 4], dataType: .float16)
        }
        let result = runGraph(g, feeds: [q: ones, k: ones, v: ones], target: out)
        #expect(result.count == 8)
        #expect(result.allSatisfy { $0.isFinite })
    }

    @Test("SDPA T>1 with causal mask produces valid output")
    func sdpaCausal() {
        let g = makeGraph()
        let T = 3
        let q = g.placeholder(shape: [1, 1, T as NSNumber, 2], dataType: .float16, name: "q")
        let k = g.placeholder(shape: [1, 1, T as NSNumber, 2], dataType: .float16, name: "k")
        let v = g.placeholder(shape: [1, 1, T as NSNumber, 2], dataType: .float16, name: "v")
        let maskPh = g.placeholder(shape: [1, 1, T as NSNumber, T as NSNumber], dataType: .float16, name: "m")
        let out = MPSGraphOps.scaledDotProductAttention(
            g, query: q, key: k, value: v, mask: maskPh, headDim: 2, name: "t")

        let data = [Float16](repeating: 1.0, count: T * 2).withUnsafeBytes { ptr in
            MPSGraphTensorData(device: MPSGraphDevice(mtlDevice: device), data: Data(ptr),
                                shape: [1, 1, T as NSNumber, 2], dataType: .float16)
        }
        let maskData = MPSGraphOps.buildCausalMask(seqLen: T)
        let maskTD = MPSGraphTensorData(
            device: MPSGraphDevice(mtlDevice: device), data: maskData,
            shape: [1, 1, T as NSNumber, T as NSNumber], dataType: .float16)
        let result = runGraph(g, feeds: [q: data, k: data, v: data, maskPh: maskTD], target: out)
        #expect(result.count == T * 2)
        #expect(result.allSatisfy { $0.isFinite })
    }

    @Test("buildCausalMask produces correct pattern")
    func causalMaskPattern() {
        let mask = MPSGraphOps.buildCausalMask(seqLen: 3)
        let values = mask.withUnsafeBytes { Array(UnsafeBufferPointer<Float16>(
            start: $0.baseAddress!.assumingMemoryBound(to: Float16.self), count: 9)) }
        // Row 0: [0, -1e4, -1e4]
        // Row 1: [0, 0, -1e4]
        // Row 2: [0, 0, 0]
        #expect(values[0] == 0)
        #expect(values[1] < -9000)
        #expect(values[2] < -9000)
        #expect(values[3] == 0)
        #expect(values[4] == 0)
        #expect(values[5] < -9000)
        #expect(values[6] == 0)
        #expect(values[7] == 0)
        #expect(values[8] == 0)
    }

    @Test("siluGate matches silu(gate) * up")
    func siluGateParity() {
        let gate = (MLXRandom.normal([1, 1, 8]) * 0.5).asType(.float16)
        let up = (MLXRandom.normal([1, 1, 8]) * 0.5).asType(.float16)
        eval(gate, up)

        let mlxOut = silu(gate) * up
        eval(mlxOut)

        let g = makeGraph()
        let gPh = g.placeholder(shape: [1, 1, 8], dataType: .float16, name: "gate")
        let uPh = g.placeholder(shape: [1, 1, 8], dataType: .float16, name: "up")
        let out = MPSGraphOps.siluGate(g, gate: gPh, up: uPh, name: "test")

        let gData = gate.asArray(Float16.self).withUnsafeBytes { ptr in
            MPSGraphTensorData(device: MPSGraphDevice(mtlDevice: device), data: Data(ptr),
                                shape: [1, 1, 8], dataType: .float16)
        }
        let uData = up.asArray(Float16.self).withUnsafeBytes { ptr in
            MPSGraphTensorData(device: MPSGraphDevice(mtlDevice: device), data: Data(ptr),
                                shape: [1, 1, 8], dataType: .float16)
        }
        let mpsOut = runGraph(g, feeds: [gPh: gData, uPh: uData], target: out)
        let mlxFlat: [Float16] = mlxOut.asArray(Float16.self)
        let maxDiff = zip(mlxFlat, mpsOut).map { abs(Float($0) - Float($1)) }.max() ?? 0
        #expect(maxDiff < 0.01, "siluGate diff \(maxDiff)")
    }

    @Test("repeatKVHeads factor=1 is identity")
    func repeatIdentity() {
        let g = makeGraph()
        let x = g.placeholder(shape: [1, 2, 3, 4], dataType: .float16, name: "x")
        let out = MPSGraphOps.repeatKVHeads(g, input: x, repeatFactor: 1, name: "t")
        // Should return same tensor (no tile op)
        #expect(out === x)
    }

    @Test("repeatKVHeads factor=7")
    func repeatNonPower() {
        let g = makeGraph()
        let x = g.placeholder(shape: [1, 2, 1, 4], dataType: .float16, name: "x")
        let out = MPSGraphOps.repeatKVHeads(g, input: x, repeatFactor: 7, name: "t")

        let data = [Float16](repeating: 1.0, count: 8).withUnsafeBytes { ptr in
            MPSGraphTensorData(device: MPSGraphDevice(mtlDevice: device), data: Data(ptr),
                                shape: [1, 2, 1, 4], dataType: .float16)
        }
        let results = g.run(with: queue, feeds: [x: data], targetTensors: [out], targetOperations: nil)
        #expect(results[out]!.shape == [1, 14, 1, 4] as [NSNumber])
    }

    @Test("rmsNorm with zero input does not produce NaN")
    func rmsNormZeroInput() {
        let g = makeGraph()
        let x = g.placeholder(shape: [1, 1, 4], dataType: .float16, name: "x")
        let w = mpsVar(g, [Float16](repeating: 1.0, count: 4), [4], "w")
        let out = MPSGraphOps.rmsNorm(g, input: x, weight: w, epsilon: 1e-5, name: "test")

        let zeros = [Float16](repeating: 0, count: 4).withUnsafeBytes { ptr in
            MPSGraphTensorData(device: MPSGraphDevice(mtlDevice: device), data: Data(ptr),
                                shape: [1, 1, 4], dataType: .float16)
        }
        let result = runGraph(g, feeds: [x: zeros], target: out)
        #expect(result.allSatisfy { $0.isFinite })
    }

    @Test("toHeads dynamic T")
    func toHeadsDynamicT() {
        let g = makeGraph()
        let x = g.placeholder(shape: [1, -1, 8], dataType: .float16, name: "x")
        let out = MPSGraphOps.toHeads(g, input: x, heads: 4, headDim: 2, name: "t")

        for T in [1, 5, 128] {
            let data = [Float16](repeating: 1.0, count: T * 8).withUnsafeBytes { ptr in
                MPSGraphTensorData(device: MPSGraphDevice(mtlDevice: device), data: Data(ptr),
                                    shape: [1, T as NSNumber, 8], dataType: .float16)
            }
            let results = g.run(with: queue, feeds: [x: data], targetTensors: [out], targetOperations: nil)
            let shape = results[out]!.shape.map { $0.intValue }
            #expect(shape == [1, 4, T, 2], "T=\(T) shape \(shape)")
        }
    }
}

// Helper for multi-target graph run
private func graph_run(
    _ g: MPSGraph, feeds: [MPSGraphTensor: MPSGraphTensorData],
    targets: [MPSGraphTensor]
) -> [MPSGraphTensor: MPSGraphTensorData] {
    g.run(with: queue, feeds: feeds, targetTensors: targets, targetOperations: nil)
}

// ============================================================================
// MARK: - Level 2: IR Compiler
// ============================================================================

@Suite("MPSGraphInferenceCompiler", .tags(.unit), .heartbeat)
struct MPSGraphInferenceCompilerTests {

    @Test("Tiny transformer compiles successfully")
    func tinyCompiles() throws {
        MLXRandom.seed(42)
        let graph = try TinyTransformer(L: 1, V: 8, D: 4, H: 2, KVH: 2, hd: 2, I: 8).makeModelGraph()
        let weights = tinyWeights(L: 1)
        let compiled = try MPSGraphInferenceCompiler().compile(graph: graph, weights: weights)
        #expect(compiled.metadata.cacheSlotCount == 1)
    }

    @Test("Metadata matches MLXInferenceCompiler")
    func metadataMatch() throws {
        MLXRandom.seed(42)
        let graph = try TinyTransformer(L: 2, V: 8, D: 4, H: 2, KVH: 2, hd: 2, I: 8).makeModelGraph()
        let weights = tinyWeights(L: 2)
        let mlx = try MLXInferenceCompiler().compile(graph: graph, weights: weights)
        let mps = try MPSGraphInferenceCompiler().compile(graph: graph, weights: weights)
        #expect(mps.metadata.cacheSlotCount == mlx.metadata.cacheSlotCount)
        #expect(mps.metadata.hasTiedOutputHead == mlx.metadata.hasTiedOutputHead)
    }

    @Test("Repeating unrolls: 2 layers produce 2 cache slots")
    func repeatingUnroll() throws {
        MLXRandom.seed(42)
        let graph = try TinyTransformer(L: 2, V: 8, D: 4, H: 2, KVH: 2, hd: 2, I: 8).makeModelGraph()
        let weights = tinyWeights(L: 2)
        let compiled = try MPSGraphInferenceCompiler().compile(graph: graph, weights: weights)
        #expect(compiled.metadata.cacheSlotCount == 2)
    }

    @Test("GQA (H=4, KVH=2) compiles")
    func gqaCompiles() throws {
        MLXRandom.seed(42)
        let graph = try TinyTransformer(L: 1, V: 8, D: 8, H: 4, KVH: 2, hd: 2, I: 16).makeModelGraph()
        let weights = tinyWeights(L: 1, D: 8, H: 4, KVH: 2, hd: 2, I: 16)
        let compiled = try MPSGraphInferenceCompiler().compile(graph: graph, weights: weights)
        #expect(compiled.metadata.cacheSlotCount == 1)
    }

    @Test("Tied output head uses embedding weight")
    func tiedHead() throws {
        MLXRandom.seed(42)
        let graph = try TinyTransformer(L: 1, V: 8, D: 4, H: 2, KVH: 2, hd: 2, I: 8).makeModelGraph()
        let weights = tinyWeights(L: 1)
        let compiled = try MPSGraphInferenceCompiler().compile(graph: graph, weights: weights)
        #expect(compiled.metadata.hasTiedOutputHead == true)
    }

    @Test("Missing weight throws error")
    func missingWeight() throws {
        let graph = try TinyTransformer(L: 1, V: 8, D: 4, H: 2, KVH: 2, hd: 2, I: 8).makeModelGraph()
        let emptyWeights = BoundWeights(tensors: [:])
        #expect(throws: (any Error).self) {
            _ = try MPSGraphInferenceCompiler().compile(graph: graph, weights: emptyWeights)
        }
    }
}

// ============================================================================
// MARK: - Level 3: Forward Pass
// ============================================================================

@Suite("MPSGraphForward", .tags(.unit), .heartbeat)
struct MPSGraphForwardTests {

    @Test("Single token forward produces finite logits")
    func singleToken() throws {
        MLXRandom.seed(42)
        let graph = try TinyTransformer(L: 1, V: 8, D: 4, H: 2, KVH: 2, hd: 2, I: 8).makeModelGraph()
        let weights = tinyWeights(L: 1)
        let compiled = try MPSGraphInferenceCompiler().compile(graph: graph, weights: weights)
        let logits = compiled.forward([Int32(1)])
        #expect(logits.shape == [1, 1, 8])
        let values: [Float16] = logits.asArray(Float16.self)
        #expect(values.allSatisfy { $0.isFinite })
    }

    @Test("Multiple tokens forward produces correct shape")
    func multipleTokens() throws {
        MLXRandom.seed(42)
        let graph = try TinyTransformer(L: 1, V: 8, D: 4, H: 2, KVH: 2, hd: 2, I: 8).makeModelGraph()
        let weights = tinyWeights(L: 1)
        let compiled = try MPSGraphInferenceCompiler().compile(graph: graph, weights: weights)
        let logits = compiled.forward([Int32(1), Int32(2), Int32(3)])
        #expect(logits.shape == [1, 3, 8])
        let values: [Float16] = logits.asArray(Float16.self)
        #expect(values.allSatisfy { $0.isFinite })
    }

    @Test("MLX vs MPSGraph logit parity (L=1 MHA)")
    func parityL1MHA() throws {
        MLXRandom.seed(42)
        let graph = try TinyTransformer(L: 1, V: 8, D: 4, H: 2, KVH: 2, hd: 2, I: 8).makeModelGraph()
        let weights = tinyWeights(L: 1)

        let mlxModel = try MLXInferenceCompiler().compile(graph: graph, weights: weights)
        let mpsModel = try MPSGraphInferenceCompiler().compile(graph: graph, weights: weights)

        let prompt = MLXArray([Int32(1)]).expandedDimensions(axis: 0)
        eval(prompt)
        let (mlxLogits, _) = mlxModel.prefill(tokenIDs: prompt, state: mlxModel.makeState())
        eval(mlxLogits)

        let mpsLogits = mpsModel.forward([Int32(1)])
        eval(mpsLogits)

        let mlxFlat: [Float] = mlxLogits.asType(.float32).asArray(Float.self)
        let mpsFlat: [Float] = mpsLogits.asType(.float32).asArray(Float.self)
        let maxDiff = zip(mlxFlat, mpsFlat).map { abs($0 - $1) }.max() ?? 0
        // fp16 tolerance — both backends use fp16 weights but different kernel paths
        #expect(maxDiff < 0.5, "MLX vs MPSGraph logit diff \(maxDiff)")
    }

    @Test("MLX vs MPSGraph parity (L=1 GQA)")
    func parityL1GQA() throws {
        MLXRandom.seed(42)
        let graph = try TinyTransformer(L: 1, V: 8, D: 8, H: 4, KVH: 2, hd: 2, I: 16).makeModelGraph()
        let weights = tinyWeights(L: 1, D: 8, H: 4, KVH: 2, hd: 2, I: 16)

        let mlxModel = try MLXInferenceCompiler().compile(graph: graph, weights: weights)
        let mpsModel = try MPSGraphInferenceCompiler().compile(graph: graph, weights: weights)

        let prompt = MLXArray([Int32(0)]).expandedDimensions(axis: 0)
        eval(prompt)
        let (mlxLogits, _) = mlxModel.prefill(tokenIDs: prompt, state: mlxModel.makeState())
        eval(mlxLogits)

        let mpsLogits = mpsModel.forward([Int32(0)])
        eval(mpsLogits)

        let mlxFlat: [Float] = mlxLogits.asType(.float32).asArray(Float.self)
        let mpsFlat: [Float] = mpsLogits.asType(.float32).asArray(Float.self)
        let maxDiff = zip(mlxFlat, mpsFlat).map { abs($0 - $1) }.max() ?? 0
        #expect(maxDiff < 0.5, "GQA parity diff \(maxDiff)")
    }

    @Test("MLX vs MPSGraph parity (L=4)")
    func parityL4() throws {
        MLXRandom.seed(42)
        let graph = try TinyTransformer(L: 4, V: 8, D: 4, H: 2, KVH: 2, hd: 2, I: 8).makeModelGraph()
        let weights = tinyWeights(L: 4)

        let mlxModel = try MLXInferenceCompiler().compile(graph: graph, weights: weights)
        let mpsModel = try MPSGraphInferenceCompiler().compile(graph: graph, weights: weights)

        let prompt = MLXArray([Int32(3)]).expandedDimensions(axis: 0)
        eval(prompt)
        let (mlxLogits, _) = mlxModel.prefill(tokenIDs: prompt, state: mlxModel.makeState())
        eval(mlxLogits)

        let mpsLogits = mpsModel.forward([Int32(3)])
        eval(mpsLogits)

        let mlxFlat: [Float] = mlxLogits.asType(.float32).asArray(Float.self)
        let mpsFlat: [Float] = mpsLogits.asType(.float32).asArray(Float.self)
        let maxDiff = zip(mlxFlat, mpsFlat).map { abs($0 - $1) }.max() ?? 0
        #expect(maxDiff < 1.0, "L=4 parity diff \(maxDiff)")
    }

    @Test("Logits no NaN/Inf")
    func logitsFinite() throws {
        MLXRandom.seed(42)
        let graph = try TinyTransformer(L: 2, V: 8, D: 4, H: 2, KVH: 2, hd: 2, I: 8).makeModelGraph()
        let weights = tinyWeights(L: 2)
        let compiled = try MPSGraphInferenceCompiler().compile(graph: graph, weights: weights)
        let logits = compiled.forward([Int32(0), Int32(1), Int32(2)])
        eval(logits)
        let values: [Float16] = logits.asArray(Float16.self)
        #expect(values.allSatisfy { $0.isFinite }, "Found NaN/Inf in logits")
    }

    @Test("Large T=64 forward works")
    func largeT() throws {
        MLXRandom.seed(42)
        let graph = try TinyTransformer(L: 1, V: 8, D: 4, H: 2, KVH: 2, hd: 2, I: 8).makeModelGraph()
        let weights = tinyWeights(L: 1)
        let compiled = try MPSGraphInferenceCompiler().compile(graph: graph, weights: weights)
        let tokens = (0..<64).map { Int32($0 % 8) }
        let logits = compiled.forward(tokens)
        #expect(logits.shape == [1, 64, 8])
    }

    @Test("Embedding lookup is correct")
    func embeddingLookup() throws {
        MLXRandom.seed(42)
        let graph = try TinyTransformer(L: 1, V: 8, D: 4, H: 2, KVH: 2, hd: 2, I: 8).makeModelGraph()
        let weights = tinyWeights(L: 1)
        let compiled = try MPSGraphInferenceCompiler().compile(graph: graph, weights: weights)
        // Different token IDs should produce different logits
        let logits0 = compiled.forward([Int32(0)])
        let logits1 = compiled.forward([Int32(1)])
        eval(logits0, logits1)
        let v0: [Float16] = logits0.asArray(Float16.self)
        let v1: [Float16] = logits1.asArray(Float16.self)
        let diff = zip(v0, v1).map { abs(Float($0) - Float($1)) }.max() ?? 0
        #expect(diff > 0.001, "Different tokens should produce different logits")
    }
}

// ============================================================================
// MARK: - RoPE Unit Tests
// ============================================================================

private func readFloat16(_ data: Data, count: Int) -> [Float16] {
    data.withUnsafeBytes { Array(UnsafeBufferPointer<Float16>(
        start: $0.baseAddress!.assumingMemoryBound(to: Float16.self), count: count)) }
}

@Suite("MPSGraphRoPE", .tags(.unit), .heartbeat)
struct MPSGraphRoPETests {

    // MARK: - buildRoPETables

    @Test("R1: position 0 produces cos=1 sin=0")
    func buildTables_pos0_identity() {
        let (cosData, sinData) = MPSGraphOps.buildRoPETables(seqLen: 1, headDim: 4, theta: 10000)
        let cos = readFloat16(cosData, count: 2)
        let sin = readFloat16(sinData, count: 2)
        // pos=0: angle = 0 → cos=1, sin=0
        #expect(abs(Float(cos[0]) - 1.0) < 0.01, "cos[0] should be ~1.0, got \(cos[0])")
        #expect(abs(Float(cos[1]) - 1.0) < 0.01, "cos[1] should be ~1.0, got \(cos[1])")
        #expect(abs(Float(sin[0])) < 0.01, "sin[0] should be ~0, got \(sin[0])")
        #expect(abs(Float(sin[1])) < 0.01, "sin[1] should be ~0, got \(sin[1])")
    }

    @Test("R2: position 1 produces non-trivial cos/sin")
    func buildTables_pos1_rotates() {
        let (cosData, sinData) = MPSGraphOps.buildRoPETables(seqLen: 2, headDim: 4, theta: 10000)
        let cos = readFloat16(cosData, count: 4) // 2 positions × 2 halfDim
        let sin = readFloat16(sinData, count: 4)
        // pos=1, dim=0: angle = 1 * (1/10000^0) = 1.0
        let cos1_0 = Float(cos[2]) // pos 1, dim 0
        let sin1_0 = Float(sin[2])
        #expect(abs(cos1_0 - Foundation.cos(Float(1.0))) < 0.01)
        #expect(abs(sin1_0 - Foundation.sin(Float(1.0))) < 0.01)
    }

    @Test("R3: different theta produces different frequencies")
    func buildTables_thetaAffects() {
        // Use larger pos to amplify angle difference in fp16
        let (cos10k, _) = MPSGraphOps.buildRoPETables(seqLen: 100, headDim: 4, theta: 100)
        let (cos1m, _) = MPSGraphOps.buildRoPETables(seqLen: 100, headDim: 4, theta: 1_000_000)
        let halfDim = 2
        // Compare pos=50, dim=0: angle = 50 * 1.0 = 50 vs 50 * 1.0 = 50 (same for dim=0)
        // Compare pos=50, dim=1: theta=100 → freq = 1/10 → angle = 5.0
        //                        theta=1M  → freq = 1/1000 → angle = 0.05
        let a = readFloat16(cos10k, count: 100 * halfDim)
        let b = readFloat16(cos1m, count: 100 * halfDim)
        let idx = 50 * halfDim + 1 // pos=50, dim=1
        let diff = abs(Float(a[idx]) - Float(b[idx]))
        #expect(diff > 0.1, "Different theta should produce different cos, diff=\(diff)")
    }

    @Test("R4: linear scaling halves the effective angle")
    func buildTables_linearScaling() {
        let scaling = RoPEScaling(kind: .linear, factor: 2.0)
        let (cosNoScale, _) = MPSGraphOps.buildRoPETables(seqLen: 3, headDim: 4, theta: 10000)
        let (cosScaled, _) = MPSGraphOps.buildRoPETables(seqLen: 3, headDim: 4, theta: 10000, scaling: scaling)
        let noScale = readFloat16(cosNoScale, count: 6)
        let scaled = readFloat16(cosScaled, count: 6)

        // With factor=2, pos=2 scaled should equal pos=1 unscaled (angle halved)
        // scaled[pos=2] uses angle = 2 * freq * 0.5 = 1 * freq = noScale[pos=1]
        let scaledPos2 = Float(scaled[4]) // pos=2, dim=0
        let noScalePos1 = Float(noScale[2]) // pos=1, dim=0
        #expect(abs(scaledPos2 - noScalePos1) < 0.02, "scaled pos=2 should ≈ unscaled pos=1")
    }

    @Test("R5: offset shifts position indices")
    func buildTables_offset() {
        let (cosOff0, sinOff0) = MPSGraphOps.buildRoPETables(seqLen: 11, headDim: 4, theta: 10000)
        let (cosOff10, sinOff10) = MPSGraphOps.buildRoPETables(seqLen: 1, headDim: 4, theta: 10000, offset: 10)
        let a = readFloat16(cosOff0, count: 22) // 11 positions × 2 halfDim
        let b = readFloat16(cosOff10, count: 2) // 1 position × 2 halfDim
        // offset=10 pos=0 should equal offset=0 pos=10
        #expect(abs(Float(a[20]) - Float(b[0])) < 0.01, "offset=10 pos=0 should match offset=0 pos=10")
        let sa = readFloat16(sinOff0, count: 22)
        let sb = readFloat16(sinOff10, count: 2)
        #expect(abs(Float(sa[20]) - Float(sb[0])) < 0.01)
    }

    // MARK: - applyRoPE (graph execution)

    @Test("R6: position 0 leaves input unchanged")
    func applyRoPE_identity() {
        let hd = 4
        let g = makeGraph()
        let x = g.placeholder(shape: [1, 2, 1, hd as NSNumber], dataType: .float16, name: "x")
        let cosPh = g.placeholder(shape: [1, 1, 1, (hd/2) as NSNumber], dataType: .float16, name: "c")
        let sinPh = g.placeholder(shape: [1, 1, 1, (hd/2) as NSNumber], dataType: .float16, name: "s")
        let out = MPSGraphOps.applyRoPE(g, input: x, cosTable: cosPh, sinTable: sinPh, headDim: hd, name: "r")

        let xData = [Float16](repeating: 0.5, count: 8)
        let (cosBytes, sinBytes) = MPSGraphOps.buildRoPETables(seqLen: 1, headDim: hd, theta: 10000)

        let dev = MPSGraphDevice(mtlDevice: device)
        let xTD = xData.withUnsafeBytes { MPSGraphTensorData(device: dev, data: Data($0), shape: [1, 2, 1, hd as NSNumber], dataType: .float16) }
        let cTD = MPSGraphTensorData(device: dev, data: cosBytes, shape: [1, 1, 1, (hd/2) as NSNumber], dataType: .float16)
        let sTD = MPSGraphTensorData(device: dev, data: sinBytes, shape: [1, 1, 1, (hd/2) as NSNumber], dataType: .float16)

        let result = runGraph(g, feeds: [x: xTD, cosPh: cTD, sinPh: sTD], target: out)
        // cos=1, sin=0 → x unchanged
        let maxDiff = zip(xData, result).map { abs(Float($0) - Float($1)) }.max() ?? 0
        #expect(maxDiff < 0.01, "pos=0 should not change input, diff=\(maxDiff)")
    }

    @Test("R7: position 1 rotates input")
    func applyRoPE_rotates() {
        let hd = 4
        let g = makeGraph()
        let x = g.placeholder(shape: [1, 1, 1, hd as NSNumber], dataType: .float16, name: "x")
        let cosPh = g.placeholder(shape: [1, 1, 1, (hd/2) as NSNumber], dataType: .float16, name: "c")
        let sinPh = g.placeholder(shape: [1, 1, 1, (hd/2) as NSNumber], dataType: .float16, name: "s")
        let out = MPSGraphOps.applyRoPE(g, input: x, cosTable: cosPh, sinTable: sinPh, headDim: hd, name: "r")

        let xData: [Float16] = [1.0, 0.0, 0.0, 1.0]
        let (cosBytes, sinBytes) = MPSGraphOps.buildRoPETables(seqLen: 2, headDim: hd, theta: 10000)
        // Extract pos=1 slice
        let cosAll = readFloat16(cosBytes, count: 4)
        let sinAll = readFloat16(sinBytes, count: 4)
        let cos1 = Array(cosAll[2..<4]).withUnsafeBytes { Data($0) }
        let sin1 = Array(sinAll[2..<4]).withUnsafeBytes { Data($0) }

        let dev = MPSGraphDevice(mtlDevice: device)
        let xTD = xData.withUnsafeBytes { MPSGraphTensorData(device: dev, data: Data($0), shape: [1, 1, 1, hd as NSNumber], dataType: .float16) }
        let cTD = MPSGraphTensorData(device: dev, data: cos1, shape: [1, 1, 1, (hd/2) as NSNumber], dataType: .float16)
        let sTD = MPSGraphTensorData(device: dev, data: sin1, shape: [1, 1, 1, (hd/2) as NSNumber], dataType: .float16)

        let result = runGraph(g, feeds: [x: xTD, cosPh: cTD, sinPh: sTD], target: out)
        let diff = zip(xData, result).map { abs(Float($0) - Float($1)) }.max() ?? 0
        #expect(diff > 0.01, "pos=1 should rotate input, diff=\(diff)")
    }

    @Test("R8: T=3 positions produce different outputs")
    func applyRoPE_T3_differs() {
        let hd = 4
        let T = 3
        let g = makeGraph()
        let x = g.placeholder(shape: [1, 1, T as NSNumber, hd as NSNumber], dataType: .float16, name: "x")
        let cosPh = g.placeholder(shape: [1, 1, T as NSNumber, (hd/2) as NSNumber], dataType: .float16, name: "c")
        let sinPh = g.placeholder(shape: [1, 1, T as NSNumber, (hd/2) as NSNumber], dataType: .float16, name: "s")
        let out = MPSGraphOps.applyRoPE(g, input: x, cosTable: cosPh, sinTable: sinPh, headDim: hd, name: "r")

        let xData = [Float16](repeating: 1.0, count: T * hd)
        let (cosBytes, sinBytes) = MPSGraphOps.buildRoPETables(seqLen: T, headDim: hd, theta: 10000)

        let dev = MPSGraphDevice(mtlDevice: device)
        let xTD = xData.withUnsafeBytes { MPSGraphTensorData(device: dev, data: Data($0), shape: [1, 1, T as NSNumber, hd as NSNumber], dataType: .float16) }
        let cTD = MPSGraphTensorData(device: dev, data: cosBytes, shape: [1, 1, T as NSNumber, (hd/2) as NSNumber], dataType: .float16)
        let sTD = MPSGraphTensorData(device: dev, data: sinBytes, shape: [1, 1, T as NSNumber, (hd/2) as NSNumber], dataType: .float16)

        let result = runGraph(g, feeds: [x: xTD, cosPh: cTD, sinPh: sTD], target: out)
        // pos 0, 1, 2 should produce different values
        let pos0 = Array(result[0..<hd])
        let pos1 = Array(result[hd..<2*hd])
        let pos2 = Array(result[2*hd..<3*hd])
        let diff01 = zip(pos0, pos1).map { abs(Float($0) - Float($1)) }.max() ?? 0
        let diff12 = zip(pos1, pos2).map { abs(Float($0) - Float($1)) }.max() ?? 0
        #expect(diff01 > 0.01, "pos 0 and 1 should differ")
        #expect(diff12 > 0.001, "pos 1 and 2 should differ")
    }

    @Test("R9: matches MLXFast.RoPE at offset=5")
    func applyRoPE_matchesMLX() {
        let hd = 4
        let x = (MLXRandom.normal([1, 2, 1, hd]) * 0.5).asType(.float16)
        eval(x)

        // MLX reference
        let mlxOut = MLXFast.RoPE(x, dimensions: hd, traditional: false,
                                   base: 10000, scale: 1.0, offset: 5)
        eval(mlxOut)
        let mlxFlat: [Float16] = mlxOut.asArray(Float16.self)

        // MPSGraph
        let g = makeGraph()
        let xPh = g.placeholder(shape: [1, 2, 1, hd as NSNumber], dataType: .float16, name: "x")
        let cosPh = g.placeholder(shape: [1, 1, 1, (hd/2) as NSNumber], dataType: .float16, name: "c")
        let sinPh = g.placeholder(shape: [1, 1, 1, (hd/2) as NSNumber], dataType: .float16, name: "s")
        let out = MPSGraphOps.applyRoPE(g, input: xPh, cosTable: cosPh, sinTable: sinPh, headDim: hd, name: "r")

        let (cosBytes, sinBytes) = MPSGraphOps.buildRoPETables(seqLen: 1, headDim: hd, theta: 10000, offset: 5)

        let dev = MPSGraphDevice(mtlDevice: device)
        let xData = x.asArray(Float16.self)
        let xTD = xData.withUnsafeBytes { MPSGraphTensorData(device: dev, data: Data($0), shape: [1, 2, 1, hd as NSNumber], dataType: .float16) }
        let cTD = MPSGraphTensorData(device: dev, data: cosBytes, shape: [1, 1, 1, (hd/2) as NSNumber], dataType: .float16)
        let sTD = MPSGraphTensorData(device: dev, data: sinBytes, shape: [1, 1, 1, (hd/2) as NSNumber], dataType: .float16)

        let mpsFlat = runGraph(g, feeds: [xPh: xTD, cosPh: cTD, sinPh: sTD], target: out)
        let maxDiff = zip(mlxFlat, mpsFlat).map { abs(Float($0) - Float($1)) }.max() ?? 0
        #expect(maxDiff < 0.01, "MPSGraph RoPE should match MLXFast.RoPE, diff=\(maxDiff)")
    }

    @Test("R10: headDim=4 output shape correct")
    func applyRoPE_headDim4() {
        let hd = 4
        let g = makeGraph()
        let x = g.placeholder(shape: [1, 2, 3, hd as NSNumber], dataType: .float16, name: "x")
        let cosPh = g.placeholder(shape: [1, 1, 3, (hd/2) as NSNumber], dataType: .float16, name: "c")
        let sinPh = g.placeholder(shape: [1, 1, 3, (hd/2) as NSNumber], dataType: .float16, name: "s")
        let out = MPSGraphOps.applyRoPE(g, input: x, cosTable: cosPh, sinTable: sinPh, headDim: hd, name: "r")

        let xData = [Float16](repeating: 1.0, count: 2 * 3 * hd)
        let (cosBytes, sinBytes) = MPSGraphOps.buildRoPETables(seqLen: 3, headDim: hd, theta: 10000)
        let dev = MPSGraphDevice(mtlDevice: device)
        let xTD = xData.withUnsafeBytes { MPSGraphTensorData(device: dev, data: Data($0), shape: [1, 2, 3, hd as NSNumber], dataType: .float16) }
        let cTD = MPSGraphTensorData(device: dev, data: cosBytes, shape: [1, 1, 3, (hd/2) as NSNumber], dataType: .float16)
        let sTD = MPSGraphTensorData(device: dev, data: sinBytes, shape: [1, 1, 3, (hd/2) as NSNumber], dataType: .float16)

        let result = runGraph(g, feeds: [x: xTD, cosPh: cTD, sinPh: sTD], target: out)
        #expect(result.count == 2 * 3 * hd, "output count should be 24, got \(result.count)")
    }
}
