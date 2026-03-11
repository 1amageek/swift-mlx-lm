import Testing
@preconcurrency import MLX
import MLXFast
@testable import SwiftLM
@testable import MLXCompiler

// MARK: - Benchmark Helpers

private func slot(
    _ components: [StructuralPathComponent], role: ParameterRole
) -> ParameterSlot {
    ParameterSlot(path: StructuralPath(components: components), role: role)
}

private func bind(_ pairs: [ParameterSlot: MLXArray]) -> BoundWeights {
    var tensors: [ParameterSlot: TensorData] = [:]
    for (slot, array) in pairs {
        tensors[slot] = TensorData(
            shape: array.shape.map { $0 },
            dtype: .float16,
            storage: array
        )
    }
    return BoundWeights(tensors: tensors)
}

/// Measure execution time with warmup.
/// Returns (median, min, max) in milliseconds.
private func benchmark(
    warmup: Int = 5,
    iterations: Int = 50,
    _ body: () throws -> Void
) rethrows -> (median: Double, min: Double, max: Double) {
    let clock = ContinuousClock()

    for _ in 0..<warmup {
        try body()
    }

    var durations: [Double] = []
    for _ in 0..<iterations {
        let start = clock.now
        try body()
        let elapsed = clock.now - start
        let ms = Double(elapsed.components.seconds) * 1_000
            + Double(elapsed.components.attoseconds) * 1e-15
        durations.append(ms)
    }

    durations.sort()
    let median = durations[durations.count / 2]
    return (median: median, min: durations.first!, max: durations.last!)
}

// MARK: - Benchmark Model: Standard Transformer

/// Llama-style transformer for benchmarking.
private struct BenchTransformer: ModelComponent {
    let layerCount: Int
    let vocabSize: Int
    let hiddenSize: Int
    let headCount: Int
    let kvHeadCount: Int
    let headDim: Int
    let intermediateSize: Int

    @ModelComponentBuilder var body: some ModelComponent {
        TokenEmbedding(vocabSize: vocabSize, embeddingSize: hiddenSize)

        Repeat(count: layerCount) {
            Residual {
                RMSNorm(dimension: hiddenSize)
                Attention(
                    hiddenSize: hiddenSize,
                    headCount: headCount,
                    kvHeadCount: kvHeadCount,
                    headDimension: headDim
                )
            }
            Residual {
                RMSNorm(dimension: hiddenSize)
                MLP(inputSize: hiddenSize, intermediateSize: intermediateSize)
            }
        }

        RMSNorm(dimension: hiddenSize)
        OutputHead(inputSize: hiddenSize, vocabSize: vocabSize, tiedToEmbedding: true)
    }
}

/// Build an inference model with float16 weights.
private func buildInferenceModel(
    layerCount: Int,
    hiddenSize: Int,
    headCount: Int,
    kvHeadCount: Int,
    headDim: Int,
    intermediateSize: Int,
    vocabSize: Int = 32000
) throws -> MLXLoweredInferenceModel {
    let model = BenchTransformer(
        layerCount: layerCount,
        vocabSize: vocabSize,
        hiddenSize: hiddenSize,
        headCount: headCount,
        kvHeadCount: kvHeadCount,
        headDim: headDim,
        intermediateSize: intermediateSize
    )
    let graph = try model.makeModelGraph()

    let D = hiddenSize
    let H = headCount
    let KVH = kvHeadCount
    let hd = headDim
    let inter = intermediateSize

    var dict: [ParameterSlot: MLXArray] = [:]

    // Embedding (float16)
    dict[slot([.operation(0)], role: .embeddingTable)] =
        (MLXRandom.normal([vocabSize, D]) * 0.02).asType(.float16)

    for i in 0..<layerCount {
        let layerPrefix: [StructuralPathComponent] = [
            .operation(1), .regionBody, .index(i)
        ]

        // Residual 0: RMSNorm + Attention
        let attnNormPath = layerPrefix + [.operation(0), .regionBody, .operation(0)]
        dict[slot(attnNormPath, role: .scale)] = MLXArray.ones([D]).asType(.float16)

        let attnPath = layerPrefix + [.operation(0), .regionBody, .operation(1)]
        dict[slot(attnPath + [.field("q_proj")], role: .weight)] =
            (MLXRandom.normal([H * hd, D]) * 0.02).asType(.float16)
        dict[slot(attnPath + [.field("k_proj")], role: .weight)] =
            (MLXRandom.normal([KVH * hd, D]) * 0.02).asType(.float16)
        dict[slot(attnPath + [.field("v_proj")], role: .weight)] =
            (MLXRandom.normal([KVH * hd, D]) * 0.02).asType(.float16)
        dict[slot(attnPath + [.field("o_proj")], role: .weight)] =
            (MLXRandom.normal([D, H * hd]) * 0.02).asType(.float16)

        // Residual 1: RMSNorm + MLP
        let mlpNormPath = layerPrefix + [.operation(1), .regionBody, .operation(0)]
        dict[slot(mlpNormPath, role: .scale)] = MLXArray.ones([D]).asType(.float16)

        let mlpPath = layerPrefix + [.operation(1), .regionBody, .operation(1)]
        dict[slot(mlpPath + [.field("gate_proj")], role: .weight)] =
            (MLXRandom.normal([inter, D]) * 0.02).asType(.float16)
        dict[slot(mlpPath + [.field("up_proj")], role: .weight)] =
            (MLXRandom.normal([inter, D]) * 0.02).asType(.float16)
        dict[slot(mlpPath + [.field("down_proj")], role: .weight)] =
            (MLXRandom.normal([D, inter]) * 0.02).asType(.float16)
    }

    // Final RMSNorm
    dict[slot([.operation(2)], role: .scale)] = MLXArray.ones([D]).asType(.float16)

    // Force all weights into GPU memory
    eval(Array(dict.values))

    let compiler = MLXInferenceCompiler()
    return try compiler.compile(graph: graph, weights: bind(dict))
}

/// Expand fused steps back to unfused form for A/B comparison.
private func expandFusedSteps(_ steps: [FlatStep]) -> [FlatStep] {
    var unfused: [FlatStep] = []
    for step in steps {
        if case .fusedSubLayer(let fused) = step {
            unfused.append(.saveResidual)
            switch fused {
            case .attention(let norm, let attn):
                unfused.append(.op(.norm(norm)))
                unfused.append(.op(.attention(attn)))
            case .mlp(let norm, let mlp):
                unfused.append(.op(.norm(norm)))
                unfused.append(.op(.mlp(mlp)))
            case .deltaNet(let norm, let dn):
                unfused.append(.op(.norm(norm)))
                unfused.append(.op(.deltaNet(dn)))
            case .moe(let norm, let moe):
                unfused.append(.op(.norm(norm)))
                unfused.append(.op(.moe(moe)))
            }
            unfused.append(.addResidual)
        } else {
            unfused.append(step)
        }
    }
    return unfused
}

// MARK: - Model Configurations (Realistic)

/// Qwen3.5-0.6B scale: D=896, H=14, KVH=2, headDim=64, inter=4864, 24L
private let smallConfig = (
    layerCount: 24, hiddenSize: 896, headCount: 14,
    kvHeadCount: 2, headDim: 64, intermediateSize: 4864
)

/// Llama-3.2-1B scale: D=2048, H=32, KVH=8, headDim=64, inter=8192, 16L
private let mediumConfig = (
    layerCount: 16, hiddenSize: 2048, headCount: 32,
    kvHeadCount: 8, headDim: 64, intermediateSize: 8192
)

// MARK: - Fusion Benchmarks (Realistic Dimensions)

@Suite("Fusion Benchmarks (Realistic)")
struct FusionRealisticBenchmarks {

    // MARK: - Small Model (Qwen3.5-0.6B scale)

    @Test("Small (D=896, 24L): decode latency — fused vs unfused")
    func smallDecodeComparison() throws {
        let model = try buildInferenceModel(
            layerCount: smallConfig.layerCount,
            hiddenSize: smallConfig.hiddenSize,
            headCount: smallConfig.headCount,
            kvHeadCount: smallConfig.kvHeadCount,
            headDim: smallConfig.headDim,
            intermediateSize: smallConfig.intermediateSize
        )

        let fusedSteps = model.decode.steps
        let unfusedSteps = expandFusedSteps(fusedSteps)

        // Prefill to warm up caches
        let prompt = MLXArray([Int32(0), Int32(1), Int32(2), Int32(3)]).reshaped(1, 4)
        let prefillState: InferenceState
        do {
            var s = model.makeState()
            let (_, ps) = model.prefill(tokenIDs: prompt, state: s)
            prefillState = ps
        }

        let decodeToken = MLXArray([Int32(5)]).reshaped(1, 1)

        // Use executeFlatSteps for BOTH paths (identical call overhead)
        // Fixed cache size: don't accumulate state across iterations

        // Benchmark UNFUSED first (cold GPU state), then fused
        let unfusedResult = benchmark(warmup: 10, iterations: 30) {
            var s = prefillState
            let logits = executeFlatSteps(unfusedSteps, input: decodeToken, state: &s)
            eval(logits)
        }

        let fusedResult = benchmark(warmup: 10, iterations: 30) {
            var s = prefillState
            let logits = executeFlatSteps(fusedSteps, input: decodeToken, state: &s)
            eval(logits)
        }

        let improvement = (unfusedResult.median - fusedResult.median) / unfusedResult.median * 100

        print("""
        [SMALL D=896 24L decode]
          fused:   \(String(format: "%.2f", fusedResult.median)) ms  (steps: \(fusedSteps.count))
          unfused: \(String(format: "%.2f", unfusedResult.median)) ms  (steps: \(unfusedSteps.count))
          diff:    \(String(format: "%+.1f", improvement))%
        """)
    }

    @Test("Small (D=896, 24L): prefill 128 tokens")
    func smallPrefill() throws {
        let model = try buildInferenceModel(
            layerCount: smallConfig.layerCount,
            hiddenSize: smallConfig.hiddenSize,
            headCount: smallConfig.headCount,
            kvHeadCount: smallConfig.kvHeadCount,
            headDim: smallConfig.headDim,
            intermediateSize: smallConfig.intermediateSize
        )

        let prompt = MLXArray((0..<128).map { Int32($0 % 32000) }).reshaped(1, 128)

        let result = benchmark(warmup: 3, iterations: 10) {
            let state = model.makeState()
            let (logits, _) = model.prefill(tokenIDs: prompt, state: state)
            eval(logits)
        }

        print("""
        [SMALL D=896 24L prefill 128tok]
          median: \(String(format: "%.2f", result.median)) ms
          min:    \(String(format: "%.2f", result.min)) ms
          max:    \(String(format: "%.2f", result.max)) ms
        """)
    }

    // MARK: - Medium Model (Llama-3.2-1B scale)

    @Test("Medium (D=2048, 16L): decode latency — fused vs unfused")
    func mediumDecodeComparison() throws {
        let model = try buildInferenceModel(
            layerCount: mediumConfig.layerCount,
            hiddenSize: mediumConfig.hiddenSize,
            headCount: mediumConfig.headCount,
            kvHeadCount: mediumConfig.kvHeadCount,
            headDim: mediumConfig.headDim,
            intermediateSize: mediumConfig.intermediateSize
        )

        let fusedSteps = model.decode.steps
        let unfusedSteps = expandFusedSteps(model.decode.steps)

        // Prefill
        let prefillState: InferenceState
        do {
            var s = model.makeState()
            let prompt = MLXArray([Int32(0), Int32(1), Int32(2), Int32(3)]).reshaped(1, 4)
            let (_, ps) = model.prefill(tokenIDs: prompt, state: s)
            prefillState = ps
        }

        let decodeToken = MLXArray([Int32(5)]).reshaped(1, 1)

        // Fixed cache: don't accumulate, run unfused first
        let unfusedResult = benchmark(warmup: 10, iterations: 30) {
            var s = prefillState
            let logits = executeFlatSteps(unfusedSteps, input: decodeToken, state: &s)
            eval(logits)
        }

        let fusedResult = benchmark(warmup: 10, iterations: 30) {
            var s = prefillState
            let logits = executeFlatSteps(fusedSteps, input: decodeToken, state: &s)
            eval(logits)
        }

        let improvement = (unfusedResult.median - fusedResult.median) / unfusedResult.median * 100

        print("""
        [MEDIUM D=2048 16L decode]
          fused:   \(String(format: "%.2f", fusedResult.median)) ms  (steps: \(fusedSteps.count))
          unfused: \(String(format: "%.2f", unfusedResult.median)) ms  (steps: \(unfusedSteps.count))
          diff:    \(String(format: "%+.1f", improvement))%
        """)
    }

    @Test("Medium (D=2048, 16L): prefill 128 tokens")
    func mediumPrefill() throws {
        let model = try buildInferenceModel(
            layerCount: mediumConfig.layerCount,
            hiddenSize: mediumConfig.hiddenSize,
            headCount: mediumConfig.headCount,
            kvHeadCount: mediumConfig.kvHeadCount,
            headDim: mediumConfig.headDim,
            intermediateSize: mediumConfig.intermediateSize
        )

        let prompt = MLXArray((0..<128).map { Int32($0 % 32000) }).reshaped(1, 128)

        let result = benchmark(warmup: 3, iterations: 10) {
            let state = model.makeState()
            let (logits, _) = model.prefill(tokenIDs: prompt, state: state)
            eval(logits)
        }

        print("""
        [MEDIUM D=2048 16L prefill 128tok]
          median: \(String(format: "%.2f", result.median)) ms
          min:    \(String(format: "%.2f", result.min)) ms
          max:    \(String(format: "%.2f", result.max)) ms
        """)
    }

    // MARK: - PackedProjection A/B (Medium)

    @Test("Medium (D=2048, 16L): packed vs individual projections")
    func mediumPackedVsIndividual() throws {
        let D = mediumConfig.hiddenSize
        let H = mediumConfig.headCount
        let KVH = mediumConfig.kvHeadCount
        let hd = mediumConfig.headDim
        let inter = mediumConfig.intermediateSize

        let x = (MLXRandom.normal([1, 1, D]) * 0.1).asType(.float16)
        eval(x)

        // Build individual Q/K/V projections
        let wQ = (MLXRandom.normal([H * hd, D]) * 0.02).asType(.float16)
        let wK = (MLXRandom.normal([KVH * hd, D]) * 0.02).asType(.float16)
        let wV = (MLXRandom.normal([KVH * hd, D]) * 0.02).asType(.float16)
        eval(wQ, wK, wV)

        let qProj = LoweredProjection(weight: wQ)
        let kProj = LoweredProjection(weight: wK)
        let vProj = LoweredProjection(weight: wV)

        // Packed
        let packed = PackedProjection.pack([qProj, kProj, vProj])!

        // Benchmark: individual
        let individualResult = benchmark(warmup: 10, iterations: 50) {
            let q = qProj.apply(x)
            let k = kProj.apply(x)
            let v = vProj.apply(x)
            eval(q, k, v)
        }

        // Benchmark: packed
        let packedResult = benchmark(warmup: 10, iterations: 50) {
            let qkv = packed.apply(x)
            eval(qkv[0], qkv[1], qkv[2])
        }

        let improvement = (individualResult.median - packedResult.median) / individualResult.median * 100

        print("""
        [MEDIUM D=2048 QKV projection]
          individual (3 matmul): \(String(format: "%.3f", individualResult.median)) ms
          packed     (1 matmul): \(String(format: "%.3f", packedResult.median)) ms
          diff:                  \(String(format: "%+.1f", improvement))%
        """)

        // Gate+Up packing
        let wGate = (MLXRandom.normal([inter, D]) * 0.02).asType(.float16)
        let wUp = (MLXRandom.normal([inter, D]) * 0.02).asType(.float16)
        eval(wGate, wUp)

        let gateProj = LoweredProjection(weight: wGate)
        let upProj = LoweredProjection(weight: wUp)
        let gateUpPacked = PackedProjection.pack([gateProj, upProj])!

        let gateUpIndividualResult = benchmark(warmup: 10, iterations: 50) {
            let g = gateProj.apply(x)
            let u = upProj.apply(x)
            eval(g, u)
        }

        let gateUpPackedResult = benchmark(warmup: 10, iterations: 50) {
            let gu = gateUpPacked.apply(x)
            eval(gu[0], gu[1])
        }

        let guImprovement = (gateUpIndividualResult.median - gateUpPackedResult.median) / gateUpIndividualResult.median * 100

        print("""
        [MEDIUM D=2048 Gate+Up projection]
          individual (2 matmul): \(String(format: "%.3f", gateUpIndividualResult.median)) ms
          packed     (1 matmul): \(String(format: "%.3f", gateUpPackedResult.median)) ms
          diff:                  \(String(format: "%+.1f", guImprovement))%
        """)
    }

    // MARK: - Layer Scaling (Realistic)

    @Test("Layer scaling: D=896 decode latency vs layer count")
    func layerScalingRealistic() throws {
        let configs = [4, 8, 16, 24]
        var results: [(layers: Int, median: Double)] = []

        for layerCount in configs {
            let model = try buildInferenceModel(
                layerCount: layerCount,
                hiddenSize: smallConfig.hiddenSize,
                headCount: smallConfig.headCount,
                kvHeadCount: smallConfig.kvHeadCount,
                headDim: smallConfig.headDim,
                intermediateSize: smallConfig.intermediateSize
            )
            var state = model.makeState()

            let prompt = MLXArray([Int32(0), Int32(1)]).reshaped(1, 2)
            let (_, prefillState) = model.prefill(tokenIDs: prompt, state: state)
            state = prefillState

            let decodeToken = MLXArray([Int32(3)]).reshaped(1, 1)

            let result = benchmark(warmup: 5, iterations: 20) {
                let (logits, newState) = model.decode(tokenIDs: decodeToken, state: state)
                eval(logits)
                state = newState
            }

            results.append((layers: layerCount, median: result.median))
        }

        print("[LAYER SCALING D=896]")
        for r in results {
            print("  \(r.layers)L: \(String(format: "%.2f", r.median)) ms")
        }
        if results.count >= 2 {
            let ratio = results.last!.median / results.first!.median
            let layerRatio = Double(results.last!.layers) / Double(results.first!.layers)
            print("  ratio \(results.last!.layers)L/\(results.first!.layers)L = \(String(format: "%.2f", ratio))x (ideal \(String(format: "%.1f", layerRatio))x)")
        }
    }

    // MARK: - Correctness Guard

    @Test("Correctness: realistic model output is finite")
    func correctnessRealistic() throws {
        let model = try buildInferenceModel(
            layerCount: 4,
            hiddenSize: smallConfig.hiddenSize,
            headCount: smallConfig.headCount,
            kvHeadCount: smallConfig.kvHeadCount,
            headDim: smallConfig.headDim,
            intermediateSize: smallConfig.intermediateSize,
            vocabSize: 1000
        )
        let state = model.makeState()

        let prompt = MLXArray([Int32(0), Int32(1), Int32(2)]).reshaped(1, 3)
        let (prefillLogits, state2) = model.prefill(tokenIDs: prompt, state: state)
        eval(prefillLogits)
        #expect(prefillLogits.shape == [1, 3, 1000])

        let (decodeLogits, _) = model.decode(
            tokenIDs: MLXArray([Int32(5)]).reshaped(1, 1), state: state2)
        eval(decodeLogits)
        #expect(decodeLogits.shape == [1, 1, 1000])

        let vals = decodeLogits.reshaped(-1).asArray(Float.self)
        for v in vals { #expect(v.isFinite) }
    }
}
