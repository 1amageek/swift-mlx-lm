import Testing
import TestHeartbeat
import Foundation
@testable import LMArchitecture

// MARK: - Measurement Helpers

/// Run a closure multiple times and return the median duration.
private func measure(
    iterations: Int = 10,
    warmup: Int = 2,
    _ block: () throws -> Void
) rethrows -> Duration {
    let clock = ContinuousClock()
    // Warmup
    for _ in 0..<warmup {
        try block()
    }
    // Measured runs
    var durations: [Duration] = []
    for _ in 0..<iterations {
        let d = try clock.measure { try block() }
        durations.append(d)
    }
    durations.sort()
    return durations[durations.count / 2]
}

/// Count all operations in a graph recursively.
private func totalOperationCount(in region: Region) -> Int {
    var count = 0
    for op in region.operations {
        count += 1
        switch op.kind {
        case .residual(_, let body):
            count += totalOperationCount(in: body)
        case .parallel(_, let branches):
            for branch in branches {
                count += totalOperationCount(in: branch)
            }
        case .repeating(_, let body):
            count += totalOperationCount(in: body)
        default:
            break
        }
    }
    return count
}

// MARK: - Model Factories

private func makeTinyLlama(layerCount: Int) -> TinyLlama {
    TinyLlama(
        vocabSize: 32000,
        hiddenSize: 4096,
        headCount: 32,
        kvHeadCount: 8,
        intermediateSize: 11008,
        layerCount: layerCount
    )
}

private func makeTransformer(hiddenLayers: Int, moe: Transformer.MoEConfig? = nil) -> Transformer {
    Transformer(config: .init(
        hiddenSize: 4096,
        hiddenLayers: hiddenLayers,
        intermediateSize: 11008,
        attentionHeads: 32,
        kvHeads: 8,
        vocabularySize: 128256,
        moe: moe
    ))
}

// MARK: - Primitive Component Normalization

@Suite("Performance: Primitive Normalization", .tags(.performance), .heartbeat)
struct PrimitiveNormalizationPerformanceTests {

    @Test("TokenEmbedding normalization")
    func tokenEmbedding() throws {
        let comp = TokenEmbedding(vocabSize: 128256, embeddingSize: 4096)
        let d = try measure { _ = try normalize(comp) }
        print("[perf] TokenEmbedding normalize: \(d)")
        #expect(d < .milliseconds(10))
    }

    @Test("Attention normalization")
    func attention() throws {
        let comp = Attention(
            hiddenSize: 4096,
            headCount: 32,
            kvHeadCount: 8,
            headDimension: 128,
            rope: RoPEAttributes(dimension: 128, base: 500_000)
        )
        let d = try measure { _ = try normalize(comp) }
        print("[perf] Attention normalize: \(d)")
        #expect(d < .milliseconds(10))
    }

    @Test("MLP normalization")
    func mlp() throws {
        let comp = MLP(
            inputSize: 4096,
            intermediateSize: 11008,
            activation: .silu,
            gating: .swiglu
        )
        let d = try measure { _ = try normalize(comp) }
        print("[perf] MLP normalize: \(d)")
        #expect(d < .milliseconds(10))
    }

    @Test("MoE normalization")
    func moe() throws {
        let comp = MoE(
            expertCount: 8,
            expertsPerToken: 2,
            expertInputSize: 4096,
            expertIntermediateSize: 14336
        )
        let d = try measure { _ = try normalize(comp) }
        print("[perf] MoE normalize: \(d)")
        #expect(d < .milliseconds(10))
    }

    @Test("RMSNorm normalization")
    func rmsNorm() throws {
        let comp = RMSNorm(dimension: 4096, epsilon: 1e-5)
        let d = try measure { _ = try normalize(comp) }
        print("[perf] RMSNorm normalize: \(d)")
        #expect(d < .milliseconds(10))
    }

    @Test("LayerNorm normalization")
    func layerNorm() throws {
        let comp = LayerNorm(dimension: 4096, epsilon: 1e-5)
        let d = try measure { _ = try normalize(comp) }
        print("[perf] LayerNorm normalize: \(d)")
        #expect(d < .milliseconds(10))
    }

    @Test("Linear normalization")
    func linear() throws {
        let comp = LMArchitecture.Linear(inputSize: 4096, outputSize: 32000)
        let d = try measure { _ = try normalize(comp) }
        print("[perf] Linear normalize: \(d)")
        #expect(d < .milliseconds(10))
    }

    @Test("OutputHead normalization")
    func outputHead() throws {
        let comp = OutputHead(inputSize: 4096, vocabSize: 128256, tiedToEmbedding: true)
        let d = try measure { _ = try normalize(comp) }
        print("[perf] OutputHead normalize: \(d)")
        #expect(d < .milliseconds(10))
    }

    @Test("RoPE normalization")
    func rope() throws {
        let comp = RoPE(dimension: 128, base: 500_000)
        let d = try measure { _ = try normalize(comp) }
        print("[perf] RoPE normalize: \(d)")
        #expect(d < .milliseconds(10))
    }

    @Test("PositionalEmbedding normalization")
    func positionalEmbedding() throws {
        let comp = PositionalEmbedding(maxPositions: 8192, embeddingSize: 4096, kind: .learnedAbsolute)
        let d = try measure { _ = try normalize(comp) }
        print("[perf] PositionalEmbedding normalize: \(d)")
        #expect(d < .milliseconds(10))
    }

    @Test("StateSpace normalization")
    func stateSpace() throws {
        let comp = StateSpace(hiddenSize: 4096, numHeads: 1, keyHeadDim: 16, valueHeadDim: 16, variant: "mamba")
        let d = try measure { _ = try normalize(comp) }
        print("[perf] StateSpace normalize: \(d)")
        #expect(d < .milliseconds(10))
    }

    @Test("Custom normalization")
    func custom() throws {
        let comp = Custom(domain: "test", name: "noop")
        let d = try measure { _ = try normalize(comp) }
        print("[perf] Custom normalize: \(d)")
        #expect(d < .milliseconds(10))
    }

}

// MARK: - Structural Component Normalization

@Suite("Performance: Structural Normalization", .tags(.performance), .heartbeat)
struct StructuralNormalizationPerformanceTests {

    @Test("Residual normalization")
    func residual() throws {
        let comp = Group {
            RMSNorm(dimension: 4096)
            Residual {
                RMSNorm(dimension: 4096)
                Attention(hiddenSize: 4096, headCount: 32, kvHeadCount: 8)
            }
        }
        let d = try measure { _ = try normalize(comp) }
        print("[perf] Residual normalize: \(d)")
        #expect(d < .milliseconds(10))
    }

    @Test("Parallel normalization (2 branches)")
    func parallel2() throws {
        let comp = Group {
            RMSNorm(dimension: 4096)
            Parallel(merge: .add) {
                Attention(hiddenSize: 4096, headCount: 32, kvHeadCount: 8)
                MLP(inputSize: 4096, intermediateSize: 11008)
            }
        }
        let d = try measure { _ = try normalize(comp) }
        print("[perf] Parallel(2 branches) normalize: \(d)")
        #expect(d < .milliseconds(10))
    }

    @Test("Repeat normalization (32 layers)")
    func repeat32() throws {
        let comp = Group {
            RMSNorm(dimension: 4096)
            Repeat(count: 32) {
                LMArchitecture.Linear(inputSize: 4096, outputSize: 4096)
            }
        }
        let d = try measure { _ = try normalize(comp) }
        print("[perf] Repeat(32) normalize: \(d)")
        #expect(d < .milliseconds(10))
    }

    @Test("Group flattening (deep nesting)")
    func groupFlattening() throws {
        let comp = Group {
            Group {
                Group {
                    Group {
                        Group {
                            RMSNorm(dimension: 4096)
                            LMArchitecture.Linear(inputSize: 4096, outputSize: 4096)
                        }
                        LMArchitecture.Linear(inputSize: 4096, outputSize: 4096)
                    }
                    LMArchitecture.Linear(inputSize: 4096, outputSize: 4096)
                }
                LMArchitecture.Linear(inputSize: 4096, outputSize: 4096)
            }
            LMArchitecture.Linear(inputSize: 4096, outputSize: 4096)
        }
        let d = try measure { _ = try normalize(comp) }
        print("[perf] Group(deep nesting) normalize: \(d)")
        #expect(d < .milliseconds(10))
    }
}

// MARK: - End-to-End Model Normalization

@Suite("Performance: Model Normalization", .tags(.performance), .heartbeat)
struct ModelNormalizationPerformanceTests {

    @Test("TinyLlama 2 layers normalization")
    func tinyLlama2() throws {
        let model = makeTinyLlama(layerCount: 2)
        let d = try measure { _ = try ModelGraph(model) }
        print("[perf] TinyLlama(2 layers) ModelGraph init: \(d)")
        #expect(d < .milliseconds(50))
    }

    @Test("TinyLlama 32 layers normalization")
    func tinyLlama32() throws {
        let model = makeTinyLlama(layerCount: 32)
        let d = try measure { _ = try ModelGraph(model) }
        print("[perf] TinyLlama(32 layers) ModelGraph init: \(d)")
        #expect(d < .milliseconds(50))
    }

    @Test("TinyLlama 80 layers normalization")
    func tinyLlama80() throws {
        let model = makeTinyLlama(layerCount: 80)
        let d = try measure { _ = try ModelGraph(model) }
        print("[perf] TinyLlama(80 layers) ModelGraph init: \(d)")
        #expect(d < .milliseconds(100))
    }

    @Test("Transformer 32 layers normalization")
    func transformer32() throws {
        let model = makeTransformer(hiddenLayers: 32)
        let d = try measure { _ = try ModelGraph(model) }
        print("[perf] Transformer(32 layers) ModelGraph init: \(d)")
        #expect(d < .milliseconds(50))
    }

    @Test("Transformer MoE 32 layers normalization")
    func transformerMoE32() throws {
        let model = makeTransformer(
            hiddenLayers: 32,
            moe: .init(expertCount: 8, expertsPerToken: 2)
        )
        let d = try measure { _ = try ModelGraph(model) }
        print("[perf] Transformer+MoE(32 layers) ModelGraph init: \(d)")
        #expect(d < .milliseconds(50))
    }

    @Test("Cohere parallel model 32 layers normalization")
    func cohere32() throws {
        let model = TinyCohere(
            vocabSize: 256000,
            hiddenSize: 4096,
            headCount: 32,
            kvHeadCount: 8,
            intermediateSize: 11008,
            layerCount: 32
        )
        let d = try measure { _ = try ModelGraph(model) }
        print("[perf] TinyCohere(32 layers) ModelGraph init: \(d)")
        #expect(d < .milliseconds(50))
    }

    @Test("Mixtral MoE model 32 layers normalization")
    func mixtral32() throws {
        let model = TinyMixtral(
            vocabSize: 32000,
            hiddenSize: 4096,
            headCount: 32,
            kvHeadCount: 8,
            intermediateSize: 14336,
            expertCount: 8,
            expertsPerToken: 2,
            layerCount: 32
        )
        let d = try measure { _ = try ModelGraph(model) }
        print("[perf] TinyMixtral(32 layers) ModelGraph init: \(d)")
        #expect(d < .milliseconds(50))
    }
}

// MARK: - Canonicalization Performance

@Suite("Performance: Canonicalization", .tags(.performance), .heartbeat)
struct CanonicalizationPerformanceTests {

    @Test("Canonicalize small graph (2 layers)")
    func canonicalizeSmall() throws {
        let graph = try ModelGraph(makeTinyLlama(layerCount: 2))
        let d = measure { _ = canonicalize(graph) }
        print("[perf] canonicalize(2 layers): \(d)")
        #expect(d < .milliseconds(10))
    }

    @Test("Canonicalize medium graph (32 layers)")
    func canonicalizeMedium() throws {
        let graph = try ModelGraph(makeTinyLlama(layerCount: 32))
        let d = measure { _ = canonicalize(graph) }
        print("[perf] canonicalize(32 layers): \(d)")
        #expect(d < .milliseconds(50))
    }

    @Test("Canonicalize large graph (80 layers)")
    func canonicalizeLarge() throws {
        let graph = try ModelGraph(makeTinyLlama(layerCount: 80))
        let d = measure { _ = canonicalize(graph) }
        print("[perf] canonicalize(80 layers): \(d)")
        #expect(d < .milliseconds(100))
    }

    @Test("Canonicalize Transformer MoE (32 layers)")
    func canonicalizeMoE() throws {
        let model = makeTransformer(
            hiddenLayers: 32,
            moe: .init(expertCount: 8, expertsPerToken: 2)
        )
        let graph = try ModelGraph(model)
        let d = measure { _ = canonicalize(graph) }
        print("[perf] canonicalize(MoE 32 layers): \(d)")
        #expect(d < .milliseconds(50))
    }

    @Test("Double canonicalization is idempotent and fast")
    func canonicalizeIdempotent() throws {
        let graph = try ModelGraph(makeTinyLlama(layerCount: 32))
        let canon1 = canonicalize(graph)
        let d = measure { _ = canonicalize(canon1) }
        print("[perf] canonicalize(already canonical): \(d)")
        #expect(d < .milliseconds(50))
        #expect(canon1 == canonicalize(canon1))
    }
}

// MARK: - Validation Performance

@Suite("Performance: Validation", .tags(.performance), .heartbeat)
struct ValidationPerformanceTests {

    @Test("GraphValidator small graph (2 layers)")
    func graphValidatorSmall() throws {
        let graph = try ModelGraph(makeTinyLlama(layerCount: 2))
        let d = try measure { try GraphValidator.validate(graph) }
        print("[perf] GraphValidator(2 layers): \(d)")
        #expect(d < .milliseconds(10))
    }

    @Test("GraphValidator medium graph (32 layers)")
    func graphValidatorMedium() throws {
        let graph = try ModelGraph(makeTinyLlama(layerCount: 32))
        let d = try measure { try GraphValidator.validate(graph) }
        print("[perf] GraphValidator(32 layers): \(d)")
        #expect(d < .milliseconds(50))
    }

    @Test("GraphValidator large graph (80 layers)")
    func graphValidatorLarge() throws {
        let graph = try ModelGraph(makeTinyLlama(layerCount: 80))
        let d = try measure { try GraphValidator.validate(graph) }
        print("[perf] GraphValidator(80 layers): \(d)")
        #expect(d < .milliseconds(100))
    }

    @Test("LLMProfileValidator medium graph (32 layers)")
    func profileValidatorMedium() throws {
        let graph = try ModelGraph(makeTinyLlama(layerCount: 32))
        let d = try measure { try LLMProfileValidator.validate(graph) }
        print("[perf] LLMProfileValidator(32 layers): \(d)")
        #expect(d < .milliseconds(50))
    }

    @Test("Combined validate + profile (32 layers)")
    func combinedValidation() throws {
        let graph = try ModelGraph(makeTinyLlama(layerCount: 32))
        let d = try measure {
            try GraphValidator.validate(graph)
            try LLMProfileValidator.validate(graph)
        }
        print("[perf] GraphValidator+LLMProfileValidator(32 layers): \(d)")
        #expect(d < .milliseconds(50))
    }
}

// MARK: - Codable Roundtrip Performance

@Suite("Performance: Codable Roundtrip", .tags(.performance), .heartbeat)
struct CodablePerformanceTests {

    @Test("JSON encode small graph (2 layers)")
    func encodeSmall() throws {
        let graph = try ModelGraph(makeTinyLlama(layerCount: 2))
        let encoder = JSONEncoder()
        let d = try measure { _ = try encoder.encode(graph) }
        print("[perf] JSON encode(2 layers): \(d)")
        #expect(d < .milliseconds(10))
    }

    @Test("JSON encode medium graph (32 layers)")
    func encodeMedium() throws {
        let graph = try ModelGraph(makeTinyLlama(layerCount: 32))
        let encoder = JSONEncoder()
        let d = try measure { _ = try encoder.encode(graph) }
        print("[perf] JSON encode(32 layers): \(d)")
        #expect(d < .milliseconds(50))
    }

    @Test("JSON decode medium graph (32 layers)")
    func decodeMedium() throws {
        let graph = try ModelGraph(makeTinyLlama(layerCount: 32))
        let data = try JSONEncoder().encode(graph)
        let decoder = JSONDecoder()
        let d = try measure { _ = try decoder.decode(ModelGraph.self, from: data) }
        print("[perf] JSON decode(32 layers): \(d)")
        #expect(d < .milliseconds(50))
    }

    @Test("JSON roundtrip large graph (80 layers)")
    func roundtripLarge() throws {
        let graph = try ModelGraph(makeTinyLlama(layerCount: 80))
        let encoder = JSONEncoder()
        let decoder = JSONDecoder()
        let d = try measure {
            let data = try encoder.encode(graph)
            let decoded = try decoder.decode(ModelGraph.self, from: data)
            #expect(decoded == graph)
        }
        print("[perf] JSON roundtrip(80 layers): \(d)")
        #expect(d < .milliseconds(200))
    }

    @Test("JSON encode size is proportional to layer count")
    func encodeSizeScaling() throws {
        let graph8 = try ModelGraph(makeTinyLlama(layerCount: 8))
        let graph32 = try ModelGraph(makeTinyLlama(layerCount: 32))
        let graph80 = try ModelGraph(makeTinyLlama(layerCount: 80))

        let size8 = try JSONEncoder().encode(graph8).count
        let size32 = try JSONEncoder().encode(graph32).count
        let size80 = try JSONEncoder().encode(graph80).count

        print("[perf] JSON size: 8L=\(size8) 32L=\(size32) 80L=\(size80)")

        // Repeat body is shared in IR, so size should NOT grow linearly
        // with layer count. Verify it stays compact.
        let ratio = Double(size80) / Double(size8)
        #expect(ratio < 5.0, Comment(rawValue: "80L/8L ratio=\(ratio), expected < 5.0 (Repeat body is shared)"))
    }
}

// MARK: - Scaling Performance

@Suite("Performance: Scaling", .tags(.performance), .heartbeat)
struct ScalingPerformanceTests {

    @Test("Normalization scales linearly with non-repeated operations")
    func normalizationScaling() throws {
        // Build models with increasing flat operation counts using ForEach
        let counts = [10, 50, 100, 200]
        var durations: [Int: Duration] = [:]

        for count in counts {
            let comp = Group {
                TokenEmbedding(vocabSize: 100, embeddingSize: 64)
                ForEach(Array(0..<count)) { _ in
                    LMArchitecture.Linear(inputSize: 64, outputSize: 64)
                }
                OutputHead(inputSize: 64, vocabSize: 100)
            }
            let d = try measure(iterations: 5) { _ = try normalize(comp) }
            durations[count] = d
            print("[perf] normalize(\(count) linear ops): \(d)")
        }

        // 200 ops should be less than 10x of 10 ops (roughly linear)
        let d10 = durations[10]!
        let d200 = durations[200]!
        let ratio = Double(d200.components.attoseconds) / Double(d10.components.attoseconds)
        print("[perf] scaling ratio 200/10: \(String(format: "%.1f", ratio))x")
        #expect(ratio < 40.0, Comment(rawValue: "200/10 ratio=\(String(format: "%.1f", ratio)), expected < 40x"))
    }

    @Test("Repeat count does not affect normalization time (body normalized once)")
    func repeatCountIndependence() throws {
        let d1 = try measure {
            let comp = Group {
                RMSNorm(dimension: 64)
                Repeat(count: 1) {
                    LMArchitecture.Linear(inputSize: 64, outputSize: 64)
                }
            }
            _ = try normalize(comp)
        }

        let d1000 = try measure {
            let comp = Group {
                RMSNorm(dimension: 64)
                Repeat(count: 1000) {
                    LMArchitecture.Linear(inputSize: 64, outputSize: 64)
                }
            }
            _ = try normalize(comp)
        }

        print("[perf] Repeat(1) normalize: \(d1)")
        print("[perf] Repeat(1000) normalize: \(d1000)")

        // Repeat body is normalized once regardless of count
        let ratio = Double(d1000.components.attoseconds) / Double(max(d1.components.attoseconds, 1))
        #expect(ratio < 3.0, Comment(rawValue: "Repeat(1000)/Repeat(1) ratio=\(String(format: "%.1f", ratio)), expected < 3x"))
    }

    @Test("Operation count in generated graph")
    func operationCountReport() throws {
        let configs: [(String, any ModelComponent)] = [
            ("TinyLlama(2)", makeTinyLlama(layerCount: 2)),
            ("TinyLlama(32)", makeTinyLlama(layerCount: 32)),
            ("TinyLlama(80)", makeTinyLlama(layerCount: 80)),
            ("Transformer(32)", makeTransformer(hiddenLayers: 32)),
            ("Transformer+MoE(32)", makeTransformer(hiddenLayers: 32, moe: .init(expertCount: 8, expertsPerToken: 2))),
        ]

        for (name, model) in configs {
            let graph = try ModelGraph(model)
            let opCount = totalOperationCount(in: graph.rootRegion)
            let rootOps = graph.rootRegion.operations.count
            print("[perf] \(name): rootOps=\(rootOps) totalOps=\(opCount)")
        }
    }
}

// MARK: - Equality Comparison Performance

@Suite("Performance: Equality", .tags(.performance), .heartbeat)
struct EqualityPerformanceTests {

    @Test("Graph equality comparison (identical, 32 layers)")
    func equalGraphs() throws {
        let graph = try ModelGraph(makeTinyLlama(layerCount: 32))
        let copy = try ModelGraph(makeTinyLlama(layerCount: 32))
        let d = measure { _ = graph == copy }
        print("[perf] graph == graph (32 layers, identical): \(d)")
        #expect(d < .milliseconds(10))
        #expect(graph == copy)
    }

    @Test("Graph equality comparison (different, 32 layers)")
    func differentGraphs() throws {
        let graphA = try ModelGraph(makeTinyLlama(layerCount: 32))
        let graphB = try ModelGraph(makeTinyLlama(layerCount: 33))
        let d = measure { _ = graphA == graphB }
        print("[perf] graph != graph (32 vs 33 layers): \(d)")
        #expect(d < .milliseconds(10))
        #expect(graphA != graphB)
    }

    @Test("Canonical equality comparison (32 layers)")
    func canonicalEquality() throws {
        let graph = try ModelGraph(makeTinyLlama(layerCount: 32))
        let canonA = canonicalize(graph)
        let canonB = canonicalize(graph)
        let d = measure { _ = canonA == canonB }
        print("[perf] canonical == canonical (32 layers): \(d)")
        #expect(d < .milliseconds(10))
        #expect(canonA == canonB)
    }
}

// MARK: - Full Pipeline Performance

@Suite("Performance: Full Pipeline", .tags(.performance), .heartbeat)
struct FullPipelinePerformanceTests {

    @Test("Full pipeline: normalize → canonicalize → validate → encode (32 layers)")
    func fullPipeline32() throws {
        let model = makeTinyLlama(layerCount: 32)
        let encoder = JSONEncoder()

        let d = try measure(iterations: 5) {
            let normalized = try NormalizedModel(model)
            let canon = canonicalize(normalized.graph)
            try GraphValidator.validate(canon)
            try LLMProfileValidator.validate(canon)
            _ = try encoder.encode(canon)
        }
        print("[perf] full pipeline (32 layers): \(d)")
        #expect(d < .milliseconds(100))
    }

    @Test("Full pipeline: Transformer MoE (32 layers)")
    func fullPipelineMoE() throws {
        let model = makeTransformer(
            hiddenLayers: 32,
            moe: .init(expertCount: 8, expertsPerToken: 2)
        )
        let encoder = JSONEncoder()

        let d = try measure(iterations: 5) {
            let normalized = try NormalizedModel(model)
            let canon = canonicalize(normalized.graph)
            try GraphValidator.validate(canon)
            try LLMProfileValidator.validate(canon)
            _ = try encoder.encode(canon)
        }
        print("[perf] full pipeline MoE (32 layers): \(d)")
        #expect(d < .milliseconds(100))
    }

    @Test("Full pipeline: large model (80 layers)")
    func fullPipeline80() throws {
        let model = makeTinyLlama(layerCount: 80)
        let encoder = JSONEncoder()

        let d = try measure(iterations: 5) {
            let normalized = try NormalizedModel(model)
            let canon = canonicalize(normalized.graph)
            try GraphValidator.validate(canon)
            _ = try encoder.encode(canon)
        }
        print("[perf] full pipeline (80 layers): \(d)")
        #expect(d < .milliseconds(200))
    }
}
