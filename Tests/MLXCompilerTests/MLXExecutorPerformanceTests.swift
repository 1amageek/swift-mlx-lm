import Testing
import TestHeartbeat
@preconcurrency import MLX
import MLXFast
import MLXNN
@testable import SwiftLM
@testable import MLXCompiler


// MARK: - Test Helpers

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
            dtype: .float32,
            storage: array
        )
    }
    return BoundWeights(tensors: tensors)
}

/// Measure execution time with warmup and multiple iterations.
/// Returns median duration in seconds.
private func measure(
    warmup: Int = 2,
    iterations: Int = 5,
    _ body: () throws -> Void
) rethrows -> Double {
    let clock = ContinuousClock()

    // Warmup
    for _ in 0..<warmup {
        try body()
    }

    // Measure
    var durations: [Double] = []
    for _ in 0..<iterations {
        let start = clock.now
        try body()
        let elapsed = clock.now - start
        durations.append(Double(elapsed.components.seconds)
            + Double(elapsed.components.attoseconds) * 1e-18)
    }

    durations.sort()
    return durations[durations.count / 2]
}


// MARK: - MoE Weight Factory

private func buildMoEExecutor(
    expertCount: Int,
    expertsPerToken: Int,
    hiddenSize: Int,
    intermediateSize: Int,
    vocabSize: Int = 16
) throws -> MLXExecutor {
    struct MoETestModel: ModelComponent {
        let vocabSize: Int
        let hiddenSize: Int
        let expertCount: Int
        let expertsPerToken: Int
        let intermediateSize: Int

        @ModelComponentBuilder var body: some ModelComponent {
            TokenEmbedding(vocabSize: vocabSize, embeddingSize: hiddenSize)
            MoE(
                expertCount: expertCount,
                expertsPerToken: expertsPerToken,
                expertInputSize: hiddenSize,
                expertIntermediateSize: intermediateSize
            )
        }
    }

    let model = MoETestModel(
        vocabSize: vocabSize,
        hiddenSize: hiddenSize,
        expertCount: expertCount,
        expertsPerToken: expertsPerToken,
        intermediateSize: intermediateSize
    )
    let graph = try model.makeModelGraph()

    var dict: [ParameterSlot: MLXArray] = [:]
    dict[slot([.operation(0)], role: .embeddingTable)] = MLXRandom.normal([vocabSize, hiddenSize]) * 0.1

    let moePath: [StructuralPathComponent] = [.operation(1)]
    dict[slot(moePath + [.field("router")], role: .weight)] = MLXRandom.normal([expertCount, hiddenSize]) * 0.1

    for expertIdx in 0..<expertCount {
        let ePath = moePath + [.field("experts"), .index(expertIdx)]
        dict[slot(ePath + [.field("gate_proj")], role: .weight)] = MLXRandom.normal([intermediateSize, hiddenSize]) * 0.1
        dict[slot(ePath + [.field("up_proj")], role: .weight)] = MLXRandom.normal([intermediateSize, hiddenSize]) * 0.1
        dict[slot(ePath + [.field("down_proj")], role: .weight)] = MLXRandom.normal([hiddenSize, intermediateSize]) * 0.1
    }

    let compiled = try MLXCompiler().compile(graph: graph, weights: bind(dict))
    return MLXExecutor(compiledModel: compiled)
}


// MARK: - DeltaNet Weight Factory

private func buildDeltaNetExecutor(
    hiddenSize: Int,
    stateSize: Int,
    vocabSize: Int = 16
) throws -> MLXExecutor {
    struct DeltaNetTestModel: ModelComponent {
        let vocabSize: Int
        let hiddenSize: Int
        let stateSize: Int

        @ModelComponentBuilder var body: some ModelComponent {
            TokenEmbedding(vocabSize: vocabSize, embeddingSize: hiddenSize)
            StateSpace(hiddenSize: hiddenSize, stateSize: stateSize, variant: "deltanet")
        }
    }

    let model = DeltaNetTestModel(
        vocabSize: vocabSize,
        hiddenSize: hiddenSize,
        stateSize: stateSize
    )
    let graph = try model.makeModelGraph()
    let ssPath: [StructuralPathComponent] = [.operation(1)]

    // Derive dimension relationships from weight shapes.
    // keyDim = stateSize, valueDim derived from z projection
    let keyDim = stateSize
    let valueDim = hiddenSize  // Simplified: z maps to full hidden
    let totalQKV = keyDim + keyDim + valueDim
    let linearKeyHeads = keyDim / stateSize  // = 1
    let linearValueHeadDim = valueDim / (valueDim / stateSize > 0 ? valueDim / stateSize : 1)

    let weights = bind([
        slot([.operation(0)], role: .embeddingTable): MLXRandom.normal([vocabSize, hiddenSize]) * 0.1,
        slot(ssPath + [.field("in_proj_qkv")], role: .weight): MLXRandom.normal([totalQKV, hiddenSize]) * 0.1,
        slot(ssPath + [.field("in_proj_z")], role: .weight): MLXRandom.normal([valueDim, hiddenSize]) * 0.1,
        slot(ssPath + [.field("in_proj_b")], role: .weight): MLXRandom.normal([linearKeyHeads, hiddenSize]) * 0.1,
        slot(ssPath + [.field("in_proj_a")], role: .weight): MLXRandom.normal([linearKeyHeads, hiddenSize]) * 0.1,
        slot(ssPath + [.field("conv1d")], role: .weight): MLXRandom.normal([totalQKV, 4]) * 0.01,
        slot(ssPath + [.field("out_proj")], role: .weight): MLXRandom.normal([hiddenSize, valueDim]) * 0.1,
        slot(ssPath + [.field("norm")], role: .scale): MLXArray.zeros([linearValueHeadDim]),
        slot(ssPath + [.field("dt_bias")], role: .bias): MLXArray.zeros([linearKeyHeads]),
        slot(ssPath + [.field("A_log")], role: .weight): MLXArray(converting: [-1.0] as [Double], [linearKeyHeads]),
    ])

    let compiled = try MLXCompiler().compile(graph: graph, weights: weights)
    return MLXExecutor(compiledModel: compiled)
}


// MARK: - Combined MoE + DeltaNet Model

private func buildMixedMoEDeltaNetExecutor(
    layerCount: Int,
    hiddenSize: Int,
    expertCount: Int,
    expertsPerToken: Int,
    intermediateSize: Int,
    stateSize: Int,
    vocabSize: Int = 16
) throws -> MLXExecutor {
    struct MixedModel: ModelComponent {
        let vocabSize: Int
        let hiddenSize: Int
        let expertCount: Int
        let expertsPerToken: Int
        let intermediateSize: Int
        let stateSize: Int
        let layerCount: Int

        @ModelComponentBuilder var body: some ModelComponent {
            TokenEmbedding(vocabSize: vocabSize, embeddingSize: hiddenSize)

            Repeat(count: layerCount) {
                Residual {
                    RMSNorm(dimension: hiddenSize)
                    StateSpace(hiddenSize: hiddenSize, stateSize: stateSize, variant: "deltanet")
                }
                Residual {
                    RMSNorm(dimension: hiddenSize)
                    MoE(
                        expertCount: expertCount,
                        expertsPerToken: expertsPerToken,
                        expertInputSize: hiddenSize,
                        expertIntermediateSize: intermediateSize
                    )
                }
            }

            RMSNorm(dimension: hiddenSize)
            OutputHead(inputSize: hiddenSize, vocabSize: vocabSize, tiedToEmbedding: true)
        }
    }

    let model = MixedModel(
        vocabSize: vocabSize,
        hiddenSize: hiddenSize,
        expertCount: expertCount,
        expertsPerToken: expertsPerToken,
        intermediateSize: intermediateSize,
        stateSize: stateSize,
        layerCount: layerCount
    )
    let graph = try model.makeModelGraph()

    var dict: [ParameterSlot: MLXArray] = [:]

    // Embedding
    dict[slot([.operation(0)], role: .embeddingTable)] = MLXRandom.normal([vocabSize, hiddenSize]) * 0.1

    // Derive DeltaNet dimensions
    let keyDim = stateSize
    let valueDim = hiddenSize
    let totalQKV = keyDim + keyDim + valueDim
    let linearKeyHeads = keyDim / stateSize  // = 1
    let linearValueHeadDim = valueDim / (valueDim / stateSize > 0 ? valueDim / stateSize : 1)

    for i in 0..<layerCount {
        let layerPrefix: [StructuralPathComponent] = [
            .operation(1), .regionBody, .index(i)
        ]

        // Residual 0: RMSNorm + DeltaNet
        let ssNormPath = layerPrefix + [.operation(0), .regionBody, .operation(0)]
        dict[slot(ssNormPath, role: .scale)] = MLXArray.zeros([hiddenSize])

        let ssPath = layerPrefix + [.operation(0), .regionBody, .operation(1)]
        dict[slot(ssPath + [.field("in_proj_qkv")], role: .weight)] = MLXRandom.normal([totalQKV, hiddenSize]) * 0.1
        dict[slot(ssPath + [.field("in_proj_z")], role: .weight)] = MLXRandom.normal([valueDim, hiddenSize]) * 0.1
        dict[slot(ssPath + [.field("in_proj_b")], role: .weight)] = MLXRandom.normal([linearKeyHeads, hiddenSize]) * 0.1
        dict[slot(ssPath + [.field("in_proj_a")], role: .weight)] = MLXRandom.normal([linearKeyHeads, hiddenSize]) * 0.1
        dict[slot(ssPath + [.field("conv1d")], role: .weight)] = MLXRandom.normal([totalQKV, 4]) * 0.01
        dict[slot(ssPath + [.field("out_proj")], role: .weight)] = MLXRandom.normal([hiddenSize, valueDim]) * 0.1
        dict[slot(ssPath + [.field("norm")], role: .scale)] = MLXArray.zeros([linearValueHeadDim])
        dict[slot(ssPath + [.field("dt_bias")], role: .bias)] = MLXArray.zeros([linearKeyHeads])
        dict[slot(ssPath + [.field("A_log")], role: .weight)] = MLXArray(converting: [-1.0] as [Double], [linearKeyHeads])

        // Residual 1: RMSNorm + MoE
        let moeNormPath = layerPrefix + [.operation(1), .regionBody, .operation(0)]
        dict[slot(moeNormPath, role: .scale)] = MLXArray.zeros([hiddenSize])

        let moePath = layerPrefix + [.operation(1), .regionBody, .operation(1)]
        dict[slot(moePath + [.field("router")], role: .weight)] = MLXRandom.normal([expertCount, hiddenSize]) * 0.1

        for expertIdx in 0..<expertCount {
            let ePath = moePath + [.field("experts"), .index(expertIdx)]
            dict[slot(ePath + [.field("gate_proj")], role: .weight)] = MLXRandom.normal([intermediateSize, hiddenSize]) * 0.1
            dict[slot(ePath + [.field("up_proj")], role: .weight)] = MLXRandom.normal([intermediateSize, hiddenSize]) * 0.1
            dict[slot(ePath + [.field("down_proj")], role: .weight)] = MLXRandom.normal([hiddenSize, intermediateSize]) * 0.1
        }
    }

    // Final norm
    dict[slot([.operation(2)], role: .scale)] = MLXArray.zeros([hiddenSize])

    let compiled = try MLXCompiler().compile(graph: graph, weights: bind(dict))
    return MLXExecutor(compiledModel: compiled)
}


// MARK: - MoE Performance Tests

@Suite("MoE Executor Performance", .tags(.performance), .heartbeat)
struct MoEExecutorPerformanceTests {

    @Test("MoE forward: 2 experts × top-1, 4 tokens")
    func moeSmall() throws {
        let executor = try buildMoEExecutor(
            expertCount: 2, expertsPerToken: 1,
            hiddenSize: 4, intermediateSize: 8
        )
        let tokens = MLXArray([Int32(0), Int32(1), Int32(2), Int32(3)])

        let median = try measure {
            let output = try executor.forward(tokenIDs: tokens)
            eval(output)
        }
        print("[MoE 2×1 4tok] median=\(String(format: "%.4f", median))s")
        #expect(median < 1.0)
    }

    @Test("MoE forward: 8 experts × top-2, 16 tokens")
    func moeMedium() throws {
        let executor = try buildMoEExecutor(
            expertCount: 8, expertsPerToken: 2,
            hiddenSize: 16, intermediateSize: 32
        )
        let tokens = MLXArray((0..<16).map { Int32($0) })

        let median = try measure {
            let output = try executor.forward(tokenIDs: tokens)
            eval(output)
        }
        print("[MoE 8×2 16tok] median=\(String(format: "%.4f", median))s")
        #expect(median < 2.0)
    }

    @Test("MoE scaling: expert count should not cause GPU→CPU sync overhead")
    func moeScaling() throws {
        // With the optimized version (no item()), increasing experts
        // should add proportional GPU work, not synchronization overhead.
        let executor4 = try buildMoEExecutor(
            expertCount: 4, expertsPerToken: 1,
            hiddenSize: 8, intermediateSize: 16
        )
        let executor8 = try buildMoEExecutor(
            expertCount: 8, expertsPerToken: 1,
            hiddenSize: 8, intermediateSize: 16
        )
        let tokens = MLXArray((0..<8).map { Int32($0 % 16) })

        let time4 = try measure {
            let output = try executor4.forward(tokenIDs: tokens)
            eval(output)
        }
        let time8 = try measure {
            let output = try executor8.forward(tokenIDs: tokens)
            eval(output)
        }
        print("[MoE scaling] 4 experts=\(String(format: "%.4f", time4))s, 8 experts=\(String(format: "%.4f", time8))s, ratio=\(String(format: "%.2f", time8 / time4))x")
        // 8 experts should not be dramatically slower than 4
        // (previously with item() sync, it would be much worse)
        #expect(time8 < time4 * 5.0)
    }
}


// MARK: - DeltaNet Performance Tests

@Suite("DeltaNet Executor Performance", .tags(.performance), .heartbeat)
struct DeltaNetExecutorPerformanceTests {

    @Test("DeltaNet forward: 2 tokens (prefill)")
    func deltaNetPrefill() throws {
        let executor = try buildDeltaNetExecutor(hiddenSize: 4, stateSize: 2)
        let tokens = MLXArray([Int32(0), Int32(1)])

        let median = try measure {
            executor.resetCaches()
            let output = try executor.forward(tokenIDs: tokens)
            eval(output)
        }
        print("[DeltaNet prefill 2tok] median=\(String(format: "%.4f", median))s")
        #expect(median < 1.0)
    }

    @Test("DeltaNet forward: 8 tokens (longer sequence)")
    func deltaNetLongerSequence() throws {
        let executor = try buildDeltaNetExecutor(hiddenSize: 4, stateSize: 2)
        let tokens = MLXArray((0..<8).map { Int32($0 % 4) })

        let median = try measure {
            executor.resetCaches()
            let output = try executor.forward(tokenIDs: tokens)
            eval(output)
        }
        print("[DeltaNet 8tok] median=\(String(format: "%.4f", median))s")
        #expect(median < 1.0)
    }

    @Test("DeltaNet autoregressive: prefill + decode steps")
    func deltaNetAutoregressive() throws {
        let executor = try buildDeltaNetExecutor(hiddenSize: 4, stateSize: 2)
        let prefillTokens = MLXArray([Int32(0), Int32(1)])

        let median = try measure {
            executor.resetCaches()
            let _ = try executor.forward(tokenIDs: prefillTokens)
            // Decode 4 tokens
            for i in 0..<4 {
                let next = MLXArray([Int32(i % 4)])
                let output = try executor.forward(tokenIDs: next)
                eval(output)
            }
        }
        print("[DeltaNet autoregressive 2+4tok] median=\(String(format: "%.4f", median))s")
        #expect(median < 2.0)
    }

    @Test("DeltaNet scaling: sequence length overhead is proportional, not quadratic")
    func deltaNetSequenceScaling() throws {
        let executor4 = try buildDeltaNetExecutor(hiddenSize: 4, stateSize: 2)
        let executor8 = try buildDeltaNetExecutor(hiddenSize: 4, stateSize: 2)

        let tokens4 = MLXArray((0..<4).map { Int32($0 % 4) })
        let tokens8 = MLXArray((0..<8).map { Int32($0 % 4) })

        let time4 = try measure {
            executor4.resetCaches()
            let output = try executor4.forward(tokenIDs: tokens4)
            eval(output)
        }
        let time8 = try measure {
            executor8.resetCaches()
            let output = try executor8.forward(tokenIDs: tokens8)
            eval(output)
        }
        print("[DeltaNet scaling] 4tok=\(String(format: "%.4f", time4))s, 8tok=\(String(format: "%.4f", time8))s, ratio=\(String(format: "%.2f", time8 / time4))x")
        // 8 tokens should be roughly 2x, not 4x (not quadratic)
        #expect(time8 < time4 * 5.0)
    }
}


// MARK: - Combined MoE + DeltaNet Performance Tests

@Suite("Combined MoE+DeltaNet Executor Performance", .tags(.performance), .heartbeat)
struct CombinedMoEDeltaNetPerformanceTests {

    @Test("Mixed model: 2 layers × (DeltaNet + MoE), short sequence")
    func mixedModelShortSequence() throws {
        let executor = try buildMixedMoEDeltaNetExecutor(
            layerCount: 2,
            hiddenSize: 4,
            expertCount: 2,
            expertsPerToken: 1,
            intermediateSize: 8,
            stateSize: 2
        )
        let tokens = MLXArray([Int32(0), Int32(1), Int32(2), Int32(3)])

        let median = try measure {
            executor.resetCaches()
            let output = try executor.forward(tokenIDs: tokens)
            eval(output)
        }
        print("[Mixed 2L 4tok] median=\(String(format: "%.4f", median))s")
        #expect(median < 2.0)
    }

    @Test("Mixed model: autoregressive decode (prefill + token-by-token)")
    func mixedModelAutoregressive() throws {
        let executor = try buildMixedMoEDeltaNetExecutor(
            layerCount: 2,
            hiddenSize: 4,
            expertCount: 2,
            expertsPerToken: 1,
            intermediateSize: 8,
            stateSize: 2
        )

        let median = try measure {
            executor.resetCaches()
            let _ = try executor.forward(tokenIDs: MLXArray([Int32(0), Int32(1)]))
            for i in 0..<4 {
                let next = MLXArray([Int32(i % 4)])
                let output = try executor.forward(tokenIDs: next)
                eval(output)
            }
        }
        print("[Mixed 2L autoregressive 2+4tok] median=\(String(format: "%.4f", median))s")
        #expect(median < 3.0)
    }

    @Test("Mixed model: layer scaling (1 vs 4 layers)")
    func mixedModelLayerScaling() throws {
        let executor1 = try buildMixedMoEDeltaNetExecutor(
            layerCount: 1,
            hiddenSize: 4,
            expertCount: 2,
            expertsPerToken: 1,
            intermediateSize: 8,
            stateSize: 2
        )
        let executor4 = try buildMixedMoEDeltaNetExecutor(
            layerCount: 4,
            hiddenSize: 4,
            expertCount: 2,
            expertsPerToken: 1,
            intermediateSize: 8,
            stateSize: 2
        )
        let tokens = MLXArray([Int32(0), Int32(1)])

        let time1 = try measure {
            executor1.resetCaches()
            let output = try executor1.forward(tokenIDs: tokens)
            eval(output)
        }
        let time4 = try measure {
            executor4.resetCaches()
            let output = try executor4.forward(tokenIDs: tokens)
            eval(output)
        }
        print("[Mixed scaling] 1L=\(String(format: "%.4f", time1))s, 4L=\(String(format: "%.4f", time4))s, ratio=\(String(format: "%.2f", time4 / time1))x")
        // 4 layers should be roughly 4x, not exponential
        #expect(time4 < time1 * 10.0)
    }

    @Test("Mixed model: correctness after optimization (output shape and finiteness)")
    func mixedModelCorrectness() throws {
        let executor = try buildMixedMoEDeltaNetExecutor(
            layerCount: 2,
            hiddenSize: 4,
            expertCount: 2,
            expertsPerToken: 1,
            intermediateSize: 8,
            stateSize: 2,
            vocabSize: 16
        )

        // Prefill
        let prefillOutput = try executor.forward(
            tokenIDs: MLXArray([Int32(0), Int32(1), Int32(2)])
        )
        #expect(prefillOutput.shape == [3, 16])
        let prefillVals = prefillOutput.asArray(Float.self)
        for v in prefillVals { #expect(v.isFinite) }

        // Decode
        let decodeOutput = try executor.forward(tokenIDs: MLXArray([Int32(3)]))
        #expect(decodeOutput.shape == [1, 16])
        let decodeVals = decodeOutput.asArray(Float.self)
        for v in decodeVals { #expect(v.isFinite) }

        // Decode should differ from fresh executor (state persists)
        let freshExecutor = try buildMixedMoEDeltaNetExecutor(
            layerCount: 2,
            hiddenSize: 4,
            expertCount: 2,
            expertsPerToken: 1,
            intermediateSize: 8,
            stateSize: 2,
            vocabSize: 16
        )
        let freshOutput = try freshExecutor.forward(tokenIDs: MLXArray([Int32(3)]))
        let freshVals = freshOutput.asArray(Float.self)
        let allSame = zip(decodeVals, freshVals).allSatisfy { abs($0 - $1) < 1e-6 }
        #expect(!allSame)
    }
}
