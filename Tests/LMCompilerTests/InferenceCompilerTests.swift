import Testing
import TestHeartbeat
@preconcurrency import MLX
import MLXFast
import MLXNN
@testable import SwiftLM
@testable import LMCompiler

// MARK: - Test Helpers

/// Build a ParameterSlot from path components and role.
private func slot(
    _ components: [StructuralPathComponent], role: ParameterRole
) -> ParameterSlot {
    ParameterSlot(path: StructuralPath(components: components), role: role)
}

/// Build BoundWeights from a dictionary of ParameterSlot to MLXArray.
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

/// Build BoundWeights supporting both MLXArray and MLXTensorStorage.
private func bindMixed(_ pairs: [ParameterSlot: any Sendable]) -> BoundWeights {
    var tensors: [ParameterSlot: TensorData] = [:]
    for (slot, storage) in pairs {
        if let array = storage as? MLXArray {
            tensors[slot] = TensorData(
                shape: array.shape.map { $0 },
                dtype: .float32,
                storage: array
            )
        } else if let ts = storage as? MLXTensorStorage {
            switch ts {
            case .dense(let array):
                tensors[slot] = TensorData(
                    shape: array.shape.map { $0 },
                    dtype: .float16,
                    storage: ts
                )
            case .affineQuantized(let qt):
                tensors[slot] = TensorData(
                    shape: qt.logicalShape,
                    dtype: .float16,
                    storage: ts
                )
            }
        }
    }
    return BoundWeights(tensors: tensors)
}

// MARK: - Tiny Model Definitions

/// Tiny Llama-style model for inference compiler tests.
private struct TinyLlama: ModelComponent {
    let layerCount: Int
    let vocabSize = 8
    let hiddenSize = 4
    let headCount = 2
    let kvHeadCount = 2
    let headDim = 2
    let intermediateSize = 8

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

/// Build weights for TinyLlama.
private func tinyLlamaWeights(layerCount: Int) -> BoundWeights {
    let D = 4
    let H = 2
    let headDim = 2
    let inter = 8
    let vocab = 8

    var dict: [ParameterSlot: MLXArray] = [:]

    // op0: TokenEmbedding
    dict[slot([.operation(0)], role: .embeddingTable)] = MLXRandom.normal([vocab, D]) * 0.1

    for i in 0..<layerCount {
        let layerPrefix: [StructuralPathComponent] = [
            .operation(1), .regionBody, .index(i)
        ]

        // Residual 0: RMSNorm + Attention
        let attnNormPath = layerPrefix + [.operation(0), .regionBody, .operation(0)]
        dict[slot(attnNormPath, role: .scale)] = MLXArray.ones([D])

        let attnPath = layerPrefix + [.operation(0), .regionBody, .operation(1)]
        dict[slot(attnPath + [.field("q_proj")], role: .weight)] = MLXRandom.normal([H * headDim, D]) * 0.1
        dict[slot(attnPath + [.field("k_proj")], role: .weight)] = MLXRandom.normal([H * headDim, D]) * 0.1
        dict[slot(attnPath + [.field("v_proj")], role: .weight)] = MLXRandom.normal([H * headDim, D]) * 0.1
        dict[slot(attnPath + [.field("o_proj")], role: .weight)] = MLXRandom.normal([D, H * headDim]) * 0.1

        // Residual 1: RMSNorm + MLP
        let mlpNormPath = layerPrefix + [.operation(1), .regionBody, .operation(0)]
        dict[slot(mlpNormPath, role: .scale)] = MLXArray.ones([D])

        let mlpPath = layerPrefix + [.operation(1), .regionBody, .operation(1)]
        dict[slot(mlpPath + [.field("gate_proj")], role: .weight)] = MLXRandom.normal([inter, D]) * 0.1
        dict[slot(mlpPath + [.field("up_proj")], role: .weight)] = MLXRandom.normal([inter, D]) * 0.1
        dict[slot(mlpPath + [.field("down_proj")], role: .weight)] = MLXRandom.normal([D, inter]) * 0.1
    }

    // op2: Final RMSNorm
    dict[slot([.operation(2)], role: .scale)] = MLXArray.ones([D])

    // op3: OutputHead (tied — no weight needed)

    return bind(dict)
}

// MARK: - LoweredProjection Tests

@Suite("LoweredProjection", .tags(.unit, .compiled), .heartbeat)
struct LoweredProjectionTests {

    @Test("Dense kernel matches matmul(x, w.T)")
    func denseKernelMatchesMatmul() throws {
        let weight = MLXRandom.normal([8, 4])
        let input = MLXRandom.normal([2, 3, 4])
        let proj = LoweredProjection(weight: weight)

        let expected = matmul(input, weight.T)
        let actual = proj.apply(input)

        let diff = abs(actual - expected).max()
        #expect(diff.item(Float.self) < 1e-5)
    }

    @Test("Dense kernel with bias")
    func denseKernelWithBias() throws {
        let weight = MLXRandom.normal([8, 4])
        let bias = MLXRandom.normal([8])
        let input = MLXRandom.normal([2, 3, 4])
        let proj = LoweredProjection(weight: weight, bias: bias)

        let expected = matmul(input, weight.T) + bias
        let actual = proj.apply(input)

        let diff = abs(actual - expected).max()
        #expect(diff.item(Float.self) < 1e-5)
    }

    @Test("Kernel selection from MLXTensorStorage.dense")
    func kernelSelectionDense() throws {
        let weight = MLXRandom.normal([8, 4])
        let storage = MLXTensorStorage.dense(weight)
        let proj = LoweredProjection(storage: storage)

        if case .dense = proj.kernel {
            // OK
        } else {
            Issue.record("Expected dense kernel")
        }
    }

    @Test("Kernel selection from MLXTensorStorage.affineQuantized")
    func kernelSelectionQuantized() throws {
        let qt = AffineQuantizedTensor(
            logicalShape: [8, 4],
            packedWeight: MLXArray.zeros([8, 1], dtype: .uint32),
            scales: MLXArray.ones([8, 1]),
            zeroBiases: MLXArray.zeros([8, 1]),
            groupSize: 32,
            bits: 4,
            origin: .ggufQ4_0
        )
        let storage = MLXTensorStorage.affineQuantized(qt)
        let proj = LoweredProjection(storage: storage)

        if case .affineQuantized = proj.kernel {
            // OK
        } else {
            Issue.record("Expected affineQuantized kernel")
        }
    }

    @Test("Kernel selection uses dequantize+matmul for groupSize<32")
    func kernelSelectionSmallGroupFallback() throws {
        let qt = AffineQuantizedTensor(
            logicalShape: [8, 16],
            packedWeight: MLXArray.zeros([8, 2], dtype: .uint32),
            scales: MLXArray.zeros([8, 1]),
            zeroBiases: MLXArray.zeros([8, 1]),
            groupSize: 16,
            bits: 4,
            origin: .ggufQ2_K
        )
        let proj = LoweredProjection(storage: .affineQuantized(qt))

        if case .dequantizeMatmul = proj.kernel {
            // OK
        } else {
            Issue.record("Expected dequantizeMatmul kernel for groupSize < 32")
        }

        let input = MLXRandom.normal([2, 3, 16]).asType(.float16)
        let expected = MLXArray.zeros([2, 3, 8], dtype: input.dtype)
        let actual = proj.apply(input)
        let diff = abs(actual - expected).max()
        #expect(diff.item(Float.self) < 0.1)
    }
}

// MARK: - InferenceWeightStore Tests

@Suite("InferenceWeightStore", .tags(.unit, .compiled), .heartbeat)
struct InferenceWeightStoreTests {

    @Test("Accepts MLXArray from BoundWeights")
    func acceptsMLXArray() throws {
        let s = slot([.operation(0)], role: .weight)
        let weights = bind([s: MLXRandom.normal([4, 4])])
        let store = try InferenceWeightStore(boundWeights: weights)

        let storage = try store.require(s)
        if case .dense = storage {
            // OK
        } else {
            Issue.record("Expected dense storage from MLXArray")
        }
    }

    @Test("Accepts MLXTensorStorage from BoundWeights")
    func acceptsTensorStorage() throws {
        let s = slot([.operation(0)], role: .weight)
        let dense = MLXTensorStorage.dense(MLXRandom.normal([4, 4]))
        let weights = bindMixed([s: dense])
        let store = try InferenceWeightStore(boundWeights: weights)

        let storage = try store.require(s)
        if case .dense = storage {
            // OK
        } else {
            Issue.record("Expected dense storage from MLXTensorStorage")
        }
    }

    @Test("requireDense throws for quantized weight")
    func requireDenseThrowsForQuantized() throws {
        let s = slot([.operation(0)], role: .weight)
        let qt = AffineQuantizedTensor(
            logicalShape: [8, 4],
            packedWeight: MLXArray.zeros([8, 1], dtype: .uint32),
            scales: MLXArray.ones([8, 1]),
            zeroBiases: MLXArray.zeros([8, 1]),
            groupSize: 32, bits: 4, origin: .unknown
        )
        let store = InferenceWeightStore(weights: [s: .affineQuantized(qt)])

        #expect(throws: CompilerError.self) {
            _ = try store.requireDense(s)
        }
    }
}

// MARK: - Inference Compiler Tests

@Suite("MLXInferenceCompiler", .tags(.unit, .compiled), .heartbeat)
struct InferenceCompilerTests {

    @Test("Compile TinyLlama: correct cache slot count")
    func compileCacheSlots() throws {
        let model = TinyLlama(layerCount: 2)
        let graph = try model.makeModelGraph()
        let weights = tinyLlamaWeights(layerCount: 2)

        let compiled = try MLXInferenceCompiler().compile(graph: graph, weights: weights)

        // 2 layers × 1 attention = 2 cache slots
        #expect(compiled.metadata.cacheSlotCount == 2)
        #expect(compiled.metadata.hasTiedOutputHead == true)
    }

    @Test("Repeating block unrolls: 2 layers produce 2 distinct cache indices")
    func repeatingBlockUnrolls() throws {
        let model = TinyLlama(layerCount: 2)
        let graph = try model.makeModelGraph()
        let weights = tinyLlamaWeights(layerCount: 2)

        let compiled = try MLXInferenceCompiler().compile(graph: graph, weights: weights)
        let state = compiled.makeState()

        // State should have 2 cache slots
        #expect(state.caches.count == 2)

        // Both should be KV caches
        for cache in state.caches {
            if case .kv = cache {
                // OK
            } else {
                Issue.record("Expected KV cache")
            }
        }
    }

    @Test("Tied output head resolves to embedding table weight")
    func tiedOutputHead() throws {
        let model = TinyLlama(layerCount: 1)
        let graph = try model.makeModelGraph()
        let weights = tinyLlamaWeights(layerCount: 1)

        let compiled = try MLXInferenceCompiler().compile(graph: graph, weights: weights)
        #expect(compiled.metadata.hasTiedOutputHead == true)
    }

    @Test("makeState creates fresh state, prefill + decode round-trip")
    func stateRoundTrip() throws {
        let model = TinyLlama(layerCount: 1)
        let graph = try model.makeModelGraph()
        let weights = tinyLlamaWeights(layerCount: 1)

        let compiled = try MLXInferenceCompiler().compile(graph: graph, weights: weights)
        let state = compiled.makeState()

        // Prefill with a 3-token prompt
        let prompt = MLXArray([1, 2, 3]).expandedDimensions(axis: 0)  // [1, 3]
        let (prefillLogits, state2) = compiled.prefill(tokenIDs: prompt, state: state)

        // Output shape: [1, 3, vocabSize=8]
        #expect(prefillLogits.dim(0) == 1)
        #expect(prefillLogits.dim(1) == 3)
        #expect(prefillLogits.dim(2) == 8)

        // Decode one more token
        let nextToken = MLXArray([4]).expandedDimensions(axis: 0)  // [1, 1]
        let (decodeLogits, state3) = compiled.decode(tokenIDs: nextToken, state: state2)

        // Output shape: [1, 1, vocabSize=8]
        #expect(decodeLogits.dim(0) == 1)
        #expect(decodeLogits.dim(1) == 1)
        #expect(decodeLogits.dim(2) == 8)

        // Cache offset should be 4 (3 prefill + 1 decode)
        if case .kv(let cache) = state3.caches[0] {
            #expect(cache.offset == 4)
        } else {
            Issue.record("Expected KV cache in slot 0")
        }
    }

    @Test("End-to-end: inference compiler matches MLXExecutor output")
    func matchesMLXExecutor() throws {
        // Use a fixed seed for reproducibility
        MLXRandom.seed(42)

        let model = TinyLlama(layerCount: 1)
        let graph = try model.makeModelGraph()
        let weights = tinyLlamaWeights(layerCount: 1)

        // Compile with both paths
        let executorModel = try MLXCompiler().compile(graph: graph, weights: weights)
        let inferenceModel = try MLXInferenceCompiler().compile(graph: graph, weights: weights)

        // Run through MLXExecutor
        let prompt = MLXArray([1, 2, 3]).expandedDimensions(axis: 0)
        let executor = MLXExecutor(compiledModel: executorModel)
        let executorLogits = try executor.forward(tokenIDs: prompt)

        // Run through inference compiler
        let state = inferenceModel.makeState()
        let (inferenceLogits, _) = inferenceModel.prefill(tokenIDs: prompt, state: state)

        // Compare outputs
        let diff = abs(inferenceLogits - executorLogits).max()
        let maxDiff = diff.item(Float.self)

        // Allow small numerical differences
        #expect(maxDiff < 1e-4, "Max diff \(maxDiff) exceeds threshold 1e-4")
    }

    @Test("CompilationStats: all projections packed for dense weights")
    func compilationStatsDense() throws {
        let model = TinyLlama(layerCount: 4)
        let graph = try model.makeModelGraph()
        let weights = tinyLlamaWeights(layerCount: 4)

        let compiled = try MLXInferenceCompiler().compile(graph: graph, weights: weights)
        let stats = compiled.metadata.compilationStats

        // 4 layers × 1 attention = 4 packed, 0 unpacked
        #expect(stats.packedAttentionCount == 4)
        #expect(stats.unpackedAttentionCount == 0)

        // 4 layers × 1 MLP (gated) = 4 packed, 0 unpacked
        #expect(stats.packedMLPCount == 4)
        #expect(stats.unpackedMLPCount == 0)
        #expect(stats.ungatedMLPCount == 0)

        // All sub-layers should be fused (4 attn + 4 mlp = 8 fused)
        #expect(stats.fusedSubLayerCount == 8)
        #expect(stats.unfusedResidualCount == 0)
    }

    @Test("CompilationStats: scaling with layer count")
    func compilationStatsScaling() throws {
        for layerCount in [1, 2, 8] {
            let model = TinyLlama(layerCount: layerCount)
            let graph = try model.makeModelGraph()
            let weights = tinyLlamaWeights(layerCount: layerCount)

            let compiled = try MLXInferenceCompiler().compile(graph: graph, weights: weights)
            let stats = compiled.metadata.compilationStats

            #expect(
                stats.packedAttentionCount == layerCount,
                "Expected \(layerCount) packed attention, got \(stats.packedAttentionCount)"
            )
            #expect(
                stats.packedMLPCount == layerCount,
                "Expected \(layerCount) packed MLP, got \(stats.packedMLPCount)"
            )
            #expect(stats.unpackedAttentionCount == 0)
            #expect(stats.unpackedMLPCount == 0)
        }
    }
}

// MARK: - Position Tracking Tests

@Suite("PositionTracking", .tags(.unit, .compiled), .heartbeat)
struct PositionTrackingTests {

    @Test("nextPosition updates after prefill and decode")
    func nextPositionUpdates() throws {
        MLXRandom.seed(99)

        let model = TinyLlama(layerCount: 1)
        let graph = try model.makeModelGraph()
        let weights = tinyLlamaWeights(layerCount: 1)

        let compiled = try MLXInferenceCompiler().compile(graph: graph, weights: weights)
        let state = compiled.makeState()

        #expect(state.nextPosition == 0)

        // Prefill with 3 tokens
        let prompt = MLXArray([1, 2, 3]).expandedDimensions(axis: 0)
        let (_, state2) = compiled.prefill(tokenIDs: prompt, state: state)
        #expect(state2.nextPosition == 3)

        // Decode 1 token
        let next = MLXArray([4]).expandedDimensions(axis: 0)
        let (_, state3) = compiled.decode(tokenIDs: next, state: state2)
        #expect(state3.nextPosition == 4)

        // Decode another token
        let next2 = MLXArray([5]).expandedDimensions(axis: 0)
        let (_, state4) = compiled.decode(tokenIDs: next2, state: state3)
        #expect(state4.nextPosition == 5)
    }

    @Test("Multi-step decode matches MLXExecutor")
    func decodeMatchesMLXExecutor() throws {
        MLXRandom.seed(42)

        let model = TinyLlama(layerCount: 1)
        let graph = try model.makeModelGraph()
        let weights = tinyLlamaWeights(layerCount: 1)

        // Compile with both paths
        let executorModel = try MLXCompiler().compile(graph: graph, weights: weights)
        let inferenceModel = try MLXInferenceCompiler().compile(graph: graph, weights: weights)

        let prompt = MLXArray([1, 2, 3]).expandedDimensions(axis: 0)

        // Prefill with executor
        let executor = MLXExecutor(compiledModel: executorModel)
        let executorPrefill = try executor.forward(tokenIDs: prompt)

        // Prefill with inference compiler
        let state = inferenceModel.makeState()
        let (infPrefill, state2) = inferenceModel.prefill(tokenIDs: prompt, state: state)

        // Prefill must match
        let prefillDiff = abs(infPrefill - executorPrefill).max().item(Float.self)
        #expect(prefillDiff < 1e-4, "Prefill diff \(prefillDiff) exceeds threshold")

        // Decode one token with executor (token 4)
        let nextToken = MLXArray([4]).expandedDimensions(axis: 0)
        let executorDecode = try executor.forward(tokenIDs: nextToken)

        // Decode with inference compiler
        let (infDecode, _) = inferenceModel.decode(tokenIDs: nextToken, state: state2)

        // Decode must match
        let decodeDiff = abs(infDecode - executorDecode).max().item(Float.self)
        #expect(decodeDiff < 1e-4, "Decode diff \(decodeDiff) exceeds threshold")
    }
}

// MARK: - Untied Output Head Tests

/// Tiny model with untied output head for regression testing.
private struct TinyUntiedModel: ModelComponent {
    let vocabSize = 8
    let hiddenSize = 4
    let headCount = 2
    let kvHeadCount = 2
    let headDim = 2
    let intermediateSize = 8

    @ModelComponentBuilder var body: some ModelComponent {
        TokenEmbedding(vocabSize: vocabSize, embeddingSize: hiddenSize)

        Residual {
            RMSNorm(dimension: hiddenSize)
            Attention(
                hiddenSize: hiddenSize,
                headCount: headCount,
                kvHeadCount: kvHeadCount,
                headDimension: headDim
            )
        }

        RMSNorm(dimension: hiddenSize)
        OutputHead(inputSize: hiddenSize, vocabSize: vocabSize, tiedToEmbedding: false)
    }
}

/// Build weights for TinyUntiedModel.
private func tinyUntiedWeights() -> BoundWeights {
    let D = 4
    let H = 2
    let headDim = 2
    let vocab = 8

    var dict: [ParameterSlot: MLXArray] = [:]

    // op0: TokenEmbedding
    dict[slot([.operation(0)], role: .embeddingTable)] = MLXRandom.normal([vocab, D]) * 0.1

    // op1: Residual (RMSNorm + Attention)
    let attnNormPath: [StructuralPathComponent] = [.operation(1), .regionBody, .operation(0)]
    dict[slot(attnNormPath, role: .scale)] = MLXArray.ones([D])

    let attnPath: [StructuralPathComponent] = [.operation(1), .regionBody, .operation(1)]
    dict[slot(attnPath + [.field("q_proj")], role: .weight)] = MLXRandom.normal([H * headDim, D]) * 0.1
    dict[slot(attnPath + [.field("k_proj")], role: .weight)] = MLXRandom.normal([H * headDim, D]) * 0.1
    dict[slot(attnPath + [.field("v_proj")], role: .weight)] = MLXRandom.normal([H * headDim, D]) * 0.1
    dict[slot(attnPath + [.field("o_proj")], role: .weight)] = MLXRandom.normal([D, H * headDim]) * 0.1

    // op2: Final RMSNorm
    dict[slot([.operation(2)], role: .scale)] = MLXArray.ones([D])

    // op3: OutputHead (untied — needs .outputProjection weight)
    dict[slot([.operation(3)], role: .outputProjection)] = MLXRandom.normal([vocab, D]) * 0.1

    return bind(dict)
}

@Suite("UntiedOutputHead", .tags(.unit, .compiled), .heartbeat)
struct UntiedOutputHeadTests {

    @Test("Untied output head compiles and produces correct output shape")
    func untiedOutputHeadCompiles() throws {
        MLXRandom.seed(77)

        let model = TinyUntiedModel()
        let graph = try model.makeModelGraph()
        let weights = tinyUntiedWeights()

        let compiled = try MLXInferenceCompiler().compile(graph: graph, weights: weights)
        #expect(compiled.metadata.hasTiedOutputHead == false)

        let state = compiled.makeState()
        let prompt = MLXArray([1, 2, 3]).expandedDimensions(axis: 0)
        let (logits, _) = compiled.prefill(tokenIDs: prompt, state: state)

        #expect(logits.dim(0) == 1)
        #expect(logits.dim(1) == 3)
        #expect(logits.dim(2) == 8)
    }
}

// MARK: - Standalone Position Op Models

/// Model with standalone RoPE for testing position offset.
private struct TinyRoPEModel: ModelComponent {
    let vocabSize = 8
    let hiddenSize = 4

    @ModelComponentBuilder var body: some ModelComponent {
        TokenEmbedding(vocabSize: vocabSize, embeddingSize: hiddenSize)
        RoPE(dimension: hiddenSize)
        RMSNorm(dimension: hiddenSize)
        OutputHead(inputSize: hiddenSize, vocabSize: vocabSize, tiedToEmbedding: true)
    }
}

/// Build weights for TinyRoPEModel.
/// op(0)=TokenEmbedding, op(1)=RoPE (no weights), op(2)=RMSNorm, op(3)=OutputHead (tied)
private func tinyRoPEWeights() -> BoundWeights {
    let D = 4
    let vocab = 8

    var dict: [ParameterSlot: MLXArray] = [:]

    // op0: TokenEmbedding
    dict[slot([.operation(0)], role: .embeddingTable)] = MLXRandom.normal([vocab, D]) * 0.1

    // op1: RoPE — no weights

    // op2: RMSNorm
    dict[slot([.operation(2)], role: .scale)] = MLXArray.ones([D])

    // op3: OutputHead (tied — no weight needed)

    return bind(dict)
}

/// Model with standalone PositionalEmbedding for testing position offset.
private struct TinyPosEmbModel: ModelComponent {
    let vocabSize = 8
    let hiddenSize = 4
    let maxPositions = 32

    @ModelComponentBuilder var body: some ModelComponent {
        TokenEmbedding(vocabSize: vocabSize, embeddingSize: hiddenSize)
        PositionalEmbedding(maxPositions: maxPositions, embeddingSize: hiddenSize, kind: .learnedAbsolute)
        RMSNorm(dimension: hiddenSize)
        OutputHead(inputSize: hiddenSize, vocabSize: vocabSize, tiedToEmbedding: true)
    }
}

/// Build weights for TinyPosEmbModel.
/// op(0)=TokenEmbedding, op(1)=PositionalEmbedding, op(2)=RMSNorm, op(3)=OutputHead (tied)
private func tinyPosEmbWeights() -> BoundWeights {
    let D = 4
    let vocab = 8
    let maxPos = 32

    var dict: [ParameterSlot: MLXArray] = [:]

    // op0: TokenEmbedding
    dict[slot([.operation(0)], role: .embeddingTable)] = MLXRandom.normal([vocab, D]) * 0.1

    // op1: PositionalEmbedding
    dict[slot([.operation(1)], role: .embeddingTable)] = MLXRandom.normal([maxPos, D]) * 0.1

    // op2: RMSNorm
    dict[slot([.operation(2)], role: .scale)] = MLXArray.ones([D])

    // op3: OutputHead (tied — no weight needed)

    return bind(dict)
}

// MARK: - Standalone Position Op Tests

@Suite("StandalonePositionOps", .tags(.unit, .compiled), .heartbeat)
struct StandalonePositionOpTests {

    @Test("Standalone RoPE uses state.nextPosition as offset")
    func standaloneRoPEUsesOffset() throws {
        MLXRandom.seed(42)

        let model = TinyRoPEModel()
        let graph = try model.makeModelGraph()
        let weights = tinyRoPEWeights()

        let compiled = try MLXInferenceCompiler().compile(graph: graph, weights: weights)

        // No attention layers → 0 cache slots
        #expect(compiled.metadata.cacheSlotCount == 0)

        // Prefill with 3 tokens → state.nextPosition == 3
        let prompt = MLXArray([1, 2, 3]).expandedDimensions(axis: 0)
        let state0 = compiled.makeState()
        let (_, state3) = compiled.prefill(tokenIDs: prompt, state: state0)
        #expect(state3.nextPosition == 3)

        // Decode token 4 with offset=3 (continuing from prefill)
        let next = MLXArray([4]).expandedDimensions(axis: 0)
        let (outputAtOffset3, _) = compiled.decode(tokenIDs: next, state: state3)

        // Decode same token 4 with offset=0 (fresh state)
        let freshState = compiled.makeState()
        let (outputAtOffset0, _) = compiled.decode(tokenIDs: next, state: freshState)

        // RoPE with different offsets must produce different outputs
        let diff = abs(outputAtOffset3 - outputAtOffset0).max().item(Float.self)
        #expect(diff > 1e-4, "RoPE output should differ at offset=3 vs offset=0, but diff=\(diff)")
    }

    @Test("Standalone PositionalEmbedding uses state.nextPosition as offset")
    func standalonePosEmbUsesOffset() throws {
        MLXRandom.seed(42)

        let model = TinyPosEmbModel()
        let graph = try model.makeModelGraph()
        let weights = tinyPosEmbWeights()

        let compiled = try MLXInferenceCompiler().compile(graph: graph, weights: weights)

        // Prefill with 3 tokens → state.nextPosition == 3
        let prompt = MLXArray([1, 2, 3]).expandedDimensions(axis: 0)
        let state0 = compiled.makeState()
        let (_, state3) = compiled.prefill(tokenIDs: prompt, state: state0)
        #expect(state3.nextPosition == 3)

        // Decode token 4 with offset=3 (adds position[3] from table)
        let next = MLXArray([4]).expandedDimensions(axis: 0)
        let (outputAtOffset3, _) = compiled.decode(tokenIDs: next, state: state3)

        // Decode same token 4 with offset=0 (adds position[0] from table)
        let freshState = compiled.makeState()
        let (outputAtOffset0, _) = compiled.decode(tokenIDs: next, state: freshState)

        // Different position embeddings must produce different outputs
        let diff = abs(outputAtOffset3 - outputAtOffset0).max().item(Float.self)
        #expect(diff > 1e-4, "PositionalEmbedding output should differ at offset=3 vs offset=0, but diff=\(diff)")
    }

    @Test("Standalone RoPE offset matches manual computation")
    func standaloneRoPEMatchesManual() throws {
        MLXRandom.seed(77)

        let model = TinyRoPEModel()
        let graph = try model.makeModelGraph()
        let weights = tinyRoPEWeights()

        let compiled = try MLXInferenceCompiler().compile(graph: graph, weights: weights)

        // Prefill 3 tokens to advance position
        let prompt = MLXArray([1, 2, 3]).expandedDimensions(axis: 0)
        let state0 = compiled.makeState()
        let (_, state3) = compiled.prefill(tokenIDs: prompt, state: state0)

        // Decode token 4 at offset=3
        let next = MLXArray([4]).expandedDimensions(axis: 0)
        let (actual, _) = compiled.decode(tokenIDs: next, state: state3)

        // Manual computation: embedding(4) → RoPE(offset=3) → RMSNorm → matmul(embedding.T)
        let embSlot = ParameterSlot(
            path: StructuralPath(components: [.operation(0)]),
            role: .embeddingTable
        )
        let normSlot = ParameterSlot(
            path: StructuralPath(components: [.operation(2)]),
            role: .scale
        )

        let store = try InferenceWeightStore(boundWeights: weights)
        let embTable = store.getDense(embSlot)!
        let normWeight = store.getDense(normSlot)!

        // Step 1: Embedding lookup
        var h = embTable[next]  // [1, 1, 4]

        // Step 2: RoPE with offset=3
        h = MLXFast.RoPE(
            h, dimensions: 4, traditional: false,
            base: 10_000.0, scale: 1.0, offset: 3
        )

        // Step 3: RMSNorm (default epsilon = 1e-6)
        h = MLXFast.rmsNorm(h, weight: normWeight, eps: 1e-6)

        // Step 4: Output head (tied to embedding) — matmul(h, embTable.T)
        let expected = matmul(h, embTable.T)

        let diff = abs(actual - expected).max().item(Float.self)
        #expect(diff < 1e-4, "Manual computation mismatch: diff=\(diff)")
    }
}

// MARK: - LoweredNorm Tests

@Suite("LoweredNorm", .tags(.unit, .compiled), .heartbeat)
struct LoweredNormTests {

    @Test("RMSNorm matches MLXFast.rmsNorm")
    func rmsNormMatches() throws {
        let weight = MLXRandom.normal([4])
        let input = MLXRandom.normal([2, 3, 4])
        let norm = LoweredNorm.rms(weight: weight, epsilon: 1e-5)

        let expected = MLXFast.rmsNorm(input, weight: weight, eps: 1e-5)
        let actual = norm.apply(input)

        let diff = abs(actual - expected).max()
        #expect(diff.item(Float.self) < 1e-6)
    }
}

// MARK: - LoweredMLP Tests

@Suite("LoweredMLP", .tags(.unit, .compiled), .heartbeat)
struct LoweredMLPTests {

    @Test("Gated MLP (SwiGLU) produces correct shape")
    func gatedMLPShape() throws {
        let D = 4
        let inter = 8
        let gateProj = LoweredProjection(weight: MLXRandom.normal([inter, D]))
        let upProj = LoweredProjection(weight: MLXRandom.normal([inter, D]))
        let downProj = LoweredProjection(weight: MLXRandom.normal([D, inter]))

        let mlp = LoweredMLP(
            gateProj: gateProj, downProj: downProj,
            upProj: upProj, activation: .silu
        )

        let input = MLXRandom.normal([2, 3, D])
        let output = mlp.apply(input)

        #expect(output.dim(0) == 2)
        #expect(output.dim(1) == 3)
        #expect(output.dim(2) == D)
    }

    @Test("Ungated MLP (no upProj) produces correct shape")
    func ungatedMLPShape() throws {
        let D = 4
        let inter = 8
        let gateProj = LoweredProjection(weight: MLXRandom.normal([inter, D]))
        let downProj = LoweredProjection(weight: MLXRandom.normal([D, inter]))

        let mlp = LoweredMLP(
            gateProj: gateProj, downProj: downProj,
            upProj: nil, activation: .gelu
        )

        let input = MLXRandom.normal([2, 3, D])
        let output = mlp.apply(input)

        #expect(output.dim(0) == 2)
        #expect(output.dim(1) == 3)
        #expect(output.dim(2) == D)
    }
}

// MARK: - LoweredCacheState Tests

@Suite("LoweredCacheState", .tags(.unit, .compiled), .heartbeat)
struct LoweredCacheStateTests {

    @Test("KV cache grows correctly")
    func kvCacheGrowth() throws {
        var cache = LoweredKVCache(step: 4)

        // First update: 2 tokens
        let k1 = MLXRandom.normal([1, 2, 2, 3])  // [B, H, L, D]
        let v1 = MLXRandom.normal([1, 2, 2, 3])
        let (_, _) = cache.update(newKeys: k1, newValues: v1)
        #expect(cache.offset == 2)

        // Second update: 3 more tokens
        let k2 = MLXRandom.normal([1, 2, 3, 3])
        let v2 = MLXRandom.normal([1, 2, 3, 3])
        let (keys, values) = cache.update(newKeys: k2, newValues: v2)
        #expect(cache.offset == 5)

        // Returned keys/values should span all 5 tokens
        #expect(keys.dim(2) == 5)
        #expect(values.dim(2) == 5)
    }

    @Test("KV cache mask: causal for prefill, none for decode")
    func kvCacheMask() throws {
        let cache = LoweredKVCache()
        // Check mask type via pattern matching (ScaledDotProductAttentionMaskMode is not Equatable)
        if case .causal = cache.makeMask(queryLength: 3) {
            // OK
        } else {
            Issue.record("Expected .causal mask for queryLength > 1")
        }
        if case .none = cache.makeMask(queryLength: 1) {
            // OK
        } else {
            Issue.record("Expected .none mask for queryLength == 1")
        }
    }
}

// MARK: - ModelGraphSlotEnumerator Tests

@Suite("ModelGraphSlotEnumerator", .tags(.unit, .compiled), .heartbeat)
struct ModelGraphSlotEnumeratorTests {

    @Test("Enumerates TinyLlama slots with correct MLX paths")
    func enumeratesTinyLlama() throws {
        let model = TinyLlama(layerCount: 2)
        let graph = try model.makeModelGraph()

        let enumerator = ModelGraphSlotEnumerator()
        let manifest = enumerator.enumerate(graph)

        // Build lookup by MLX path for easy assertions
        let byPath = Dictionary(
            manifest.map { ($0.mlxWeightPath, $0.slot) },
            uniquingKeysWith: { first, _ in first }
        )

        // Token embedding
        #expect(byPath["model.embed_tokens.weight"] != nil)
        #expect(byPath["model.embed_tokens.weight"]?.role == .embeddingTable)

        // Layer 0 attention norm
        #expect(byPath["model.layers.0.input_layernorm.weight"] != nil)
        #expect(byPath["model.layers.0.input_layernorm.weight"]?.role == .scale)

        // Layer 0 attention projections
        #expect(byPath["model.layers.0.self_attn.q_proj.weight"] != nil)
        #expect(byPath["model.layers.0.self_attn.k_proj.weight"] != nil)
        #expect(byPath["model.layers.0.self_attn.v_proj.weight"] != nil)
        #expect(byPath["model.layers.0.self_attn.o_proj.weight"] != nil)

        // Layer 0 MLP norm
        #expect(byPath["model.layers.0.post_attention_layernorm.weight"] != nil)
        #expect(byPath["model.layers.0.post_attention_layernorm.weight"]?.role == .scale)

        // Layer 0 MLP projections
        #expect(byPath["model.layers.0.mlp.gate_proj.weight"] != nil)
        #expect(byPath["model.layers.0.mlp.up_proj.weight"] != nil)
        #expect(byPath["model.layers.0.mlp.down_proj.weight"] != nil)

        // Layer 1 (same structure)
        #expect(byPath["model.layers.1.self_attn.q_proj.weight"] != nil)
        #expect(byPath["model.layers.1.mlp.gate_proj.weight"] != nil)

        // Final norm
        #expect(byPath["model.norm.weight"] != nil)
        #expect(byPath["model.norm.weight"]?.role == .scale)

        // Output head is tied — should NOT appear as lm_head
        #expect(byPath["lm_head.weight"] == nil)
    }

    @Test("Enumerates untied output head as lm_head.weight")
    func enumeratesUntiedOutputHead() throws {
        let model = TinyUntiedModel()
        let graph = try model.makeModelGraph()

        let enumerator = ModelGraphSlotEnumerator()
        let manifest = enumerator.enumerate(graph)

        let byPath = Dictionary(
            manifest.map { ($0.mlxWeightPath, $0.slot) },
            uniquingKeysWith: { first, _ in first }
        )

        #expect(byPath["lm_head.weight"] != nil)
        #expect(byPath["lm_head.weight"]?.role == .outputProjection)
    }

    @Test("Enumerated slots match compiler expectations")
    func slotsMatchCompiler() throws {
        let model = TinyLlama(layerCount: 1)
        let graph = try model.makeModelGraph()

        let enumerator = ModelGraphSlotEnumerator()
        let manifest = enumerator.enumerate(graph)

        // All slots in the manifest should be usable by the compiler
        // Build weights from manifest slots
        var weightDict: [ParameterSlot: MLXArray] = [:]
        for entry in manifest {
            switch entry.slot.role {
            case .embeddingTable:
                weightDict[entry.slot] = MLXRandom.normal([8, 4]) * 0.1
            case .scale:
                weightDict[entry.slot] = MLXArray.zeros([4])
            case .weight:
                // Determine shape from field name
                if entry.mlxWeightPath.contains("gate_proj") || entry.mlxWeightPath.contains("up_proj") {
                    weightDict[entry.slot] = MLXRandom.normal([8, 4]) * 0.1
                } else if entry.mlxWeightPath.contains("down_proj") {
                    weightDict[entry.slot] = MLXRandom.normal([4, 8]) * 0.1
                } else {
                    weightDict[entry.slot] = MLXRandom.normal([4, 4]) * 0.1
                }
            default:
                break
            }
        }

        let weights = bind(weightDict)

        // Compiler should accept these slots without error
        let compiled = try MLXInferenceCompiler().compile(graph: graph, weights: weights)
        #expect(compiled.metadata.cacheSlotCount == 1)
    }
}
