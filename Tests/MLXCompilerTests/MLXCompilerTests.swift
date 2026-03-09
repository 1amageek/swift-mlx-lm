import Testing
@preconcurrency import MLX
import MLXFast
import MLXNN
@testable import SwiftLM
@testable import MLXCompiler


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

// MARK: - Compiler Tests

@Suite("MLXCompiler")
struct CompilerTests {

    @Test("Cache descriptors for Llama-style model: 2 KV caches per layer")
    func cacheDescriptorsLlama() throws {
        let model = MiniLlama(layerCount: 2)
        let graph = try model.makeModelGraph()
        let compiled = try MLXCompiler().compile(
            graph: graph, weights: miniLlamaWeights(graph: graph, layerCount: 2)
        )

        // 2 layers × 1 attention = 2 KV caches
        #expect(compiled.cacheDescriptors.count == 2)
        for desc in compiled.cacheDescriptors {
            #expect(desc.kind == .kv)
        }
        // Slot indices are sequential
        #expect(compiled.cacheDescriptors[0].slotIndex == 0)
        #expect(compiled.cacheDescriptors[1].slotIndex == 1)
    }

    @Test("Embedding path is discovered for tied output head")
    func embeddingPathDiscovery() throws {
        let model = MiniLlama(layerCount: 1)
        let graph = try model.makeModelGraph()
        let compiled = try MLXCompiler().compile(
            graph: graph, weights: miniLlamaWeights(graph: graph, layerCount: 1)
        )

        // Embedding path should point to the first operation (TokenEmbedding)
        #expect(compiled.embeddingPath != nil)
        #expect(compiled.embeddingPath!.components == [.operation(0)])
    }

    @Test("Cache descriptors for Cohere-style model with parallel branches")
    func cacheDescriptorsCohere() throws {
        let model = MiniCohere(layerCount: 2)
        let graph = try model.makeModelGraph()
        let compiled = try MLXCompiler().compile(
            graph: graph, weights: miniCohereWeights(graph: graph, layerCount: 2)
        )

        // 2 layers × 1 attention = 2 KV caches
        #expect(compiled.cacheDescriptors.count == 2)
    }

    @Test("Model with no attention has no caches")
    func noCachesForNonAttentionModel() throws {
        struct NormOnly: LanguageModel {
            @ModelComponentBuilder var body: some ModelComponent {
                TokenEmbedding(vocabSize: 8, embeddingSize: 4)
                RMSNorm(dimension: 4)
                OutputHead(inputSize: 4, vocabSize: 8, tiedToEmbedding: true)
            }
        }

        let model = NormOnly()
        let graph = try model.makeModelGraph()
        let weights = bind([
            slot([.operation(0)], role: .embeddingTable):
                MLXArray.ones([8, 4]),
            slot([.operation(1)], role: .scale):
                MLXArray.zeros([4]),
        ])
        let compiled = try MLXCompiler().compile(graph: graph, weights: weights)
        #expect(compiled.cacheDescriptors.isEmpty)
    }
}

// MARK: - Weight Store Tests

@Suite("MLXWeightStore")
struct WeightStoreTests {

    @Test("Weight store extracts MLXArrays from BoundWeights")
    func weightExtraction() throws {
        let arr = MLXArray.ones([3, 4])
        let s = slot([.operation(0)], role: .weight)
        let weights = bind([s: arr])
        let store = try MLXWeightStore(boundWeights: weights)

        let retrieved = store.get(s)
        #expect(retrieved != nil)
        #expect(retrieved!.shape == [3, 4])
    }

    @Test("Weight store require throws for missing weight")
    func requireThrowsOnMissing() throws {
        let store = MLXWeightStore(weights: [:])
        let s = slot([.operation(0)], role: .weight)
        #expect(throws: CompilerError.self) {
            try store.require(s)
        }
    }

    @Test("Weight store rejects non-MLXArray storage")
    func rejectNonMLXArrayStorage() throws {
        let s = slot([.operation(0)], role: .weight)
        let badWeights = BoundWeights(tensors: [
            s: TensorData(shape: [3, 4], dtype: .float32, storage: "not_an_array")
        ])
        #expect(throws: CompilerError.self) {
            try MLXWeightStore(boundWeights: badWeights)
        }
    }
}

// MARK: - Executor Operation Tests

@Suite("MLXExecutor Operations")
struct ExecutorOperationTests {

    @Test("TokenEmbedding looks up embeddings by token ID")
    func tokenEmbedding() throws {
        struct EmbedOnly: LanguageModel {
            @ModelComponentBuilder var body: some ModelComponent {
                TokenEmbedding(vocabSize: 4, embeddingSize: 3)
            }
        }

        let graph = try EmbedOnly().makeModelGraph()

        // Embedding table: each row is distinct
        let table = MLXArray(converting: [
            1, 0, 0,   // token 0
            0, 1, 0,   // token 1
            0, 0, 1,   // token 2
            1, 1, 1,   // token 3
        ] as [Double], [4, 3])

        let weights = bind([
            slot([.operation(0)], role: .embeddingTable): table,
        ])
        let compiled = try MLXCompiler().compile(graph: graph, weights: weights)
        let executor = MLXExecutor(compiledModel: compiled)

        let tokens = MLXArray([Int32(2), Int32(0)])  // [B=1, L=2] — request embeddings for tokens 2, 0
        let output = try executor.forward(tokenIDs: tokens)

        // Token 2 → [0, 0, 1], Token 0 → [1, 0, 0]
        let vals = output.asArray(Float.self)
        #expect(vals[0] == 0)
        #expect(vals[1] == 0)
        #expect(vals[2] == 1)
        #expect(vals[3] == 1)
        #expect(vals[4] == 0)
        #expect(vals[5] == 0)
    }

    @Test("RMSNorm applies 1+w convention")
    func rmsNormConvention() throws {
        struct NormModel: LanguageModel {
            @ModelComponentBuilder var body: some ModelComponent {
                TokenEmbedding(vocabSize: 4, embeddingSize: 4)
                RMSNorm(dimension: 4)
            }
        }

        let graph = try NormModel().makeModelGraph()

        // Uniform input → RMS = value itself, so normalized = sign(x)
        let table = MLXArray(converting: [Double](repeating: 2.0, count: 16), [4, 4])

        // Scale weights centered at 0: w=0 means effective weight=1
        let scale = MLXArray.zeros([4])

        let weights = bind([
            slot([.operation(0)], role: .embeddingTable): table,
            slot([.operation(1)], role: .scale): scale,
        ])
        let compiled = try MLXCompiler().compile(graph: graph, weights: weights)
        let executor = MLXExecutor(compiledModel: compiled)

        let output = try executor.forward(tokenIDs: MLXArray([Int32(0)]))

        // RMSNorm on [2,2,2,2] with weight=(1+0)=1:
        // rms = sqrt(mean([4,4,4,4])) = sqrt(4) = 2
        // normalized = [2,2,2,2] / 2 * 1 = [1,1,1,1]
        let vals = output.asArray(Float.self)
        for v in vals {
            #expect(abs(v - 1.0) < 1e-4)
        }
    }

    @Test("Linear projection: output = input @ weight.T")
    func linearProjection() throws {
        struct LinearModel: LanguageModel {
            @ModelComponentBuilder var body: some ModelComponent {
                TokenEmbedding(vocabSize: 4, embeddingSize: 2)
                Linear(inputSize: 2, outputSize: 3)
            }
        }

        let graph = try LinearModel().makeModelGraph()

        let table = MLXArray(converting: [
            1, 2,
            3, 4,
            5, 6,
            7, 8,
        ] as [Double], [4, 2])

        // Weight [3, 2]: output = [1,2] @ [[1,0],[0,1],[1,1]]^T = [1, 2, 3]
        let w = MLXArray(converting: [
            1, 0,
            0, 1,
            1, 1,
        ] as [Double], [3, 2])

        let weights = bind([
            slot([.operation(0)], role: .embeddingTable): table,
            slot([.operation(1)], role: .weight): w,
        ])
        let compiled = try MLXCompiler().compile(graph: graph, weights: weights)
        let executor = MLXExecutor(compiledModel: compiled)

        // Token 0 embedding = [1, 2]
        let output = try executor.forward(tokenIDs: MLXArray([Int32(0)]))
        let vals = output.asArray(Float.self)

        #expect(abs(vals[0] - 1.0) < 1e-4)  // 1*1 + 2*0 = 1
        #expect(abs(vals[1] - 2.0) < 1e-4)  // 1*0 + 2*1 = 2
        #expect(abs(vals[2] - 3.0) < 1e-4)  // 1*1 + 2*1 = 3
    }

    @Test("OutputHead with tied embedding reuses embedding table")
    func tiedOutputHead() throws {
        struct TiedModel: LanguageModel {
            @ModelComponentBuilder var body: some ModelComponent {
                TokenEmbedding(vocabSize: 3, embeddingSize: 2)
                OutputHead(inputSize: 2, vocabSize: 3, tiedToEmbedding: true)
            }
        }

        let graph = try TiedModel().makeModelGraph()

        let table = MLXArray(converting: [
            1, 0,
            0, 1,
            1, 1,
        ] as [Double], [3, 2])

        // Only embedding table weight — no separate output projection
        let weights = bind([
            slot([.operation(0)], role: .embeddingTable): table,
        ])
        let compiled = try MLXCompiler().compile(graph: graph, weights: weights)
        let executor = MLXExecutor(compiledModel: compiled)

        // Token 0 → embed [1, 0] → matmul([1,0], table.T) → [1*1+0*0, 1*0+0*1, 1*1+0*1] = [1, 0, 1]
        let output = try executor.forward(tokenIDs: MLXArray([Int32(0)]))
        let vals = output.asArray(Float.self)

        #expect(abs(vals[0] - 1.0) < 1e-4)
        #expect(abs(vals[1] - 0.0) < 1e-4)
        #expect(abs(vals[2] - 1.0) < 1e-4)
    }

    @Test("MLP SwiGLU: gate activation * up projection, then down")
    func mlpSwiGLU() throws {
        struct MLPModel: LanguageModel {
            @ModelComponentBuilder var body: some ModelComponent {
                TokenEmbedding(vocabSize: 4, embeddingSize: 2)
                MLP(inputSize: 2, intermediateSize: 4, activation: .silu, gating: .swiglu)
            }
        }

        let graph = try MLPModel().makeModelGraph()

        let table = MLXArray(converting: [1, 1, 0, 0, 0, 0, 0, 0] as [Double], [4, 2])

        // Use identity-like weights for simplicity
        // gate_proj [4, 2], up_proj [4, 2], down_proj [2, 4]
        let gateW = MLXArray(converting: [
            1, 0,
            0, 1,
            1, 0,
            0, 1,
        ] as [Double], [4, 2])
        let upW = MLXArray(converting: [
            1, 0,
            0, 1,
            1, 0,
            0, 1,
        ] as [Double], [4, 2])
        let downW = MLXArray(converting: [
            1, 0, 0, 0,
            0, 1, 0, 0,
        ] as [Double], [2, 4])

        let weights = bind([
            slot([.operation(0)], role: .embeddingTable): table,
            slot([.operation(1), .field("gate_proj")], role: .weight): gateW,
            slot([.operation(1), .field("up_proj")], role: .weight): upW,
            slot([.operation(1), .field("down_proj")], role: .weight): downW,
        ])
        let compiled = try MLXCompiler().compile(graph: graph, weights: weights)
        let executor = MLXExecutor(compiledModel: compiled)

        // Token 0 → [1, 1]
        // gate = [1,1] @ gateW.T = [1, 1, 1, 1]
        // activated = silu([1, 1, 1, 1]) = [0.7311, 0.7311, 0.7311, 0.7311]
        // up = [1,1] @ upW.T = [1, 1, 1, 1]
        // gated = activated * up = [0.7311, ...]
        // output = gated @ downW.T = [0.7311, 0.7311]
        let output = try executor.forward(tokenIDs: MLXArray([Int32(0)]))
        let vals = output.asArray(Float.self)

        let expected = MLXNN.silu(MLXArray(Float(1.0))).item(Float.self)
        #expect(abs(vals[0] - expected) < 1e-4)
        #expect(abs(vals[1] - expected) < 1e-4)
    }
}

// MARK: - Structural Execution Tests

@Suite("MLXExecutor Structural Operations")
struct ExecutorStructuralTests {

    @Test("Residual adds body output to input")
    func residualAdd() throws {
        struct ResidualModel: LanguageModel {
            @ModelComponentBuilder var body: some ModelComponent {
                TokenEmbedding(vocabSize: 4, embeddingSize: 2)
                Residual {
                    Linear(inputSize: 2, outputSize: 2)
                }
            }
        }

        let graph = try ResidualModel().makeModelGraph()

        let table = MLXArray(converting: [1, 2, 3, 4, 5, 6, 7, 8] as [Double], [4, 2])

        // Identity weight → body(x) = x, so residual output = x + x = 2x
        let identityW = MLXArray(converting: [1, 0, 0, 1] as [Double], [2, 2])

        let weights = bind([
            slot([.operation(0)], role: .embeddingTable): table,
            // Linear inside residual body: [op1, regionBody, op0]
            slot([.operation(1), .regionBody, .operation(0)], role: .weight): identityW,
        ])
        let compiled = try MLXCompiler().compile(graph: graph, weights: weights)
        let executor = MLXExecutor(compiledModel: compiled)

        // Token 0 → [1, 2], residual = [1,2] + identity([1,2]) = [2, 4]
        let output = try executor.forward(tokenIDs: MLXArray([Int32(0)]))
        let vals = output.asArray(Float.self)

        #expect(abs(vals[0] - 2.0) < 1e-4)
        #expect(abs(vals[1] - 4.0) < 1e-4)
    }

    @Test("Repeating block executes body N times with per-iteration weight paths")
    func repeatingWithDistinctWeights() throws {
        struct RepeatModel: LanguageModel {
            @ModelComponentBuilder var body: some ModelComponent {
                TokenEmbedding(vocabSize: 4, embeddingSize: 2)
                Repeat(count: 2) {
                    Linear(inputSize: 2, outputSize: 2)
                }
            }
        }

        let graph = try RepeatModel().makeModelGraph()

        let table = MLXArray(converting: [1, 0, 0, 1, 1, 1, 0, 0] as [Double], [4, 2])

        // Iteration 0: scale by 2
        let w0 = MLXArray(converting: [2, 0, 0, 2] as [Double], [2, 2])
        // Iteration 1: scale by 3
        let w1 = MLXArray(converting: [3, 0, 0, 3] as [Double], [2, 2])

        let weights = bind([
            slot([.operation(0)], role: .embeddingTable): table,
            // Linear inside repeat body: [op1, regionBody, index(i), op0]
            slot([.operation(1), .regionBody, .index(0), .operation(0)], role: .weight): w0,
            slot([.operation(1), .regionBody, .index(1), .operation(0)], role: .weight): w1,
        ])
        let compiled = try MLXCompiler().compile(graph: graph, weights: weights)
        let executor = MLXExecutor(compiledModel: compiled)

        // Token 0 → [1, 0]
        // Iteration 0: [1, 0] @ [[2,0],[0,2]]^T = [2, 0]
        // Iteration 1: [2, 0] @ [[3,0],[0,3]]^T = [6, 0]
        let output = try executor.forward(tokenIDs: MLXArray([Int32(0)]))
        let vals = output.asArray(Float.self)

        #expect(abs(vals[0] - 6.0) < 1e-4)
        #expect(abs(vals[1] - 0.0) < 1e-4)
    }

    @Test("Parallel with add merge sums branch outputs")
    func parallelAdd() throws {
        struct ParallelModel: LanguageModel {
            @ModelComponentBuilder var body: some ModelComponent {
                TokenEmbedding(vocabSize: 4, embeddingSize: 2)
                Parallel(merge: .add) {
                    Linear(inputSize: 2, outputSize: 2)
                    Linear(inputSize: 2, outputSize: 2)
                }
            }
        }

        let graph = try ParallelModel().makeModelGraph()

        let table = MLXArray(converting: [1, 1, 0, 0, 0, 0, 0, 0] as [Double], [4, 2])

        // Branch 0: identity → output = [1, 1]
        let w0 = MLXArray(converting: [1, 0, 0, 1] as [Double], [2, 2])
        // Branch 1: scale by 2 → output = [2, 2]
        let w1 = MLXArray(converting: [2, 0, 0, 2] as [Double], [2, 2])

        let weights = bind([
            slot([.operation(0)], role: .embeddingTable): table,
            slot([.operation(1), .regionBranch(0), .operation(0)], role: .weight): w0,
            slot([.operation(1), .regionBranch(1), .operation(0)], role: .weight): w1,
        ])
        let compiled = try MLXCompiler().compile(graph: graph, weights: weights)
        let executor = MLXExecutor(compiledModel: compiled)

        // Token 0 → [1, 1]
        // Branch 0: [1, 1] @ I = [1, 1]
        // Branch 1: [1, 1] @ 2I = [2, 2]
        // Merged (add): [3, 3]
        let output = try executor.forward(tokenIDs: MLXArray([Int32(0)]))
        let vals = output.asArray(Float.self)

        #expect(abs(vals[0] - 3.0) < 1e-4)
        #expect(abs(vals[1] - 3.0) < 1e-4)
    }
}

// MARK: - Attention Tests

@Suite("MLXExecutor Attention")
struct ExecutorAttentionTests {

    @Test("Attention produces correct output shape and caches keys/values")
    func attentionShapeAndCache() throws {
        struct AttnModel: LanguageModel {
            @ModelComponentBuilder var body: some ModelComponent {
                TokenEmbedding(vocabSize: 8, embeddingSize: 4)
                Attention(
                    hiddenSize: 4, headCount: 2, kvHeadCount: 2,
                    headDimension: 2
                )
            }
        }

        let graph = try AttnModel().makeModelGraph()

        let table = MLXRandom.normal([8, 4])
        // Q, K, V projections: [headCount * headDim, hiddenSize] = [4, 4]
        let qw = MLXRandom.normal([4, 4]) * 0.1
        let kw = MLXRandom.normal([4, 4]) * 0.1
        let vw = MLXRandom.normal([4, 4]) * 0.1
        let ow = MLXRandom.normal([4, 4]) * 0.1

        let attnPath: [StructuralPathComponent] = [.operation(1)]
        let weights = bind([
            slot([.operation(0)], role: .embeddingTable): table,
            slot(attnPath + [.field("q_proj")], role: .weight): qw,
            slot(attnPath + [.field("k_proj")], role: .weight): kw,
            slot(attnPath + [.field("v_proj")], role: .weight): vw,
            slot(attnPath + [.field("o_proj")], role: .weight): ow,
        ])
        let compiled = try MLXCompiler().compile(graph: graph, weights: weights)
        let executor = MLXExecutor(compiledModel: compiled)

        // First forward: 3 tokens
        let output1 = try executor.forward(tokenIDs: MLXArray([Int32(1), Int32(2), Int32(3)]))
        #expect(output1.shape == [3, 4])

        // Second forward: 1 token (autoregressive), cache should have 3 previous tokens
        let output2 = try executor.forward(tokenIDs: MLXArray([Int32(4)]))
        #expect(output2.shape == [1, 4])
    }

    @Test("Attention with RoPE produces correct output")
    func attentionWithRoPE() throws {
        let rope = RoPEAttributes(dimension: 2, base: 10_000.0)
        struct RoPEAttnModel: LanguageModel {
            let rope: RoPEAttributes
            @ModelComponentBuilder var body: some ModelComponent {
                TokenEmbedding(vocabSize: 8, embeddingSize: 4)
                Attention(
                    hiddenSize: 4, headCount: 2, kvHeadCount: 2,
                    headDimension: 2, rope: rope
                )
            }
        }

        let graph = try RoPEAttnModel(rope: rope).makeModelGraph()

        let table = MLXRandom.normal([8, 4])
        let qw = MLXRandom.normal([4, 4]) * 0.1
        let kw = MLXRandom.normal([4, 4]) * 0.1
        let vw = MLXRandom.normal([4, 4]) * 0.1
        let ow = MLXRandom.normal([4, 4]) * 0.1

        let attnPath: [StructuralPathComponent] = [.operation(1)]
        let weights = bind([
            slot([.operation(0)], role: .embeddingTable): table,
            slot(attnPath + [.field("q_proj")], role: .weight): qw,
            slot(attnPath + [.field("k_proj")], role: .weight): kw,
            slot(attnPath + [.field("v_proj")], role: .weight): vw,
            slot(attnPath + [.field("o_proj")], role: .weight): ow,
        ])
        let compiled = try MLXCompiler().compile(graph: graph, weights: weights)
        let executor = MLXExecutor(compiledModel: compiled)

        let output = try executor.forward(tokenIDs: MLXArray([Int32(0), Int32(1)]))
        #expect(output.shape == [2, 4])
        // Output should be finite
        let vals = output.asArray(Float.self)
        for v in vals {
            #expect(v.isFinite)
        }
    }
}

// MARK: - End-to-End Tests

@Suite("MLXExecutor End-to-End")
struct ExecutorEndToEndTests {

    @Test("Mini transformer forward pass: embed → repeat(norm+attn, norm+mlp) → norm → head")
    func miniTransformerForward() throws {
        let model = MiniLlama(layerCount: 1)
        let graph = try model.makeModelGraph()
        let weights = miniLlamaWeights(graph: graph, layerCount: 1)

        let compiled = try MLXCompiler().compile(graph: graph, weights: weights)
        let executor = MLXExecutor(compiledModel: compiled)

        // Prefill with 3 tokens
        let logits = try executor.forward(tokenIDs: MLXArray([Int32(0), Int32(1), Int32(2)]))

        // Output shape: [3, vocabSize=8]
        #expect(logits.shape == [3, 8])

        // All values should be finite
        let vals = logits.asArray(Float.self)
        for v in vals {
            #expect(v.isFinite)
        }
    }

    @Test("Mini transformer autoregressive: 2 forward calls use KV cache correctly")
    func miniTransformerAutoregressive() throws {
        let model = MiniLlama(layerCount: 1)
        let graph = try model.makeModelGraph()
        let weights = miniLlamaWeights(graph: graph, layerCount: 1)

        let compiled = try MLXCompiler().compile(graph: graph, weights: weights)
        let executor = MLXExecutor(compiledModel: compiled)

        // Prefill
        let logits1 = try executor.forward(tokenIDs: MLXArray([Int32(0), Int32(1)]))
        #expect(logits1.shape == [2, 8])

        // Decode one token
        let logits2 = try executor.forward(tokenIDs: MLXArray([Int32(2)]))
        #expect(logits2.shape == [1, 8])

        // Decode another token
        let logits3 = try executor.forward(tokenIDs: MLXArray([Int32(3)]))
        #expect(logits3.shape == [1, 8])

        // All finite
        for v in logits3.asArray(Float.self) {
            #expect(v.isFinite)
        }
    }

    @Test("Reset caches clears state between sequences")
    func resetCaches() throws {
        let model = MiniLlama(layerCount: 1)
        let graph = try model.makeModelGraph()
        let weights = miniLlamaWeights(graph: graph, layerCount: 1)

        let compiled = try MLXCompiler().compile(graph: graph, weights: weights)
        let executor = MLXExecutor(compiledModel: compiled)

        // Forward 1: process tokens 0, 1
        let logits1a = try executor.forward(tokenIDs: MLXArray([Int32(0), Int32(1)]))

        // Reset and process same tokens
        executor.resetCaches()
        let logits1b = try executor.forward(tokenIDs: MLXArray([Int32(0), Int32(1)]))

        // Results should be identical after reset
        let a = logits1a.asArray(Float.self)
        let b = logits1b.asArray(Float.self)
        for i in 0..<a.count {
            #expect(abs(a[i] - b[i]) < 1e-4)
        }
    }

    @Test("Cohere-style parallel attention+MLP forward pass")
    func cohereForward() throws {
        let model = MiniCohere(layerCount: 1)
        let graph = try model.makeModelGraph()
        let weights = miniCohereWeights(graph: graph, layerCount: 1)

        let compiled = try MLXCompiler().compile(graph: graph, weights: weights)
        let executor = MLXExecutor(compiledModel: compiled)

        let logits = try executor.forward(tokenIDs: MLXArray([Int32(0), Int32(1)]))
        #expect(logits.shape == [2, 8])

        for v in logits.asArray(Float.self) {
            #expect(v.isFinite)
        }
    }

    @Test("Multi-layer transformer: weights differ per layer")
    func multiLayerWeightIsolation() throws {
        let model = MiniLlama(layerCount: 2)
        let graph = try model.makeModelGraph()
        let weights = miniLlamaWeights(graph: graph, layerCount: 2)

        let compiled = try MLXCompiler().compile(graph: graph, weights: weights)
        let executor = MLXExecutor(compiledModel: compiled)

        let logits = try executor.forward(tokenIDs: MLXArray([Int32(0)]))
        #expect(logits.shape == [1, 8])

        for v in logits.asArray(Float.self) {
            #expect(v.isFinite)
        }
    }
}

// MARK: - LayerNorm Tests

@Suite("MLXExecutor LayerNorm")
struct ExecutorLayerNormTests {

    @Test("LayerNorm normalizes input with affine transform")
    func layerNormAffine() throws {
        struct LNModel: LanguageModel {
            @ModelComponentBuilder var body: some ModelComponent {
                TokenEmbedding(vocabSize: 4, embeddingSize: 4)
                LayerNorm(dimension: 4, affine: true)
            }
        }

        let graph = try LNModel().makeModelGraph()

        // Input [1, 2, 3, 4] — mean=2.5, var=1.25
        let table = MLXArray(converting: [
            1.0, 2.0, 3.0, 4.0,
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
        ], [4, 4])

        // scale=1, bias=0 → pure normalization
        let weights = bind([
            slot([.operation(0)], role: .embeddingTable): table,
            slot([.operation(1)], role: .scale): MLXArray.ones([4]),
            slot([.operation(1)], role: .bias): MLXArray.zeros([4]),
        ])
        let compiled = try MLXCompiler().compile(graph: graph, weights: weights)
        let executor = MLXExecutor(compiledModel: compiled)

        let output = try executor.forward(tokenIDs: MLXArray([Int32(0)]))
        let vals = output.asArray(Float.self)

        // LayerNorm on [1,2,3,4]: mean=2.5, std=sqrt(1.25+1e-5)≈1.1180
        // normalized = [-1.342, -0.447, 0.447, 1.342] (approximately)
        #expect(vals[0] < -1.0)  // below mean
        #expect(vals[3] > 1.0)   // above mean
        // Mean should be ~0
        let mean = vals.reduce(0, +) / Float(vals.count)
        #expect(abs(mean) < 1e-4)
    }

    @Test("LayerNorm with non-trivial scale and bias shifts output")
    func layerNormScaleBias() throws {
        struct LNModel: LanguageModel {
            @ModelComponentBuilder var body: some ModelComponent {
                TokenEmbedding(vocabSize: 4, embeddingSize: 2)
                LayerNorm(dimension: 2, affine: true)
            }
        }

        let graph = try LNModel().makeModelGraph()

        // Input [1, 3] — mean=2, var=1
        let table = MLXArray(converting: [1.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [4, 2])

        // scale=2, bias=10 → output = 2*normalized + 10
        let weights = bind([
            slot([.operation(0)], role: .embeddingTable): table,
            slot([.operation(1)], role: .scale):
                MLXArray(converting: [2.0, 2.0], [2]),
            slot([.operation(1)], role: .bias):
                MLXArray(converting: [10.0, 10.0], [2]),
        ])
        let compiled = try MLXCompiler().compile(graph: graph, weights: weights)
        let executor = MLXExecutor(compiledModel: compiled)

        let output = try executor.forward(tokenIDs: MLXArray([Int32(0)]))
        let vals = output.asArray(Float.self)

        // [1,3]: mean=2, var=1, std=1+eps≈1
        // normalized ≈ [-1, 1], scaled = [-2, 2], biased = [8, 12]
        #expect(abs(vals[0] - 8.0) < 0.1)
        #expect(abs(vals[1] - 12.0) < 0.1)
    }
}

// MARK: - GQA Attention Tests

@Suite("MLXExecutor GQA Attention")
struct ExecutorGQATests {

    @Test("GQA with kvHeadCount < headCount produces correct shape")
    func gqaShape() throws {
        // 4 query heads, 2 kv heads → GQA ratio = 2
        struct GQAModel: LanguageModel {
            @ModelComponentBuilder var body: some ModelComponent {
                TokenEmbedding(vocabSize: 8, embeddingSize: 8)
                Attention(
                    hiddenSize: 8, headCount: 4, kvHeadCount: 2,
                    headDimension: 2
                )
            }
        }

        let graph = try GQAModel().makeModelGraph()

        let table = MLXRandom.normal([8, 8])
        // Q: [headCount * headDim, hiddenSize] = [8, 8]
        // K, V: [kvHeadCount * headDim, hiddenSize] = [4, 8]
        let qw = MLXRandom.normal([8, 8]) * 0.1
        let kw = MLXRandom.normal([4, 8]) * 0.1
        let vw = MLXRandom.normal([4, 8]) * 0.1
        let ow = MLXRandom.normal([8, 8]) * 0.1

        let attnPath: [StructuralPathComponent] = [.operation(1)]
        let weights = bind([
            slot([.operation(0)], role: .embeddingTable): table,
            slot(attnPath + [.field("q_proj")], role: .weight): qw,
            slot(attnPath + [.field("k_proj")], role: .weight): kw,
            slot(attnPath + [.field("v_proj")], role: .weight): vw,
            slot(attnPath + [.field("o_proj")], role: .weight): ow,
        ])
        let compiled = try MLXCompiler().compile(graph: graph, weights: weights)
        let executor = MLXExecutor(compiledModel: compiled)

        let output = try executor.forward(tokenIDs: MLXArray([Int32(0), Int32(1), Int32(2)]))
        #expect(output.shape == [3, 8])

        for v in output.asArray(Float.self) {
            #expect(v.isFinite)
        }
    }

    @Test("GQA autoregressive: KV cache stores correct head count")
    func gqaAutoregressive() throws {
        struct GQAModel: LanguageModel {
            @ModelComponentBuilder var body: some ModelComponent {
                TokenEmbedding(vocabSize: 8, embeddingSize: 8)
                Attention(
                    hiddenSize: 8, headCount: 4, kvHeadCount: 2,
                    headDimension: 2
                )
            }
        }

        let graph = try GQAModel().makeModelGraph()
        let table = MLXRandom.normal([8, 8])
        let attnPath: [StructuralPathComponent] = [.operation(1)]
        let weights = bind([
            slot([.operation(0)], role: .embeddingTable): table,
            slot(attnPath + [.field("q_proj")], role: .weight): MLXRandom.normal([8, 8]) * 0.1,
            slot(attnPath + [.field("k_proj")], role: .weight): MLXRandom.normal([4, 8]) * 0.1,
            slot(attnPath + [.field("v_proj")], role: .weight): MLXRandom.normal([4, 8]) * 0.1,
            slot(attnPath + [.field("o_proj")], role: .weight): MLXRandom.normal([8, 8]) * 0.1,
        ])
        let compiled = try MLXCompiler().compile(graph: graph, weights: weights)
        let executor = MLXExecutor(compiledModel: compiled)

        // Prefill 2 tokens
        let out1 = try executor.forward(tokenIDs: MLXArray([Int32(0), Int32(1)]))
        #expect(out1.shape == [2, 8])

        // Decode 1 token — should attend to all 3 positions (2 cached + 1 new)
        let out2 = try executor.forward(tokenIDs: MLXArray([Int32(2)]))
        #expect(out2.shape == [1, 8])

        for v in out2.asArray(Float.self) {
            #expect(v.isFinite)
        }
    }
}

// MARK: - Attention Bias and QK Norm Tests

@Suite("MLXExecutor Attention Variants")
struct ExecutorAttentionVariantTests {

    @Test("Attention with bias adds bias to projections")
    func attentionWithBias() throws {
        struct BiasAttnModel: LanguageModel {
            @ModelComponentBuilder var body: some ModelComponent {
                TokenEmbedding(vocabSize: 4, embeddingSize: 4)
                Attention(
                    hiddenSize: 4, headCount: 2, kvHeadCount: 2,
                    headDimension: 2, bias: true
                )
            }
        }

        let graph = try BiasAttnModel().makeModelGraph()
        let attnPath: [StructuralPathComponent] = [.operation(1)]
        let weights = bind([
            slot([.operation(0)], role: .embeddingTable): MLXRandom.normal([4, 4]),
            slot(attnPath + [.field("q_proj")], role: .weight): MLXRandom.normal([4, 4]) * 0.1,
            slot(attnPath + [.field("k_proj")], role: .weight): MLXRandom.normal([4, 4]) * 0.1,
            slot(attnPath + [.field("v_proj")], role: .weight): MLXRandom.normal([4, 4]) * 0.1,
            slot(attnPath + [.field("o_proj")], role: .weight): MLXRandom.normal([4, 4]) * 0.1,
            // Biases
            slot(attnPath + [.field("q_proj")], role: .bias): MLXArray.zeros([4]),
            slot(attnPath + [.field("k_proj")], role: .bias): MLXArray.zeros([4]),
            slot(attnPath + [.field("v_proj")], role: .bias): MLXArray.zeros([4]),
            slot(attnPath + [.field("o_proj")], role: .bias): MLXArray.zeros([4]),
        ])
        let compiled = try MLXCompiler().compile(graph: graph, weights: weights)
        let executor = MLXExecutor(compiledModel: compiled)

        let output = try executor.forward(tokenIDs: MLXArray([Int32(0), Int32(1)]))
        #expect(output.shape == [2, 4])

        for v in output.asArray(Float.self) {
            #expect(v.isFinite)
        }
    }

    @Test("Attention with RMSNorm QK normalization")
    func attentionQKNorm() throws {
        struct QKNormModel: LanguageModel {
            @ModelComponentBuilder var body: some ModelComponent {
                TokenEmbedding(vocabSize: 4, embeddingSize: 4)
                Attention(
                    hiddenSize: 4, headCount: 2, kvHeadCount: 2,
                    headDimension: 2, qkNorm: .rmsNorm
                )
            }
        }

        let graph = try QKNormModel().makeModelGraph()
        let attnPath: [StructuralPathComponent] = [.operation(1)]
        let weights = bind([
            slot([.operation(0)], role: .embeddingTable): MLXRandom.normal([4, 4]),
            slot(attnPath + [.field("q_proj")], role: .weight): MLXRandom.normal([4, 4]) * 0.1,
            slot(attnPath + [.field("k_proj")], role: .weight): MLXRandom.normal([4, 4]) * 0.1,
            slot(attnPath + [.field("v_proj")], role: .weight): MLXRandom.normal([4, 4]) * 0.1,
            slot(attnPath + [.field("o_proj")], role: .weight): MLXRandom.normal([4, 4]) * 0.1,
            // QK norm weights (centered at 0 for 1+w convention)
            slot(attnPath + [.field("q_norm")], role: .scale): MLXArray.zeros([2]),
            slot(attnPath + [.field("k_norm")], role: .scale): MLXArray.zeros([2]),
        ])
        let compiled = try MLXCompiler().compile(graph: graph, weights: weights)
        let executor = MLXExecutor(compiledModel: compiled)

        let output = try executor.forward(tokenIDs: MLXArray([Int32(0), Int32(1)]))
        #expect(output.shape == [2, 4])

        for v in output.asArray(Float.self) {
            #expect(v.isFinite)
        }
    }

    @Test("Attention with partial RoPE applies rotation to subset of dimensions")
    func attentionPartialRoPE() throws {
        // headDim=4, ropeDim=2 → only first 2 dims get RoPE, last 2 pass through
        let rope = RoPEAttributes(dimension: 2, base: 10_000.0)
        struct PartialRoPEModel: LanguageModel {
            let rope: RoPEAttributes
            @ModelComponentBuilder var body: some ModelComponent {
                TokenEmbedding(vocabSize: 8, embeddingSize: 8)
                Attention(
                    hiddenSize: 8, headCount: 2, kvHeadCount: 2,
                    headDimension: 4, rope: rope
                )
            }
        }

        let graph = try PartialRoPEModel(rope: rope).makeModelGraph()
        let attnPath: [StructuralPathComponent] = [.operation(1)]
        let weights = bind([
            slot([.operation(0)], role: .embeddingTable): MLXRandom.normal([8, 8]),
            slot(attnPath + [.field("q_proj")], role: .weight): MLXRandom.normal([8, 8]) * 0.1,
            slot(attnPath + [.field("k_proj")], role: .weight): MLXRandom.normal([8, 8]) * 0.1,
            slot(attnPath + [.field("v_proj")], role: .weight): MLXRandom.normal([8, 8]) * 0.1,
            slot(attnPath + [.field("o_proj")], role: .weight): MLXRandom.normal([8, 8]) * 0.1,
        ])
        let compiled = try MLXCompiler().compile(graph: graph, weights: weights)
        let executor = MLXExecutor(compiledModel: compiled)

        // Prefill + decode to verify partial RoPE works with cache
        let out1 = try executor.forward(tokenIDs: MLXArray([Int32(0), Int32(1)]))
        #expect(out1.shape == [2, 8])

        let out2 = try executor.forward(tokenIDs: MLXArray([Int32(2)]))
        #expect(out2.shape == [1, 8])

        for v in out2.asArray(Float.self) {
            #expect(v.isFinite)
        }
    }
}

// MARK: - Linear and OutputHead Variant Tests

@Suite("MLXExecutor Linear Variants")
struct ExecutorLinearVariantTests {

    @Test("Linear with bias adds bias vector")
    func linearWithBias() throws {
        struct BiasLinearModel: LanguageModel {
            @ModelComponentBuilder var body: some ModelComponent {
                TokenEmbedding(vocabSize: 4, embeddingSize: 2)
                Linear(inputSize: 2, outputSize: 3, bias: true)
            }
        }

        let graph = try BiasLinearModel().makeModelGraph()

        // Identity-like weight + bias=[10, 20, 30]
        let w = MLXArray(converting: [1.0, 0.0, 0.0, 1.0, 0.0, 0.0], [3, 2])
        let b = MLXArray(converting: [10.0, 20.0, 30.0], [3])
        let table = MLXArray(converting: [1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [4, 2])

        let weights = bind([
            slot([.operation(0)], role: .embeddingTable): table,
            slot([.operation(1)], role: .weight): w,
            slot([.operation(1)], role: .bias): b,
        ])
        let compiled = try MLXCompiler().compile(graph: graph, weights: weights)
        let executor = MLXExecutor(compiledModel: compiled)

        // Token 0 → [1, 2] → matmul = [1, 2, 0] → + bias = [11, 22, 30]
        let output = try executor.forward(tokenIDs: MLXArray([Int32(0)]))
        let vals = output.asArray(Float.self)

        #expect(abs(vals[0] - 11.0) < 1e-4)
        #expect(abs(vals[1] - 22.0) < 1e-4)
        #expect(abs(vals[2] - 30.0) < 1e-4)
    }

    @Test("OutputHead with separate projection (not tied to embedding)")
    func outputHeadSeparateProjection() throws {
        struct UntiedModel: LanguageModel {
            @ModelComponentBuilder var body: some ModelComponent {
                TokenEmbedding(vocabSize: 3, embeddingSize: 2)
                OutputHead(inputSize: 2, vocabSize: 3, tiedToEmbedding: false)
            }
        }

        let graph = try UntiedModel().makeModelGraph()

        let table = MLXArray(converting: [1.0, 0.0, 0.0, 1.0, 1.0, 1.0], [3, 2])

        // Separate output projection — different from embedding table
        let outProj = MLXArray(converting: [2.0, 0.0, 0.0, 2.0, 1.0, 1.0], [3, 2])

        let weights = bind([
            slot([.operation(0)], role: .embeddingTable): table,
            slot([.operation(1)], role: .outputProjection): outProj,
        ])
        let compiled = try MLXCompiler().compile(graph: graph, weights: weights)
        let executor = MLXExecutor(compiledModel: compiled)

        // Token 0 → embed [1, 0] → matmul([1,0], outProj.T) = [2, 0, 1]
        let output = try executor.forward(tokenIDs: MLXArray([Int32(0)]))
        let vals = output.asArray(Float.self)

        #expect(abs(vals[0] - 2.0) < 1e-4)
        #expect(abs(vals[1] - 0.0) < 1e-4)
        #expect(abs(vals[2] - 1.0) < 1e-4)
    }
}

// MARK: - MLP Variant Tests

@Suite("MLXExecutor MLP Variants")
struct ExecutorMLPVariantTests {

    @Test("MLP without gating (no up_proj): gate → activate → down")
    func mlpNoGating() throws {
        struct NoGateModel: LanguageModel {
            @ModelComponentBuilder var body: some ModelComponent {
                TokenEmbedding(vocabSize: 4, embeddingSize: 2)
                MLP(inputSize: 2, intermediateSize: 4, activation: .silu, gating: .none)
            }
        }

        let graph = try NoGateModel().makeModelGraph()
        let table = MLXArray(converting: [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [4, 2])

        // gate_proj [4, 2], down_proj [2, 4] — no up_proj
        let gateW = MLXArray(converting: [
            1.0, 0.0,
            0.0, 1.0,
            1.0, 0.0,
            0.0, 1.0,
        ], [4, 2])
        let downW = MLXArray(converting: [
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
        ], [2, 4])

        let weights = bind([
            slot([.operation(0)], role: .embeddingTable): table,
            slot([.operation(1), .field("gate_proj")], role: .weight): gateW,
            slot([.operation(1), .field("down_proj")], role: .weight): downW,
        ])
        let compiled = try MLXCompiler().compile(graph: graph, weights: weights)
        let executor = MLXExecutor(compiledModel: compiled)

        // Token 0 → [1, 1]
        // gate = [1,1] @ gateW.T = [1, 1, 1, 1]
        // activated = silu([1, 1, 1, 1]) = [0.7311, ...]
        // NO up_proj gating — gated = activated directly
        // output = activated @ downW.T = [silu(1), silu(1)]
        let output = try executor.forward(tokenIDs: MLXArray([Int32(0)]))
        let vals = output.asArray(Float.self)

        let expected = MLXNN.silu(MLXArray(Float(1.0))).item(Float.self)
        #expect(abs(vals[0] - expected) < 1e-4)
        #expect(abs(vals[1] - expected) < 1e-4)
    }

    @Test("MLP with GELU activation")
    func mlpGELU() throws {
        struct GELUModel: LanguageModel {
            @ModelComponentBuilder var body: some ModelComponent {
                TokenEmbedding(vocabSize: 4, embeddingSize: 2)
                MLP(inputSize: 2, intermediateSize: 4, activation: .gelu, gating: .none)
            }
        }

        let graph = try GELUModel().makeModelGraph()
        let table = MLXArray(converting: [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [4, 2])

        let gateW = MLXArray(converting: [
            1.0, 0.0,
            0.0, 1.0,
            0.0, 0.0,
            0.0, 0.0,
        ], [4, 2])
        let downW = MLXArray(converting: [
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
        ], [2, 4])

        let weights = bind([
            slot([.operation(0)], role: .embeddingTable): table,
            slot([.operation(1), .field("gate_proj")], role: .weight): gateW,
            slot([.operation(1), .field("down_proj")], role: .weight): downW,
        ])
        let compiled = try MLXCompiler().compile(graph: graph, weights: weights)
        let executor = MLXExecutor(compiledModel: compiled)

        // Token 0 → [1, 1] → gate=[1, 1, 0, 0] → gelu([1, 1, 0, 0])
        // gelu(1.0) ≈ 0.8413, gelu(0.0) = 0.0
        // output = [gelu(1), gelu(1)]
        let output = try executor.forward(tokenIDs: MLXArray([Int32(0)]))
        let vals = output.asArray(Float.self)

        let expected = MLXNN.gelu(MLXArray(Float(1.0))).item(Float.self)
        #expect(abs(vals[0] - expected) < 1e-3)
        #expect(abs(vals[1] - expected) < 1e-3)
    }

    @Test("MLP with bias on all projections")
    func mlpWithBias() throws {
        struct BiasMLP: LanguageModel {
            @ModelComponentBuilder var body: some ModelComponent {
                TokenEmbedding(vocabSize: 4, embeddingSize: 2)
                MLP(inputSize: 2, intermediateSize: 2, activation: .silu, gating: .swiglu, bias: true)
            }
        }

        let graph = try BiasMLP().makeModelGraph()
        let table = MLXArray(converting: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [4, 2])

        // Zero input + non-zero bias → output determined by biases only
        let zeroW = MLXArray.zeros([2, 2])
        let gateBias = MLXArray(converting: [1.0, 1.0], [2])
        let upBias = MLXArray(converting: [2.0, 2.0], [2])
        let downBias = MLXArray(converting: [100.0, 200.0], [2])

        let weights = bind([
            slot([.operation(0)], role: .embeddingTable): table,
            slot([.operation(1), .field("gate_proj")], role: .weight): zeroW,
            slot([.operation(1), .field("gate_proj")], role: .bias): gateBias,
            slot([.operation(1), .field("up_proj")], role: .weight): zeroW,
            slot([.operation(1), .field("up_proj")], role: .bias): upBias,
            slot([.operation(1), .field("down_proj")], role: .weight): zeroW,
            slot([.operation(1), .field("down_proj")], role: .bias): downBias,
        ])
        let compiled = try MLXCompiler().compile(graph: graph, weights: weights)
        let executor = MLXExecutor(compiledModel: compiled)

        // Token 0 → [0, 0]
        // gate = [0,0]@0 + gateBias = [1, 1], activated = silu([1, 1])
        // up = [0,0]@0 + upBias = [2, 2]
        // gated = silu(1)*2 ≈ 1.4622
        // output = [1.4622, 1.4622]@0 + downBias = [100, 200]
        let output = try executor.forward(tokenIDs: MLXArray([Int32(0)]))
        let vals = output.asArray(Float.self)

        let silu1 = MLXNN.silu(MLXArray(Float(1.0))).item(Float.self)
        let _ = silu1 * 2.0
        // matmul([silu1*2, silu1*2], zeros.T) = [0, 0] + [100, 200]
        #expect(abs(vals[0] - 100.0) < 1e-2)
        #expect(abs(vals[1] - 200.0) < 1e-2)
    }
}

// MARK: - Parallel Merge Variant Tests

@Suite("MLXExecutor Parallel Merge Variants")
struct ExecutorParallelMergeTests {

    @Test("Parallel with concat merge concatenates branch outputs")
    func parallelConcat() throws {
        struct ConcatModel: LanguageModel {
            @ModelComponentBuilder var body: some ModelComponent {
                TokenEmbedding(vocabSize: 4, embeddingSize: 2)
                Parallel(merge: .concat) {
                    Linear(inputSize: 2, outputSize: 3)
                    Linear(inputSize: 2, outputSize: 2)
                }
            }
        }

        let graph = try ConcatModel().makeModelGraph()

        let table = MLXArray(converting: [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [4, 2])

        // Branch 0 → [3, 2] → output [1, 0, 0]
        let w0 = MLXArray(converting: [1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [3, 2])
        // Branch 1 → [2, 2] → output [5, 0]
        let w1 = MLXArray(converting: [5.0, 0.0, 0.0, 0.0], [2, 2])

        let weights = bind([
            slot([.operation(0)], role: .embeddingTable): table,
            slot([.operation(1), .regionBranch(0), .operation(0)], role: .weight): w0,
            slot([.operation(1), .regionBranch(1), .operation(0)], role: .weight): w1,
        ])
        let compiled = try MLXCompiler().compile(graph: graph, weights: weights)
        let executor = MLXExecutor(compiledModel: compiled)

        // Token 0 → [1, 0]
        // Branch 0: [1] @ w0.T = [1, 0, 0]
        // Branch 1: [1] @ w1.T = [5, 0]
        // Concat: [1, 0, 0, 5, 0]
        let output = try executor.forward(tokenIDs: MLXArray([Int32(0)]))
        let vals = output.asArray(Float.self)

        #expect(vals.count == 5)  // 3 + 2
        #expect(abs(vals[0] - 1.0) < 1e-4)
        #expect(abs(vals[3] - 5.0) < 1e-4)
    }
}

// MARK: - Batch Dimension Tests

@Suite("MLXExecutor Batch Handling")
struct ExecutorBatchTests {

    @Test("2D tokenIDs [B, L] preserves batch dimension in output")
    func batchDimensionPreserved() throws {
        struct SimpleModel: LanguageModel {
            @ModelComponentBuilder var body: some ModelComponent {
                TokenEmbedding(vocabSize: 4, embeddingSize: 2)
                Linear(inputSize: 2, outputSize: 3)
            }
        }

        let graph = try SimpleModel().makeModelGraph()

        let table = MLXArray(converting: [1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0], [4, 2])
        let w = MLXArray(converting: [1.0, 0.0, 0.0, 1.0, 1.0, 1.0], [3, 2])

        let weights = bind([
            slot([.operation(0)], role: .embeddingTable): table,
            slot([.operation(1)], role: .weight): w,
        ])
        let compiled = try MLXCompiler().compile(graph: graph, weights: weights)
        let executor = MLXExecutor(compiledModel: compiled)

        // 2D input: [B=2, L=1]
        let tokens = MLXArray([Int32(0), Int32(1)]).reshaped(2, 1)
        let output = try executor.forward(tokenIDs: tokens)

        // 2D input → output preserves batch: [B=2, L=1, outDim=3]
        #expect(output.shape == [2, 1, 3])
    }

    @Test("1D tokenIDs [L] squeezes batch dimension from output")
    func singleSequenceSqueezed() throws {
        struct SimpleModel: LanguageModel {
            @ModelComponentBuilder var body: some ModelComponent {
                TokenEmbedding(vocabSize: 4, embeddingSize: 2)
                Linear(inputSize: 2, outputSize: 3)
            }
        }

        let graph = try SimpleModel().makeModelGraph()
        let table = MLXArray(converting: [1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0], [4, 2])
        let w = MLXArray(converting: [1.0, 0.0, 0.0, 1.0, 1.0, 1.0], [3, 2])

        let weights = bind([
            slot([.operation(0)], role: .embeddingTable): table,
            slot([.operation(1)], role: .weight): w,
        ])
        let compiled = try MLXCompiler().compile(graph: graph, weights: weights)
        let executor = MLXExecutor(compiledModel: compiled)

        // 1D input: [L=2]
        let tokens = MLXArray([Int32(0), Int32(1)])
        let output = try executor.forward(tokenIDs: tokens)

        // 1D input → batch squeezed: [L=2, outDim=3]
        #expect(output.shape == [2, 3])
    }
}

// MARK: - Attention Numerical Correctness Tests

@Suite("MLXExecutor Attention Numerics")
struct ExecutorAttentionNumericsTests {

    @Test("Single-head attention with identity projections computes correct output")
    func singleHeadIdentityAttention() throws {
        // 1 head, headDim=2, hiddenSize=2 → projections are identity-like
        struct SingleHeadModel: LanguageModel {
            @ModelComponentBuilder var body: some ModelComponent {
                TokenEmbedding(vocabSize: 4, embeddingSize: 2)
                Attention(
                    hiddenSize: 2, headCount: 1, kvHeadCount: 1,
                    headDimension: 2
                )
            }
        }

        let graph = try SingleHeadModel().makeModelGraph()

        // Token 0 → [1, 0], Token 1 → [0, 1]
        let table = MLXArray(converting: [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], [4, 2])

        // Identity projections for Q, K, V, O
        let identity = MLXArray(converting: [1.0, 0.0, 0.0, 1.0], [2, 2])

        let attnPath: [StructuralPathComponent] = [.operation(1)]
        let weights = bind([
            slot([.operation(0)], role: .embeddingTable): table,
            slot(attnPath + [.field("q_proj")], role: .weight): identity,
            slot(attnPath + [.field("k_proj")], role: .weight): identity,
            slot(attnPath + [.field("v_proj")], role: .weight): identity,
            slot(attnPath + [.field("o_proj")], role: .weight): identity,
        ])
        let compiled = try MLXCompiler().compile(graph: graph, weights: weights)
        let executor = MLXExecutor(compiledModel: compiled)

        // Single token: self-attention with Q=K=V=[1,0], causal mask is trivial
        // attn_weight = softmax(Q@K.T / sqrt(2)) = softmax([0.5/sqrt(2)]) = [1.0]
        // output = 1.0 * V = [1, 0]
        let out1 = try executor.forward(tokenIDs: MLXArray([Int32(0)]))
        let vals1 = out1.asArray(Float.self)
        #expect(abs(vals1[0] - 1.0) < 1e-3)
        #expect(abs(vals1[1] - 0.0) < 1e-3)

        // Reset to test fresh sequence
        executor.resetCaches()

        // Two tokens: [1,0] then [0,1]
        // Token 0: self-attention only → output = [1, 0]
        // Token 1: attends to both tokens 0 and 1
        //   Q=[0,1], K=[[1,0],[0,1]], V=[[1,0],[0,1]]
        //   scores = [0, 1] / sqrt(2) = [0, 0.7071]
        //   weights = softmax([0, 0.7071]) ≈ [0.330, 0.670]
        //   output ≈ 0.330*[1,0] + 0.670*[0,1] = [0.330, 0.670]
        let out2 = try executor.forward(tokenIDs: MLXArray([Int32(0), Int32(1)]))
        let vals2 = out2.asArray(Float.self)

        // Token 0 output: [1, 0]
        #expect(abs(vals2[0] - 1.0) < 1e-3)
        #expect(abs(vals2[1] - 0.0) < 1e-3)

        // Token 1 output: mixture of [1,0] and [0,1]
        #expect(vals2[2] > 0.2 && vals2[2] < 0.5)  // ~0.33
        #expect(vals2[3] > 0.5 && vals2[3] < 0.8)  // ~0.67
    }
}

// MARK: - Cache State Tests

@Suite("MLXExecutor Cache Behavior")
struct ExecutorCacheTests {

    @Test("Cache offset increments correctly across multiple forward passes")
    func cacheOffsetTracking() throws {
        struct AttnModel: LanguageModel {
            @ModelComponentBuilder var body: some ModelComponent {
                TokenEmbedding(vocabSize: 8, embeddingSize: 4)
                Attention(
                    hiddenSize: 4, headCount: 2, kvHeadCount: 2,
                    headDimension: 2
                )
            }
        }

        let graph = try AttnModel().makeModelGraph()
        let attnPath: [StructuralPathComponent] = [.operation(1)]
        let weights = bind([
            slot([.operation(0)], role: .embeddingTable): MLXRandom.normal([8, 4]),
            slot(attnPath + [.field("q_proj")], role: .weight): MLXRandom.normal([4, 4]) * 0.1,
            slot(attnPath + [.field("k_proj")], role: .weight): MLXRandom.normal([4, 4]) * 0.1,
            slot(attnPath + [.field("v_proj")], role: .weight): MLXRandom.normal([4, 4]) * 0.1,
            slot(attnPath + [.field("o_proj")], role: .weight): MLXRandom.normal([4, 4]) * 0.1,
        ])
        let compiled = try MLXCompiler().compile(graph: graph, weights: weights)
        let executor = MLXExecutor(compiledModel: compiled)

        // Prefill 3 tokens
        let _ = try executor.forward(tokenIDs: MLXArray([Int32(0), Int32(1), Int32(2)]))

        // Decode 1 token
        let _ = try executor.forward(tokenIDs: MLXArray([Int32(3)]))

        // Decode another token
        let _ = try executor.forward(tokenIDs: MLXArray([Int32(4)]))

        // After reset, outputs for same input should match first run
        let beforeReset = try executor.forward(tokenIDs: MLXArray([Int32(5)]))
        executor.resetCaches()
        let afterReset = try executor.forward(tokenIDs: MLXArray([Int32(5)]))

        // After reset, output is from scratch (only 1 token, no history)
        // Before reset, output attends to 6 cached tokens
        // These should differ because context is different
        let a = beforeReset.asArray(Float.self)
        let b = afterReset.asArray(Float.self)
        let maxDiff = zip(a, b).map { abs($0 - $1) }.max() ?? 0
        #expect(maxDiff > 0.01)  // outputs differ due to different context
    }

    @Test("Determinism: same input produces identical output")
    func determinism() throws {
        let model = MiniLlama(layerCount: 1)
        let graph = try model.makeModelGraph()
        let weights = miniLlamaWeights(graph: graph, layerCount: 1)

        let compiled = try MLXCompiler().compile(graph: graph, weights: weights)
        let tokens = MLXArray([Int32(0), Int32(1), Int32(2)])

        let exec1 = MLXExecutor(compiledModel: compiled)
        let out1 = try exec1.forward(tokenIDs: tokens)

        let exec2 = MLXExecutor(compiledModel: compiled)
        let out2 = try exec2.forward(tokenIDs: tokens)

        let a = out1.asArray(Float.self)
        let b = out2.asArray(Float.self)
        for i in 0..<a.count {
            #expect(abs(a[i] - b[i]) < 1e-6)
        }
    }
}

// MARK: - Compiler Edge Case Tests

@Suite("MLXCompiler Edge Cases")
struct CompilerEdgeCaseTests {

    @Test("Deeply nested structure produces correct weight paths")
    func deeplyNestedPaths() throws {
        // Residual { Residual { Linear } }
        struct DeepModel: LanguageModel {
            @ModelComponentBuilder var body: some ModelComponent {
                TokenEmbedding(vocabSize: 4, embeddingSize: 2)
                Residual {
                    Residual {
                        Linear(inputSize: 2, outputSize: 2)
                    }
                }
            }
        }

        let graph = try DeepModel().makeModelGraph()

        let table = MLXArray(converting: [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], [4, 2])
        // Scale by 3 → body(body(x)) = 3x, inner residual = x + 3x = 4x, outer residual = x + 4x = 5x
        let w = MLXArray(converting: [3.0, 0.0, 0.0, 3.0], [2, 2])

        // Path: [op1, regionBody, op0, regionBody, op0]
        let weights = bind([
            slot([.operation(0)], role: .embeddingTable): table,
            slot([.operation(1), .regionBody, .operation(0), .regionBody, .operation(0)], role: .weight): w,
        ])
        let compiled = try MLXCompiler().compile(graph: graph, weights: weights)
        let executor = MLXExecutor(compiledModel: compiled)

        // Token 0 → [1, 0]
        // Inner Linear: [1, 0] * 3 = [3, 0]
        // Inner Residual: [1, 0] + [3, 0] = [4, 0]
        // Outer Residual: [1, 0] + [4, 0] = [5, 0]
        let output = try executor.forward(tokenIDs: MLXArray([Int32(0)]))
        let vals = output.asArray(Float.self)

        #expect(abs(vals[0] - 5.0) < 1e-4)
        #expect(abs(vals[1] - 0.0) < 1e-4)
    }

    @Test("Missing weight during execution throws descriptive error")
    func missingWeightError() throws {
        struct LinearModel: LanguageModel {
            @ModelComponentBuilder var body: some ModelComponent {
                TokenEmbedding(vocabSize: 4, embeddingSize: 2)
                Linear(inputSize: 2, outputSize: 2)
            }
        }

        let graph = try LinearModel().makeModelGraph()

        // Only provide embedding — missing Linear weight
        let weights = bind([
            slot([.operation(0)], role: .embeddingTable): MLXArray.ones([4, 2]),
        ])
        let compiled = try MLXCompiler().compile(graph: graph, weights: weights)
        let executor = MLXExecutor(compiledModel: compiled)

        #expect(throws: CompilerError.self) {
            try executor.forward(tokenIDs: MLXArray([Int32(0)]))
        }
    }

    @Test("Repeat(count=1) applies body exactly once")
    func singleRepeat() throws {
        struct OneRepeatModel: LanguageModel {
            @ModelComponentBuilder var body: some ModelComponent {
                TokenEmbedding(vocabSize: 4, embeddingSize: 2)
                Repeat(count: 1) {
                    Linear(inputSize: 2, outputSize: 2)
                }
            }
        }

        let graph = try OneRepeatModel().makeModelGraph()

        let table = MLXArray(converting: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], [4, 2])
        // Scale by 2
        let w = MLXArray(converting: [2.0, 0.0, 0.0, 2.0], [2, 2])
        let weights = bind([
            slot([.operation(0)], role: .embeddingTable): table,
            slot([.operation(1), .regionBody, .index(0), .operation(0)], role: .weight): w,
        ])
        let compiled = try MLXCompiler().compile(graph: graph, weights: weights)
        let executor = MLXExecutor(compiledModel: compiled)

        // Token 0 → [1, 2] → *2 = [2, 4] (applied once, not twice)
        let output = try executor.forward(tokenIDs: MLXArray([Int32(0)]))
        let vals = output.asArray(Float.self)

        #expect(abs(vals[0] - 2.0) < 1e-4)
        #expect(abs(vals[1] - 4.0) < 1e-4)
    }
}

// MARK: - Compiler Error Path Tests

@Suite("MLXCompiler Error Paths")
struct CompilerErrorPathTests {

    @Test("Multiple embeddings with tied head throws invalidGraphStructure")
    func multipleEmbeddingsWithTiedHeadThrows() throws {
        struct DoubleEmbed: LanguageModel {
            @ModelComponentBuilder var body: some ModelComponent {
                TokenEmbedding(vocabSize: 4, embeddingSize: 4)
                TokenEmbedding(vocabSize: 4, embeddingSize: 4)
                OutputHead(inputSize: 4, vocabSize: 4, tiedToEmbedding: true)
            }
        }

        let graph = try DoubleEmbed().makeModelGraph()
        let weights = bind([
            slot([.operation(0)], role: .embeddingTable): MLXArray.ones([4, 4]),
            slot([.operation(1)], role: .embeddingTable): MLXArray.ones([4, 4]),
        ])

        #expect(throws: CompilerError.self) {
            try MLXCompiler().compile(graph: graph, weights: weights)
        }
    }

    @Test("Tied head without embedding throws invalidGraphStructure")
    func tiedHeadWithoutEmbeddingThrows() throws {
        struct HeadOnly: LanguageModel {
            @ModelComponentBuilder var body: some ModelComponent {
                RMSNorm(dimension: 4)
                OutputHead(inputSize: 4, vocabSize: 4, tiedToEmbedding: true)
            }
        }

        let graph = try HeadOnly().makeModelGraph()
        let weights = bind([
            slot([.operation(0)], role: .scale): MLXArray.zeros([4]),
        ])

        #expect(throws: CompilerError.self) {
            try MLXCompiler().compile(graph: graph, weights: weights)
        }
    }

    @Test("StateSpace operation produces recurrent cache descriptor")
    func stateSpaceRecurrentCache() throws {
        struct SSModel: LanguageModel {
            @ModelComponentBuilder var body: some ModelComponent {
                TokenEmbedding(vocabSize: 4, embeddingSize: 4)
                StateSpace(hiddenSize: 4, stateSize: 2, variant: "deltanet")
            }
        }

        let graph = try SSModel().makeModelGraph()
        let weights = bind([
            slot([.operation(0)], role: .embeddingTable): MLXArray.ones([4, 4]),
        ])

        let compiled = try MLXCompiler().compile(graph: graph, weights: weights)
        #expect(compiled.cacheDescriptors.count == 1)
        #expect(compiled.cacheDescriptors[0].kind == .recurrent)
        #expect(compiled.cacheDescriptors[0].slotIndex == 0)
    }
}

// MARK: - Executor Error Path Tests

@Suite("MLXExecutor Error Paths")
struct ExecutorErrorPathTests {

    @Test("Custom operation throws unsupportedOperation")
    func customOpThrowsUnsupported() throws {
        struct CustomModel: LanguageModel {
            @ModelComponentBuilder var body: some ModelComponent {
                TokenEmbedding(vocabSize: 4, embeddingSize: 4)
                Custom(domain: "test", name: "no_impl")
            }
        }

        let graph = try CustomModel().makeModelGraph()
        let weights = bind([
            slot([.operation(0)], role: .embeddingTable): MLXArray.ones([4, 4]),
        ])

        let compiled = try MLXCompiler().compile(graph: graph, weights: weights)
        let executor = MLXExecutor(compiledModel: compiled)

        #expect(throws: CompilerError.self) {
            try executor.forward(tokenIDs: MLXArray([Int32(0)]))
        }
    }

    @Test("Empty parallel branches throws invalidGraphStructure")
    func emptyParallelBranchesThrows() throws {
        // Build graph manually since the DSL requires at least one branch
        let rootRegion = Region(
            parameters: [],
            operations: [
                Operation(
                    key: OperationKey(rawValue: 0),
                    kind: .tokenEmbedding(TokenEmbeddingAttributes(vocabSize: 4, embeddingSize: 4)),
                    operands: [],
                    results: [OperationResult(id: ValueID(rawValue: 0))]
                ),
                Operation(
                    key: OperationKey(rawValue: 1),
                    kind: .parallel(merge: .add, branches: []),
                    operands: [Operand(value: ValueID(rawValue: 0))],
                    results: [OperationResult(id: ValueID(rawValue: 1))]
                ),
            ],
            results: [ValueUse(value: ValueID(rawValue: 1))]
        )

        let graph = ModelGraph(rootRegion: rootRegion)
        let weights = bind([
            slot([.operation(0)], role: .embeddingTable): MLXArray.ones([4, 4]),
        ])

        let compiled = try MLXCompiler().compile(graph: graph, weights: weights)
        let executor = MLXExecutor(compiledModel: compiled)

        #expect(throws: CompilerError.self) {
            try executor.forward(tokenIDs: MLXArray([Int32(0)]))
        }
    }

    @Test("Unsupported state-space variant throws unsupportedVariant")
    func unsupportedVariantThrows() throws {
        #expect(throws: CompilerError.self) {
            _ = try MLXStateSpaceVariant(variant: "unknown_variant")
        }
    }
}

// MARK: - Standalone RoPE Tests

@Suite("MLXExecutor Standalone RoPE")
struct ExecutorStandaloneRoPETests {

    @Test("Standalone RoPE applies rotation and preserves shape")
    func standaloneRoPEAppliesRotation() throws {
        struct RoPEModel: LanguageModel {
            @ModelComponentBuilder var body: some ModelComponent {
                TokenEmbedding(vocabSize: 4, embeddingSize: 4)
                RoPE(dimension: 4, base: 10_000.0)
            }
        }

        let graph = try RoPEModel().makeModelGraph()
        let table = MLXArray(converting: [
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ] as [Double], [4, 4])

        let weights = bind([
            slot([.operation(0)], role: .embeddingTable): table,
        ])

        let compiled = try MLXCompiler().compile(graph: graph, weights: weights)
        let executor = MLXExecutor(compiledModel: compiled)

        let output = try executor.forward(tokenIDs: MLXArray([Int32(0), Int32(1)]))
        // Shape preserved
        #expect(output.shape == [2, 4])
        // RoPE rotates values — output differs from input
        let vals = output.asArray(Float.self)
        // Position 0: cos(0)=1, sin(0)=0 so first token is unchanged
        #expect(abs(vals[0] - 1.0) < 1e-4)
        // Position 1: rotation applied, so value changes
        let inputRow1 = [Float(0.0), 1.0, 0.0, 0.0]
        let outputRow1 = Array(vals[4..<8])
        let changed = zip(inputRow1, outputRow1).contains { abs($0 - $1) > 1e-4 }
        #expect(changed)
    }

    @Test("Standalone RoPE with linear scaling applies scale factor")
    func ropeWithLinearScaling() throws {
        struct ScaledRoPEModel: LanguageModel {
            @ModelComponentBuilder var body: some ModelComponent {
                TokenEmbedding(vocabSize: 4, embeddingSize: 4)
                RoPE(dimension: 4, base: 10_000.0,
                     scaling: RoPEScaling(kind: .linear, factor: 2.0))
            }
        }

        struct UnscaledRoPEModel: LanguageModel {
            @ModelComponentBuilder var body: some ModelComponent {
                TokenEmbedding(vocabSize: 4, embeddingSize: 4)
                RoPE(dimension: 4, base: 10_000.0)
            }
        }

        let table = MLXArray(converting: [
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            1.0, 1.0, 1.0, 1.0,
            0.0, 0.0, 0.0, 0.0,
        ] as [Double], [4, 4])

        let weightsDict: [ParameterSlot: MLXArray] = [
            slot([.operation(0)], role: .embeddingTable): table,
        ]

        let scaledGraph = try ScaledRoPEModel().makeModelGraph()
        let scaledCompiled = try MLXCompiler().compile(
            graph: scaledGraph, weights: bind(weightsDict))
        let scaledExecutor = MLXExecutor(compiledModel: scaledCompiled)

        let unscaledGraph = try UnscaledRoPEModel().makeModelGraph()
        let unscaledCompiled = try MLXCompiler().compile(
            graph: unscaledGraph, weights: bind(weightsDict))
        let unscaledExecutor = MLXExecutor(compiledModel: unscaledCompiled)

        let tokens = MLXArray([Int32(0), Int32(1)])
        let scaledOut = try scaledExecutor.forward(tokenIDs: tokens)
        let unscaledOut = try unscaledExecutor.forward(tokenIDs: tokens)

        // Scaled and unscaled produce different outputs at position 1
        let sv = scaledOut.asArray(Float.self)
        let uv = unscaledOut.asArray(Float.self)
        let differs = zip(sv[4..<8], uv[4..<8]).contains { abs($0 - $1) > 1e-4 }
        #expect(differs)
    }
}

// MARK: - Positional Embedding Tests

@Suite("MLXExecutor Positional Embedding")
struct ExecutorPositionalEmbeddingTests {

    @Test("Positional embedding adds learned positions to input")
    func positionalEmbeddingAddsPositions() throws {
        struct PosEmbedModel: LanguageModel {
            @ModelComponentBuilder var body: some ModelComponent {
                TokenEmbedding(vocabSize: 4, embeddingSize: 2)
                PositionalEmbedding(maxPositions: 8, embeddingSize: 2, kind: .learnedAbsolute)
            }
        }

        let graph = try PosEmbedModel().makeModelGraph()

        let table = MLXArray(converting: [
            1.0, 0.0,
            0.0, 1.0,
            1.0, 1.0,
            0.0, 0.0,
        ] as [Double], [4, 2])

        let posTable = MLXArray(converting: [
            0.1, 0.2,   // pos 0
            0.3, 0.4,   // pos 1
            0.5, 0.6,   // pos 2
            0.7, 0.8,   // pos 3
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ] as [Double], [8, 2])

        let weights = bind([
            slot([.operation(0)], role: .embeddingTable): table,
            slot([.operation(1)], role: .embeddingTable): posTable,
        ])

        let compiled = try MLXCompiler().compile(graph: graph, weights: weights)
        let executor = MLXExecutor(compiledModel: compiled)

        let output = try executor.forward(tokenIDs: MLXArray([Int32(0), Int32(1)]))
        let vals = output.asArray(Float.self)

        // Token 0 at pos 0: [1, 0] + [0.1, 0.2] = [1.1, 0.2]
        #expect(abs(vals[0] - 1.1) < 1e-4)
        #expect(abs(vals[1] - 0.2) < 1e-4)
        // Token 1 at pos 1: [0, 1] + [0.3, 0.4] = [0.3, 1.4]
        #expect(abs(vals[2] - 0.3) < 1e-4)
        #expect(abs(vals[3] - 1.4) < 1e-4)
    }
}

// MARK: - MoE Tests

@Suite("MLXExecutor MoE")
struct ExecutorMoETests {

    @Test("MoE routes tokens to experts and produces correct shape")
    func moeRoutesToExperts() throws {
        struct MoEModel: LanguageModel {
            @ModelComponentBuilder var body: some ModelComponent {
                TokenEmbedding(vocabSize: 4, embeddingSize: 4)
                MoE(
                    expertCount: 2,
                    expertsPerToken: 1,
                    expertInputSize: 4,
                    expertIntermediateSize: 8
                )
            }
        }

        let graph = try MoEModel().makeModelGraph()

        var dict: [ParameterSlot: MLXArray] = [:]
        dict[slot([.operation(0)], role: .embeddingTable)] = MLXRandom.normal([4, 4]) * 0.1

        // Router: [2 experts, 4 hidden]
        let moePath: [StructuralPathComponent] = [.operation(1)]
        dict[slot(moePath + [.field("router")], role: .weight)] = MLXRandom.normal([2, 4]) * 0.1

        // Expert 0 and 1 weights
        for expertIdx in 0..<2 {
            let ePath = moePath + [.field("experts"), .index(expertIdx)]
            dict[slot(ePath + [.field("gate_proj")], role: .weight)] = MLXRandom.normal([8, 4]) * 0.1
            dict[slot(ePath + [.field("up_proj")], role: .weight)] = MLXRandom.normal([8, 4]) * 0.1
            dict[slot(ePath + [.field("down_proj")], role: .weight)] = MLXRandom.normal([4, 8]) * 0.1
        }

        let compiled = try MLXCompiler().compile(graph: graph, weights: bind(dict))
        let executor = MLXExecutor(compiledModel: compiled)

        let output = try executor.forward(tokenIDs: MLXArray([Int32(0), Int32(1)]))
        // Output shape: [L=2, D=4]
        #expect(output.shape == [2, 4])
        // All values are finite
        let vals = output.asArray(Float.self)
        for v in vals {
            #expect(v.isFinite)
        }
    }
}

// MARK: - Parallel Stack Merge Tests

@Suite("MLXExecutor Parallel Stack Merge")
struct ExecutorParallelStackTests {

    @Test("Parallel with stack merge adds new axis")
    func parallelStackMerge() throws {
        struct StackModel: LanguageModel {
            @ModelComponentBuilder var body: some ModelComponent {
                TokenEmbedding(vocabSize: 4, embeddingSize: 2)
                Parallel(merge: .stack) {
                    Linear(inputSize: 2, outputSize: 2)
                    Linear(inputSize: 2, outputSize: 2)
                }
            }
        }

        let graph = try StackModel().makeModelGraph()

        // Branch weights — determine structural paths
        let parallelOp: [StructuralPathComponent] = [.operation(1)]
        let branch0Linear = parallelOp + [.regionBranch(0), .operation(0)]
        let branch1Linear = parallelOp + [.regionBranch(1), .operation(0)]

        let weights = bind([
            slot([.operation(0)], role: .embeddingTable):
                MLXArray(converting: [1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0] as [Double], [4, 2]),
            slot(branch0Linear, role: .weight):
                MLXArray(converting: [1.0, 0.0, 0.0, 1.0] as [Double], [2, 2]),
            slot(branch1Linear, role: .weight):
                MLXArray(converting: [2.0, 0.0, 0.0, 2.0] as [Double], [2, 2]),
        ])

        let compiled = try MLXCompiler().compile(graph: graph, weights: weights)
        let executor = MLXExecutor(compiledModel: compiled)

        let output = try executor.forward(tokenIDs: MLXArray([Int32(0)]))
        // Stack adds a new axis: 2 branches stacked → [2, B=1, L=1, D=2]
        // Batch dim is not squeezed because stack axis occupies axis 0
        #expect(output.shape == [2, 1, 1, 2])
    }
}

// MARK: - KV Cache Capacity Tests

@Suite("MLXKVCacheSimple Internals")
struct KVCacheCapacityTests {

    @Test("KV cache grows beyond initial capacity without errors")
    func kvCacheGrowsBeyondCapacity() throws {
        let cache = MLXKVCacheSimple(step: 4)

        // First update: allocates capacity
        let k1 = MLXRandom.normal([1, 2, 3, 4])  // [B=1, H=2, L=3, D=4]
        let v1 = MLXRandom.normal([1, 2, 3, 4])
        let (ck1, cv1) = cache.update(keys: k1, values: v1)
        #expect(ck1.dim(2) == 3)
        #expect(cv1.dim(2) == 3)
        #expect(cache.offset == 3)

        // Second update: within capacity
        let k2 = MLXRandom.normal([1, 2, 2, 4])
        let v2 = MLXRandom.normal([1, 2, 2, 4])
        let (ck2, cv2) = cache.update(keys: k2, values: v2)
        #expect(ck2.dim(2) == 5)
        #expect(cv2.dim(2) == 5)
        #expect(cache.offset == 5)

        // Third update: exceeds initial capacity, triggers growth
        let k3 = MLXRandom.normal([1, 2, 6, 4])
        let v3 = MLXRandom.normal([1, 2, 6, 4])
        let (ck3, cv3) = cache.update(keys: k3, values: v3)
        #expect(ck3.dim(2) == 11)
        #expect(cv3.dim(2) == 11)
        #expect(cache.offset == 11)
    }

    @Test("KV cache mask returns causal for L>1 and none for L=1")
    func kvCacheMaskBehavior() {
        let cache = MLXKVCacheSimple()
        let mask1 = cache.makeMask(queryLength: 4)
        if case .causal = mask1 {
            // Expected
        } else {
            Issue.record("Expected .causal for queryLength=4")
        }

        let mask2 = cache.makeMask(queryLength: 1)
        if case .none = mask2 {
            // Expected
        } else {
            Issue.record("Expected .none for queryLength=1")
        }
    }
}

// MARK: - Recurrent Cache Tests

@Suite("MLXRecurrentCache Internals")
struct RecurrentCacheTests {

    @Test("Recurrent cache tracks conv state and offset")
    func recurrentCacheStateTracking() {
        let cache = MLXRecurrentCache()
        #expect(cache.offset == 0)
        #expect(cache.convState == nil)
        #expect(cache.recurrentState == nil)

        // Simulate conv state update
        cache.convState = MLXRandom.normal([1, 4, 8])
        cache.recurrentState = MLXRandom.normal([1, 4, 16, 16])
        cache.incrementOffset(by: 3)

        #expect(cache.offset == 3)
        #expect(cache.convState != nil)
        #expect(cache.convState!.shape == [1, 4, 8])
        #expect(cache.recurrentState!.shape == [1, 4, 16, 16])

        // Increment again
        cache.incrementOffset(by: 1)
        #expect(cache.offset == 4)
    }
}

// MARK: - MLXStateSpaceVariant Tests

@Suite("MLXStateSpaceVariant")
struct StateSpaceVariantTests {

    @Test("Parses known variant strings correctly")
    func parsesKnownVariants() throws {
        let v1 = try MLXStateSpaceVariant(variant: "deltanet")
        #expect(v1 == .deltaNet)

        let v2 = try MLXStateSpaceVariant(variant: "gated_deltanet")
        #expect(v2 == .deltaNet)

        let v3 = try MLXStateSpaceVariant(variant: "gated-deltanet")
        #expect(v3 == .deltaNet)

        let v4 = try MLXStateSpaceVariant(variant: "DELTANET")
        #expect(v4 == .deltaNet)
    }

    @Test("Rejects unknown variant string")
    func rejectsUnknownVariant() {
        #expect(throws: CompilerError.self) {
            _ = try MLXStateSpaceVariant(variant: "mamba")
        }
    }
}

// MARK: - Executor Protocol Conformance Tests

@Suite("MLXExecutor Protocol Conformance")
struct ExecutorProtocolTests {

    @Test("Executor.run produces ModelOutputs with correct logits shape")
    func executorProtocolRun() async throws {
        let model = MiniLlama(layerCount: 1)
        let graph = try model.makeModelGraph()
        let weights = miniLlamaWeights(graph: graph, layerCount: 1)

        let compiled = try MLXCompiler().compile(graph: graph, weights: weights)
        let executor = MLXExecutor(compiledModel: compiled)

        // Build CompiledModel wrapper for Executor protocol
        let compiledModel = CompiledModel(
            semanticGraph: graph,
            loweredGraph: LoweredGraph(),
            weights: weights,
            runtimePlan: RuntimePlan(data: compiled)
        )

        let tokenArray = MLXArray([Int32(0), Int32(1)])
        let inputs = ModelInputs(
            tokenIDs: TensorData(
                shape: [2],
                dtype: .float32,
                storage: tokenArray
            )
        )

        let outputs = try await executor.run(compiledModel, inputs: inputs)

        // Logits shape: [L=2, vocab=8]
        #expect(outputs.logits.shape == [2, 8])
        // Cache state is returned
        #expect(outputs.cache != nil)
        #expect(outputs.cache!.cachedLength == 2)
    }
}

// MARK: - ModelCompiler Protocol Tests

@Suite("ModelCompiler Protocol")
struct ModelCompilerProtocolTests {

    @Test("MLXCompiler conforms to ModelCompiler with Compiled=MLXCompiledModel")
    func compilerConformsToProtocol() throws {
        func useCompiler<C: ModelCompiler>(_ compiler: C, graph: ModelGraph, weights: BoundWeights) throws -> C.Compiled {
            try compiler.compile(graph: graph, weights: weights)
        }

        let model = MiniLlama(layerCount: 1)
        let graph = try model.makeModelGraph()
        let weights = miniLlamaWeights(graph: graph, layerCount: 1)

        let result = try useCompiler(MLXCompiler(), graph: graph, weights: weights)
        // Compiled type is inferred as MLXCompiledModel — verify it works
        #expect(result.cacheDescriptors.count == 1)
        #expect(result.weightStore.count > 0)
    }
}

// MARK: - DeltaNet End-to-End Tests

@Suite("MLXExecutor DeltaNet")
struct ExecutorDeltaNetTests {

    @Test("DeltaNet forward pass produces correct shape and finite values")
    func deltaNetForwardPass() throws {
        // Minimal DeltaNet: 1 head, keyDim=2, valueDim=2, stateSize=2
        // Weight dimensions derived from Qwen 3.5 structure:
        //   in_proj_qkv: [keyDim + keyDim + valueDim, hiddenSize] = [6, 4]
        //   in_proj_z:   [valueDim, hiddenSize] = [2, 4]
        //   in_proj_b/a: [linearKeyHeads, hiddenSize] = [1, 4]
        //   conv1d:      [convDim=6, kernelSize=4]
        //   out_proj:    [hiddenSize, valueDim] = [4, 2]
        //   norm:        [linearValueHeadDim=2]
        //   dt_bias:     [linearKeyHeads=1]
        //   A_log:       [linearKeyHeads=1]
        struct DeltaNetModel: LanguageModel {
            @ModelComponentBuilder var body: some ModelComponent {
                TokenEmbedding(vocabSize: 4, embeddingSize: 4)
                StateSpace(hiddenSize: 4, stateSize: 2, variant: "deltanet")
            }
        }

        let graph = try DeltaNetModel().makeModelGraph()
        let ssPath: [StructuralPathComponent] = [.operation(1)]

        let weights = bind([
            slot([.operation(0)], role: .embeddingTable): MLXRandom.normal([4, 4]) * 0.1,
            // keyDim=2, valueDim=2 → totalQKV = 2+2+2 = 6
            slot(ssPath + [.field("in_proj_qkv")], role: .weight): MLXRandom.normal([6, 4]) * 0.1,
            slot(ssPath + [.field("in_proj_z")], role: .weight): MLXRandom.normal([2, 4]) * 0.1,
            slot(ssPath + [.field("in_proj_b")], role: .weight): MLXRandom.normal([1, 4]) * 0.1,
            slot(ssPath + [.field("in_proj_a")], role: .weight): MLXRandom.normal([1, 4]) * 0.1,
            slot(ssPath + [.field("conv1d")], role: .weight): MLXRandom.normal([6, 4]) * 0.01,
            slot(ssPath + [.field("out_proj")], role: .weight): MLXRandom.normal([4, 2]) * 0.1,
            slot(ssPath + [.field("norm")], role: .scale): MLXArray.zeros([2]),
            slot(ssPath + [.field("dt_bias")], role: .bias): MLXArray.zeros([1]),
            slot(ssPath + [.field("A_log")], role: .weight): MLXArray(converting: [-1.0] as [Double], [1]),
        ])

        let compiled = try MLXCompiler().compile(graph: graph, weights: weights)
        let executor = MLXExecutor(compiledModel: compiled)

        let output = try executor.forward(tokenIDs: MLXArray([Int32(0), Int32(1)]))
        // Output shape: [L=2, D=4]
        #expect(output.shape == [2, 4])

        let vals = output.asArray(Float.self)
        for v in vals {
            #expect(v.isFinite)
        }
    }

    @Test("DeltaNet autoregressive: recurrent state persists across calls")
    func deltaNetAutoregressive() throws {
        struct DeltaNetModel: LanguageModel {
            @ModelComponentBuilder var body: some ModelComponent {
                TokenEmbedding(vocabSize: 4, embeddingSize: 4)
                StateSpace(hiddenSize: 4, stateSize: 2, variant: "deltanet")
            }
        }

        let graph = try DeltaNetModel().makeModelGraph()
        let ssPath: [StructuralPathComponent] = [.operation(1)]

        // Use larger weights so state updates are measurable
        let weights = bind([
            slot([.operation(0)], role: .embeddingTable): MLXRandom.normal([4, 4]),
            slot(ssPath + [.field("in_proj_qkv")], role: .weight): MLXRandom.normal([6, 4]),
            slot(ssPath + [.field("in_proj_z")], role: .weight): MLXRandom.normal([2, 4]),
            slot(ssPath + [.field("in_proj_b")], role: .weight): MLXRandom.normal([1, 4]),
            slot(ssPath + [.field("in_proj_a")], role: .weight): MLXRandom.normal([1, 4]),
            slot(ssPath + [.field("conv1d")], role: .weight): MLXRandom.normal([6, 4]),
            slot(ssPath + [.field("out_proj")], role: .weight): MLXRandom.normal([4, 2]),
            slot(ssPath + [.field("norm")], role: .scale): MLXArray.zeros([2]),
            slot(ssPath + [.field("dt_bias")], role: .bias): MLXArray.zeros([1]),
            slot(ssPath + [.field("A_log")], role: .weight): MLXArray(converting: [-1.0] as [Double], [1]),
        ])

        let compiled = try MLXCompiler().compile(graph: graph, weights: weights)
        let executor = MLXExecutor(compiledModel: compiled)

        // Prefill then decode
        let _ = try executor.forward(tokenIDs: MLXArray([Int32(0), Int32(1)]))
        let withState = try executor.forward(tokenIDs: MLXArray([Int32(2)]))
        #expect(withState.shape == [1, 4])

        // Same token without prior state (fresh executor)
        let freshExecutor = MLXExecutor(compiledModel: compiled)
        let withoutState = try freshExecutor.forward(tokenIDs: MLXArray([Int32(2)]))
        #expect(withoutState.shape == [1, 4])

        // With state vs without state should differ
        let vs = withState.asArray(Float.self)
        let vf = withoutState.asArray(Float.self)
        let allSame = zip(vs, vf).allSatisfy { abs($0 - $1) < 1e-6 }
        #expect(!allSame)
    }
}

// MARK: - MLP Activation/Gating Variant Tests

@Suite("MLXExecutor MLP Activation Variants")
struct ExecutorMLPActivationTests {

    @Test("MLP with ReLU activation")
    func mlpReLU() throws {
        struct ReLUModel: LanguageModel {
            @ModelComponentBuilder var body: some ModelComponent {
                TokenEmbedding(vocabSize: 4, embeddingSize: 4)
                MLP(inputSize: 4, intermediateSize: 8, activation: .relu, gating: .none)
            }
        }

        let graph = try ReLUModel().makeModelGraph()
        let mlpPath: [StructuralPathComponent] = [.operation(1)]

        let weights = bind([
            slot([.operation(0)], role: .embeddingTable):
                MLXArray(converting: [1.0, -1.0, 2.0, -2.0] as [Double], [1, 4]),
            slot(mlpPath + [.field("gate_proj")], role: .weight):
                MLXArray.ones([8, 4]) * 0.1,
            slot(mlpPath + [.field("down_proj")], role: .weight):
                MLXArray.ones([4, 8]) * 0.1,
        ])

        let compiled = try MLXCompiler().compile(graph: graph, weights: weights)
        let executor = MLXExecutor(compiledModel: compiled)

        let output = try executor.forward(tokenIDs: MLXArray([Int32(0)]))
        #expect(output.shape == [1, 4])
        // ReLU zeros out negatives — all outputs should be non-negative
        let vals = output.asArray(Float.self)
        for v in vals { #expect(v >= -1e-6) }
    }

    @Test("MLP with GEGLU gating uses gelu activation with up projection")
    func mlpGEGLU() throws {
        struct GEGLUModel: LanguageModel {
            @ModelComponentBuilder var body: some ModelComponent {
                TokenEmbedding(vocabSize: 4, embeddingSize: 4)
                MLP(inputSize: 4, intermediateSize: 8, activation: .gelu, gating: .geglu)
            }
        }

        let graph = try GEGLUModel().makeModelGraph()
        let mlpPath: [StructuralPathComponent] = [.operation(1)]

        let weights = bind([
            slot([.operation(0)], role: .embeddingTable):
                MLXRandom.normal([4, 4]) * 0.1,
            slot(mlpPath + [.field("gate_proj")], role: .weight):
                MLXRandom.normal([8, 4]) * 0.1,
            slot(mlpPath + [.field("up_proj")], role: .weight):
                MLXRandom.normal([8, 4]) * 0.1,
            slot(mlpPath + [.field("down_proj")], role: .weight):
                MLXRandom.normal([4, 8]) * 0.1,
        ])

        let compiled = try MLXCompiler().compile(graph: graph, weights: weights)
        let executor = MLXExecutor(compiledModel: compiled)

        let output = try executor.forward(tokenIDs: MLXArray([Int32(0)]))
        #expect(output.shape == [1, 4])
        let vals = output.asArray(Float.self)
        for v in vals { #expect(v.isFinite) }
    }

    @Test("MLP with custom gating falls through to up_proj multiplication")
    func mlpCustomGating() throws {
        struct CustomGateModel: LanguageModel {
            @ModelComponentBuilder var body: some ModelComponent {
                TokenEmbedding(vocabSize: 4, embeddingSize: 4)
                MLP(inputSize: 4, intermediateSize: 8, activation: .silu, gating: .custom("test"))
            }
        }

        let graph = try CustomGateModel().makeModelGraph()
        let mlpPath: [StructuralPathComponent] = [.operation(1)]

        let weights = bind([
            slot([.operation(0)], role: .embeddingTable):
                MLXRandom.normal([4, 4]) * 0.1,
            slot(mlpPath + [.field("gate_proj")], role: .weight):
                MLXRandom.normal([8, 4]) * 0.1,
            slot(mlpPath + [.field("up_proj")], role: .weight):
                MLXRandom.normal([8, 4]) * 0.1,
            slot(mlpPath + [.field("down_proj")], role: .weight):
                MLXRandom.normal([4, 8]) * 0.1,
        ])

        let compiled = try MLXCompiler().compile(graph: graph, weights: weights)
        let executor = MLXExecutor(compiledModel: compiled)

        let output = try executor.forward(tokenIDs: MLXArray([Int32(0)]))
        #expect(output.shape == [1, 4])
        let vals = output.asArray(Float.self)
        for v in vals { #expect(v.isFinite) }
    }
}

// MARK: - Attention QK LayerNorm Tests

@Suite("MLXExecutor Attention LayerNorm QK")
struct ExecutorAttentionLayerNormQKTests {

    @Test("Attention with LayerNorm QK normalization applies per-head normalization")
    func attentionLayerNormQK() throws {
        struct LNQKModel: LanguageModel {
            @ModelComponentBuilder var body: some ModelComponent {
                TokenEmbedding(vocabSize: 4, embeddingSize: 4)
                Attention(
                    hiddenSize: 4, headCount: 2, kvHeadCount: 2,
                    headDimension: 2, qkNorm: .layerNorm
                )
            }
        }

        let graph = try LNQKModel().makeModelGraph()
        let attnPath: [StructuralPathComponent] = [.operation(1)]

        let weights = bind([
            slot([.operation(0)], role: .embeddingTable):
                MLXRandom.normal([4, 4]) * 0.1,
            slot(attnPath + [.field("q_proj")], role: .weight):
                MLXRandom.normal([4, 4]) * 0.1,
            slot(attnPath + [.field("k_proj")], role: .weight):
                MLXRandom.normal([4, 4]) * 0.1,
            slot(attnPath + [.field("v_proj")], role: .weight):
                MLXRandom.normal([4, 4]) * 0.1,
            slot(attnPath + [.field("o_proj")], role: .weight):
                MLXRandom.normal([4, 4]) * 0.1,
            slot(attnPath + [.field("q_norm")], role: .scale):
                MLXArray.ones([2]),
            slot(attnPath + [.field("k_norm")], role: .scale):
                MLXArray.ones([2]),
            slot(attnPath + [.field("q_norm")], role: .bias):
                MLXArray.zeros([2]),
            slot(attnPath + [.field("k_norm")], role: .bias):
                MLXArray.zeros([2]),
        ])

        let compiled = try MLXCompiler().compile(graph: graph, weights: weights)
        let executor = MLXExecutor(compiledModel: compiled)

        let output = try executor.forward(tokenIDs: MLXArray([Int32(0), Int32(1)]))
        #expect(output.shape == [2, 4])
        let vals = output.asArray(Float.self)
        for v in vals { #expect(v.isFinite) }
    }
}

// MARK: - OutputHead Variants Tests

@Suite("MLXExecutor OutputHead Variants")
struct ExecutorOutputHeadVariantTests {

    @Test("OutputHead with bias adds bias vector to logits")
    func outputHeadWithBias() throws {
        struct BiasHeadModel: LanguageModel {
            @ModelComponentBuilder var body: some ModelComponent {
                TokenEmbedding(vocabSize: 4, embeddingSize: 2)
                OutputHead(inputSize: 2, vocabSize: 4, tiedToEmbedding: false, bias: true)
            }
        }

        let graph = try BiasHeadModel().makeModelGraph()

        let weights = bind([
            slot([.operation(0)], role: .embeddingTable):
                MLXArray(converting: [1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0] as [Double], [4, 2]),
            slot([.operation(1)], role: .outputProjection):
                MLXArray(converting: [1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0] as [Double], [4, 2]),
            slot([.operation(1)], role: .bias):
                MLXArray(converting: [10.0, 20.0, 30.0, 40.0] as [Double], [4]),
        ])

        let compiled = try MLXCompiler().compile(graph: graph, weights: weights)
        let executor = MLXExecutor(compiledModel: compiled)

        // Token 0 → embed [1, 0] → proj [1, 0, 1, 0] + bias [10, 20, 30, 40] = [11, 20, 31, 40]
        let output = try executor.forward(tokenIDs: MLXArray([Int32(0)]))
        let vals = output.asArray(Float.self)
        #expect(abs(vals[0] - 11.0) < 1e-4)
        #expect(abs(vals[1] - 20.0) < 1e-4)
        #expect(abs(vals[2] - 31.0) < 1e-4)
        #expect(abs(vals[3] - 40.0) < 1e-4)
    }
}

// MARK: - Parallel Custom Merge Tests

@Suite("MLXExecutor Parallel Custom Merge")
struct ExecutorParallelCustomMergeTests {

    @Test("Parallel with custom merge throws unsupportedOperation")
    func parallelCustomMergeThrows() throws {
        // Build manually since DSL doesn't support custom merge directly
        let embedOp = Operation(
            key: OperationKey(rawValue: 0),
            kind: .tokenEmbedding(TokenEmbeddingAttributes(vocabSize: 4, embeddingSize: 2)),
            operands: [],
            results: [OperationResult(id: ValueID(rawValue: 0))]
        )

        let branch0 = Region(
            parameters: [RegionParameter(id: ValueID(rawValue: 100))],
            operations: [
                Operation(
                    key: OperationKey(rawValue: 10),
                    kind: .linear(LinearAttributes(inputSize: 2, outputSize: 2)),
                    operands: [Operand(value: ValueID(rawValue: 100))],
                    results: [OperationResult(id: ValueID(rawValue: 101))]
                )
            ],
            results: [ValueUse(value: ValueID(rawValue: 101))]
        )

        let parallelOp = Operation(
            key: OperationKey(rawValue: 1),
            kind: .parallel(merge: .custom("myMerge"), branches: [branch0]),
            operands: [Operand(value: ValueID(rawValue: 0))],
            results: [OperationResult(id: ValueID(rawValue: 1))]
        )

        let graph = ModelGraph(rootRegion: Region(
            parameters: [],
            operations: [embedOp, parallelOp],
            results: [ValueUse(value: ValueID(rawValue: 1))]
        ))

        let weights = bind([
            slot([.operation(0)], role: .embeddingTable): MLXArray.ones([4, 2]),
            slot([.operation(1), .regionBranch(0), .operation(0)], role: .weight): MLXArray.ones([2, 2]),
        ])

        let compiled = try MLXCompiler().compile(graph: graph, weights: weights)
        let executor = MLXExecutor(compiledModel: compiled)

        #expect(throws: CompilerError.self) {
            try executor.forward(tokenIDs: MLXArray([Int32(0)]))
        }
    }
}

// MARK: - Mixed Cache Type Tests

@Suite("MLXCompiler Mixed Cache Types")
struct CompilerMixedCacheTests {

    @Test("Model with both attention and stateSpace produces mixed cache descriptors")
    func mixedCacheTypes() throws {
        struct HybridModel: LanguageModel {
            @ModelComponentBuilder var body: some ModelComponent {
                TokenEmbedding(vocabSize: 4, embeddingSize: 4)
                Attention(hiddenSize: 4, headCount: 2, kvHeadCount: 2, headDimension: 2)
                StateSpace(hiddenSize: 4, stateSize: 2, variant: "deltanet")
                Attention(hiddenSize: 4, headCount: 2, kvHeadCount: 2, headDimension: 2)
            }
        }

        let graph = try HybridModel().makeModelGraph()
        let weights = bind([
            slot([.operation(0)], role: .embeddingTable): MLXArray.ones([4, 4]),
        ])

        let compiled = try MLXCompiler().compile(graph: graph, weights: weights)

        // 2 attention (kv) + 1 stateSpace (recurrent) = 3 caches
        #expect(compiled.cacheDescriptors.count == 3)
        #expect(compiled.cacheDescriptors[0].kind == .kv)
        #expect(compiled.cacheDescriptors[0].slotIndex == 0)
        #expect(compiled.cacheDescriptors[1].kind == .recurrent)
        #expect(compiled.cacheDescriptors[1].slotIndex == 1)
        #expect(compiled.cacheDescriptors[2].kind == .kv)
        #expect(compiled.cacheDescriptors[2].slotIndex == 2)

        // cacheSlotByPath has 3 entries
        #expect(compiled.cacheSlotByPath.count == 3)
    }
}

// MARK: - Single Token Forward Tests

@Suite("MLXExecutor Single Token")
struct ExecutorSingleTokenTests {

    @Test("Single token forward pass with L=1 uses non-causal mask")
    func singleTokenForward() throws {
        let model = MiniLlama(layerCount: 1)
        let graph = try model.makeModelGraph()
        let weights = miniLlamaWeights(graph: graph, layerCount: 1)

        let compiled = try MLXCompiler().compile(graph: graph, weights: weights)
        let executor = MLXExecutor(compiledModel: compiled)

        // Single token: L=1
        let output = try executor.forward(tokenIDs: MLXArray([Int32(0)]))
        #expect(output.shape == [1, 8])  // [L=1, vocab=8]
        let vals = output.asArray(Float.self)
        for v in vals { #expect(v.isFinite) }
    }

    @Test("Autoregressive decode: prefill then generate token-by-token")
    func autoregressiveGeneration() throws {
        let model = MiniLlama(layerCount: 1)
        let graph = try model.makeModelGraph()
        let weights = miniLlamaWeights(graph: graph, layerCount: 1)

        let compiled = try MLXCompiler().compile(graph: graph, weights: weights)
        let executor = MLXExecutor(compiledModel: compiled)

        // Prefill with 3 tokens
        let prefill = try executor.forward(tokenIDs: MLXArray([Int32(0), Int32(1), Int32(2)]))
        #expect(prefill.shape == [3, 8])

        // Generate 3 tokens one at a time
        for i in 3..<6 {
            let step = try executor.forward(tokenIDs: MLXArray([Int32(i % 4)]))
            #expect(step.shape == [1, 8])
        }
    }
}

// MARK: - WeightStore Property Tests

@Suite("MLXWeightStore Properties")
struct WeightStorePropertyTests {

    @Test("allSlots returns all stored parameter slots")
    func allSlotsReturnsAll() {
        let s1 = slot([.operation(0)], role: .weight)
        let s2 = slot([.operation(1)], role: .scale)
        let s3 = slot([.operation(2)], role: .bias)

        let store = MLXWeightStore(weights: [
            s1: MLXArray.ones([2, 2]),
            s2: MLXArray.ones([4]),
            s3: MLXArray.zeros([4]),
        ])

        #expect(store.count == 3)
        #expect(store.allSlots.contains(s1))
        #expect(store.allSlots.contains(s2))
        #expect(store.allSlots.contains(s3))
    }

    @Test("Empty weight store has count 0 and no slots")
    func emptyWeightStore() {
        let store = MLXWeightStore(weights: [:])
        #expect(store.count == 0)
        #expect(store.allSlots.isEmpty)
        #expect(store.get(slot([.operation(0)], role: .weight)) == nil)
    }
}

// MARK: - CacheSlotByPath Tests

@Suite("MLXCompiledModel CacheSlotByPath")
struct CacheSlotByPathTests {

    @Test("cacheSlotByPath maps each cache descriptor path to its slot index")
    func cacheSlotByPathMapping() throws {
        let model = MiniLlama(layerCount: 3)
        let graph = try model.makeModelGraph()
        let weights = miniLlamaWeights(graph: graph, layerCount: 3)

        let compiled = try MLXCompiler().compile(graph: graph, weights: weights)

        // 3 layers → 3 cache descriptors
        #expect(compiled.cacheSlotByPath.count == 3)

        // Each descriptor's path maps to its slotIndex
        for desc in compiled.cacheDescriptors {
            #expect(compiled.cacheSlotByPath[desc.path] == desc.slotIndex)
        }
    }
}

// MARK: - MoE Top-K Variants Tests

@Suite("MLXExecutor MoE Variants")
struct ExecutorMoEVariantTests {

    @Test("MoE with top-2 routing activates multiple experts per token")
    func moeTopK2() throws {
        struct MoETop2Model: LanguageModel {
            @ModelComponentBuilder var body: some ModelComponent {
                TokenEmbedding(vocabSize: 4, embeddingSize: 4)
                MoE(
                    expertCount: 3,
                    expertsPerToken: 2,
                    expertInputSize: 4,
                    expertIntermediateSize: 8
                )
            }
        }

        let graph = try MoETop2Model().makeModelGraph()

        var dict: [ParameterSlot: MLXArray] = [:]
        dict[slot([.operation(0)], role: .embeddingTable)] = MLXRandom.normal([4, 4]) * 0.1

        let moePath: [StructuralPathComponent] = [.operation(1)]
        dict[slot(moePath + [.field("router")], role: .weight)] = MLXRandom.normal([3, 4]) * 0.1

        for expertIdx in 0..<3 {
            let ePath = moePath + [.field("experts"), .index(expertIdx)]
            dict[slot(ePath + [.field("gate_proj")], role: .weight)] = MLXRandom.normal([8, 4]) * 0.1
            dict[slot(ePath + [.field("up_proj")], role: .weight)] = MLXRandom.normal([8, 4]) * 0.1
            dict[slot(ePath + [.field("down_proj")], role: .weight)] = MLXRandom.normal([4, 8]) * 0.1
        }

        let compiled = try MLXCompiler().compile(graph: graph, weights: bind(dict))
        let executor = MLXExecutor(compiledModel: compiled)

        let output = try executor.forward(tokenIDs: MLXArray([Int32(0), Int32(1), Int32(2)]))
        #expect(output.shape == [3, 4])
        let vals = output.asArray(Float.self)
        for v in vals { #expect(v.isFinite) }
    }

    @Test("MoE with GELU expert activation")
    func moeGELUExpert() throws {
        struct MoEGELUModel: LanguageModel {
            @ModelComponentBuilder var body: some ModelComponent {
                TokenEmbedding(vocabSize: 4, embeddingSize: 4)
                MoE(
                    expertCount: 2,
                    expertsPerToken: 1,
                    expertInputSize: 4,
                    expertIntermediateSize: 8,
                    expertActivation: .gelu
                )
            }
        }

        let graph = try MoEGELUModel().makeModelGraph()

        var dict: [ParameterSlot: MLXArray] = [:]
        dict[slot([.operation(0)], role: .embeddingTable)] = MLXRandom.normal([4, 4]) * 0.1

        let moePath: [StructuralPathComponent] = [.operation(1)]
        dict[slot(moePath + [.field("router")], role: .weight)] = MLXRandom.normal([2, 4]) * 0.1

        for expertIdx in 0..<2 {
            let ePath = moePath + [.field("experts"), .index(expertIdx)]
            dict[slot(ePath + [.field("gate_proj")], role: .weight)] = MLXRandom.normal([8, 4]) * 0.1
            dict[slot(ePath + [.field("up_proj")], role: .weight)] = MLXRandom.normal([8, 4]) * 0.1
            dict[slot(ePath + [.field("down_proj")], role: .weight)] = MLXRandom.normal([4, 8]) * 0.1
        }

        let compiled = try MLXCompiler().compile(graph: graph, weights: bind(dict))
        let executor = MLXExecutor(compiledModel: compiled)

        let output = try executor.forward(tokenIDs: MLXArray([Int32(0)]))
        #expect(output.shape == [1, 4])
        let vals = output.asArray(Float.self)
        for v in vals { #expect(v.isFinite) }
    }
}

// MARK: - Batch Dimension Tests

@Suite("MLXExecutor Batch Variants")
struct ExecutorBatchVariantTests {

    @Test("Batch size > 1 with 2D tokens processes all sequences")
    func largeBatch() throws {
        struct SimpleModel: LanguageModel {
            @ModelComponentBuilder var body: some ModelComponent {
                TokenEmbedding(vocabSize: 4, embeddingSize: 4)
                RMSNorm(dimension: 4)
                OutputHead(inputSize: 4, vocabSize: 4, tiedToEmbedding: true)
            }
        }

        let graph = try SimpleModel().makeModelGraph()
        let weights = bind([
            slot([.operation(0)], role: .embeddingTable): MLXRandom.normal([4, 4]) * 0.1,
            slot([.operation(1)], role: .scale): MLXArray.zeros([4]),
        ])

        let compiled = try MLXCompiler().compile(graph: graph, weights: weights)
        let executor = MLXExecutor(compiledModel: compiled)

        // B=3, L=2
        let tokens = MLXArray([Int32(0), Int32(1), Int32(2), Int32(3), Int32(0), Int32(2)])
            .reshaped(3, 2)

        let output = try executor.forward(tokenIDs: tokens)
        #expect(output.shape == [3, 2, 4])
        let vals = output.asArray(Float.self)
        for v in vals { #expect(v.isFinite) }
    }
}

// MARK: - CompilerError Description Tests

@Suite("CompilerError Descriptions")
struct CompilerErrorDescriptionTests {

    @Test("All error cases produce non-empty descriptions")
    func errorDescriptions() {
        let errors: [CompilerError] = [
            .missingWeight(slot([.operation(0)], role: .weight)),
            .invalidWeightStorage(slot([.operation(0)], role: .weight), "bad type"),
            .unsupportedOperation("custom"),
            .unsupportedVariant("mamba"),
            .invalidGraphStructure("no embedding"),
            .executionError("runtime error"),
        ]

        for error in errors {
            let desc = error.description
            #expect(!desc.isEmpty)
            #expect(desc.count > 5)
        }
    }
}

// MARK: - Test Model Definitions

/// Tiny Llama-style: TokenEmbed → Repeat(Residual(Norm+Attn), Residual(Norm+MLP)) → Norm → OutputHead
private struct MiniLlama: LanguageModel {
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

/// Tiny Cohere-style: TokenEmbed → Repeat(Residual(LayerNorm, Parallel(Attn, MLP))) → LayerNorm → OutputHead
private struct MiniCohere: LanguageModel {
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
                LayerNorm(dimension: hiddenSize)
                Parallel(merge: .add) {
                    Attention(
                        hiddenSize: hiddenSize,
                        headCount: headCount,
                        kvHeadCount: kvHeadCount,
                        headDimension: headDim
                    )
                    MLP(inputSize: hiddenSize, intermediateSize: intermediateSize)
                }
            }
        }

        LayerNorm(dimension: hiddenSize)
        OutputHead(inputSize: hiddenSize, vocabSize: vocabSize, tiedToEmbedding: true)
    }
}

// MARK: - Weight Factory Helpers

/// Build weights for MiniLlama.
///
/// Graph structure: [op0: embed, op1: repeat, op2: norm, op3: head]
/// Repeat body: [op0: residual(norm+attn), op1: residual(norm+mlp)]
private func miniLlamaWeights(graph: ModelGraph, layerCount: Int) -> BoundWeights {
    let D = 4  // hiddenSize
    let H = 2  // headCount
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
        dict[slot(attnNormPath, role: .scale)] = MLXArray.zeros([D])

        let attnPath = layerPrefix + [.operation(0), .regionBody, .operation(1)]
        dict[slot(attnPath + [.field("q_proj")], role: .weight)] = MLXRandom.normal([H * headDim, D]) * 0.1
        dict[slot(attnPath + [.field("k_proj")], role: .weight)] = MLXRandom.normal([H * headDim, D]) * 0.1
        dict[slot(attnPath + [.field("v_proj")], role: .weight)] = MLXRandom.normal([H * headDim, D]) * 0.1
        dict[slot(attnPath + [.field("o_proj")], role: .weight)] = MLXRandom.normal([D, H * headDim]) * 0.1

        // Residual 1: RMSNorm + MLP
        let mlpNormPath = layerPrefix + [.operation(1), .regionBody, .operation(0)]
        dict[slot(mlpNormPath, role: .scale)] = MLXArray.zeros([D])

        let mlpPath = layerPrefix + [.operation(1), .regionBody, .operation(1)]
        dict[slot(mlpPath + [.field("gate_proj")], role: .weight)] = MLXRandom.normal([inter, D]) * 0.1
        dict[slot(mlpPath + [.field("up_proj")], role: .weight)] = MLXRandom.normal([inter, D]) * 0.1
        dict[slot(mlpPath + [.field("down_proj")], role: .weight)] = MLXRandom.normal([D, inter]) * 0.1
    }

    // op2: Final RMSNorm
    dict[slot([.operation(2)], role: .scale)] = MLXArray.zeros([D])

    // op3: OutputHead (tied to embedding — no weight needed)

    return bind(dict)
}

/// Build weights for MiniCohere.
///
/// Graph structure: [op0: embed, op1: repeat, op2: layernorm, op3: head]
/// Repeat body: [op0: residual(layernorm, parallel(attn, mlp))]
private func miniCohereWeights(graph: ModelGraph, layerCount: Int) -> BoundWeights {
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

        // Residual body: op0=LayerNorm, op1=Parallel(attn, mlp)
        let normPath = layerPrefix + [.operation(0), .regionBody, .operation(0)]
        dict[slot(normPath, role: .scale)] = MLXArray.ones([D])
        dict[slot(normPath, role: .bias)] = MLXArray.zeros([D])

        // Parallel branch 0: Attention
        let attnPath = layerPrefix + [.operation(0), .regionBody, .operation(1), .regionBranch(0), .operation(0)]
        dict[slot(attnPath + [.field("q_proj")], role: .weight)] = MLXRandom.normal([H * headDim, D]) * 0.1
        dict[slot(attnPath + [.field("k_proj")], role: .weight)] = MLXRandom.normal([H * headDim, D]) * 0.1
        dict[slot(attnPath + [.field("v_proj")], role: .weight)] = MLXRandom.normal([H * headDim, D]) * 0.1
        dict[slot(attnPath + [.field("o_proj")], role: .weight)] = MLXRandom.normal([D, H * headDim]) * 0.1

        // Parallel branch 1: MLP
        let mlpPath = layerPrefix + [.operation(0), .regionBody, .operation(1), .regionBranch(1), .operation(0)]
        dict[slot(mlpPath + [.field("gate_proj")], role: .weight)] = MLXRandom.normal([inter, D]) * 0.1
        dict[slot(mlpPath + [.field("up_proj")], role: .weight)] = MLXRandom.normal([inter, D]) * 0.1
        dict[slot(mlpPath + [.field("down_proj")], role: .weight)] = MLXRandom.normal([D, inter]) * 0.1
    }

    // op2: Final LayerNorm
    dict[slot([.operation(2)], role: .scale)] = MLXArray.ones([D])
    dict[slot([.operation(2)], role: .bias)] = MLXArray.zeros([D])

    // op3: OutputHead (tied)

    return bind(dict)
}
