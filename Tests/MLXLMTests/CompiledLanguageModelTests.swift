import Testing
@preconcurrency import MLX
import MLXFast
import MLXNN
@testable import MLXLM
@testable import MLXCompiler
@testable import SwiftLM

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

// MARK: - Tiny Model Definitions (duplicated from InferenceCompilerTests)

/// Tiny Llama-style model for compiled language model adapter tests.
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

    dict[slot([.operation(0)], role: .embeddingTable)] = MLXRandom.normal([vocab, D]) * 0.1

    for i in 0..<layerCount {
        let layerPrefix: [StructuralPathComponent] = [
            .operation(1), .regionBody, .index(i)
        ]

        let attnNormPath = layerPrefix + [.operation(0), .regionBody, .operation(0)]
        dict[slot(attnNormPath, role: .scale)] = MLXArray.zeros([D])

        let attnPath = layerPrefix + [.operation(0), .regionBody, .operation(1)]
        dict[slot(attnPath + [.field("q_proj")], role: .weight)] = MLXRandom.normal([H * headDim, D]) * 0.1
        dict[slot(attnPath + [.field("k_proj")], role: .weight)] = MLXRandom.normal([H * headDim, D]) * 0.1
        dict[slot(attnPath + [.field("v_proj")], role: .weight)] = MLXRandom.normal([H * headDim, D]) * 0.1
        dict[slot(attnPath + [.field("o_proj")], role: .weight)] = MLXRandom.normal([D, H * headDim]) * 0.1

        let mlpNormPath = layerPrefix + [.operation(1), .regionBody, .operation(0)]
        dict[slot(mlpNormPath, role: .scale)] = MLXArray.zeros([D])

        let mlpPath = layerPrefix + [.operation(1), .regionBody, .operation(1)]
        dict[slot(mlpPath + [.field("gate_proj")], role: .weight)] = MLXRandom.normal([inter, D]) * 0.1
        dict[slot(mlpPath + [.field("up_proj")], role: .weight)] = MLXRandom.normal([inter, D]) * 0.1
        dict[slot(mlpPath + [.field("down_proj")], role: .weight)] = MLXRandom.normal([D, inter]) * 0.1
    }

    dict[slot([.operation(2)], role: .scale)] = MLXArray.zeros([D])

    return bind(dict)
}

/// Compile a TinyLlama into a CompiledLanguageModel.
private func compileModel(layerCount: Int) throws -> CompiledLanguageModel {
    let model = TinyLlama(layerCount: layerCount)
    let graph = try model.makeModelGraph()
    let weights = tinyLlamaWeights(layerCount: layerCount)
    let lowered = try MLXInferenceCompiler().compile(graph: graph, weights: weights)
    return CompiledLanguageModel(lowered: lowered)
}

// MARK: - CompiledKVCache Tests

@Suite("CompiledKVCache")
struct CompiledKVCacheTests {

    @Test("Offset reflects nextPosition from InferenceState")
    func offsetReflectsPosition() throws {
        MLXRandom.seed(42)
        let model = try compileModel(layerCount: 1)
        let caches = model.newCache(parameters: nil)

        #expect(caches.count == 1)
        guard let compiledCache = caches.first as? CompiledKVCache else {
            Issue.record("Expected CompiledKVCache")
            return
        }

        #expect(compiledCache.offset == 0)

        // Run prefill to advance position
        let prompt = LMInput.Text(tokens: MLXArray([1, 2, 3]).reshaped([1, 3]))
        _ = model.callAsFunction(prompt, cache: caches, state: nil)

        #expect(compiledCache.offset == 3)
    }

    @Test("innerState collects MLXArrays from KV caches")
    func innerStateCollectsArrays() throws {
        MLXRandom.seed(42)
        let model = try compileModel(layerCount: 2)
        let caches = model.newCache(parameters: nil)
        guard let compiledCache = caches.first as? CompiledKVCache else {
            Issue.record("Expected CompiledKVCache")
            return
        }

        // Before any forward pass, caches are empty
        #expect(compiledCache.innerState().isEmpty)

        // After prefill, caches should contain arrays
        let prompt = LMInput.Text(tokens: MLXArray([1, 2]).reshaped([1, 2]))
        _ = model.callAsFunction(prompt, cache: caches, state: nil)

        let inner = compiledCache.innerState()
        // 2 layers × 2 arrays (keys + values) = 4
        #expect(inner.count == 4)
    }

    @Test("makeMask returns causal for prefill, none for decode")
    func maskBehavior() throws {
        let model = try compileModel(layerCount: 1)
        let caches = model.newCache(parameters: nil)
        guard let compiledCache = caches.first as? CompiledKVCache else {
            Issue.record("Expected CompiledKVCache")
            return
        }

        if case .causal = compiledCache.makeMask(queryLength: 3) {
            // OK
        } else {
            Issue.record("Expected .causal for queryLength > 1")
        }

        if case .none = compiledCache.makeMask(queryLength: 1) {
            // OK
        } else {
            Issue.record("Expected .none for queryLength == 1")
        }
    }
}

// MARK: - CompiledLanguageModel Protocol Conformance Tests

@Suite("CompiledLanguageModel")
struct CompiledLanguageModelTests {

    @Test("Conforms to LanguageModel protocol")
    func conformsToProtocol() throws {
        let model = try compileModel(layerCount: 1)
        let _: any LanguageModel = model
    }

    @Test("newCache returns single CompiledKVCache")
    func newCacheReturnsSingleCache() throws {
        let model = try compileModel(layerCount: 2)
        let caches = model.newCache(parameters: nil)

        #expect(caches.count == 1)
        #expect(caches.first is CompiledKVCache)
    }

    @Test("layerCount returns cache slot count")
    func layerCountMatchesCacheSlots() throws {
        let model1 = try compileModel(layerCount: 1)
        #expect(model1.layerCount == 1)

        let model2 = try compileModel(layerCount: 3)
        #expect(model2.layerCount == 3)
    }

    @Test("callAsFunction with multi-token input runs prefill")
    func prefillPath() throws {
        MLXRandom.seed(42)
        let model = try compileModel(layerCount: 1)
        let caches = model.newCache(parameters: nil)

        let input = LMInput.Text(tokens: MLXArray([1, 2, 3]).reshaped([1, 3]))
        let output = model.callAsFunction(input, cache: caches, state: nil)

        // Output shape: [1, 3, vocabSize=8]
        #expect(output.logits.dim(0) == 1)
        #expect(output.logits.dim(1) == 3)
        #expect(output.logits.dim(2) == 8)

        // Cache should reflect 3 prefilled tokens
        guard let compiledCache = caches.first as? CompiledKVCache else {
            Issue.record("Expected CompiledKVCache")
            return
        }
        #expect(compiledCache.offset == 3)
    }

    @Test("callAsFunction with single-token input runs decode")
    func decodePath() throws {
        MLXRandom.seed(42)
        let model = try compileModel(layerCount: 1)
        let caches = model.newCache(parameters: nil)

        // Prefill first
        let prompt = LMInput.Text(tokens: MLXArray([1, 2, 3]).reshaped([1, 3]))
        _ = model.callAsFunction(prompt, cache: caches, state: nil)

        // Decode single token
        let next = LMInput.Text(tokens: MLXArray([4]).reshaped([1, 1]))
        let output = model.callAsFunction(next, cache: caches, state: nil)

        // Output shape: [1, 1, vocabSize=8]
        #expect(output.logits.dim(0) == 1)
        #expect(output.logits.dim(1) == 1)
        #expect(output.logits.dim(2) == 8)

        // Position should be 4 (3 prefill + 1 decode)
        guard let compiledCache = caches.first as? CompiledKVCache else {
            Issue.record("Expected CompiledKVCache")
            return
        }
        #expect(compiledCache.offset == 4)
    }

    @Test("Prefill + multi-step decode produces finite logits")
    func multiStepDecode() throws {
        MLXRandom.seed(42)
        let model = try compileModel(layerCount: 2)
        let caches = model.newCache(parameters: nil)

        // Prefill
        let prompt = LMInput.Text(tokens: MLXArray([1, 2]).reshaped([1, 2]))
        let prefillOutput = model.callAsFunction(prompt, cache: caches, state: nil)
        eval(prefillOutput.logits)
        let prefillVals: [Float] = prefillOutput.logits.flattened().asArray(Float.self)
        for v in prefillVals { #expect(v.isFinite) }

        // 5 decode steps
        for i in 3...7 {
            let token = LMInput.Text(tokens: MLXArray([Int32(i)]).reshaped([1, 1]))
            let output = model.callAsFunction(token, cache: caches, state: nil)
            eval(output.logits)
            let vals: [Float] = output.logits.flattened().asArray(Float.self)
            for v in vals { #expect(v.isFinite, "Non-finite at decode step \(i - 2)") }
        }

        guard let compiledCache = caches.first as? CompiledKVCache else {
            Issue.record("Expected CompiledKVCache")
            return
        }
        #expect(compiledCache.offset == 7)
    }

    @Test("Compiled model output matches direct MLXLoweredInferenceModel output")
    func outputMatchesDirect() throws {
        MLXRandom.seed(42)

        let model = TinyLlama(layerCount: 1)
        let graph = try model.makeModelGraph()
        let weights = tinyLlamaWeights(layerCount: 1)
        let lowered = try MLXInferenceCompiler().compile(graph: graph, weights: weights)

        // Direct path
        let directState = lowered.makeState()
        let prompt = MLXArray([1, 2, 3]).reshaped([1, 3])
        let (directLogits, directState2) = lowered.prefill(tokenIDs: prompt, state: directState)
        let nextToken = MLXArray([4]).reshaped([1, 1])
        let (directDecode, _) = lowered.decode(tokenIDs: nextToken, state: directState2)

        // Adapter path
        let compiled = CompiledLanguageModel(lowered: lowered)
        let caches = compiled.newCache(parameters: nil)
        let adapterPrefill = compiled.callAsFunction(
            LMInput.Text(tokens: prompt), cache: caches, state: nil)
        let adapterDecode = compiled.callAsFunction(
            LMInput.Text(tokens: nextToken), cache: caches, state: nil)

        // Prefill must match exactly
        let prefillDiff = abs(adapterPrefill.logits - directLogits).max().item(Float.self)
        #expect(prefillDiff < 1e-6, "Prefill diff \(prefillDiff) — adapter output must match direct")

        // Decode must match exactly
        let decodeDiff = abs(adapterDecode.logits - directDecode).max().item(Float.self)
        #expect(decodeDiff < 1e-6, "Decode diff \(decodeDiff) — adapter output must match direct")
    }

    @Test("prepare() chunks long prefills")
    func prepareChunking() throws {
        MLXRandom.seed(42)
        let model = try compileModel(layerCount: 1)
        let caches = model.newCache(parameters: nil)

        // 6-token input with windowSize=2 → should chunk
        let tokens = MLXArray([1, 2, 3, 4, 5, 6]).reshaped([1, 6])
        let input = LMInput(tokens: tokens)

        // First call: processes tokens[0..<2], returns .tokens
        let result1 = try model.prepare(input, cache: caches, windowSize: 2)
        switch result1 {
        case .tokens:
            break  // Expected — more chunks remain
        case .logits:
            Issue.record("Expected .tokens on first chunk")
        }

        // Second call: processes tokens[2..<4], returns .tokens
        let result2 = try model.prepare(input, cache: caches, windowSize: 2)
        switch result2 {
        case .tokens:
            break
        case .logits:
            Issue.record("Expected .tokens on second chunk")
        }

        // Third call: processes tokens[4..<6], returns .logits
        let result3 = try model.prepare(input, cache: caches, windowSize: 2)
        switch result3 {
        case .tokens:
            Issue.record("Expected .logits on final chunk")
        case .logits(let output):
            eval(output.logits)
            let chunkVals: [Float] = output.logits.flattened().asArray(Float.self)
            for v in chunkVals { #expect(v.isFinite) }
        }
    }

    @Test("Can be used as 'any LanguageModel' in ModelContext")
    func worksInModelContext() throws {
        MLXRandom.seed(42)
        let model = try compileModel(layerCount: 1)

        // Verify the model can be assigned to ModelContext.model field type
        let _: any LanguageModel = model
        let cache = model.newCache(parameters: GenerateParameters())
        #expect(cache.count == 1)

        // Verify forward pass works through the protocol interface
        let lm: any LanguageModel = model
        let input = LMInput.Text(tokens: MLXArray([1, 2]).reshaped([1, 2]))
        let output = lm.callAsFunction(input, cache: cache, state: nil)
        #expect(output.logits.dim(2) == 8)
    }
}

// MARK: - P1: Sanitize Equivalence Tests

@Suite("CompiledPathSanitize")
struct CompiledPathSanitizeTests {

    @Test("Default sanitizeCompiledWeights filters rotary_emb.inv_freq")
    func defaultSanitizeFiltersInvFreq() throws {
        let invFreq = MLXRandom.normal([2])
        let normalWeight = MLXRandom.normal([4, 4])

        var weights: [String: TensorData] = [:]
        weights["model.layers.0.self_attn.q_proj.weight"] = TensorData(
            shape: [4, 4], dtype: .float32, storage: MLXTensorStorage.dense(normalWeight))
        weights["model.layers.0.self_attn.rotary_emb.inv_freq"] = TensorData(
            shape: [2], dtype: .float32, storage: MLXTensorStorage.dense(invFreq))

        let sanitized = TransformerModel.sanitizeCompiledWeights(weights)

        #expect(sanitized.count == 1)
        #expect(sanitized["model.layers.0.self_attn.q_proj.weight"] != nil)
        #expect(sanitized["model.layers.0.self_attn.rotary_emb.inv_freq"] == nil)
    }

    @Test("Qwen35 sanitizeCompiledWeights reshapes conv1d.weight from 2D to 3D")
    func qwen35SanitizeReshapesConv1d() throws {
        let conv2d = MLXRandom.normal([6, 4])  // [C, K]
        let normalWeight = MLXRandom.normal([4, 4])

        var weights: [String: TensorData] = [:]
        weights["model.layers.0.linear_attn.conv1d.weight"] = TensorData(
            shape: [6, 4], dtype: .float16, storage: MLXTensorStorage.dense(conv2d))
        weights["model.layers.0.self_attn.q_proj.weight"] = TensorData(
            shape: [4, 4], dtype: .float16, storage: MLXTensorStorage.dense(normalWeight))

        let sanitized = Qwen35Model.sanitizeCompiledWeights(weights)

        #expect(sanitized.count == 2)

        // conv1d.weight should now be 3D: [C, K, 1]
        guard let convTD = sanitized["model.layers.0.linear_attn.conv1d.weight"] else {
            Issue.record("conv1d.weight missing after sanitize")
            return
        }
        #expect(convTD.shape == [6, 4, 1])

        guard let storage = convTD.storage as? MLXTensorStorage,
              case .dense(let array) = storage
        else {
            Issue.record("Expected dense MLXTensorStorage")
            return
        }
        #expect(array.ndim == 3)
        #expect(array.dim(0) == 6)
        #expect(array.dim(1) == 4)
        #expect(array.dim(2) == 1)
    }

    @Test("Qwen35 sanitizeCompiledWeights transposes conv1d.weight from 3D")
    func qwen35SanitizeTransposesConv1d3D() throws {
        let conv3d = MLXRandom.normal([6, 2, 4])  // PyTorch [O, I/G, K]

        var weights: [String: TensorData] = [:]
        weights["model.layers.0.linear_attn.conv1d.weight"] = TensorData(
            shape: [6, 2, 4], dtype: .float16, storage: MLXTensorStorage.dense(conv3d))

        let sanitized = Qwen35Model.sanitizeCompiledWeights(weights)

        guard let convTD = sanitized["model.layers.0.linear_attn.conv1d.weight"] else {
            Issue.record("conv1d.weight missing after sanitize")
            return
        }
        // Transposed: [O, K, I/G]
        #expect(convTD.shape == [6, 4, 2])
    }

    @Test("Qwen35 sanitizeCompiledWeights also filters rotary_emb.inv_freq")
    func qwen35SanitizeFiltersInvFreq() throws {
        var weights: [String: TensorData] = [:]
        weights["model.layers.0.self_attn.rotary_emb.inv_freq"] = TensorData(
            shape: [2], dtype: .float32,
            storage: MLXTensorStorage.dense(MLXRandom.normal([2])))
        weights["model.layers.0.self_attn.q_proj.weight"] = TensorData(
            shape: [4, 4], dtype: .float16,
            storage: MLXTensorStorage.dense(MLXRandom.normal([4, 4])))

        let sanitized = Qwen35Model.sanitizeCompiledWeights(weights)
        #expect(sanitized["model.layers.0.self_attn.rotary_emb.inv_freq"] == nil)
        #expect(sanitized["model.layers.0.self_attn.q_proj.weight"] != nil)
    }

    @Test("Compiled model sanitize() is identity — weights are pre-bound")
    func sanitizeIsIdentity() throws {
        let model = try compileModel(layerCount: 1)

        let testWeights: [String: MLXArray] = [
            "test.weight": MLXRandom.normal([4, 4]),
            "test.bias": MLXRandom.normal([4]),
        ]

        let sanitized = model.sanitize(weights: testWeights)

        #expect(sanitized.count == testWeights.count)
        for (key, value) in testWeights {
            #expect(sanitized[key] != nil)
            let diff = abs(sanitized[key]! - value).max().item(Float.self)
            #expect(diff == 0.0, "sanitize() should not modify weights")
        }
    }
}

// MARK: - P2: PromptCacheSnapshot Interop Tests

@Suite("CompiledKVCacheSnapshot")
struct CompiledKVCacheSnapshotTests {

    @Test("state getter returns non-empty arrays after prefill")
    func stateGetterAfterPrefill() throws {
        MLXRandom.seed(42)
        let model = try compileModel(layerCount: 2)
        let caches = model.newCache(parameters: nil)
        guard let compiledCache = caches.first as? CompiledKVCache else {
            Issue.record("Expected CompiledKVCache")
            return
        }

        // Before prefill: state has arrays but they are empty (size 0)
        // After prefill: state should have real data
        let prompt = LMInput.Text(tokens: MLXArray([1, 2, 3]).reshaped([1, 3]))
        _ = model.callAsFunction(prompt, cache: caches, state: nil)

        let state = compiledCache.state
        // 2 layers × 2 arrays (keys + values) = 4
        #expect(state.count == 4)
        for array in state {
            #expect(array.size > 0, "state arrays should be non-empty after prefill")
        }
    }

    @Test("metaState encodes cache structure correctly")
    func metaStateEncodesDecode() throws {
        MLXRandom.seed(42)
        let model = try compileModel(layerCount: 2)
        let caches = model.newCache(parameters: nil)
        guard let compiledCache = caches.first as? CompiledKVCache else {
            Issue.record("Expected CompiledKVCache")
            return
        }

        // Prefill to set some state
        let prompt = LMInput.Text(tokens: MLXArray([1, 2, 3]).reshaped([1, 3]))
        _ = model.callAsFunction(prompt, cache: caches, state: nil)

        let meta = compiledCache.metaState
        // Format: [nextPosition, cacheCount, type0, offset0, step0, type1, offset1, step1]
        #expect(meta.count >= 2)
        #expect(meta[0] == "3", "nextPosition should be 3")
        #expect(meta[1] == "2", "cacheCount should be 2")
        // Each cache slot has 3 entries: type, offset, step
        #expect(meta.count == 2 + 2 * 3)
    }

    @Test("capturePromptCache produces valid snapshot for CompiledKVCache")
    func captureProducesValidSnapshot() throws {
        MLXRandom.seed(42)
        let model = try compileModel(layerCount: 2)
        let caches = model.newCache(parameters: nil)

        // Prefill
        let prompt = LMInput.Text(tokens: MLXArray([1, 2, 3]).reshaped([1, 3]))
        _ = model.callAsFunction(prompt, cache: caches, state: nil)
        eval(caches.first!.innerState())

        let snapshot = capturePromptCache(cache: caches, prefixTokenCount: 3)

        #expect(snapshot.prefixTokenCount == 3)
        #expect(snapshot.cacheClasses.count == 1)
        #expect(snapshot.cacheClasses[0] == "CompiledKVCache")
        #expect(snapshot.cacheState[0].count > 0, "Snapshot should capture non-empty state")
    }

    @Test("materializePromptCache restores CompiledKVCache from snapshot")
    func materializeRestoresCompiledCache() throws {
        MLXRandom.seed(42)
        let model = try compileModel(layerCount: 2)
        let caches = model.newCache(parameters: nil)

        // Prefill
        let prompt = LMInput.Text(tokens: MLXArray([1, 2, 3]).reshaped([1, 3]))
        _ = model.callAsFunction(prompt, cache: caches, state: nil)
        eval(caches.first!.innerState())

        // Capture
        let snapshot = capturePromptCache(cache: caches, prefixTokenCount: 3)

        // Materialize
        let restored = materializePromptCache(from: snapshot)
        #expect(restored.count == 1)
        guard let restoredCache = restored.first as? CompiledKVCache else {
            Issue.record("Expected restored cache to be CompiledKVCache")
            return
        }

        #expect(restoredCache.offset == 3, "Restored nextPosition should be 3")
        #expect(restoredCache.inferenceState.caches.count == 2, "Should have 2 cache slots")

        // Verify state arrays were restored
        let restoredState = restoredCache.state
        let originalState = (caches.first as! CompiledKVCache).state
        #expect(restoredState.count == originalState.count)

        for (orig, rest) in zip(originalState, restoredState) {
            if orig.size > 0 {
                let diff = abs(orig - rest).max().item(Float.self)
                #expect(diff < 1e-6, "Restored state should match original")
            }
        }
    }

    @Test("Decode after materialize produces same output as decode without snapshot")
    func decodeAfterMaterializeMatchesContinuousDecode() throws {
        MLXRandom.seed(42)
        let model = try compileModel(layerCount: 1)

        // Path A: continuous prefill + decode
        let cachesA = model.newCache(parameters: nil)
        let prompt = LMInput.Text(tokens: MLXArray([1, 2, 3]).reshaped([1, 3]))
        _ = model.callAsFunction(prompt, cache: cachesA, state: nil)
        let nextToken = LMInput.Text(tokens: MLXArray([4]).reshaped([1, 1]))
        let outputA = model.callAsFunction(nextToken, cache: cachesA, state: nil)
        eval(outputA.logits)

        // Path B: prefill, snapshot, restore, decode
        let cachesB = model.newCache(parameters: nil)
        _ = model.callAsFunction(prompt, cache: cachesB, state: nil)
        eval(cachesB.first!.innerState())

        let snapshot = capturePromptCache(cache: cachesB, prefixTokenCount: 3)
        let restoredCaches = materializePromptCache(from: snapshot)
        let outputB = model.callAsFunction(nextToken, cache: restoredCaches, state: nil)
        eval(outputB.logits)

        let diff = abs(outputA.logits - outputB.logits).max().item(Float.self)
        #expect(diff < 1e-5, "Decode after restore should match continuous decode, diff=\(diff)")
    }
}

// MARK: - Binder Tests (retained from original)

@Suite("CompiledPathBinder")
struct CompiledPathBinderTests {

    @Test("MLXWeightPathBinder skips RawWeights tensors not matching any slot")
    func binderSkipsUnmatchedTensors() throws {
        MLXRandom.seed(42)
        let model = TinyLlama(layerCount: 1)
        let graph = try model.makeModelGraph()

        let enumerator = ModelGraphSlotEnumerator()
        let manifest = enumerator.enumerate(graph)

        var tensors: [String: TensorData] = [:]

        for entry in manifest {
            let shape: [Int]
            switch entry.slot.role {
            case .embeddingTable:
                shape = [8, 4]
            case .scale:
                shape = [4]
            case .weight:
                if entry.mlxWeightPath.contains("gate_proj") || entry.mlxWeightPath.contains("up_proj") {
                    shape = [8, 4]
                } else if entry.mlxWeightPath.contains("down_proj") {
                    shape = [4, 8]
                } else {
                    shape = [4, 4]
                }
            default:
                continue
            }
            let array = MLXRandom.normal(shape) * 0.1
            tensors[entry.mlxWeightPath] = TensorData(
                shape: shape, dtype: .float32, storage: array)
        }

        // Extra tensor that simulates rotary_emb.inv_freq
        tensors["model.layers.0.self_attn.rotary_emb.inv_freq"] = TensorData(
            shape: [2], dtype: .float32, storage: MLXRandom.normal([2]))

        let rawWeights = RawWeights(tensors: tensors)

        let binder = MLXWeightPathBinder()
        let boundWeights = try binder.bind(rawWeights, to: graph)

        let compiled = try MLXInferenceCompiler().compile(graph: graph, weights: boundWeights)
        let state = compiled.makeState()
        let prompt = MLXArray([1, 2]).reshaped([1, 2])
        let (logits, _) = compiled.prefill(tokenIDs: prompt, state: state)

        let vals: [Float] = logits.flattened().asArray(Float.self)
        for v in vals { #expect(v.isFinite) }
    }
}

// MARK: - Integration: End-to-End Pipeline Tests

@Suite("CompiledPipelineIntegration")
struct CompiledPipelineIntegrationTests {

    @Test("Compiled model works with TokenIterator-style usage pattern")
    func tokenIteratorPattern() throws {
        MLXRandom.seed(42)
        let model = try compileModel(layerCount: 1)
        let caches = model.newCache(parameters: GenerateParameters())

        // Simulate TokenIterator flow:
        // 1. prepare() for prefill
        let tokens = MLXArray([1, 2, 3]).reshaped([1, 3])
        let input = LMInput(tokens: tokens)
        let result = try model.prepare(input, cache: caches, windowSize: nil)

        let prefillLogits: MLXArray
        switch result {
        case .logits(let output):
            prefillLogits = output.logits
        case .tokens:
            Issue.record("Expected .logits from prepare()")
            return
        }
        eval(prefillLogits)
        let prefillVals: [Float] = prefillLogits.flattened().asArray(Float.self)
        for v in prefillVals { #expect(v.isFinite) }

        // 2. Autoregressive decode (simulate 3 steps)
        for step in 0..<3 {
            let tokenID: Int32 = Int32(step + 4)
            let decodeInput = LMInput.Text(
                tokens: MLXArray([tokenID]).reshaped([1, 1]))
            let output = model.callAsFunction(decodeInput, cache: caches, state: nil)
            eval(output.logits)
            let decVals: [Float] = output.logits.flattened().asArray(Float.self)
            for v in decVals { #expect(v.isFinite, "Non-finite at decode step \(step)") }
        }

        // Verify position tracking
        guard let compiledCache = caches.first as? CompiledKVCache else {
            Issue.record("Expected CompiledKVCache")
            return
        }
        #expect(compiledCache.offset == 6, "3 prefill + 3 decode = 6")
    }

    @Test("Multiple independent caches produce independent outputs")
    func independentCaches() throws {
        MLXRandom.seed(42)
        let model = try compileModel(layerCount: 1)

        // Create two independent cache instances
        let cache1 = model.newCache(parameters: nil)
        let cache2 = model.newCache(parameters: nil)

        // Prefill with different prompts
        let prompt1 = LMInput.Text(tokens: MLXArray([1, 2, 3]).reshaped([1, 3]))
        let prompt2 = LMInput.Text(tokens: MLXArray([5, 6, 7]).reshaped([1, 3]))

        let out1 = model.callAsFunction(prompt1, cache: cache1, state: nil)
        let out2 = model.callAsFunction(prompt2, cache: cache2, state: nil)

        eval(out1.logits, out2.logits)

        // Different prompts → different outputs
        let diff = abs(out1.logits - out2.logits).max().item(Float.self)
        #expect(diff > 1e-4, "Different prompts should produce different outputs")

        // Decode same token from different caches → different outputs
        let next = LMInput.Text(tokens: MLXArray([4]).reshaped([1, 1]))
        let dec1 = model.callAsFunction(next, cache: cache1, state: nil)
        let dec2 = model.callAsFunction(next, cache: cache2, state: nil)

        eval(dec1.logits, dec2.logits)
        let decodeDiff = abs(dec1.logits - dec2.logits).max().item(Float.self)
        #expect(decodeDiff > 1e-4, "Same token with different history should differ")
    }
}
