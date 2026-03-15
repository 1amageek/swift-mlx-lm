import Foundation
import MLX
import Testing
import TestHeartbeat
@testable import LMInference

/// Diagnostic tests for HFDirectoryBundle loading with real cached models.
///
/// Uses locally cached mlx-community models to validate the full
/// HF bundle → IR → compile → inference pipeline.
@Suite("HFDirectoryBundle Diagnostics", .tags(.diagnostic), .heartbeat)
struct HFBundleDiagnosticTests {

    // MARK: - Model Paths

    private static let qwen35_4B_dir = URL(fileURLWithPath: NSHomeDirectory())
        .appendingPathComponent("Library/Containers/team.stamp.JARDIS.ml/Data/Library/Caches/huggingface/models/mlx-community/Qwen3.5-4B-MLX-4bit")

    private static let lfm25_1_2B_dir = URL(fileURLWithPath: NSHomeDirectory())
        .appendingPathComponent("Library/Containers/team.stamp.JARDIS.ml/Data/.cache/huggingface/hub/LiquidAI--LFM2.5-1.2B-Thinking/main")

    private func requireModel(at dir: URL) throws -> URL {
        let configPath = dir.appendingPathComponent("config.json").path
        guard FileManager.default.fileExists(atPath: configPath) else {
            throw SkipInfo("Model not cached at \(dir.path)")
        }
        return dir
    }

    private struct SkipInfo: Error, CustomStringConvertible {
        let description: String
        init(_ message: String) { self.description = message }
    }

    // MARK: - Step 1: Config Decoding

    @Test("HFDirectoryBundle: decode Qwen3.5-4B config")
    func decodeConfig() throws {
        let dir = try requireModel(at: Self.qwen35_4B_dir)
        let bundle = try HFDirectoryBundle(directory: dir)
        let config = try bundle.configuration()

        print("[HFBundle] config: hiddenSize=\(config.hiddenSize) layers=\(config.layerCount) heads=\(config.attentionHeads) kvHeads=\(config.kvHeads) headDim=\(config.headDim)")
        print("[HFBundle] intermediateSize=\(config.intermediateSize) vocabSize=\(config.vocabSize)")
        print("[HFBundle] ropeTheta=\(config.ropeTheta) ropeDim=\(config.ropeDimension) partialRotary=\(config.partialRotaryFactor ?? -1)")
        print("[HFBundle] layerTypes count=\(config.layerTypes?.count ?? 0)")
        print("[HFBundle] mropeAxes=\(String(describing: config.mropeAxes))")
        print("[HFBundle] DeltaNet: ssmNumHeads=\(config.ssmNumHeads ?? -1) ssmGroupCount=\(config.ssmGroupCount ?? -1) ssmKeyHeadDim=\(config.ssmKeyHeadDim ?? -1) ssmValueHeadDim=\(config.ssmValueHeadDim ?? -1)")

        // Qwen3.5-4B specific assertions
        #expect(config.hiddenSize == 2560)
        #expect(config.layerCount == 32)
        #expect(config.attentionHeads == 16)
        #expect(config.kvHeads == 4)
        #expect(config.headDim == 256)
        #expect(config.intermediateSize == 9216)
        #expect(config.vocabSize == 248320)
        #expect(config.ropeTheta == 10000000.0)
        #expect(config.partialRotaryFactor == 0.25)
        #expect(config.ropeDimension == 64)  // 256 * 0.25
        #expect(config.tiedEmbeddings == true)
        #expect(config.normEps == 1e-06)
        #expect(config.normKind == .rmsNorm)

        // DeltaNet fields
        #expect(config.ssmNumHeads == 32)      // linear_num_value_heads
        #expect(config.ssmGroupCount == 16)    // linear_num_key_heads
        #expect(config.ssmKeyHeadDim == 128)
        #expect(config.ssmValueHeadDim == 128)
        #expect(config.convKernelSize == 4)
        #expect(config.fullAttentionInterval == 4)

        // Layer schedule
        #expect(config.layerTypes?.count == 32)
        #expect(config.layerTypes?[0] == "linear_attention")
        #expect(config.layerTypes?[3] == "full_attention")
        #expect(config.layerTypes?[31] == "full_attention")

        // M-RoPE
        #expect(config.mropeAxes != nil)
        #expect(config.mropeAxes?.sections == [11, 11, 10])
        #expect(config.mropeAxes?.interleaved == true)
    }

    // MARK: - Step 2: Model Type Detection

    @Test("HFDirectoryBundle: detect Qwen3.5-4B model type")
    func detectModelType() throws {
        let dir = try requireModel(at: Self.qwen35_4B_dir)
        let bundle = try HFDirectoryBundle(directory: dir)
        let configData = try bundle.rawConfigData()
        let modelType = try HFConfigDecoder().modelType(from: configData)

        print("[HFBundle] detected model_type: \(modelType)")
        #expect(modelType == "qwen3_5")
    }

    // MARK: - Step 3: Weight Loading

    @Test("HFDirectoryBundle: load Qwen3.5-4B weights and inspect")
    func loadWeightsInspect() throws {
        let dir = try requireModel(at: Self.qwen35_4B_dir)
        let bundle = try HFDirectoryBundle(directory: dir)
        let manifest = try bundle.loadWeights()

        let totalTensors = manifest.weights.count
        let quantizedCount = manifest.quantizationInfo.count
        let mappedCount = manifest.nameMapping.count

        print("[HFBundle] weights: \(totalTensors) tensors, \(quantizedCount) quantized, \(mappedCount) mapped")

        // Verify text model weights are loaded (no language_model. prefix)
        let hasEmbedding = manifest.weights["model.embed_tokens.weight"] != nil
        let hasNorm = manifest.weights["model.norm.weight"] != nil
        let hasLayer0Norm = manifest.weights["model.layers.0.input_layernorm.weight"] != nil

        print("[HFBundle] embed_tokens: \(hasEmbedding), norm: \(hasNorm), layer0.input_layernorm: \(hasLayer0Norm)")
        #expect(hasEmbedding)
        #expect(hasNorm)
        #expect(hasLayer0Norm)

        // Verify DeltaNet layer 0 weights
        let dnPrefix = "model.layers.0.linear_attn"
        let hasDNQKV = manifest.weights["\(dnPrefix).in_proj_qkv.weight"] != nil
        let hasDNZ = manifest.weights["\(dnPrefix).in_proj_z.weight"] != nil
        let hasDNALog = manifest.weights["\(dnPrefix).A_log"] != nil
        let hasDNConv1d = manifest.weights["\(dnPrefix).conv1d.weight"] != nil
        let hasDNDtBias = manifest.weights["\(dnPrefix).dt_bias"] != nil

        print("[HFBundle] DeltaNet layer 0: in_proj_qkv=\(hasDNQKV) in_proj_z=\(hasDNZ) A_log=\(hasDNALog) conv1d=\(hasDNConv1d) dt_bias=\(hasDNDtBias)")
        #expect(hasDNQKV)
        #expect(hasDNZ)
        #expect(hasDNALog)
        #expect(hasDNConv1d)
        #expect(hasDNDtBias)

        // Verify full attention layer 3 weights
        let attnPrefix = "model.layers.3.self_attn"
        let hasQProj = manifest.weights["\(attnPrefix).q_proj.weight"] != nil
        let hasKProj = manifest.weights["\(attnPrefix).k_proj.weight"] != nil
        let hasVProj = manifest.weights["\(attnPrefix).v_proj.weight"] != nil
        let hasOProj = manifest.weights["\(attnPrefix).o_proj.weight"] != nil

        print("[HFBundle] FullAttn layer 3: q=\(hasQProj) k=\(hasKProj) v=\(hasVProj) o=\(hasOProj)")
        #expect(hasQProj)
        #expect(hasKProj)
        #expect(hasVProj)
        #expect(hasOProj)

        // Verify no vision tower tensors leaked through
        let visionKeys = manifest.weights.keys.filter { $0.hasPrefix("vision_tower.") }
        print("[HFBundle] vision tower tensors: \(visionKeys.count) (should be 0)")
        #expect(visionKeys.isEmpty)

        // Verify no language_model. prefix remains
        let prefixedKeys = manifest.weights.keys.filter { $0.hasPrefix("language_model.") }
        print("[HFBundle] language_model. prefixed tensors: \(prefixedKeys.count) (should be 0)")
        #expect(prefixedKeys.isEmpty)

        // Quantization info
        if !manifest.quantizationInfo.isEmpty {
            let sample = manifest.quantizationInfo.first!
            print("[HFBundle] quantization sample: \(sample.key) → groupSize=\(sample.value.groupSize) bits=\(sample.value.bits)")
        }
    }

    // MARK: - Step 4: Tokenizer

    @Test("HFDirectoryBundle: create Qwen3.5-4B tokenizer")
    func createTokenizer() throws {
        let dir = try requireModel(at: Self.qwen35_4B_dir)
        let bundle = try HFDirectoryBundle(directory: dir)
        let tokenizer = try bundle.tokenizer()

        let text = "Hello, world!"
        let tokens = tokenizer.encode(text: text)
        let decoded = tokenizer.decode(tokens: tokens)

        print("[HFBundle] tokenizer: vocabSize=\(tokenizer.vocabularySize) bos=\(tokenizer.bosTokenID ?? -1) eos=\(tokenizer.eosTokenID ?? -1)")
        print("[HFBundle] encode('\(text)') → \(tokens)")
        print("[HFBundle] decode(\(tokens)) → '\(decoded)'")

        #expect(tokenizer.vocabularySize == 248320)
        #expect(!tokens.isEmpty)
        #expect(decoded.contains("Hello"))
    }

    // MARK: - Step 5: Chat Template

    @Test("HFDirectoryBundle: load Qwen3.5-4B chat template")
    func loadChatTemplate() throws {
        let dir = try requireModel(at: Self.qwen35_4B_dir)
        let bundle = try HFDirectoryBundle(directory: dir)
        let template = try bundle.chatTemplate()

        print("[HFBundle] chat template: \(template != nil ? "\(template!.count) chars" : "nil")")
        #expect(template != nil)
        #expect(template!.contains("im_start"))
    }

    // MARK: - Step 6: Full Pipeline (Config → IR → Compile → Forward)

    @Test("HFDirectoryBundle: Qwen3.5-4B full compiled pipeline")
    func fullCompiledPipeline() throws {
        let dir = try requireModel(at: Self.qwen35_4B_dir)
        let bundle = try HFDirectoryBundle(directory: dir)

        // Step A: Config + Registry resolve
        let t0 = CFAbsoluteTimeGetCurrent()
        let config = try bundle.configuration()
        let configData = try bundle.rawConfigData()
        let modelType = try HFConfigDecoder().modelType(from: configData)
        guard let rawConfig = try JSONSerialization.jsonObject(with: configData) as? [String: Any] else {
            throw SkipInfo("Failed to parse config.json")
        }
        let resolved = try ModelRegistry().resolve(
            modelType: modelType, config: config, rawConfig: rawConfig)
        let graph = resolved.graph
        let configTime = CFAbsoluteTimeGetCurrent() - t0
        print("[HFBundle] config + resolve: \(String(format: "%.3f", configTime))s")

        // Step B: Verify IR
        let t1 = CFAbsoluteTimeGetCurrent()
        print("[HFBundle] IR assembled: \(graph.rootRegion.operations.count) ops [\(String(format: "%.3f", CFAbsoluteTimeGetCurrent() - t1))s]")
        #expect(!graph.rootRegion.operations.isEmpty)

        // Step C: Load weights
        let t2 = CFAbsoluteTimeGetCurrent()
        let manifest = try bundle.loadWeights()
        let loadTime = CFAbsoluteTimeGetCurrent() - t2
        print("[HFBundle] weights loaded: \(manifest.weights.count) tensors [\(String(format: "%.3f", loadTime))s]")

        // Step D: Convert to RawWeights
        let loader = ModelBundleLoader()
        let t3 = CFAbsoluteTimeGetCurrent()
        let context = try loader.loadCompiled(bundle: bundle)
        let totalTime = CFAbsoluteTimeGetCurrent() - t0
        print("[HFBundle] full pipeline: [\(String(format: "%.3f", totalTime))s]")

        // Step E: Verify model produces output
        let tokenizer = context.tokenizer
        let inputText = "Hello"
        let tokens = tokenizer.encode(text: inputText)
        print("[HFBundle] input tokens: \(tokens)")
        #expect(!tokens.isEmpty)
    }

    // MARK: - Step 7: Forward Pass Verification

    @Test("HFDirectoryBundle: Qwen3.5-4B forward pass produces valid logits")
    func forwardPassVerification() throws {
        let dir = try requireModel(at: Self.qwen35_4B_dir)
        let bundle = try HFDirectoryBundle(directory: dir)

        // Load the compiled model
        let loader = ModelBundleLoader()
        let context = try loader.loadCompiled(bundle: bundle)
        let model = context.model

        // Encode a short prompt
        let tokenizer = context.tokenizer
        let promptTokens = tokenizer.encode(text: "Hello")
        print("[HFBundle] forward pass: prompt=\(promptTokens.count) tokens")
        #expect(!promptTokens.isEmpty)

        // Create cache
        let caches = model.newCache(parameters: nil)

        // Prefill: multi-token forward pass
        let tokenArray = MLXArray(promptTokens.map { Int32($0) }).reshaped([1, promptTokens.count])
        let input = LMInput.Text(tokens: tokenArray)
        let t0 = CFAbsoluteTimeGetCurrent()
        let output = model.callAsFunction(input, cache: caches, state: nil)
        eval(output.logits)
        let prefillTime = CFAbsoluteTimeGetCurrent() - t0
        print("[HFBundle] prefill: \(String(format: "%.3f", prefillTime))s → logits shape=\(output.logits.shape)")

        // Verify logits shape: [1, seqLen, vocabSize]
        #expect(output.logits.dim(0) == 1)
        #expect(output.logits.dim(1) == promptTokens.count)
        #expect(output.logits.dim(2) == 248320)  // Qwen3.5 vocab size

        // Verify all logits are finite (no NaN/Inf)
        let lastLogits = output.logits[0..., (promptTokens.count - 1)..., 0...]
        eval(lastLogits)
        let hasNaN = MLX.any(MLX.isNaN(lastLogits)).item(Bool.self)
        let hasInf = MLX.any(abs(lastLogits) .== Float.infinity).item(Bool.self)
        print("[HFBundle] logits finite check: hasNaN=\(hasNaN) hasInf=\(hasInf)")
        #expect(!hasNaN, "Logits contain NaN values")
        #expect(!hasInf, "Logits contain Inf values")

        // Verify top token is reasonable (not always 0 or constant)
        let topToken = MLX.argMax(lastLogits, axis: -1).item(Int.self)
        print("[HFBundle] top predicted token: \(topToken) → '\(tokenizer.decode(tokens: [topToken]))'")
        #expect(topToken >= 0 && topToken < 248320)

        // Decode: single-token step
        let nextTokenArray = MLXArray([Int32(topToken)]).reshaped([1, 1])
        let decodeInput = LMInput.Text(tokens: nextTokenArray)
        let t1 = CFAbsoluteTimeGetCurrent()
        let decodeOutput = model.callAsFunction(decodeInput, cache: caches, state: nil)
        eval(decodeOutput.logits)
        let decodeTime = CFAbsoluteTimeGetCurrent() - t1
        print("[HFBundle] decode: \(String(format: "%.3f", decodeTime))s → logits shape=\(decodeOutput.logits.shape)")

        // Verify decode output shape
        #expect(decodeOutput.logits.dim(0) == 1)
        #expect(decodeOutput.logits.dim(1) == 1)
        #expect(decodeOutput.logits.dim(2) == 248320)

        // Verify decode logits are finite
        let decodeHasNaN = MLX.any(MLX.isNaN(decodeOutput.logits)).item(Bool.self)
        #expect(!decodeHasNaN, "Decode logits contain NaN values")

        // Generate a few more tokens to verify consistent output
        var generatedTokens = [topToken]
        for _ in 0..<4 {
            let nextInput = LMInput.Text(
                tokens: MLXArray([Int32(generatedTokens.last!)]).reshaped([1, 1])
            )
            let stepOutput = model.callAsFunction(nextInput, cache: caches, state: nil)
            eval(stepOutput.logits)
            let nextToken = MLX.argMax(stepOutput.logits[0..., 0..., 0...], axis: -1).item(Int.self)
            generatedTokens.append(nextToken)
        }
        let generatedText = tokenizer.decode(tokens: generatedTokens)
        print("[HFBundle] generated: \(generatedTokens) → '\(generatedText)'")
        #expect(generatedTokens.count == 5)
        #expect(!generatedText.isEmpty)
    }

    // MARK: - Step 8: Chat Template Generation

    @Test("HFDirectoryBundle: Qwen3.5-4B chat template generation")
    func chatTemplateGeneration() throws {
        let dir = try requireModel(at: Self.qwen35_4B_dir)
        let bundle = try HFDirectoryBundle(directory: dir)

        let loader = ModelBundleLoader()
        let context = try loader.loadCompiled(bundle: bundle)

        // Build prompt via chat template directly (avoiding async)
        let tokenizer = context.tokenizer
        let template = try bundle.chatTemplate()
        #expect(template != nil, "Chat template is required")

        let eosToken = tokenizer.eosTokenID.flatMap { tokenizer.tokenToString($0) }
        let renderer = try ChatTemplateRenderer(
            templateString: template!,
            bosToken: nil,
            eosToken: eosToken
        )
        let prompt = try renderer.render(
            messages: [Chat.Message(role: .user, content: "What is 2+2?")],
            tools: nil,
            additionalContext: nil,
            addGenerationPrompt: true
        )
        let promptTokenIDs = tokenizer.encode(text: prompt)
        print("[HFBundle] chat template prompt: \(promptTokenIDs.count) tokens")
        #expect(promptTokenIDs.count > 5, "Chat template should produce more than raw prompt tokens")

        // Run forward pass with chat-template-formatted input
        let tokenArray = MLXArray(promptTokenIDs.map { Int32($0) }).reshaped([1, promptTokenIDs.count])
        let input = LMInput.Text(tokens: tokenArray)
        let model = context.model
        let caches = model.newCache(parameters: nil)
        let output = model.callAsFunction(input, cache: caches, state: nil)
        eval(output.logits)

        // Verify logits shape
        let promptTokenCount = promptTokenIDs.count
        #expect(output.logits.dim(0) == 1)
        #expect(output.logits.dim(1) == promptTokenCount)
        #expect(output.logits.dim(2) == 248320)

        let lastLogits = output.logits[0..., (promptTokenCount - 1)..<promptTokenCount, 0...]
        eval(lastLogits)
        let hasNaN = MLX.any(MLX.isNaN(lastLogits)).item(Bool.self)
        #expect(!hasNaN, "Chat template logits contain NaN")

        // Greedy decode 20 tokens
        var tokens: [Int] = []
        let eosId = tokenizer.eosTokenID ?? -1
        for _ in 0..<20 {
            let logits: MLXArray
            if tokens.isEmpty {
                logits = lastLogits
            } else {
                let nextInput = LMInput.Text(
                    tokens: MLXArray([Int32(tokens.last!)]).reshaped([1, 1])
                )
                let stepOutput = model.callAsFunction(nextInput, cache: caches, state: nil)
                eval(stepOutput.logits)
                logits = stepOutput.logits
            }
            let nextToken = MLX.argMax(logits.reshaped(-1), axis: 0).item(Int.self)
            if nextToken == eosId { break }
            tokens.append(nextToken)
        }
        let generatedText = tokenizer.decode(tokens: tokens)
        print("[HFBundle] chat generation (\(tokens.count) tokens): '\(generatedText)'")
        #expect(!tokens.isEmpty, "Model should generate at least one token")
        #expect(!generatedText.isEmpty, "Generated text should not be empty")
    }

    // MARK: - LFM2.5-1.2B-Thinking Tests

    @Test("HFDirectoryBundle: LFM2.5-1.2B full compiled pipeline")
    func lfm2FullCompiledPipeline() throws {
        let dir = try requireModel(at: Self.lfm25_1_2B_dir)
        let bundle = try HFDirectoryBundle(directory: dir)

        let loader = ModelBundleLoader()
        let context = try loader.loadCompiled(bundle: bundle)

        let tokenizer = context.tokenizer
        let tokens = tokenizer.encode(text: "Hello, world!")
        print("[HFBundle/LFM2] input tokens: \(tokens)")
        #expect(!tokens.isEmpty)

        // Forward pass: prefill
        let tokenArray = MLXArray(tokens.map { Int32($0) }).reshaped([1, tokens.count])
        let input = LMInput.Text(tokens: tokenArray)
        let model = context.model
        let caches = model.newCache(parameters: nil)
        let output = model.callAsFunction(input, cache: caches, state: nil)
        eval(output.logits)

        // Verify logits shape
        #expect(output.logits.dim(0) == 1)
        #expect(output.logits.dim(1) == tokens.count)
        #expect(output.logits.dim(2) == 65536)

        let hasNaN = MLX.any(MLX.isNaN(output.logits)).item(Bool.self)
        #expect(!hasNaN, "LFM2 logits contain NaN")

        // Decode one token
        let lastLogits = output.logits[0..., (tokens.count - 1)..<tokens.count, 0...]
        eval(lastLogits)
        let nextToken = MLX.argMax(lastLogits.reshaped(-1), axis: 0).item(Int.self)
        print("[HFBundle/LFM2] next token: \(nextToken) = '\(tokenizer.decode(tokens: [nextToken]))'")
        #expect(nextToken >= 0 && nextToken < 65536)
    }
}
