import Testing
import Foundation
import MLX
import GGUFParser
import GGUFTokenizer
@testable import MLXLM

// MARK: - GGUFTensorNameMapper Tests

@Suite("LlamaTensorNameMapper")
struct LlamaTensorNameMapperTests {

    let mapper = LlamaTensorNameMapper()

    @Test("Global tensor: token_embd.weight")
    func tokenEmbd() {
        #expect(mapper.mlxName(for: "token_embd.weight") == "model.embed_tokens.weight")
    }

    @Test("Global tensor: output_norm.weight")
    func outputNorm() {
        #expect(mapper.mlxName(for: "output_norm.weight") == "model.norm.weight")
    }

    @Test("Global tensor: output.weight")
    func output() {
        #expect(mapper.mlxName(for: "output.weight") == "lm_head.weight")
    }

    @Test("Block attention query")
    func attnQ() {
        #expect(mapper.mlxName(for: "blk.0.attn_q.weight") == "model.layers.0.self_attn.q_proj.weight")
        #expect(mapper.mlxName(for: "blk.15.attn_q.weight") == "model.layers.15.self_attn.q_proj.weight")
    }

    @Test("Block attention key/value/output")
    func attnKVO() {
        #expect(mapper.mlxName(for: "blk.3.attn_k.weight") == "model.layers.3.self_attn.k_proj.weight")
        #expect(mapper.mlxName(for: "blk.3.attn_v.weight") == "model.layers.3.self_attn.v_proj.weight")
        #expect(mapper.mlxName(for: "blk.3.attn_output.weight") == "model.layers.3.self_attn.o_proj.weight")
    }

    @Test("Block MLP projections")
    func mlpProj() {
        #expect(mapper.mlxName(for: "blk.7.ffn_gate.weight") == "model.layers.7.mlp.gate_proj.weight")
        #expect(mapper.mlxName(for: "blk.7.ffn_up.weight") == "model.layers.7.mlp.up_proj.weight")
        #expect(mapper.mlxName(for: "blk.7.ffn_down.weight") == "model.layers.7.mlp.down_proj.weight")
    }

    @Test("Block norms")
    func blockNorms() {
        #expect(mapper.mlxName(for: "blk.0.attn_norm.weight") == "model.layers.0.input_layernorm.weight")
        #expect(mapper.mlxName(for: "blk.0.ffn_norm.weight") == "model.layers.0.post_attention_layernorm.weight")
    }

    @Test("Bias variants")
    func biasVariants() {
        #expect(mapper.mlxName(for: "blk.0.attn_q.bias") == "model.layers.0.self_attn.q_proj.bias")
        #expect(mapper.mlxName(for: "blk.0.attn_k.bias") == "model.layers.0.self_attn.k_proj.bias")
        #expect(mapper.mlxName(for: "blk.0.attn_v.bias") == "model.layers.0.self_attn.v_proj.bias")
        #expect(mapper.mlxName(for: "blk.0.attn_output.bias") == "model.layers.0.self_attn.o_proj.bias")
        #expect(mapper.mlxName(for: "blk.0.ffn_gate.bias") == "model.layers.0.mlp.gate_proj.bias")
        #expect(mapper.mlxName(for: "blk.0.ffn_up.bias") == "model.layers.0.mlp.up_proj.bias")
        #expect(mapper.mlxName(for: "blk.0.ffn_down.bias") == "model.layers.0.mlp.down_proj.bias")
    }

    @Test("Unknown tensor returns nil")
    func unknownTensor() {
        #expect(mapper.mlxName(for: "unknown.tensor") == nil)
        #expect(mapper.mlxName(for: "blk.0.unknown.weight") == nil)
    }

    @Test("Invalid block format returns nil")
    func invalidBlock() {
        #expect(mapper.mlxName(for: "blk.abc.attn_q.weight") == nil)
        #expect(mapper.mlxName(for: "blk.") == nil)
    }
}

// MARK: - GGUFConfigExtractor Tests

@Suite("GGUFConfigExtractor")
struct GGUFConfigExtractorTests {

    /// Create a minimal GGUF file with specified metadata for testing.
    private func makeGGUFData(
        architecture: String = "llama",
        embeddingLength: Int = 256,
        blockCount: Int = 2,
        headCount: Int = 4,
        feedForwardLength: Int? = nil,
        headCountKV: Int? = nil,
        rmsEps: Float? = nil,
        ropeFreqBase: Float? = nil,
        vocabSize: Int = 100,
        includeLmHead: Bool = false
    ) throws -> GGUFFile {
        // Build a minimal GGUF binary
        var builder = GGUFTestBuilder()
        builder.addMetadata("general.architecture", value: .string(architecture))
        builder.addMetadata("\(architecture).embedding_length", value: .uint32(UInt32(embeddingLength)))
        builder.addMetadata("\(architecture).block_count", value: .uint32(UInt32(blockCount)))
        builder.addMetadata("\(architecture).attention.head_count", value: .uint32(UInt32(headCount)))

        if let ff = feedForwardLength {
            builder.addMetadata("\(architecture).feed_forward_length", value: .uint32(UInt32(ff)))
        }
        if let kv = headCountKV {
            builder.addMetadata("\(architecture).attention.head_count_kv", value: .uint32(UInt32(kv)))
        }
        if let eps = rmsEps {
            builder.addMetadata("\(architecture).attention.layer_norm_rms_epsilon", value: .float32(eps))
        }
        if let base = ropeFreqBase {
            builder.addMetadata("\(architecture).rope.freq_base", value: .float32(base))
        }

        // Vocabulary tokens
        let tokens = (0..<vocabSize).map { "tok\($0)" }
        builder.addMetadata("tokenizer.ggml.tokens", value: .array(tokens.map { .string($0) }))

        if includeLmHead {
            builder.addTensor(name: "output.weight", shape: [UInt64(vocabSize), UInt64(embeddingLength)], type: .f32)
        }

        let data = builder.build()
        return try GGUFFile.parse(data: data)
    }

    @Test("Extract basic config")
    func basicConfig() throws {
        let file = try makeGGUFData()
        let config = try GGUFConfigExtractor.extractTransformerConfig(from: file, archHint: "llama", isMoE: false)

        #expect(config.hiddenSize == 256)
        #expect(config.hiddenLayers == 2)
        #expect(config.attentionHeads == 4)
        #expect(config.vocabularySize == 100)
        #expect(config.resolvedHeadDimensions == 64) // 256 / 4
    }

    @Test("Custom feed forward length")
    func customFFN() throws {
        let file = try makeGGUFData(feedForwardLength: 512)
        let config = try GGUFConfigExtractor.extractTransformerConfig(from: file, archHint: "llama", isMoE: false)
        #expect(config.intermediateSize == 512)
    }

    @Test("Default feed forward length is 4x hidden")
    func defaultFFN() throws {
        let file = try makeGGUFData()
        let config = try GGUFConfigExtractor.extractTransformerConfig(from: file, archHint: "llama", isMoE: false)
        #expect(config.intermediateSize == 1024) // 256 * 4
    }

    @Test("KV heads override")
    func kvHeads() throws {
        let file = try makeGGUFData(headCountKV: 2)
        let config = try GGUFConfigExtractor.extractTransformerConfig(from: file, archHint: "llama", isMoE: false)
        #expect(config.kvHeads == 2)
    }

    @Test("RMS norm eps")
    func rmsEps() throws {
        let file = try makeGGUFData(rmsEps: 1e-6)
        let config = try GGUFConfigExtractor.extractTransformerConfig(from: file, archHint: "llama", isMoE: false)
        #expect(config.normEps == 1e-6)
    }

    @Test("RoPE theta")
    func ropeTheta() throws {
        let file = try makeGGUFData(ropeFreqBase: 500_000.0)
        let config = try GGUFConfigExtractor.extractTransformerConfig(from: file, archHint: "llama", isMoE: false)
        #expect(config.ropeTheta == 500_000.0)
    }

    @Test("Tie word embeddings when no output.weight tensor")
    func tieEmbeddings() throws {
        let file = try makeGGUFData(includeLmHead: false)
        let config = try GGUFConfigExtractor.extractTransformerConfig(from: file, archHint: "llama", isMoE: false)
        #expect(config.tieWordEmbeddings == true)
    }

    @Test("No tie when output.weight tensor exists")
    func noTieEmbeddings() throws {
        let file = try makeGGUFData(includeLmHead: true)
        let config = try GGUFConfigExtractor.extractTransformerConfig(from: file, archHint: "llama", isMoE: false)
        #expect(config.tieWordEmbeddings == false)
    }

    @Test("Missing embedding_length throws")
    func missingEmbedding() throws {
        var builder = GGUFTestBuilder()
        builder.addMetadata("general.architecture", value: .string("llama"))
        builder.addMetadata("llama.block_count", value: .uint32(2))
        builder.addMetadata("llama.attention.head_count", value: .uint32(4))
        let file = try GGUFFile.parse(data: builder.build())

        #expect(throws: GGUFLoadError.self) {
            try GGUFConfigExtractor.extractTransformerConfig(from: file, archHint: "llama", isMoE: false)
        }
    }
}

// MARK: - GGUFTensorBridge Tests

@Suite("GGUFTensorBridge")
struct GGUFTensorBridgeTests {

    let bridge = GGUFTensorBridge()

    @Test("F32 tensor loading")
    func f32Loading() throws {
        let values: [Float] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        let data = values.withUnsafeBytes { Data($0) }
        let tensor = GGUFTensorInfo(
            name: "test",
            dimensions: [3, 2], // GGUF: inner-first
            quantizationType: .f32,
            offset: 0
        )

        let result = try bridge.convert(tensor: tensor, data: data)
        // GGUF dims [3,2] reversed to MLX [2,3]
        #expect(result.shape == [2, 3])
        #expect(result.dtype == .float16)
    }

    @Test("F16 tensor loading")
    func f16Loading() throws {
        // Create F16 data: 4 float16 values
        let float16Values: [UInt16] = [
            0x3C00, // 1.0
            0x4000, // 2.0
            0x4200, // 3.0
            0x4400, // 4.0
        ]
        let data = float16Values.withUnsafeBytes { Data($0) }
        let tensor = GGUFTensorInfo(
            name: "test",
            dimensions: [4],
            quantizationType: .f16,
            offset: 0
        )

        let result = try bridge.convert(tensor: tensor, data: data)
        #expect(result.shape == [4])
        #expect(result.dtype == .float16)

        let asFloat = result.asType(.float32)
        let vals: [Float] = [asFloat[0].item(), asFloat[1].item(), asFloat[2].item(), asFloat[3].item()]
        #expect(abs(vals[0] - 1.0) < 0.01)
        #expect(abs(vals[1] - 2.0) < 0.01)
        #expect(abs(vals[2] - 3.0) < 0.01)
        #expect(abs(vals[3] - 4.0) < 0.01)
    }

    @Test("Q4_0 dequantization produces correct shape")
    func q4_0Shape() throws {
        // Q4_0: 32 elements per block = 18 bytes (2 scale + 16 data)
        // 1 block = 32 elements
        let blockData = Data(repeating: 0, count: 18)
        let tensor = GGUFTensorInfo(
            name: "test",
            dimensions: [32],
            quantizationType: .q4_0,
            offset: 0
        )

        let result = try bridge.convert(tensor: tensor, data: blockData)
        #expect(result.shape == [32])
        #expect(result.dtype == .float16)
    }

    @Test("Q8_0 dequantization produces correct shape")
    func q8_0Shape() throws {
        // Q8_0: 32 elements per block = 34 bytes (2 scale + 32 data)
        let blockData = Data(repeating: 0, count: 34)
        let tensor = GGUFTensorInfo(
            name: "test",
            dimensions: [32],
            quantizationType: .q8_0,
            offset: 0
        )

        let result = try bridge.convert(tensor: tensor, data: blockData)
        #expect(result.shape == [32])
        #expect(result.dtype == .float16)
    }

    @Test("Q4_K dequantization produces correct shape")
    func q4_KShape() throws {
        // Q4_K: 256 elements per super-block = 144 bytes
        let blockData = Data(repeating: 0, count: 144)
        let tensor = GGUFTensorInfo(
            name: "test",
            dimensions: [256],
            quantizationType: .q4_K,
            offset: 0
        )

        let result = try bridge.convert(tensor: tensor, data: blockData)
        #expect(result.shape == [256])
        #expect(result.dtype == .float16)
    }

    @Test("Q6_K dequantization produces correct shape")
    func q6_KShape() throws {
        // Q6_K: 256 elements per super-block = 210 bytes
        let blockData = Data(repeating: 0, count: 210)
        let tensor = GGUFTensorInfo(
            name: "test",
            dimensions: [256],
            quantizationType: .q6_K,
            offset: 0
        )

        let result = try bridge.convert(tensor: tensor, data: blockData)
        #expect(result.shape == [256])
    }

    @Test("Q2_K dequantization produces correct shape")
    func q2_KShape() throws {
        let blockData = Data(repeating: 0, count: 84)
        let tensor = GGUFTensorInfo(
            name: "test",
            dimensions: [256],
            quantizationType: .q2_K,
            offset: 0
        )

        let result = try bridge.convert(tensor: tensor, data: blockData)
        #expect(result.shape == [256])
    }

    @Test("Q3_K dequantization produces correct shape")
    func q3_KShape() throws {
        let blockData = Data(repeating: 0, count: 110)
        let tensor = GGUFTensorInfo(
            name: "test",
            dimensions: [256],
            quantizationType: .q3_K,
            offset: 0
        )

        let result = try bridge.convert(tensor: tensor, data: blockData)
        #expect(result.shape == [256])
    }

    @Test("Q5_K dequantization produces correct shape")
    func q5_KShape() throws {
        let blockData = Data(repeating: 0, count: 176)
        let tensor = GGUFTensorInfo(
            name: "test",
            dimensions: [256],
            quantizationType: .q5_K,
            offset: 0
        )

        let result = try bridge.convert(tensor: tensor, data: blockData)
        #expect(result.shape == [256])
    }

    @Test("Unsupported quantization throws")
    func unsupportedQuant() throws {
        let tensor = GGUFTensorInfo(
            name: "test",
            dimensions: [32],
            quantizationType: .q4_1,
            offset: 0
        )

        #expect(throws: GGUFLoadError.self) {
            try bridge.convert(tensor: tensor, data: Data(repeating: 0, count: 100))
        }
    }

    @Test("2D tensor shape reversal")
    func shapeReversal() throws {
        // GGUF dimensions [8, 4] means 4 rows x 8 cols in GGUF convention
        // MLX should be [4, 8] (rows first)
        let values = [Float](repeating: 1.0, count: 32)
        let data = values.withUnsafeBytes { Data($0) }
        let tensor = GGUFTensorInfo(
            name: "test",
            dimensions: [8, 4], // GGUF: ne[0]=8, ne[1]=4
            quantizationType: .f32,
            offset: 0
        )

        let result = try bridge.convert(tensor: tensor, data: data)
        #expect(result.shape == [4, 8]) // MLX: reversed
    }

    @Test("Q4_0 dequantization values are reasonable")
    func q4_0Values() throws {
        // Build a single Q4_0 block with known values:
        // scale = 1.0 (f16), all nibbles = 8 (zero after subtraction)
        var blockData = Data(count: 18)

        // f16 representation of 1.0 = 0x3C00
        blockData[0] = 0x00  // lo byte
        blockData[1] = 0x3C  // hi byte

        // All data bytes = 0x88 → lo nibble = 8, hi nibble = 8
        // q = 8 - 8 = 0, so all values should be 0
        for i in 2..<18 {
            blockData[i] = 0x88
        }

        let tensor = GGUFTensorInfo(
            name: "test",
            dimensions: [32],
            quantizationType: .q4_0,
            offset: 0
        )

        let result = try bridge.convert(tensor: tensor, data: blockData)
        let floatResult = result.asType(.float32)
        eval(floatResult)

        // All values should be 0 (scale * (8-8) = 0)
        let sum: Float = floatResult.sum().item()
        #expect(abs(sum) < 0.01)
    }
}

// MARK: - TransformerModel Tests

@Suite("TransformerModel")
struct TransformerModelTests {

    private func makeSmallConfig() -> TransformerConfiguration {
        TransformerConfiguration(
            hiddenSize: 64,
            hiddenLayers: 2,
            intermediateSize: 128,
            attentionHeads: 4,
            vocabularySize: 100,
            kvHeads: 2
        )
    }

    @Test("Model creation")
    func modelCreation() {
        let config = makeSmallConfig()
        let model = TransformerModel(config)
        #expect(model.vocabularySize == 100)
        #expect(model.layerCount == 2)
        #expect(model.kvHeads == [2, 2])
    }

    @Test("Forward pass shape")
    func forwardPassShape() {
        let config = makeSmallConfig()
        let model = TransformerModel(config)

        let tokens = MLXArray([1, 2, 3]).reshaped([1, 3])
        let cache = model.newCache(parameters: nil)

        let output = model.callAsFunction(tokens, cache: cache)
        #expect(output.shape == [1, 3, 100]) // [batch, seq, vocab]
    }

    @Test("Tied embeddings use embed_tokens as lm_head")
    func tiedEmbeddings() {
        let config = TransformerConfiguration(
            hiddenSize: 64,
            hiddenLayers: 1,
            intermediateSize: 128,
            attentionHeads: 4,
            vocabularySize: 100,
            tieWordEmbeddings: true
        )
        let model = TransformerModel(config)
        // lmHead should be nil when tied
        #expect(model.lmHead == nil)
    }

    @Test("Untied embeddings have separate lm_head")
    func untiedEmbeddings() {
        let config = TransformerConfiguration(
            hiddenSize: 64,
            hiddenLayers: 1,
            intermediateSize: 128,
            attentionHeads: 4,
            vocabularySize: 100,
            tieWordEmbeddings: false
        )
        let model = TransformerModel(config)
        #expect(model.lmHead != nil)
    }

    @Test("Sanitize filters rotary embeddings")
    func sanitizeWeights() {
        let config = makeSmallConfig()
        let model = TransformerModel(config)

        let weights: [String: MLXArray] = [
            "model.layers.0.self_attn.rotary_emb.inv_freq": MLXArray([1.0]),
            "model.layers.0.self_attn.q_proj.weight": MLXArray([1.0]),
        ]

        let sanitized = model.sanitize(weights: weights)
        #expect(sanitized.count == 1)
        #expect(sanitized["model.layers.0.self_attn.q_proj.weight"] != nil)
    }
}

// MARK: - ChatTemplateRenderer Tests

@Suite("ChatTemplateRenderer")
struct ChatTemplateRendererTests {

    @Test("Simple template rendering")
    func simpleTemplate() throws {
        let template = "{% for message in messages %}{{ message.role }}: {{ message.content }}\n{% endfor %}"
        let renderer = try ChatTemplateRenderer(
            templateString: template,
            bosToken: "<s>",
            eosToken: "</s>"
        )

        let result = try renderer.render(messages: [
            .system("Be helpful"),
            .user("Hello"),
        ])

        #expect(result.contains("system: Be helpful"))
        #expect(result.contains("user: Hello"))
    }

    @Test("BOS/EOS tokens available in template")
    func specialTokens() throws {
        let template = "{{ bos_token }}{% for m in messages %}{{ m.content }}{% endfor %}{{ eos_token }}"
        let renderer = try ChatTemplateRenderer(
            templateString: template,
            bosToken: "<BOS>",
            eosToken: "<EOS>"
        )

        let result = try renderer.render(messages: [.user("Hi")])
        #expect(result == "<BOS>Hi<EOS>")
    }

    @Test("add_generation_prompt flag")
    func addGenPrompt() throws {
        let template = "{% for m in messages %}{{ m.content }}{% endfor %}{% if add_generation_prompt %}ASSISTANT:{% endif %}"
        let renderer = try ChatTemplateRenderer(
            templateString: template,
            bosToken: nil,
            eosToken: nil
        )

        let withPrompt = try renderer.render(messages: [.user("Hi")], addGenerationPrompt: true)
        #expect(withPrompt.contains("ASSISTANT:"))

        let withoutPrompt = try renderer.render(messages: [.user("Hi")], addGenerationPrompt: false)
        #expect(!withoutPrompt.contains("ASSISTANT:"))
    }

    @Test("Nil BOS/EOS renders empty strings")
    func nilSpecialTokens() throws {
        let template = "[{{ bos_token }}]{{ messages[0].content }}[{{ eos_token }}]"
        let renderer = try ChatTemplateRenderer(
            templateString: template,
            bosToken: nil,
            eosToken: nil
        )

        let result = try renderer.render(messages: [.user("test")])
        #expect(result == "[]test[]")
    }

    @Test("Additional context merged into template")
    func additionalContext() throws {
        let template = "{% if enable_thinking %}THINK{% endif %}{{ messages[0].content }}"
        let renderer = try ChatTemplateRenderer(
            templateString: template,
            bosToken: nil,
            eosToken: nil
        )

        let withThinking = try renderer.render(
            messages: [.user("Hi")],
            additionalContext: ["enable_thinking": true]
        )
        #expect(withThinking == "THINKHi")

        let withoutThinking = try renderer.render(
            messages: [.user("Hi")],
            additionalContext: ["enable_thinking": false]
        )
        #expect(withoutThinking == "Hi")
    }
}

// MARK: - GGUFUserInputProcessor Tests

@Suite("GGUFUserInputProcessor")
struct GGUFUserInputProcessorTests {

    @Test("Fallback formatting without template")
    func fallbackFormatting() async throws {
        let tokenizer = MockTokenizer()
        let processor = GGUFUserInputProcessor(
            tokenizer: tokenizer,
            chatTemplate: nil,
            bosToken: nil,
            eosToken: nil,
            addBosToken: false
        )

        let input = UserInput(chat: [
            .system("Be helpful"),
            .user("Hello"),
        ])

        let result = try await processor.prepare(input: input)
        #expect(result.text.tokens.shape[0] == 1) // batch dim
        #expect(result.text.tokens.dim(1) > 0)    // has tokens
    }

    @Test("Template-based processing")
    func templateProcessing() async throws {
        let tokenizer = MockTokenizer()
        let processor = GGUFUserInputProcessor(
            tokenizer: tokenizer,
            chatTemplate: "{% for m in messages %}{{ m.role }}: {{ m.content }}\n{% endfor %}",
            bosToken: "<s>",
            eosToken: "</s>",
            addBosToken: false
        )

        let input = UserInput(prompt: "Hello")
        let result = try await processor.prepare(input: input)
        #expect(result.text.tokens.dim(1) > 0)
    }

    @Test("Additional context passed to template")
    func additionalContext() async throws {
        let tokenizer = MockTokenizer()
        let processor = GGUFUserInputProcessor(
            tokenizer: tokenizer,
            chatTemplate: "{% if enable_thinking %}THINK{% endif %}{{ messages[0].content }}",
            bosToken: nil,
            eosToken: nil,
            addBosToken: false
        )

        let input = UserInput(
            chat: [.user("Hello")],
            additionalContext: ["enable_thinking": true]
        )
        let result = try await processor.prepare(input: input)
        #expect(result.text.tokens.dim(1) > 0)
    }
}

// MARK: - KVCache Tests

@Suite("KVCache")
struct KVCacheTests {

    @Test("Simple cache creation")
    func simpleCreation() {
        let params = GenerateParameters()
        let caches = createKVCaches(layerCount: 4, parameters: params)
        #expect(caches.count == 4)
        #expect(caches[0].offset == 0)
    }

    @Test("Rotating cache with max size")
    func rotatingCache() {
        var params = GenerateParameters()
        params.maxKVSize = 256
        let caches = createKVCaches(layerCount: 2, parameters: params)
        #expect(caches.count == 2)
    }
}

// MARK: - GGUFConfigExtractor Bias Detection Tests

@Suite("GGUFConfigExtractor Bias Detection")
struct GGUFConfigExtractorBiasTests {

    @Test("Detect attention bias from tensor names")
    func detectAttentionBias() throws {
        var builder = GGUFTestBuilder()
        builder.addMetadata("general.architecture", value: .string("qwen2"))
        builder.addMetadata("qwen2.embedding_length", value: .uint32(64))
        builder.addMetadata("qwen2.block_count", value: .uint32(1))
        builder.addMetadata("qwen2.attention.head_count", value: .uint32(4))

        let tokens = (0..<16).map { "tok\($0)" }
        builder.addMetadata("tokenizer.ggml.tokens", value: .array(tokens.map { .string($0) }))

        // Add attention bias tensors
        builder.addTensor(name: "blk.0.attn_q.bias", shape: [64], type: .f32)

        let file = try GGUFFile.parse(data: builder.build())
        let config = try GGUFConfigExtractor.extractTransformerConfig(from: file, archHint: "qwen2", isMoE: false)
        #expect(config.attentionBias == true)
    }

    @Test("No attention bias when tensor absent")
    func noAttentionBias() throws {
        var builder = GGUFTestBuilder()
        builder.addMetadata("general.architecture", value: .string("llama"))
        builder.addMetadata("llama.embedding_length", value: .uint32(64))
        builder.addMetadata("llama.block_count", value: .uint32(1))
        builder.addMetadata("llama.attention.head_count", value: .uint32(4))

        let tokens = (0..<16).map { "tok\($0)" }
        builder.addMetadata("tokenizer.ggml.tokens", value: .array(tokens.map { .string($0) }))

        let file = try GGUFFile.parse(data: builder.build())
        let config = try GGUFConfigExtractor.extractTransformerConfig(from: file, archHint: "llama", isMoE: false)
        #expect(config.attentionBias == false)
    }

    @Test("Detect MLP bias from tensor names")
    func detectMlpBias() throws {
        var builder = GGUFTestBuilder()
        builder.addMetadata("general.architecture", value: .string("qwen2"))
        builder.addMetadata("qwen2.embedding_length", value: .uint32(64))
        builder.addMetadata("qwen2.block_count", value: .uint32(1))
        builder.addMetadata("qwen2.attention.head_count", value: .uint32(4))

        let tokens = (0..<16).map { "tok\($0)" }
        builder.addMetadata("tokenizer.ggml.tokens", value: .array(tokens.map { .string($0) }))

        builder.addTensor(name: "blk.0.ffn_gate.bias", shape: [64], type: .f32)

        let file = try GGUFFile.parse(data: builder.build())
        let config = try GGUFConfigExtractor.extractTransformerConfig(from: file, archHint: "qwen2", isMoE: false)
        #expect(config.mlpBias == true)
    }
}

// MARK: - PromptCacheSnapshot Tests

@Suite("PromptCacheSnapshot")
struct PromptCacheSnapshotTests {

    @Test("Capture and materialize simple cache")
    func captureAndMaterialize() {
        let caches: [KVCache] = [KVCacheSimple(), KVCacheSimple()]

        let snapshot = capturePromptCache(cache: caches, prefixTokenCount: 10)
        #expect(snapshot.prefixTokenCount == 10)
        #expect(snapshot.cacheClasses.count == 2)
        #expect(snapshot.cacheClasses[0] == "KVCacheSimple")

        let restored = materializePromptCache(from: snapshot)
        #expect(restored.count == 2)
        #expect(restored[0] is KVCacheSimple)
    }

    @Test("Capture and materialize rotating cache")
    func captureRotating() {
        let caches: [KVCache] = [RotatingKVCache(maxSize: 512, keep: 4, step: 128)]

        let snapshot = capturePromptCache(cache: caches, prefixTokenCount: 20)
        #expect(snapshot.cacheClasses[0] == "RotatingKVCache")

        let restored = materializePromptCache(from: snapshot)
        #expect(restored.count == 1)
        #expect(restored[0] is RotatingKVCache)
    }

    @Test("Capture and materialize quantized cache")
    func captureQuantized() {
        let caches: [KVCache] = [QuantizedKVCache(groupSize: 64, bits: 4)]

        let snapshot = capturePromptCache(cache: caches, prefixTokenCount: 5)
        #expect(snapshot.cacheClasses[0] == "QuantizedKVCache")

        let restored = materializePromptCache(from: snapshot)
        #expect(restored.count == 1)
        #expect(restored[0] is QuantizedKVCache)
    }
}

// MARK: - E2E Integration Test

@Suite("E2E Integration")
struct E2EIntegrationTests {

    /// Build a complete minimal GGUF binary suitable for GGUFModelLoader.
    private func buildTinyModelGGUF() -> Data {
        // Tiny model: hidden=32, layers=1, heads=2, kvHeads=2, intermediate=64, vocab=16
        let hidden = 32
        let intermediate = 64
        let vocabSize = 16
        let heads = 2

        var builder = GGUFTestBuilder()

        // Architecture metadata
        builder.addMetadata("general.architecture", value: .string("llama"))
        builder.addMetadata("llama.embedding_length", value: .uint32(UInt32(hidden)))
        builder.addMetadata("llama.block_count", value: .uint32(1))
        builder.addMetadata("llama.attention.head_count", value: .uint32(UInt32(heads)))
        builder.addMetadata("llama.attention.head_count_kv", value: .uint32(UInt32(heads)))
        builder.addMetadata("llama.feed_forward_length", value: .uint32(UInt32(intermediate)))
        builder.addMetadata("llama.context_length", value: .uint32(512))

        // Tokenizer metadata (SentencePiece style)
        let tokens = (0..<vocabSize).map { "<tok\($0)>" }
        builder.addMetadata("tokenizer.ggml.tokens", value: .array(tokens.map { .string($0) }))
        builder.addMetadata("tokenizer.ggml.scores", value: .array((0..<vocabSize).map { .float32(Float(vocabSize - $0)) }))
        builder.addMetadata("tokenizer.ggml.model", value: .string("llama"))
        builder.addMetadata("tokenizer.ggml.bos_token_id", value: .uint32(0))
        builder.addMetadata("tokenizer.ggml.eos_token_id", value: .uint32(UInt32(vocabSize - 1)))

        // Tensors (GGUF dims: inner-first)
        // Embedding: MLX [vocab, hidden] → GGUF [hidden, vocab]
        builder.addTensor(name: "token_embd.weight", shape: [UInt64(hidden), UInt64(vocabSize)], type: .f32)

        // Final norm: [hidden]
        builder.addTensor(name: "output_norm.weight", shape: [UInt64(hidden)], type: .f32)

        // Layer 0 norms
        builder.addTensor(name: "blk.0.attn_norm.weight", shape: [UInt64(hidden)], type: .f32)
        builder.addTensor(name: "blk.0.ffn_norm.weight", shape: [UInt64(hidden)], type: .f32)

        // Layer 0 attention: all [hidden, hidden] → GGUF [hidden, hidden]
        builder.addTensor(name: "blk.0.attn_q.weight", shape: [UInt64(hidden), UInt64(hidden)], type: .f32)
        builder.addTensor(name: "blk.0.attn_k.weight", shape: [UInt64(hidden), UInt64(hidden)], type: .f32)
        builder.addTensor(name: "blk.0.attn_v.weight", shape: [UInt64(hidden), UInt64(hidden)], type: .f32)
        builder.addTensor(name: "blk.0.attn_output.weight", shape: [UInt64(hidden), UInt64(hidden)], type: .f32)

        // Layer 0 MLP
        builder.addTensor(name: "blk.0.ffn_gate.weight", shape: [UInt64(hidden), UInt64(intermediate)], type: .f32)
        builder.addTensor(name: "blk.0.ffn_up.weight", shape: [UInt64(hidden), UInt64(intermediate)], type: .f32)
        builder.addTensor(name: "blk.0.ffn_down.weight", shape: [UInt64(intermediate), UInt64(hidden)], type: .f32)

        return builder.build()
    }

    @Test("Load tiny model from GGUF data")
    func loadTinyModel() throws {
        let ggufData = buildTinyModelGGUF()
        let file = try GGUFFile.parse(data: ggufData)

        // Verify metadata
        #expect(file.architecture == "llama")
        #expect(file.embeddingLength == 32)
        #expect(file.blockCount == 1)

        // Extract config
        let config = try GGUFConfigExtractor.extractTransformerConfig(from: file, archHint: "llama", isMoE: false)
        #expect(config.hiddenSize == 32)
        #expect(config.hiddenLayers == 1)
        #expect(config.intermediateSize == 64)
        #expect(config.tieWordEmbeddings == true) // no output.weight tensor
    }

    @Test("Full pipeline: GGUF → Model → Forward pass")
    func fullPipeline() throws {
        let ggufData = buildTinyModelGGUF()

        // Write to temp file for GGUFModelLoader
        let tempURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("test_tiny_model.gguf")
        try ggufData.write(to: tempURL)
        defer { try? FileManager.default.removeItem(at: tempURL) }

        // Load model
        let loader = GGUFModelLoader()
        let context = try loader.loadContext(url: tempURL)

        // Verify context
        #expect(context.configuration.name == "test_tiny_model")
        #expect(context.model.layerCount == 1)

        // Forward pass with token input
        let tokens = MLXArray([Int32(1), Int32(2), Int32(3)]).reshaped([1, 3])
        let cache = context.model.newCache(parameters: nil)
        let output = context.model.callAsFunction(
            LMInput.Text(tokens: tokens),
            cache: cache,
            state: nil
        )

        // Should produce logits of shape [1, 3, vocab_size=16]
        #expect(output.logits.shape == [1, 3, 16])
    }

    @Test("Token generation from tiny model")
    func tokenGeneration() throws {
        let ggufData = buildTinyModelGGUF()

        let tempURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("test_tiny_gen.gguf")
        try ggufData.write(to: tempURL)
        defer { try? FileManager.default.removeItem(at: tempURL) }

        let loader = GGUFModelLoader()
        let context = try loader.loadContext(url: tempURL)

        // Create input
        let tokens = MLXArray([Int32(1)]).reshaped([1, 1])
        let input = LMInput(tokens: tokens)

        // Generate with argmax (temperature=0) and limited tokens
        var params = GenerateParameters()
        params.temperature = 0
        params.maxTokens = 3

        let iterator = try TokenIterator(
            input: input,
            model: context.model,
            cache: context.model.newCache(parameters: params),
            parameters: params,
            eosTokenIds: context.configuration.eosTokenIds
        )

        var generated: [Int] = []
        var iter = iterator
        while let token = iter.next() {
            generated.append(token)
        }

        // Should have generated tokens (up to maxTokens, unless EOS hit)
        #expect(generated.count > 0)
        #expect(generated.count <= 3)
    }
}

// MARK: - Test Helpers

/// Minimal mock tokenizer for testing.
struct MockTokenizer: Tokenizer, @unchecked Sendable {
    var bosTokenID: Int? { 1 }
    var eosTokenID: Int? { 2 }
    var vocabularySize: Int { 100 }

    func encode(text: String) -> [Int] {
        // Simple byte-level encoding
        Array(text.utf8).map { Int($0) }
    }

    func decode(tokens: [Int]) -> String {
        String(tokens.compactMap { UnicodeScalar($0) }.map { Character($0) })
    }

    func tokenToString(_ id: Int) -> String? {
        if id == 1 { return "<s>" }
        if id == 2 { return "</s>" }
        return String(UnicodeScalar(id) ?? UnicodeScalar(0))
    }
}

// MARK: - GGUFTestBuilder

/// Builds minimal GGUF binary data for testing.
struct GGUFTestBuilder {
    private var metadata: [(String, GGUFMetadataValue)] = []
    private var tensors: [(name: String, shape: [UInt64], type: GGUFQuantizationType)] = []

    mutating func addMetadata(_ key: String, value: GGUFMetadataValue) {
        metadata.append((key, value))
    }

    mutating func addTensor(name: String, shape: [UInt64], type: GGUFQuantizationType) {
        tensors.append((name, shape, type))
    }

    func build() -> Data {
        var data = Data()

        // Magic
        appendUInt32(&data, GGUFFile.magic)
        // Version 3
        appendUInt32(&data, 3)
        // Tensor count
        appendUInt64(&data, UInt64(tensors.count))
        // Metadata KV count
        appendUInt64(&data, UInt64(metadata.count))

        // Metadata
        for (key, value) in metadata {
            appendString(&data, key)
            appendMetadataValue(&data, value)
        }

        // Tensor directory
        var tensorDataSize = 0
        for (name, shape, type) in tensors {
            appendString(&data, name)
            appendUInt32(&data, UInt32(shape.count))
            for dim in shape {
                appendUInt64(&data, dim)
            }
            appendUInt32(&data, type.rawValue)
            appendUInt64(&data, UInt64(tensorDataSize))

            let elements = shape.reduce(1, *)
            let bytesPerElement: Int
            switch type {
            case .f32: bytesPerElement = 4
            case .f16: bytesPerElement = 2
            default: bytesPerElement = 2
            }
            tensorDataSize += Int(elements) * bytesPerElement
        }

        // Pad to 32-byte alignment
        let headerEnd = data.count
        let alignment = 32
        let padding = (alignment - (headerEnd % alignment)) % alignment
        data.append(Data(repeating: 0, count: padding))

        // Tensor data (zeros)
        data.append(Data(repeating: 0, count: tensorDataSize))

        return data
    }

    private func appendUInt32(_ data: inout Data, _ value: UInt32) {
        withUnsafeBytes(of: value.littleEndian) { data.append(contentsOf: $0) }
    }

    private func appendUInt64(_ data: inout Data, _ value: UInt64) {
        withUnsafeBytes(of: value.littleEndian) { data.append(contentsOf: $0) }
    }

    private func appendString(_ data: inout Data, _ value: String) {
        let utf8 = Array(value.utf8)
        appendUInt64(&data, UInt64(utf8.count))
        data.append(contentsOf: utf8)
    }

    private func appendMetadataValue(_ data: inout Data, _ value: GGUFMetadataValue) {
        switch value {
        case .uint32(let v):
            appendUInt32(&data, 4) // type tag for UINT32
            appendUInt32(&data, v)
        case .float32(let v):
            appendUInt32(&data, 6) // type tag for FLOAT32
            withUnsafeBytes(of: v) { data.append(contentsOf: $0) }
        case .string(let s):
            appendUInt32(&data, 8) // type tag for STRING
            appendString(&data, s)
        case .array(let arr):
            appendUInt32(&data, 9) // type tag for ARRAY
            // Determine element type from first element
            if let first = arr.first {
                switch first {
                case .string:
                    appendUInt32(&data, 8) // STRING elements
                case .float32:
                    appendUInt32(&data, 6) // FLOAT32 elements
                default:
                    appendUInt32(&data, 4) // UINT32 elements
                }
            } else {
                appendUInt32(&data, 8)
            }
            appendUInt64(&data, UInt64(arr.count))
            for element in arr {
                switch element {
                case .string(let s):
                    appendString(&data, s)
                case .uint32(let v):
                    appendUInt32(&data, v)
                case .float32(let v):
                    withUnsafeBytes(of: v) { data.append(contentsOf: $0) }
                default:
                    break
                }
            }
        default:
            break
        }
    }
}
