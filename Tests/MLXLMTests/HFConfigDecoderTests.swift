import Testing
import SwiftLM
@testable import MLXLM

@Suite("HFConfigDecoder")
struct HFConfigDecoderTests {

    let decoder = HFConfigDecoder()

    // MARK: - Flat Config (Qwen2, Phi3)

    @Test("Decode Qwen2.5-0.5B flat config")
    func decodeQwen2FlatConfig() throws {
        let json = """
        {
          "model_type": "qwen2",
          "hidden_size": 896,
          "num_hidden_layers": 24,
          "num_attention_heads": 14,
          "num_key_value_heads": 2,
          "intermediate_size": 4864,
          "vocab_size": 151936,
          "rms_norm_eps": 1e-06,
          "rope_theta": 1000000.0,
          "tie_word_embeddings": true,
          "sliding_window": 32768
        }
        """.data(using: .utf8)!

        let config = try decoder.decode(from: json)

        #expect(config.hiddenSize == 896)
        #expect(config.layerCount == 24)
        #expect(config.attentionHeads == 14)
        #expect(config.kvHeads == 2)
        #expect(config.headDim == 64)  // 896 / 14
        #expect(config.intermediateSize == 4864)
        #expect(config.vocabSize == 151936)
        #expect(config.normEps == 1e-06)
        #expect(config.normKind == .rmsNorm)
        #expect(config.ropeTheta == 1000000.0)
        #expect(config.tiedEmbeddings == true)
        #expect(config.slidingWindow == 32768)
        #expect(config.attentionBias == false)
        #expect(config.layerTypes == nil)
    }

    @Test("Decode Phi3 flat config with explicit attention_bias")
    func decodePhi3FlatConfig() throws {
        let json = """
        {
          "model_type": "phi3",
          "hidden_size": 3072,
          "num_hidden_layers": 32,
          "num_attention_heads": 32,
          "num_key_value_heads": 32,
          "intermediate_size": 8192,
          "vocab_size": 32064,
          "rms_norm_eps": 1e-05,
          "rope_theta": 10000.0,
          "tie_word_embeddings": false,
          "attention_bias": false,
          "sliding_window": 2047
        }
        """.data(using: .utf8)!

        let config = try decoder.decode(from: json)

        #expect(config.hiddenSize == 3072)
        #expect(config.attentionHeads == 32)
        #expect(config.kvHeads == 32)
        #expect(config.headDim == 96)  // 3072 / 32
        #expect(config.tiedEmbeddings == false)
        #expect(config.slidingWindow == 2047)
    }

    // MARK: - Nested Config (Qwen3.5 VLM)

    @Test("Decode Qwen3.5 nested text_config with DeltaNet fields")
    func decodeQwen35NestedConfig() throws {
        let json = """
        {
          "model_type": "qwen3_5",
          "tie_word_embeddings": true,
          "text_config": {
            "hidden_size": 2048,
            "num_hidden_layers": 24,
            "num_attention_heads": 8,
            "num_key_value_heads": 2,
            "head_dim": 256,
            "intermediate_size": 6144,
            "vocab_size": 248320,
            "rms_norm_eps": 1e-06,
            "full_attention_interval": 4,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 16,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4,
            "attention_bias": false,
            "tie_word_embeddings": true,
            "layer_types": [
              "linear_attention", "linear_attention", "linear_attention", "full_attention",
              "linear_attention", "linear_attention", "linear_attention", "full_attention",
              "linear_attention", "linear_attention", "linear_attention", "full_attention",
              "linear_attention", "linear_attention", "linear_attention", "full_attention",
              "linear_attention", "linear_attention", "linear_attention", "full_attention",
              "linear_attention", "linear_attention", "linear_attention", "full_attention"
            ],
            "rope_parameters": {
              "rope_theta": 10000000,
              "partial_rotary_factor": 0.25,
              "mrope_interleaved": true,
              "mrope_section": [11, 11, 10],
              "rope_type": "default"
            }
          }
        }
        """.data(using: .utf8)!

        let config = try decoder.decode(from: json)

        // Core dimensions from text_config
        #expect(config.hiddenSize == 2048)
        #expect(config.layerCount == 24)
        #expect(config.attentionHeads == 8)
        #expect(config.kvHeads == 2)
        #expect(config.headDim == 256)  // Explicit head_dim
        #expect(config.intermediateSize == 6144)
        #expect(config.vocabSize == 248320)

        // RoPE from nested rope_parameters
        #expect(config.ropeTheta == 10000000.0)
        #expect(config.partialRotaryFactor == 0.25)

        // DeltaNet fields
        #expect(config.fullAttentionInterval == 4)
        #expect(config.ssmNumHeads == 16)       // linear_num_value_heads
        #expect(config.ssmGroupCount == 16)     // linear_num_key_heads
        #expect(config.ssmKeyHeadDim == 128)
        #expect(config.ssmValueHeadDim == 128)
        #expect(config.convKernelSize == 4)

        // Layer schedule
        #expect(config.layerTypes?.count == 24)
        #expect(config.layerTypes?[0] == "linear_attention")
        #expect(config.layerTypes?[3] == "full_attention")

        // M-RoPE
        #expect(config.mropeAxes != nil)
        #expect(config.mropeAxes?.sections == [11, 11, 10])
        #expect(config.mropeAxes?.interleaved == true)

        // Tied embeddings (inherited from top level)
        #expect(config.tiedEmbeddings == true)
    }

    // MARK: - Model Type Detection

    @Test("Extract model_type from flat config")
    func extractModelTypeFlat() throws {
        let json = """
        {"model_type": "qwen2", "hidden_size": 896, "num_hidden_layers": 24,
         "num_attention_heads": 14, "intermediate_size": 4864, "vocab_size": 151936}
        """.data(using: .utf8)!

        let modelType = try decoder.modelType(from: json)
        #expect(modelType == "qwen2")
    }

    @Test("Extract model_type from nested VLM config")
    func extractModelTypeNested() throws {
        let json = """
        {"model_type": "qwen3_5", "text_config": {"model_type": "qwen3_5_text", "hidden_size": 2048,
         "num_hidden_layers": 24, "num_attention_heads": 8, "intermediate_size": 6144, "vocab_size": 248320}}
        """.data(using: .utf8)!

        let modelType = try decoder.modelType(from: json)
        #expect(modelType == "qwen3_5")
    }

    // MARK: - Defaults and Edge Cases

    @Test("Missing kvHeads defaults to attentionHeads")
    func missingKVHeadsDefaultsToAttentionHeads() throws {
        let json = """
        {"model_type": "llama", "hidden_size": 4096, "num_hidden_layers": 32,
         "num_attention_heads": 32, "intermediate_size": 11008, "vocab_size": 32000}
        """.data(using: .utf8)!

        let config = try decoder.decode(from: json)
        #expect(config.kvHeads == 32)  // Same as attentionHeads
    }

    @Test("Missing rope_theta defaults to 10000")
    func missingRopeThetaDefaults() throws {
        let json = """
        {"model_type": "llama", "hidden_size": 4096, "num_hidden_layers": 32,
         "num_attention_heads": 32, "intermediate_size": 11008, "vocab_size": 32000}
        """.data(using: .utf8)!

        let config = try decoder.decode(from: json)
        #expect(config.ropeTheta == 10000.0)
    }

    @Test("Missing required field throws")
    func missingRequiredFieldThrows() throws {
        let json = """
        {"model_type": "llama", "num_hidden_layers": 32}
        """.data(using: .utf8)!

        #expect(throws: HFConfigError.self) {
            _ = try decoder.decode(from: json)
        }
    }

    @Test("LayerNorm detection from layer_norm_eps")
    func layerNormDetection() throws {
        let json = """
        {"model_type": "gpt2", "hidden_size": 768, "num_hidden_layers": 12,
         "num_attention_heads": 12, "intermediate_size": 3072, "vocab_size": 50257,
         "layer_norm_eps": 1e-05}
        """.data(using: .utf8)!

        let config = try decoder.decode(from: json)
        #expect(config.normKind == .layerNorm)
        #expect(config.normEps == 1e-05)
    }

    // MARK: - RoPE Dimension

    @Test("ropeDimension defaults to headDim when no partial_rotary_factor")
    func ropeDimensionDefaultsToHeadDim() throws {
        let json = """
        {"model_type": "llama", "hidden_size": 4096, "num_hidden_layers": 32,
         "num_attention_heads": 32, "intermediate_size": 11008, "vocab_size": 32000}
        """.data(using: .utf8)!

        let config = try decoder.decode(from: json)
        #expect(config.ropeDimension == 128)  // 4096 / 32 = 128 = headDim
    }

    @Test("ropeDimension computed from partial_rotary_factor")
    func ropeDimensionFromPartialRotary() throws {
        let json = """
        {"model_type": "phi3", "hidden_size": 3072, "num_hidden_layers": 32,
         "num_attention_heads": 32, "intermediate_size": 8192, "vocab_size": 32064,
         "partial_rotary_factor": 0.5}
        """.data(using: .utf8)!

        let config = try decoder.decode(from: json)
        #expect(config.headDim == 96)  // 3072 / 32
        #expect(config.partialRotaryFactor == 0.5)
        #expect(config.ropeDimension == 48)  // 96 * 0.5 = 48
    }

    @Test("ropeDimension computed from nested partial_rotary_factor")
    func ropeDimensionFromNestedPartialRotary() throws {
        let json = """
        {"model_type": "qwen3_5",
         "text_config": {
           "hidden_size": 2048, "num_hidden_layers": 4, "num_attention_heads": 8,
           "head_dim": 256, "intermediate_size": 6144, "vocab_size": 248320,
           "rope_parameters": {"rope_theta": 10000000, "partial_rotary_factor": 0.25}
         }}
        """.data(using: .utf8)!

        let config = try decoder.decode(from: json)
        #expect(config.headDim == 256)
        #expect(config.partialRotaryFactor == 0.25)
        #expect(config.ropeDimension == 64)  // 256 * 0.25 = 64
    }

    // MARK: - M-RoPE

    @Test("M-RoPE extraction from rope_scaling section (Qwen2-VL style)")
    func mropeFromRopeScaling() throws {
        let json = """
        {"model_type": "qwen2_vl", "hidden_size": 1536, "num_hidden_layers": 28,
         "num_attention_heads": 12, "num_key_value_heads": 2,
         "intermediate_size": 8960, "vocab_size": 151936,
         "rope_scaling": {"type": "mrope", "mrope_section": [16, 24, 24]}}
        """.data(using: .utf8)!

        let config = try decoder.decode(from: json)
        #expect(config.mropeAxes != nil)
        #expect(config.mropeAxes?.sections == [16, 24, 24])
        #expect(config.mropeAxes?.interleaved == false)
    }

    // MARK: - RoPE Scaling

    @Test("RoPE scaling extraction from flat config")
    func ropeScalingFlat() throws {
        let json = """
        {"model_type": "llama", "hidden_size": 4096, "num_hidden_layers": 32,
         "num_attention_heads": 32, "intermediate_size": 11008, "vocab_size": 32000,
         "rope_scaling": {"type": "linear", "factor": 2.0}}
        """.data(using: .utf8)!

        let config = try decoder.decode(from: json)
        #expect(config.ropeScaling != nil)
        #expect(config.ropeScaling?.kind == .linear)
        #expect(config.ropeScaling?.factor == 2.0)
    }

    @Test("Null rope_scaling returns nil")
    func nullRopeScaling() throws {
        let json = """
        {"model_type": "phi3", "hidden_size": 3072, "num_hidden_layers": 32,
         "num_attention_heads": 32, "intermediate_size": 8192, "vocab_size": 32064,
         "rope_scaling": null}
        """.data(using: .utf8)!

        let config = try decoder.decode(from: json)
        #expect(config.ropeScaling == nil)
    }

    @Test("Default rope_type returns nil scaling")
    func defaultRopeTypeReturnsNilScaling() throws {
        let json = """
        {"model_type": "qwen3_5",
         "text_config": {
           "hidden_size": 2048, "num_hidden_layers": 24, "num_attention_heads": 8,
           "intermediate_size": 6144, "vocab_size": 248320,
           "rope_parameters": {"rope_type": "default", "rope_theta": 10000000}
         }}
        """.data(using: .utf8)!

        let config = try decoder.decode(from: json)
        #expect(config.ropeScaling == nil)
        #expect(config.ropeTheta == 10000000.0)
    }
}

// MARK: - HFArchitectureDetector Tests

@Suite("HFArchitectureDetector")
struct HFArchitectureDetectorTests {

    let detector = HFArchitectureDetector()

    @Test("Standard transformer model_types")
    func standardTransformerTypes() {
        #expect(detector.detect(modelType: "llama") == .transformer)
        #expect(detector.detect(modelType: "qwen2") == .transformer)
        #expect(detector.detect(modelType: "mistral") == .transformer)
        #expect(detector.detect(modelType: "gemma") == .transformer)
        #expect(detector.detect(modelType: "gemma2") == .transformer)
        #expect(detector.detect(modelType: "phi3") == .transformer)
        #expect(detector.detect(modelType: "starcoder2") == .transformer)
    }

    @Test("Parallel attention + MLP model_types")
    func parallelAttentionMLPTypes() {
        #expect(detector.detect(modelType: "cohere") == .parallelAttentionMLP)
    }

    @Test("MoE model_types")
    func moeTypes() {
        #expect(detector.detect(modelType: "mixtral") == .moe)
        #expect(detector.detect(modelType: "qwen2_moe") == .moe)
    }

    @Test("Hybrid DeltaNet model_types")
    func hybridDeltaNetTypes() {
        #expect(detector.detect(modelType: "qwen3_5") == .hybridDeltaNetAttention)
    }

    @Test("Unknown model_type with layer_types falls back to field inspection")
    func unknownTypeWithLayerTypes() {
        let config: [String: Any] = [
            "text_config": [
                "layer_types": ["linear_attention", "linear_attention", "full_attention"]
            ]
        ]
        #expect(detector.detect(modelType: "unknown_hybrid", config: config) == .hybridDeltaNetAttention)
    }

    @Test("Unknown model_type with num_local_experts → .moe")
    func unknownTypeWithExperts() {
        let config: [String: Any] = [
            "num_local_experts": 8
        ]
        #expect(detector.detect(modelType: "unknown_moe", config: config) == .moe)
    }

    @Test("Completely unknown model_type → .transformer fallback")
    func unknownTypeFallback() {
        #expect(detector.detect(modelType: "totally_new_model") == .transformer)
    }

    @Test("Case-insensitive model_type matching")
    func caseInsensitive() {
        #expect(detector.detect(modelType: "Llama") == .transformer)
        #expect(detector.detect(modelType: "QWEN2") == .transformer)
    }

    @Test("Detect from raw JSON data")
    func detectFromData() throws {
        let json = """
        {"model_type": "qwen3_5", "text_config": {"model_type": "qwen3_5_text"}}
        """.data(using: .utf8)!

        let arch = try detector.detect(from: json)
        #expect(arch == .hybridDeltaNetAttention)
    }
}
