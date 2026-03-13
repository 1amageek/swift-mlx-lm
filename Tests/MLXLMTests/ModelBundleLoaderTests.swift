import Foundation
import Testing
import SwiftLM
@testable import MLXLM

/// Tests for ModelBundleLoader's WeightManifest → RawWeights conversion
/// and pipeline assembly (config → IR → compile).
///
/// These tests use mock bundles with synthetic data to verify the pipeline
/// without requiring real model files.
@Suite("ModelBundleLoader")
struct ModelBundleLoaderTests {

    // MARK: - HFConfigDecoder + IRGraphAssembler Integration

    @Test("Config → IR assembly for standard transformer")
    func configToIRTransformer() throws {
        let json = """
        {"model_type": "llama", "hidden_size": 256, "num_hidden_layers": 4,
         "num_attention_heads": 4, "intermediate_size": 512, "vocab_size": 1000,
         "rms_norm_eps": 1e-06, "rope_theta": 10000.0, "tie_word_embeddings": true}
        """.data(using: .utf8)!

        let decoder = HFConfigDecoder()
        let config = try decoder.decode(from: json)

        let detector = HFArchitectureDetector()
        let arch = try detector.detect(from: json)
        #expect(arch == .transformer)

        let assembler = IRGraphAssembler()
        let graph = try assembler.assemble(config: config, architecture: arch)

        // Verify graph has operations
        #expect(!graph.rootRegion.operations.isEmpty)

        // Should have tokenEmbedding, norm, outputHead at minimum
        let opKinds = graph.rootRegion.operations.map { $0.kind }
        let hasEmbedding = opKinds.contains { if case .tokenEmbedding = $0 { return true }; return false }
        let hasOutputHead = opKinds.contains { if case .outputHead = $0 { return true }; return false }
        #expect(hasEmbedding)
        #expect(hasOutputHead)
    }

    @Test("Config → IR assembly for hybrid DeltaNet attention")
    func configToIRHybridDeltaNet() throws {
        let json = """
        {"model_type": "qwen3_5",
         "text_config": {
           "hidden_size": 2048, "num_hidden_layers": 4, "num_attention_heads": 8,
           "num_key_value_heads": 2, "head_dim": 256, "intermediate_size": 6144,
           "vocab_size": 248320, "rms_norm_eps": 1e-06, "full_attention_interval": 4,
           "linear_num_key_heads": 16, "linear_num_value_heads": 16,
           "linear_key_head_dim": 128, "linear_value_head_dim": 128,
           "linear_conv_kernel_dim": 4,
           "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"],
           "rope_parameters": {"rope_theta": 10000000, "partial_rotary_factor": 0.25, "rope_type": "default"}
         }}
        """.data(using: .utf8)!

        let decoder = HFConfigDecoder()
        let config = try decoder.decode(from: json)

        let detector = HFArchitectureDetector()
        let arch = try detector.detect(from: json)
        #expect(arch == .hybridDeltaNetAttention)

        let assembler = IRGraphAssembler()
        let graph = try assembler.assemble(config: config, architecture: arch)
        #expect(!graph.rootRegion.operations.isEmpty)
    }

    @Test("Config → IR assembly for MoE")
    func configToIRMoE() throws {
        let json = """
        {"model_type": "mixtral", "hidden_size": 4096, "num_hidden_layers": 4,
         "num_attention_heads": 32, "num_key_value_heads": 8,
         "intermediate_size": 14336, "vocab_size": 32000,
         "rms_norm_eps": 1e-05, "num_local_experts": 8, "num_experts_per_tok": 2}
        """.data(using: .utf8)!

        let decoder = HFConfigDecoder()
        let config = try decoder.decode(from: json)

        let detector = HFArchitectureDetector()
        let arch = try detector.detect(from: json)
        #expect(arch == .moe)

        let assembler = IRGraphAssembler()
        let graph = try assembler.assemble(config: config, architecture: arch)
        #expect(!graph.rootRegion.operations.isEmpty)
    }

    // MARK: - HFDirectoryBundle Validation

    @Test("HFDirectoryBundle throws on missing config.json")
    func bundleMissingConfig() {
        let tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
        try? FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tempDir) }

        #expect(throws: (any Error).self) {
            _ = try HFDirectoryBundle(directory: tempDir)
        }
    }

    // MARK: - HFTensorNameMapper

    @Test("HFTensorNameMapper passes through standard names")
    func tensorMapperPassthrough() {
        let mapper = HFTensorNameMapper()
        #expect(mapper.mlxPath(for: "model.layers.0.self_attn.q_proj.weight") ==
                "model.layers.0.self_attn.q_proj.weight")
        #expect(mapper.mlxPath(for: "model.embed_tokens.weight") ==
                "model.embed_tokens.weight")
        #expect(mapper.mlxPath(for: "lm_head.weight") == "lm_head.weight")
    }

    @Test("HFTensorNameMapper skips rotary_emb")
    func tensorMapperSkipsRotary() {
        let mapper = HFTensorNameMapper()
        #expect(mapper.mlxPath(for: "model.layers.0.self_attn.rotary_emb.inv_freq") == nil)
    }

    @Test("HFTensorNameMapper with tied embeddings skips lm_head")
    func tensorMapperTiedEmbeddings() {
        let mapper = HFTensorNameMapper(tiedEmbeddings: true)
        #expect(mapper.mlxPath(for: "lm_head.weight") == nil)
        #expect(mapper.mlxPath(for: "model.embed_tokens.weight") ==
                "model.embed_tokens.weight")
    }

    @Test("HFTensorNameMapper without tied embeddings keeps lm_head")
    func tensorMapperUntiedEmbeddings() {
        let mapper = HFTensorNameMapper(tiedEmbeddings: false)
        #expect(mapper.mlxPath(for: "lm_head.weight") == "lm_head.weight")
    }

    @Test("HFTensorNameMapper strips VLM language_model prefix")
    func tensorMapperStripsVLMPrefix() {
        let mapper = HFTensorNameMapper(tiedEmbeddings: true)
        #expect(mapper.mlxPath(for: "language_model.model.layers.0.self_attn.q_proj.weight") ==
                "model.layers.0.self_attn.q_proj.weight")
        #expect(mapper.mlxPath(for: "language_model.model.embed_tokens.weight") ==
                "model.embed_tokens.weight")
        #expect(mapper.mlxPath(for: "language_model.model.norm.weight") ==
                "model.norm.weight")
        #expect(mapper.mlxPath(for: "language_model.lm_head.weight") == nil)  // tied
    }

    @Test("HFTensorNameMapper skips vision tower tensors")
    func tensorMapperSkipsVisionTower() {
        let mapper = HFTensorNameMapper()
        #expect(mapper.mlxPath(for: "vision_tower.blocks.0.attn.proj.weight") == nil)
        #expect(mapper.mlxPath(for: "vision_tower.merger.linear_fc1.weight") == nil)
        #expect(mapper.mlxPath(for: "vision_tower.patch_embed.proj.weight") == nil)
    }

    // MARK: - ModelFormat Detection

    @Test("ModelFormat equality")
    func modelFormatEquality() {
        #expect(ModelFormat.safetensors == ModelFormat.safetensors)
        #expect(ModelFormat.auto == ModelFormat.auto)
        #expect(ModelFormat.safetensors != ModelFormat.auto)
    }

    // MARK: - WeightManifest

    @Test("WeightManifest default empty quantization info")
    func weightManifestDefaults() {
        let manifest = WeightManifest(weights: [:])
        #expect(manifest.quantizationInfo.isEmpty)
        #expect(manifest.nameMapping.isEmpty)
    }

    // MARK: - QuantizationSpec

    @Test("QuantizationSpec equatable")
    func quantizationSpecEquatable() {
        let a = QuantizationSpec(groupSize: 64, bits: 4)
        let b = QuantizationSpec(groupSize: 64, bits: 4)
        let c = QuantizationSpec(groupSize: 32, bits: 8)
        #expect(a == b)
        #expect(a != c)
    }

    // MARK: - resolveLayerSchedule

    @Test("resolveLayerSchedule uses layerTypes when available")
    func resolveLayerScheduleFromTypes() throws {
        let assembler = IRGraphAssembler()
        let config = makeMinimalHybridConfig(layerTypes: [
            "linear_attention", "linear_attention", "linear_attention", "full_attention"
        ], fullAttentionInterval: nil)

        let schedule = try assembler.resolveLayerSchedule(config: config)
        #expect(schedule == [false, false, false, true])
    }

    @Test("resolveLayerSchedule falls back to fullAttentionInterval")
    func resolveLayerScheduleFromInterval() throws {
        let assembler = IRGraphAssembler()
        let config = makeMinimalHybridConfig(layerTypes: nil, fullAttentionInterval: 4)

        let schedule = try assembler.resolveLayerSchedule(config: config)
        // (i+1) % 4 == 0 → indices 3, 7, 11, ... are true
        #expect(schedule == [false, false, false, true])
    }

    @Test("resolveLayerSchedule throws when neither available")
    func resolveLayerScheduleThrows() {
        let assembler = IRGraphAssembler()
        let config = makeMinimalHybridConfig(layerTypes: nil, fullAttentionInterval: nil)

        #expect(throws: (any Error).self) {
            _ = try assembler.resolveLayerSchedule(config: config)
        }
    }

    @Test("resolveLayerSchedule throws on count mismatch")
    func resolveLayerScheduleCountMismatch() {
        let assembler = IRGraphAssembler()
        let config = makeMinimalHybridConfig(layerTypes: [
            "linear_attention", "full_attention"
        ], fullAttentionInterval: nil)
        // config has 4 layers but layerTypes has 2

        #expect(throws: (any Error).self) {
            _ = try assembler.resolveLayerSchedule(config: config)
        }
    }

    // MARK: - Helpers

    private func makeMinimalHybridConfig(
        layerTypes: [String]?,
        fullAttentionInterval: Int?
    ) -> ModelConfig {
        ModelConfig(
            hiddenSize: 2048,
            layerCount: 4,
            intermediateSize: 6144,
            vocabSize: 248320,
            attentionHeads: 8,
            kvHeads: 2,
            headDim: 256,
            attentionBias: false,
            mlpBias: false,
            normEps: 1e-6,
            normKind: .rmsNorm,
            ropeTheta: 10000000,
            ropeDimension: 64,
            ropeScaling: nil,
            tiedEmbeddings: true,
            expertCount: nil,
            expertsPerToken: nil,
            qkNorm: false,
            fullAttentionInterval: fullAttentionInterval,
            ssmNumHeads: 16,
            ssmGroupCount: 16,
            ssmKeyHeadDim: 128,
            ssmValueHeadDim: 128,
            convKernelSize: 4,
            partialRotaryFactor: 0.25,
            slidingWindow: nil,
            layerTypes: layerTypes,
            mropeAxes: nil
        )
    }
}
