import Foundation
import Testing
import LMIR
@testable import MetalCompiler

@Suite("STAF Model Bundle Metadata Builder")
struct STAFModelBundleMetadataBuilderTests {
    @Test("Builder emits provenance and prefers chat_template.jinja")
    func builderEmitsBundleProvenance() throws {
        let directory = makeTempDirectory()
        defer { cleanup(directory) }

        let configData = Data("""
        {"model_type":"lfm2","hidden_size":2048}
        """.utf8)
        let tokenizerData = Data(#"{"version":"1.0"}"#.utf8)
        let tokenizerConfigData = Data(#"{"chat_template":"from tokenizer config"}"#.utf8)
        let specialTokensData = Data(#"{"eos_token":"<eos>"}"#.utf8)
        let chatTemplateData = Data("from jinja".utf8)
        let safetensorsURL = directory.appendingPathComponent("model.safetensors")

        try configData.write(to: directory.appendingPathComponent("config.json"))
        try tokenizerData.write(to: directory.appendingPathComponent("tokenizer.json"))
        try tokenizerConfigData.write(to: directory.appendingPathComponent("tokenizer_config.json"))
        try specialTokensData.write(to: directory.appendingPathComponent("special_tokens_map.json"))
        try chatTemplateData.write(to: directory.appendingPathComponent("chat_template.jinja"))
        try Data([0, 1, 2, 3]).write(to: safetensorsURL)

        let metadata = try STAFModelBundleMetadataBuilder().build(
            directory: directory,
            modelType: "lfm2",
            config: makeConfig(),
            configData: configData,
            safetensorsURLs: [safetensorsURL]
        )

        #expect(metadata[STAFMetadataKey.sourceFormat] == .string("safetensors"))
        #expect(metadata[STAFMetadataKey.converterVersion] == .uint32(1))
        #expect(metadata[STAFMetadataKey.metadataSchemaVersion] == .uint32(1))
        #expect(metadata[STAFMetadataKey.sourceShardCount] == .uint64(1))

        #expect(metadata[STAFMetadataKey.modelArchitectureFamily] == .string("lfm2"))
        #expect(metadata[STAFMetadataKey.modelHiddenSize] == .uint64(2048))
        #expect(metadata[STAFMetadataKey.modelLayerCount] == .uint64(24))

        #expect(metadata[STAFMetadataKey.sourceTokenizerPresent] == .bool(true))
        #expect(metadata[STAFMetadataKey.sourceTokenizerConfigPresent] == .bool(true))
        #expect(metadata[STAFMetadataKey.sourceSpecialTokensPresent] == .bool(true))
        #expect(metadata[STAFMetadataKey.sourceChatTemplatePresent] == .bool(true))
        #expect(metadata[STAFMetadataKey.sourceChatTemplateSource] == .string(STAFMetadataKey.chatTemplateJinjaSource))

        expectUInt64(metadata[STAFMetadataKey.sourceConfigHash], "config hash")
        expectUInt64(metadata[STAFMetadataKey.sourceTokenizerHash], "tokenizer hash")
        expectUInt64(metadata[STAFMetadataKey.sourceTokenizerConfigHash], "tokenizer config hash")
        expectUInt64(metadata[STAFMetadataKey.sourceSpecialTokensHash], "special tokens hash")
        expectUInt64(metadata[STAFMetadataKey.sourceChatTemplateHash], "chat template hash")
        expectUInt64(metadata[STAFMetadataKey.sourceSafetensorsManifestHash], "safetensors manifest hash")
    }

    private func makeTempDirectory() -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("staf_bundle_metadata_\(UUID().uuidString)")
        try! FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    private func cleanup(_ url: URL) {
        try? FileManager.default.removeItem(at: url)
    }

    private func makeConfig() -> ModelConfig {
        ModelConfig(
            hiddenSize: 2048,
            layerCount: 24,
            intermediateSize: 6144,
            vocabSize: 32768,
            attentionHeads: 16,
            kvHeads: 8,
            headDim: 128,
            attentionBias: false,
            mlpBias: false,
            normEps: 1e-5,
            normKind: .rmsNorm,
            ropeTheta: 10000,
            ropeDimension: 128,
            ropeScaling: nil,
            tiedEmbeddings: false,
            expertCount: nil,
            expertsPerToken: nil,
            qkNorm: false,
            fullAttentionInterval: nil,
            ssmNumHeads: nil,
            ssmKeyHeadDim: nil,
            ssmValueHeadDim: nil,
            convKernelSize: nil,
            partialRotaryFactor: nil,
            slidingWindow: nil
        )
    }

    private func expectUInt64(_ value: STAFMetadataValue?, _ label: String) {
        guard case .uint64 = value else {
            Issue.record("Expected uint64 metadata for \(label), got \(String(describing: value))")
            return
        }
    }
}
