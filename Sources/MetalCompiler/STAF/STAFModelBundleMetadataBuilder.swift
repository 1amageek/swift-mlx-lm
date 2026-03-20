import Foundation
import LMIR

public struct STAFModelBundleMetadataBuilder: Sendable {
    public init() {}

    public func build(
        directory: URL,
        modelType: String,
        config: ModelConfig,
        configData: Data,
        safetensorsURLs: [URL]
    ) throws -> STAFFileMetadata {
        let tokenizerURL = directory.appendingPathComponent("tokenizer.json")
        let tokenizerConfigURL = directory.appendingPathComponent("tokenizer_config.json")
        let specialTokensURL = directory.appendingPathComponent("special_tokens_map.json")
        let chatTemplateURL = directory.appendingPathComponent("chat_template.jinja")

        let tokenizerData = try loadOptionalData(at: tokenizerURL)
        let tokenizerConfigData = try loadOptionalData(at: tokenizerConfigURL)
        let specialTokensData = try loadOptionalData(at: specialTokensURL)
        let chatTemplate = try resolveChatTemplate(
            chatTemplateURL: chatTemplateURL,
            tokenizerConfigData: tokenizerConfigData
        )

        var values: [String: STAFMetadataValue] = [
            STAFMetadataKey.modelArchitectureFamily: .string(modelType),
            STAFMetadataKey.modelHiddenSize: .uint64(UInt64(config.hiddenSize)),
            STAFMetadataKey.modelLayerCount: .uint64(UInt64(config.layerCount)),
            STAFMetadataKey.modelIntermediateSize: .uint64(UInt64(config.intermediateSize)),
            STAFMetadataKey.modelVocabSize: .uint64(UInt64(config.vocabSize)),
            STAFMetadataKey.modelAttentionHeads: .uint64(UInt64(config.attentionHeads)),
            STAFMetadataKey.modelKVHeads: .uint64(UInt64(config.kvHeads)),
            STAFMetadataKey.modelHeadDimension: .uint64(UInt64(config.headDim)),
            STAFMetadataKey.modelTiedEmbeddings: .bool(config.tiedEmbeddings),
            STAFMetadataKey.modelRopeDimension: .uint64(UInt64(config.ropeDimension)),
            STAFMetadataKey.modelRopeTheta: .float64(Double(config.ropeTheta)),

            STAFMetadataKey.sourceTokenizerPresent: .bool(tokenizerData != nil),
            STAFMetadataKey.sourceTokenizerConfigPresent: .bool(tokenizerConfigData != nil),
            STAFMetadataKey.sourceSpecialTokensPresent: .bool(specialTokensData != nil),
            STAFMetadataKey.sourceChatTemplatePresent: .bool(chatTemplate != nil)
        ]

        values[STAFMetadataKey.sourceConfigHash] = .uint64(hash(data: configData))
        values[STAFMetadataKey.sourceSafetensorsManifestHash] = .uint64(
            try hashSafetensorsManifest(urls: safetensorsURLs)
        )

        if let tokenizerData {
            values[STAFMetadataKey.sourceTokenizerHash] = .uint64(hash(data: tokenizerData))
        }
        if let tokenizerConfigData {
            values[STAFMetadataKey.sourceTokenizerConfigHash] = .uint64(hash(data: tokenizerConfigData))
        }
        if let specialTokensData {
            values[STAFMetadataKey.sourceSpecialTokensHash] = .uint64(hash(data: specialTokensData))
        }
        if let chatTemplate {
            values[STAFMetadataKey.sourceChatTemplateHash] = .uint64(hash(data: chatTemplate.contents))
            values[STAFMetadataKey.sourceChatTemplateSource] = .string(chatTemplate.source)
        }

        let bundleMetadata = STAFFileMetadata(values: values)
        return STAFFileMetadata.defaultCacheMetadata(sourceShardCount: safetensorsURLs.count)
            .merged(with: bundleMetadata)
    }

    private func loadOptionalData(at url: URL) throws -> Data? {
        guard FileManager.default.fileExists(atPath: url.path) else {
            return nil
        }
        return try Data(contentsOf: url)
    }

    private func resolveChatTemplate(
        chatTemplateURL: URL,
        tokenizerConfigData: Data?
    ) throws -> (contents: Data, source: String)? {
        if FileManager.default.fileExists(atPath: chatTemplateURL.path) {
            return (
                contents: try Data(contentsOf: chatTemplateURL),
                source: STAFMetadataKey.chatTemplateJinjaSource
            )
        }

        guard let tokenizerConfigData else {
            return nil
        }
        guard let jsonObject = try JSONSerialization.jsonObject(with: tokenizerConfigData) as? [String: Any],
              let templateString = jsonObject["chat_template"] as? String else {
            return nil
        }
        return (
            contents: Data(templateString.utf8),
            source: STAFMetadataKey.tokenizerConfigSource
        )
    }

    private func hash(data: Data) -> UInt64 {
        var hash = STAFStableHash64()
        hash.update(data)
        return hash.finalize()
    }

    private func hashSafetensorsManifest(urls: [URL]) throws -> UInt64 {
        var hash = STAFStableHash64()
        for url in urls.sorted(by: { $0.lastPathComponent < $1.lastPathComponent }) {
            let attributes = try FileManager.default.attributesOfItem(atPath: url.path)
            let size = attributes[.size] as? UInt64 ?? 0
            let modificationDate = attributes[.modificationDate] as? Date
            let modifiedAtNanoseconds: UInt64
            if let modificationDate {
                modifiedAtNanoseconds = UInt64(
                    max(0, modificationDate.timeIntervalSince1970 * 1_000_000_000)
                )
            } else {
                modifiedAtNanoseconds = 0
            }

            hash.update(url.lastPathComponent)
            hash.update(uint64: size)
            hash.update(uint64: modifiedAtNanoseconds)
        }
        return hash.finalize()
    }
}
