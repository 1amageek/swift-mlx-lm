import Foundation
import MetalCompiler
import ModelDeclarations
import Tokenizers

final class Gemma4Runtime {
    private let padTokenID: Int
    private let textRuntime: Gemma4TextRuntime
    private let visionEncoder: Gemma4VisionEncoder?

    init(
        padTokenID: Int,
        textRuntime: Gemma4TextRuntime,
        visionEncoder: Gemma4VisionEncoder?
    ) {
        self.padTokenID = padTokenID
        self.textRuntime = textRuntime
        self.visionEncoder = visionEncoder
    }

    static func makeIfSupported(
        resources: ModelBundleResources,
        tokenizer: any Tokenizer,
        weights: STAFWeightStore
    ) throws -> Gemma4Runtime? {
        let modelType = resources.modelType.lowercased()
        guard modelType == "gemma4" || modelType == "gemma4_text" else {
            return nil
        }
        try Gemma4.validate(resources.config)
        let weightStore = Gemma4WeightStore(weights: weights)
        let textRuntime = try Gemma4TextRuntime(config: resources.config, weights: weightStore)
        let padTokenID = try resolvePadTokenID(
            configData: resources.configData,
            tokenizer: tokenizer
        )
        let visionEncoder: Gemma4VisionEncoder?
        if Gemma4Support.supportsImagePromptPreparation(vision: resources.visionConfiguration),
           let visionConfig = resources.visionConfiguration {
            visionEncoder = try Gemma4VisionEncoder(
                configuration: visionConfig,
                textHiddenSize: resources.config.hiddenSize,
                weights: weightStore
            )
        } else {
            visionEncoder = nil
        }
        return Gemma4Runtime(
            padTokenID: padTokenID,
            textRuntime: textRuntime,
            visionEncoder: visionEncoder
        )
    }

    func makePromptContext(from prepared: PreparedPrompt) throws -> Gemma4PromptContext {
        var promptEmbeddings = try textRuntime.tokenEmbeddings(tokenIDs: prepared.tokenIDs)
        var perLayerTokenIDs = prepared.tokenIDs
        var usesEmbeddingOverrides = false

        if let multimodal = prepared.multimodalMetadata {
            if !multimodal.videos.isEmpty {
                throw InferenceSessionError.multimodalInputNotSupported(
                    "Gemma4 video execution is not implemented yet."
                )
            }
            if !multimodal.images.isEmpty {
                guard let visionEncoder else {
                    throw InferenceSessionError.multimodalInputNotSupported(
                        "This Gemma4 bundle does not have an active vision encoder."
                    )
                }
                let imageEmbeddings = try visionEncoder.encode(images: multimodal.images)
                let imageIndices = multimodal.mmTokenTypeIDs.enumerated().compactMap { index, type in
                    type == 1 ? index : nil
                }
                guard imageIndices.count == imageEmbeddings.count else {
                    throw InferenceSessionError.multimodalInputNotSupported(
                        "Gemma4 image soft token count does not match the encoded image feature count."
                    )
                }
                for (index, embedding) in zip(imageIndices, imageEmbeddings) {
                    promptEmbeddings[index] = embedding
                    perLayerTokenIDs[index] = padTokenID
                    usesEmbeddingOverrides = true
                }
            }
        }

        return Gemma4PromptContext(
            promptEmbeddings: promptEmbeddings,
            perLayerInputs: try textRuntime.buildPrefillPerLayerInputs(
                tokenIDs: perLayerTokenIDs,
                promptEmbeddings: promptEmbeddings
            ),
            usesEmbeddingOverrides: usesEmbeddingOverrides
        )
    }

    func buildDecodePerLayerInputs(tokenID: Int) throws -> [[Float]] {
        try textRuntime.buildDecodePerLayerInputs(tokenID: tokenID)
    }

    private static func resolvePadTokenID(
        configData: Data,
        tokenizer: any Tokenizer
    ) throws -> Int {
        if let explicit = tokenizer.convertTokenToId("<pad>") {
            return explicit
        }
        guard let raw = try JSONSerialization.jsonObject(with: configData) as? [String: Any] else {
            throw ModelBundleLoaderError.invalidConfig("Gemma4 config.json is not a JSON object")
        }
        if let textConfig = raw["text_config"] as? [String: Any],
           let padTokenID = textConfig["pad_token_id"] as? Int {
            return padTokenID
        }
        if let padTokenID = raw["pad_token_id"] as? Int {
            return padTokenID
        }
        return 0
    }
}
