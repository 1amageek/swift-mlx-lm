import Metal
import MetalCompiler
import Foundation
import Darwin

final class QwenVisionRuntime {
    private let encoder: QwenVisionEncoder

    init(encoder: QwenVisionEncoder) {
        self.encoder = encoder
    }

    static func makeIfSupported(
        resources: ModelBundleResources,
        device: MTLDevice
    ) throws -> QwenVisionRuntime? {
        guard let vision = resources.visionConfiguration else {
            return nil
        }
        let supportsImages = QwenVisionSupport.supportsImagePromptPreparation(vision: vision)
        let supportsVideos = QwenVisionSupport.supportsVideoPromptPreparation(vision: vision)
        guard supportsImages || supportsVideos else {
            return nil
        }

        let weights = try SafetensorsLoader().loadAll(urls: resources.safetensorsURLs, device: device)
        let visionWeights = QwenVisionWeightStore(weights: weights)
        let encoder = try QwenVisionEncoder(configuration: vision, weights: visionWeights)
        return QwenVisionRuntime(encoder: encoder)
    }

    func makeVisualContext(from prepared: PreparedPrompt) throws -> VisualContext {
        guard let multimodal = prepared.multimodalMetadata else {
            throw LanguageModelContextError.unsupportedInputForModel(
                "Missing multimodal prompt metadata."
            )
        }

        let layout = try QwenVisionExecutionLayoutBuilder().makeLayout(for: prepared)
        let encodedImages = try encoder.encode(images: multimodal.images)
        let encodedVideos = try encoder.encode(videos: multimodal.videos)

        let visualTokenCount = layout.layout.tokenTypeIDs.filter { $0 == 1 }.count
        guard visualTokenCount == encodedImages.visualTokenEmbeddings.count else {
            throw LanguageModelContextError.multimodalInputNotSupported(
                "Vision encoder output count does not match image placeholder count."
            )
        }
        for (layerIndex, features) in encodedImages.deepstackFeaturesByLayer {
            guard features.count == visualTokenCount else {
                throw LanguageModelContextError.multimodalInputNotSupported(
                    "Deepstack feature count mismatch at visual layer \(layerIndex)."
                )
            }
        }
        let videoTokenCount = layout.layout.tokenTypeIDs.filter { $0 == 2 }.count
        guard videoTokenCount == encodedVideos.visualTokenEmbeddings.count else {
            throw LanguageModelContextError.multimodalInputNotSupported(
                "Vision encoder output count does not match video placeholder count."
            )
        }
        for (layerIndex, features) in encodedVideos.deepstackFeaturesByLayer {
            guard features.count == videoTokenCount else {
                throw LanguageModelContextError.multimodalInputNotSupported(
                    "Deepstack feature count mismatch at video layer \(layerIndex)."
                )
            }
        }

        return VisualContext(
            layout: layout.layout,
            imageTokenEmbeddings: encodedImages.visualTokenEmbeddings,
            imageDeepstackFeaturesByLayer: encodedImages.deepstackFeaturesByLayer,
            videoTokenEmbeddings: encodedVideos.visualTokenEmbeddings,
            videoDeepstackFeaturesByLayer: encodedVideos.deepstackFeaturesByLayer
        )
    }
}
