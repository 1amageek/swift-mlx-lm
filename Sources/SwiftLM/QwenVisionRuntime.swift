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

    func makeVisualContext(from prepared: PreparedInput) throws -> VisualContext {
        guard let multimodal = prepared.multimodalMetadata else {
            throw ModelContainerError.unsupportedInputForModel(
                "Missing multimodal prompt metadata."
            )
        }

        let shouldProfile = getenv("SWIFTLM_PROFILE_MULTIMODAL").map { String(cString: $0) == "1" } ?? false
        let totalStart = shouldProfile ? CFAbsoluteTimeGetCurrent() : 0

        let layoutStart = shouldProfile ? CFAbsoluteTimeGetCurrent() : 0
        let layout = try QwenVisionExecutionLayoutBuilder().makeLayout(for: prepared)
        if shouldProfile {
            let elapsed = CFAbsoluteTimeGetCurrent() - layoutStart
            print("[QwenVisionRuntime] layout=\(String(format: "%.3f", elapsed))s tokens=\(prepared.tokenIDs.count)")
        }

        let imagesStart = shouldProfile ? CFAbsoluteTimeGetCurrent() : 0
        let encodedImages = try encoder.encode(images: multimodal.images)
        if shouldProfile {
            let elapsed = CFAbsoluteTimeGetCurrent() - imagesStart
            print("[QwenVisionRuntime] encode-images=\(String(format: "%.3f", elapsed))s samples=\(multimodal.images.count) tokens=\(encodedImages.visualTokenEmbeddings.count)")
        }

        let videosStart = shouldProfile ? CFAbsoluteTimeGetCurrent() : 0
        let encodedVideos = try encoder.encode(videos: multimodal.videos)
        if shouldProfile {
            let elapsed = CFAbsoluteTimeGetCurrent() - videosStart
            print("[QwenVisionRuntime] encode-videos=\(String(format: "%.3f", elapsed))s samples=\(multimodal.videos.count) tokens=\(encodedVideos.visualTokenEmbeddings.count)")
        }

        let visualTokenCount = layout.layout.tokenTypeIDs.filter { $0 == 1 }.count
        guard visualTokenCount == encodedImages.visualTokenEmbeddings.count else {
            throw ModelContainerError.multimodalInputNotSupported(
                "Vision encoder output count does not match image placeholder count."
            )
        }
        for (layerIndex, features) in encodedImages.deepstackFeaturesByLayer {
            guard features.count == visualTokenCount else {
                throw ModelContainerError.multimodalInputNotSupported(
                    "Deepstack feature count mismatch at visual layer \(layerIndex)."
                )
            }
        }
        let videoTokenCount = layout.layout.tokenTypeIDs.filter { $0 == 2 }.count
        guard videoTokenCount == encodedVideos.visualTokenEmbeddings.count else {
            throw ModelContainerError.multimodalInputNotSupported(
                "Vision encoder output count does not match video placeholder count."
            )
        }
        for (layerIndex, features) in encodedVideos.deepstackFeaturesByLayer {
            guard features.count == videoTokenCount else {
                throw ModelContainerError.multimodalInputNotSupported(
                    "Deepstack feature count mismatch at video layer \(layerIndex)."
                )
            }
        }

        if shouldProfile {
            let elapsed = CFAbsoluteTimeGetCurrent() - totalStart
            print("[QwenVisionRuntime] executable-total=\(String(format: "%.3f", elapsed))s")
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
