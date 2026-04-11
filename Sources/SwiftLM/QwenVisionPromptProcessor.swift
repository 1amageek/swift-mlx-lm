import Foundation

struct QwenVisionPromptProcessor {
    private let configuration: ModelConfiguration

    init(configuration: ModelConfiguration) {
        self.configuration = configuration
    }

    func prepare(renderedText: String, messages: [InputMessage]) async throws -> RenderedPrompt {
        let images = messages.flatMap(\.images)
        let videos = messages.flatMap(\.videos)
        guard !images.isEmpty else {
            if videos.isEmpty {
                return RenderedPrompt(text: renderedText, multimodal: nil)
            }
            return try await prepareMultimodalPrompt(renderedText: renderedText, messages: messages)
        }
        guard canProcessImages else {
            throw LanguageModelContextError.multimodalInputNotSupported(
                "This model declares image input support, but SwiftLM only knows how to expand Qwen3.5/Qwen3-VL style image placeholders today."
            )
        }
        return try await prepareMultimodalPrompt(renderedText: renderedText, messages: messages)
    }

    private func prepareMultimodalPrompt(
        renderedText: String,
        messages: [InputMessage]
    ) async throws -> RenderedPrompt {
        let images = messages.flatMap(\.images)
        let videos = messages.flatMap(\.videos)
        guard images.isEmpty || canProcessImages else {
            throw LanguageModelContextError.multimodalInputNotSupported(
                "This model declares image input support, but SwiftLM only knows how to expand Qwen3.5/Qwen3-VL style image placeholders today."
            )
        }
        guard videos.isEmpty || canProcessVideos else {
            throw LanguageModelContextError.multimodalInputNotSupported(
                "This model declares video input support, but SwiftLM only knows how to expand Qwen3.5/Qwen3-VL style video placeholders today."
            )
        }

        let imagePlaceholderCount = renderedText.components(separatedBy: imageToken).count - 1
        if imagePlaceholderCount != images.count {
            throw LanguageModelContextError.multimodalInputNotSupported(
                imagePlaceholderCount < images.count
                    ? "The rendered chat template does not contain enough Qwen image placeholders for the supplied images."
                    : "The rendered chat template contains more Qwen image placeholders than supplied images."
            )
        }
        let videoPlaceholderCount = renderedText.components(separatedBy: videoToken).count - 1
        if videoPlaceholderCount != videos.count {
            throw LanguageModelContextError.multimodalInputNotSupported(
                videoPlaceholderCount < videos.count
                    ? "The rendered chat template does not contain enough Qwen video placeholders for the supplied videos."
                    : "The rendered chat template contains more Qwen video placeholders than supplied videos."
            )
        }

        var expandedText = renderedText
        var preparedImages: [PreparedPrompt.Multimodal.Image] = []
        var preparedVideos: [PreparedPrompt.Multimodal.Video] = []

        for message in messages {
            for item in message.content {
                switch item {
                case .text:
                    continue
                case .image(let image):
                    let preparedImage = try prepareImage(image)
                    let replacement = String(
                        repeating: imageToken,
                        count: preparedImage.placeholderTokenCount
                    )
                    guard let range = expandedText.range(of: imageToken) else {
                        throw LanguageModelContextError.multimodalInputNotSupported(
                            "The rendered chat template does not contain enough Qwen image placeholders for the supplied images."
                        )
                    }
                    expandedText.replaceSubrange(range, with: replacement)
                    preparedImages.append(preparedImage)
                case .video(let video):
                    let preparedVideo = try await prepareVideo(video)
                    let replacement = makeVideoReplacement(for: preparedVideo)
                    if let wrappedRange = expandedText.range(
                        of: "\(visionStartToken)\(videoToken)\(visionEndToken)"
                    ) {
                        expandedText.replaceSubrange(wrappedRange, with: replacement)
                    } else if let plainRange = expandedText.range(of: videoToken) {
                        expandedText.replaceSubrange(plainRange, with: replacement)
                    } else {
                        throw LanguageModelContextError.multimodalInputNotSupported(
                            "The rendered chat template does not contain enough Qwen video placeholders for the supplied videos."
                        )
                    }
                    preparedVideos.append(preparedVideo)
                }
            }
        }

        return RenderedPrompt(
            text: expandedText,
            multimodal: PreparedPrompt.Multimodal(
                images: preparedImages,
                videos: preparedVideos
            )
        )
    }

    func multimodalTokenTypes(for tokens: [Int]) -> [Int] {
        let imageTokenID = configuration.vision?.imageTokenID
        let videoTokenID = configuration.vision?.videoTokenID
        return tokens.map { token in
            if let imageTokenID, token == imageTokenID {
                return 1
            }
            if let videoTokenID, token == videoTokenID {
                return 2
            }
            return 0
        }
    }

    private func prepareImage(_ image: InputImage) throws -> PreparedPrompt.Multimodal.Image {
        let vision = configuration.vision ?? ModelVisionConfiguration()
        return try QwenVisionImagePreprocessor(configuration: vision).prepare(image)
    }

    private func prepareVideo(_ video: InputVideo) async throws -> PreparedPrompt.Multimodal.Video {
        let vision = configuration.vision ?? ModelVisionConfiguration()
        return try await QwenVisionVideoPreprocessor(configuration: vision).prepare(video)
    }

    private var canProcessImages: Bool {
        QwenVisionSupport.supportsImagePromptPreparation(vision: configuration.vision)
    }

    private var canProcessVideos: Bool {
        QwenVisionSupport.supportsVideoPromptPreparation(vision: configuration.vision)
    }

    private func makeVideoReplacement(
        for video: PreparedPrompt.Multimodal.Video
    ) -> String {
        let frameTokenCount = video.placeholderTokenCount / max(1, video.gridTHW.first ?? 1)
        return video.frameTimestamps.map { timestamp in
            let formatted = String(format: "<%.1f seconds>", timestamp)
            return formatted
                + visionStartToken
                + String(repeating: videoToken, count: frameTokenCount)
                + visionEndToken
        }
        .joined()
    }

    private var visionStartToken: String { "<|vision_start|>" }
    private var visionEndToken: String { "<|vision_end|>" }
    private var imageToken: String { "<|image_pad|>" }
    private var videoToken: String { "<|video_pad|>" }
}

private extension InputMessage {
    var images: [InputImage] {
        content.compactMap { item in
            if case .image(let image) = item {
                return image
            }
            return nil
        }
    }

    var videos: [InputVideo] {
        content.compactMap { item in
            if case .video(let video) = item {
                return video
            }
            return nil
        }
    }
}
