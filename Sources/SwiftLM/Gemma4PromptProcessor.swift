import Foundation

struct Gemma4PromptProcessor {
    private let configuration: ModelConfiguration

    init(configuration: ModelConfiguration) {
        self.configuration = configuration
    }

    func prepare(renderedText: String, messages: [InputMessage]) async throws -> RenderedPrompt {
        let images = messages.flatMap(\.images)
        guard !images.isEmpty else {
            return RenderedPrompt(text: renderedText, multimodal: nil)
        }
        guard Gemma4Support.supportsImagePromptPreparation(vision: configuration.vision) else {
            throw InferenceSessionError.multimodalInputNotSupported(
                "This model declares image input support, but SwiftLM does not have an active Gemma4 image processor."
            )
        }

        let imagePlaceholderCount = renderedText.components(separatedBy: imageToken).count - 1
        if imagePlaceholderCount != images.count {
            throw InferenceSessionError.multimodalInputNotSupported(
                imagePlaceholderCount < images.count
                    ? "The rendered chat template does not contain enough Gemma4 image placeholders for the supplied images."
                    : "The rendered chat template contains more Gemma4 image placeholders than supplied images."
            )
        }

        var expandedText = renderedText
        var preparedImages: [PreparedPrompt.Multimodal.Image] = []
        let preprocessor = Gemma4ImagePreprocessor(configuration: configuration.vision ?? .init())
        for message in messages {
            for item in message.content {
                switch item {
                case .text, .video:
                    continue
                case .image(let image):
                    let preparedImage = try preprocessor.prepare(image)
                    let replacement =
                        boiToken
                        + String(repeating: imageToken, count: preparedImage.placeholderTokenCount)
                        + eoiToken
                    guard let range = expandedText.range(of: imageToken) else {
                        throw InferenceSessionError.multimodalInputNotSupported(
                            "The rendered chat template does not contain enough Gemma4 image placeholders for the supplied images."
                        )
                    }
                    expandedText.replaceSubrange(range, with: replacement)
                    preparedImages.append(preparedImage)
                }
            }
        }

        return RenderedPrompt(
            text: expandedText,
            multimodal: PreparedPrompt.Multimodal(
                images: preparedImages,
                videos: []
            )
        )
    }

    func multimodalTokenTypes(for tokens: [Int]) -> [Int] {
        let imageTokenID = configuration.vision?.imageTokenID
        return tokens.map { token in
            if let imageTokenID, token == imageTokenID {
                return 1
            }
            return 0
        }
    }

    private var imageToken: String { "<|image|>" }
    private var boiToken: String { "<|image>" }
    private var eoiToken: String { "<image|>" }
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
}
