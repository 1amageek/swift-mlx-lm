import CoreImage
import MLX

/// Vision-aware input processor for VLMs.
///
/// Extends the standard text processing with image preprocessing:
/// inserts vision placeholder tokens and builds ``LMInput/ProcessedImage``
/// for the vision encoder. All token IDs come from ``VLMInputConfig``.
struct VLMUserInputProcessor: UserInputProcessor {

    /// Closure that preprocesses a single image into pixel tensor + grid info.
    typealias ImagePreprocessor = @Sendable (CIImage) throws -> (MLXArray, LMInput.THW)

    private let textProcessor: ChatTemplateInputProcessor
    private let preprocessImage: ImagePreprocessor
    private let tokenizer: any Tokenizer

    /// Token IDs for vision placeholder markup.
    private let visionStartTokenId: Int
    private let visionEndTokenId: Int
    private let imagePadTokenId: Int
    private let videoPadTokenId: Int
    private let spatialMergeSize: Int

    /// Create a VLM input processor from any ``VLMInputConfig`` and image preprocessor.
    init(
        tokenizer: any Tokenizer,
        chatTemplate: String?,
        bosToken: String?,
        eosToken: String?,
        addBosToken: Bool,
        vlmInputConfig: any VLMInputConfig,
        preprocessImage: @escaping ImagePreprocessor
    ) {
        self.tokenizer = tokenizer
        self.preprocessImage = preprocessImage

        self.textProcessor = ChatTemplateInputProcessor(
            tokenizer: tokenizer,
            chatTemplate: chatTemplate,
            bosToken: bosToken,
            eosToken: eosToken,
            addBosToken: addBosToken
        )

        self.visionStartTokenId = vlmInputConfig.visionStartTokenId
        self.visionEndTokenId = vlmInputConfig.visionEndTokenId
        self.imagePadTokenId = vlmInputConfig.imageTokenId
        self.videoPadTokenId = vlmInputConfig.videoTokenId
        self.spatialMergeSize = vlmInputConfig.spatialMergeSize
    }

    func prepare(input: UserInput) async throws -> LMInput {
        let images = input.images

        // No images — delegate to text-only processor
        if images.isEmpty {
            return try await textProcessor.prepare(input: input)
        }

        // Preprocess images
        var allPixels = [MLXArray]()
        var allTHW = [LMInput.THW]()

        for inputImage in images {
            let ciImage = try inputImage.asCIImage()
            let (pixels, thw) = try preprocessImage(ciImage)
            allPixels.append(pixels)
            allTHW.append(thw)
        }

        // Tokenize text with vision placeholders
        let textLMInput = try await textProcessor.prepare(input: input)

        // Replace <|image_pad|> counts to match actual vision token counts
        let adjustedTokens = adjustImagePadTokens(
            tokens: textLMInput.text.tokens,
            grids: allTHW
        )

        let combinedPixels = concatenated(allPixels, axis: 0)
        let processedImage = LMInput.ProcessedImage(
            pixels: combinedPixels,
            frames: allTHW
        )

        return LMInput(
            text: LMInput.Text(tokens: adjustedTokens),
            image: processedImage
        )
    }

    // MARK: - Private

    /// Adjust image pad token counts to match the actual number of vision tokens
    /// produced by the vision encoder (after spatial merge).
    private func adjustImagePadTokens(
        tokens: MLXArray,
        grids: [LMInput.THW]
    ) -> MLXArray {
        let spatialMerge = spatialMergeSize
        var tokenList: [Int32] = []
        let flatTokens: [Int32] = (0..<tokens.dim(tokens.ndim - 1)).map { i in
            tokens[0, i].item()
        }

        var gridIdx = 0
        var i = 0

        while i < flatTokens.count {
            let token = flatTokens[i]

            if token == Int32(visionStartTokenId) && gridIdx < grids.count {
                // Found vision_start — emit it
                tokenList.append(token)
                i += 1

                // Compute how many pad tokens this image needs
                let grid = grids[gridIdx]
                let mergedH = grid.h / spatialMerge
                let mergedW = grid.w / spatialMerge
                let numPadTokens = grid.t * mergedH * mergedW

                // Emit the correct number of image_pad tokens
                for _ in 0..<numPadTokens {
                    tokenList.append(Int32(imagePadTokenId))
                }

                // Skip existing image_pad or video_pad tokens in the original sequence
                while i < flatTokens.count
                    && (flatTokens[i] == Int32(imagePadTokenId) || flatTokens[i] == Int32(videoPadTokenId))
                {
                    i += 1
                }

                // Emit vision_end if present
                if i < flatTokens.count && flatTokens[i] == Int32(visionEndTokenId) {
                    tokenList.append(flatTokens[i])
                    i += 1
                }

                gridIdx += 1
            } else {
                tokenList.append(token)
                i += 1
            }
        }

        return MLXArray(tokenList).reshaped(1, tokenList.count)
    }
}
