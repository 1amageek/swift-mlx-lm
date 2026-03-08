import CoreImage
import MLX
import GGUFTokenizer

/// Vision-aware input processor for Qwen2.5-VL.
///
/// Extends the standard GGUF text processing with image preprocessing:
/// inserts `<|vision_start|><|image_pad|>...<|vision_end|>` tokens and
/// builds ``LMInput/ProcessedImage`` for the vision encoder.
struct VLMUserInputProcessor: UserInputProcessor {

    private let textProcessor: GGUFUserInputProcessor
    private let imageProcessor: Qwen25VLImageProcessor
    private let tokenizer: any Tokenizer

    /// Token IDs for vision placeholder markup.
    private let visionStartTokenId: Int
    private let visionEndTokenId: Int
    private let imagePadTokenId: Int

    init(
        tokenizer: any Tokenizer,
        chatTemplate: String?,
        bosToken: String?,
        eosToken: String?,
        addBosToken: Bool,
        visionConfig: Qwen25VLConfiguration.VisionConfiguration
    ) {
        self.tokenizer = tokenizer
        self.imageProcessor = Qwen25VLImageProcessor(config: visionConfig)

        self.textProcessor = GGUFUserInputProcessor(
            tokenizer: tokenizer,
            chatTemplate: chatTemplate,
            bosToken: bosToken,
            eosToken: eosToken,
            addBosToken: addBosToken
        )

        // Qwen2.5-VL special tokens
        // <|vision_start|> = 151652, <|vision_end|> = 151653, <|image_pad|> = 151655
        self.visionStartTokenId = 151652
        self.visionEndTokenId = 151653
        self.imagePadTokenId = 151655
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
            let (pixels, thw) = try imageProcessor.preprocess(image: ciImage)
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
        let spatialMerge = imageProcessor.imageFactor / 14  // spatialMergeSize
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

                // Skip existing image_pad tokens in the original sequence
                while i < flatTokens.count && flatTokens[i] == Int32(imagePadTokenId) {
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
