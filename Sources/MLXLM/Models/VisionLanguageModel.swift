import MLX
import MLXNN

/// A language model that can process visual input alongside text.
///
/// VLMs encode images/videos through a vision encoder and merge the resulting
/// embeddings into the text sequence before autoregressive generation.
/// Conforms to ``LanguageModel`` so the generation pipeline works unchanged —
/// vision encoding happens inside the ``LanguageModel/prepare(_:cache:windowSize:)``
/// override.
public protocol VisionLanguageModel: LanguageModel {

    /// Encode visual input into embeddings aligned with the LLM's hidden dimension.
    ///
    /// Called during ``prepare(_:cache:windowSize:)`` on the first prefill pass.
    /// The returned embeddings are scattered into the text sequence at placeholder
    /// token positions (``imageTokenId`` / ``videoTokenId``).
    ///
    /// - Parameters:
    ///   - image: Preprocessed image data with pixel values and spatial grid info.
    ///   - video: Preprocessed video data with pixel values and temporal grid info.
    /// - Returns: Vision embeddings of shape `[totalTokens, hiddenSize]`, or `nil` if no visual input.
    func encodeVision(
        image: LMInput.ProcessedImage?,
        video: LMInput.ProcessedVideo?
    ) throws -> MLXArray?

    /// Token ID used as placeholder for image content in the text sequence.
    var imageTokenId: Int { get }

    /// Token ID used as placeholder for video content in the text sequence.
    var videoTokenId: Int { get }
}

/// A vision encoder that transforms pixel data into feature embeddings.
///
/// Implementations include ViT variants (CLIP, SigLIP, custom ViTs)
/// paired with optional spatial merging and projection layers.
protocol VisionEncoder: Module {

    /// Encode pixel data into feature embeddings.
    ///
    /// - Parameters:
    ///   - pixels: Pixel tensor (layout is model-specific, e.g. `[N, C, T, H, W]`).
    ///   - gridTHW: Per-image/video grid dimensions for position encoding.
    /// - Returns: Feature embeddings of shape `[totalTokens, outputDim]`.
    func callAsFunction(_ pixels: MLXArray, gridTHW: [LMInput.THW]) -> MLXArray
}
