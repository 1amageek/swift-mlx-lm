import MLX
import MLXNN

/// Configuration interface for VLM input processors.
///
/// Provides the token IDs and spatial merge information needed to
/// process vision placeholders in the text token sequence.
protocol VLMInputConfig {
    var visionStartTokenId: Int { get }
    var visionEndTokenId: Int { get }
    var imageTokenId: Int { get }
    var videoTokenId: Int { get }
    var spatialMergeSize: Int { get }
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
