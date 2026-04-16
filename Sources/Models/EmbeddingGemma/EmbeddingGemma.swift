import LMArchitecture

/// EmbeddingGemma: bidirectional text embedding model based on Gemma 3 backbone.
///
/// Uses the same architecture as Gemma 3 text (sandwich norms, mixed RoPE schedule)
/// but with bidirectional attention (non-causal) and no output head.
/// The model produces dense hidden states that are pooled and projected by the
/// sentence-transformer runtime into normalized embeddings.
///
/// Supported HuggingFace repos:
/// - `google/embeddinggemma-300m` (official FP32)
/// - `mlx-community/embeddinggemma-300m-bf16`
/// - `mlx-community/embeddinggemma-300m-4bit`
public struct EmbeddingGemma: ModelComponent {

    public let config: ModelConfig

    public init(config: ModelConfig) throws {
        guard config.useBidirectionalAttention else {
            throw ModelGraphBuildError.invalidConfig(
                "EmbeddingGemma requires use_bidirectional_attention=true in config.json"
            )
        }
        try Gemma3Text.validate(config)
        self.config = config
    }

    @ModelComponentBuilder
    public var body: some ModelComponent {
        Gemma3TextBackbone(validatedConfig: config)
    }
}
