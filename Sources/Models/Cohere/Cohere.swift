import LMArchitecture

/// Cohere Command-R architecture.
///
/// Key differences from standard transformer:
/// - Uses LayerNorm instead of RMSNorm
/// - QK normalization in attention
///
/// Accepts `ModelConfig` directly. Norm kind and QK norm are read from config fields.
public struct Cohere: ModelComponent {

    public let config: ModelConfig

    public init(config: ModelConfig) {
        self.config = config
    }

    @ModelComponentBuilder
    public var body: some ModelComponent {
        TokenEmbedding(vocabSize: config.vocabSize, embeddingSize: config.hiddenSize)

        Repeat(count: config.layerCount, label: "layers") {
            Residual {
                LayerNorm(dimension: config.hiddenSize, epsilon: config.normEps)
                Attention(
                    hiddenSize: config.hiddenSize,
                    headCount: config.attentionHeads,
                    kvHeadCount: config.kvHeads,
                    headDimension: config.headDim,
                    rope: RoPEAttributes(
                        dimension: config.ropeDimension,
                        base: config.ropeTheta,
                        scaling: config.ropeScaling
                    ),
                    qkNorm: config.qkNorm ? .layerNorm : nil
                )
            }
            Residual {
                LayerNorm(dimension: config.hiddenSize, epsilon: config.normEps)
                MLP(inputSize: config.hiddenSize, intermediateSize: config.intermediateSize)
            }
        }

        LayerNorm(dimension: config.hiddenSize, epsilon: config.normEps)
        OutputHead(
            inputSize: config.hiddenSize,
            vocabSize: config.vocabSize,
            tiedToEmbedding: config.tiedEmbeddings
        )
    }
}
