import SwiftLM

/// Cohere Command-R architecture.
///
/// Key differences from standard transformer:
/// - Uses LayerNorm instead of RMSNorm
/// - QK normalization in attention
///
/// ```swift
/// let cohere = Cohere(config: .init(
///     hiddenSize: 8192,
///     hiddenLayers: 40,
///     intermediateSize: 22528,
///     attentionHeads: 64,
///     kvHeads: 8,
///     vocabularySize: 256000
/// ))
/// let graph = try cohere.makeModelGraph()
/// ```
public struct Cohere: ModelComponent {

    /// Architecture configuration for Cohere models.
    public struct Config: Sendable {

        // MARK: - Core Dimensions

        public let hiddenSize: Int
        public let hiddenLayers: Int
        public let intermediateSize: Int
        public let attentionHeads: Int
        public let kvHeads: Int
        public let headDimension: Int?
        public let vocabularySize: Int

        // MARK: - Normalization

        public let layerNormEps: Float

        // MARK: - RoPE

        public let ropeTheta: Float
        public let ropeScaling: RoPEScaling?

        // MARK: - Features

        public let useQKNorm: Bool

        // MARK: - Output

        public let tieWordEmbeddings: Bool

        // MARK: - Computed

        public var resolvedHeadDimension: Int {
            headDimension ?? (hiddenSize / attentionHeads)
        }

        public init(
            hiddenSize: Int,
            hiddenLayers: Int,
            intermediateSize: Int,
            attentionHeads: Int,
            kvHeads: Int,
            headDimension: Int? = nil,
            vocabularySize: Int,
            layerNormEps: Float = 1e-5,
            ropeTheta: Float = 10_000.0,
            ropeScaling: RoPEScaling? = nil,
            useQKNorm: Bool = true,
            tieWordEmbeddings: Bool = true
        ) {
            self.hiddenSize = hiddenSize
            self.hiddenLayers = hiddenLayers
            self.intermediateSize = intermediateSize
            self.attentionHeads = attentionHeads
            self.kvHeads = kvHeads
            self.headDimension = headDimension
            self.vocabularySize = vocabularySize
            self.layerNormEps = layerNormEps
            self.ropeTheta = ropeTheta
            self.ropeScaling = ropeScaling
            self.useQKNorm = useQKNorm
            self.tieWordEmbeddings = tieWordEmbeddings
        }
    }

    public let config: Config

    public init(config: Config) {
        self.config = config
    }

    @ModelComponentBuilder
    public var body: some ModelComponent {
        TokenEmbedding(vocabSize: config.vocabularySize, embeddingSize: config.hiddenSize)

        Repeat(count: config.hiddenLayers, label: "layers") {
            Residual {
                LayerNorm(dimension: config.hiddenSize, epsilon: config.layerNormEps)
                Attention(
                    hiddenSize: config.hiddenSize,
                    headCount: config.attentionHeads,
                    kvHeadCount: config.kvHeads,
                    headDimension: config.headDimension,
                    rope: RoPEAttributes(
                        dimension: config.resolvedHeadDimension,
                        base: config.ropeTheta,
                        scaling: config.ropeScaling
                    ),
                    qkNorm: config.useQKNorm ? .layerNorm : nil
                )
            }
            Residual {
                LayerNorm(dimension: config.hiddenSize, epsilon: config.layerNormEps)
                MLP(inputSize: config.hiddenSize, intermediateSize: config.intermediateSize)
            }
        }

        LayerNorm(dimension: config.hiddenSize, epsilon: config.layerNormEps)
        OutputHead(
            inputSize: config.hiddenSize,
            vocabSize: config.vocabularySize,
            tiedToEmbedding: config.tieWordEmbeddings
        )
    }
}
