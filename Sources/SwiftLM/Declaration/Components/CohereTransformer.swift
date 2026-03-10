/// Cohere-style transformer with parallel attention + MLP.
///
/// Each layer runs Attention and MLP in parallel from a shared LayerNorm,
/// then adds both results to the residual. This is unlike the standard
/// sequential residual structure.
///
/// Architecture: LayerNorm -> Parallel(Attention, MLP) -> Add
///
/// ```swift
/// let cohere = CohereTransformer(config: .init(
///     hiddenSize: 4096,
///     hiddenLayers: 32,
///     intermediateSize: 14336,
///     attentionHeads: 32,
///     kvHeads: 8,
///     vocabularySize: 256000,
///     useQKNorm: true
/// ))
/// ```
public struct CohereTransformer: ModelComponent {

    /// Configuration for Cohere-style transformers.
    public struct Config: Sendable {

        public let hiddenSize: Int
        public let hiddenLayers: Int
        public let intermediateSize: Int
        public let attentionHeads: Int
        public let kvHeads: Int
        public let headDimension: Int?
        public let vocabularySize: Int
        public let normEps: Float
        public let activation: ActivationKind
        public let gating: GatingKind
        public let attentionBias: Bool
        public let mlpBias: Bool
        public let ropeTheta: Float
        public let ropeScaling: RoPEScaling?
        public let tieWordEmbeddings: Bool
        public let useQKNorm: Bool

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
            normEps: Float = 1e-5,
            activation: ActivationKind = .silu,
            gating: GatingKind = .swiglu,
            attentionBias: Bool = false,
            mlpBias: Bool = false,
            ropeTheta: Float = 10_000.0,
            ropeScaling: RoPEScaling? = nil,
            tieWordEmbeddings: Bool = true,
            useQKNorm: Bool = false
        ) {
            self.hiddenSize = hiddenSize
            self.hiddenLayers = hiddenLayers
            self.intermediateSize = intermediateSize
            self.attentionHeads = attentionHeads
            self.kvHeads = kvHeads
            self.headDimension = headDimension
            self.vocabularySize = vocabularySize
            self.normEps = normEps
            self.activation = activation
            self.gating = gating
            self.attentionBias = attentionBias
            self.mlpBias = mlpBias
            self.ropeTheta = ropeTheta
            self.ropeScaling = ropeScaling
            self.tieWordEmbeddings = tieWordEmbeddings
            self.useQKNorm = useQKNorm
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
            // Cohere uses a single LayerNorm shared across both branches
            // with parallel Attention + MLP, added to residual.
            Residual {
                LayerNorm(dimension: config.hiddenSize, epsilon: config.normEps)
                Parallel(merge: .add) {
                    Attention(
                        hiddenSize: config.hiddenSize,
                        headCount: config.attentionHeads,
                        kvHeadCount: config.kvHeads,
                        headDimension: config.headDimension,
                        bias: config.attentionBias,
                        rope: RoPEAttributes(
                            dimension: config.resolvedHeadDimension,
                            base: config.ropeTheta,
                            scaling: config.ropeScaling
                        ),
                        qkNorm: config.useQKNorm ? .layerNorm : nil
                    )
                    MLP(
                        inputSize: config.hiddenSize,
                        intermediateSize: config.intermediateSize,
                        activation: config.activation,
                        gating: config.gating,
                        bias: config.mlpBias
                    )
                }
            }
        }

        LayerNorm(dimension: config.hiddenSize, epsilon: config.normEps)
        OutputHead(
            inputSize: config.hiddenSize,
            vocabSize: config.vocabularySize,
            tiedToEmbedding: config.tieWordEmbeddings
        )
    }
}
