/// Hybrid DeltaNet / full-attention transformer.
///
/// Alternates between gated DeltaNet layers and full attention layers.
/// The architecture family requires its routing and DeltaNet dimensions
/// to be provided explicitly by the caller.
public struct HybridDeltaNetAttentionTransformer: ModelComponent {

    public struct Config: Sendable {

        public let hiddenSize: Int
        public let hiddenLayers: Int
        public let intermediateSize: Int
        public let vocabularySize: Int
        public let normEps: Float

        public let attentionHeads: Int
        public let kvHeads: Int
        public let headDimension: Int
        public let ropeTheta: Float
        public let ropeScaling: RoPEScaling?
        public let partialRotaryFactor: Float

        public let linearKeyHeads: Int
        public let linearValueHeads: Int
        public let linearKeyHeadDim: Int
        public let linearValueHeadDim: Int
        public let convKernelSize: Int

        public let fullAttentionInterval: Int

        public let tieWordEmbeddings: Bool

        public let activation: ActivationKind
        public let gating: GatingKind

        public var ropePartialDim: Int {
            Int(Float(headDimension) * partialRotaryFactor)
        }

        public func isFullAttentionLayer(_ index: Int) -> Bool {
            (index + 1) % fullAttentionInterval == 0
        }

        public init(
            hiddenSize: Int,
            hiddenLayers: Int,
            intermediateSize: Int,
            vocabularySize: Int,
            normEps: Float,
            attentionHeads: Int,
            kvHeads: Int,
            headDimension: Int,
            ropeTheta: Float,
            ropeScaling: RoPEScaling?,
            partialRotaryFactor: Float,
            linearKeyHeads: Int,
            linearValueHeads: Int,
            linearKeyHeadDim: Int,
            linearValueHeadDim: Int,
            convKernelSize: Int,
            fullAttentionInterval: Int,
            tieWordEmbeddings: Bool,
            activation: ActivationKind,
            gating: GatingKind
        ) {
            self.hiddenSize = hiddenSize
            self.hiddenLayers = hiddenLayers
            self.intermediateSize = intermediateSize
            self.vocabularySize = vocabularySize
            self.normEps = normEps
            self.attentionHeads = attentionHeads
            self.kvHeads = kvHeads
            self.headDimension = headDimension
            self.ropeTheta = ropeTheta
            self.ropeScaling = ropeScaling
            self.partialRotaryFactor = partialRotaryFactor
            self.linearKeyHeads = linearKeyHeads
            self.linearValueHeads = linearValueHeads
            self.linearKeyHeadDim = linearKeyHeadDim
            self.linearValueHeadDim = linearValueHeadDim
            self.convKernelSize = convKernelSize
            self.fullAttentionInterval = fullAttentionInterval
            self.tieWordEmbeddings = tieWordEmbeddings
            self.activation = activation
            self.gating = gating
        }
    }

    public let config: Config

    public init(config: Config) {
        self.config = config
    }

    @ModelComponentBuilder
    public var body: some ModelComponent {
        TokenEmbedding(vocabSize: config.vocabularySize, embeddingSize: config.hiddenSize)

        LayerStack(0..<config.hiddenLayers) { layerIndex in
            if config.isFullAttentionLayer(layerIndex) {
                Residual {
                    RMSNorm(dimension: config.hiddenSize, epsilon: config.normEps)
                    Attention(
                        hiddenSize: config.hiddenSize,
                        headCount: config.attentionHeads,
                        kvHeadCount: config.kvHeads,
                        headDimension: config.headDimension,
                        rope: RoPEAttributes(
                            dimension: config.ropePartialDim,
                            base: config.ropeTheta,
                            scaling: config.ropeScaling
                        ),
                        qkNorm: .rmsNorm,
                        outputGate: .sigmoidPackedInQProj
                    )
                }
                Residual {
                    RMSNorm(dimension: config.hiddenSize, epsilon: config.normEps)
                    MLP(
                        inputSize: config.hiddenSize,
                        intermediateSize: config.intermediateSize,
                        activation: config.activation,
                        gating: config.gating
                    )
                }
            } else {
                Residual {
                    RMSNorm(dimension: config.hiddenSize, epsilon: config.normEps)
                    DeltaNet(
                        hiddenSize: config.hiddenSize,
                        stateSize: config.linearKeyHeadDim,
                        variant: .gated
                    )
                }
                Residual {
                    RMSNorm(dimension: config.hiddenSize, epsilon: config.normEps)
                    MLP(
                        inputSize: config.hiddenSize,
                        intermediateSize: config.intermediateSize,
                        activation: config.activation,
                        gating: config.gating
                    )
                }
            }
        }

        RMSNorm(dimension: config.hiddenSize, epsilon: config.normEps)
        OutputHead(
            inputSize: config.hiddenSize,
            vocabSize: config.vocabularySize,
            tiedToEmbedding: config.tieWordEmbeddings
        )
    }
}
