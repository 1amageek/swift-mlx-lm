/// Qwen 3.5 hybrid DeltaNet + Full Attention transformer.
///
/// Alternates between Gated DeltaNet (linear attention, O(1) per token)
/// and standard Full Attention layers in a configurable ratio
/// (default 3:1 via `fullAttentionInterval=4`).
///
/// Key architectural features:
/// - DeltaNet layers: Conv1D + recurrent state update, no growing KV cache
/// - Full attention layers: QK norm, partial RoPE, sigmoid output gate packed in Q proj
/// - SwiGLU FFN shared across both layer types
///
/// ```swift
/// let qwen35 = Qwen35HybridTransformer(config: .init(
///     hiddenSize: 1024,
///     hiddenLayers: 24,
///     intermediateSize: 3584,
///     vocabularySize: 248320,
///     attentionHeads: 8,
///     kvHeads: 2,
///     headDimension: 256,
///     linearKeyHeads: 16,
///     linearKeyHeadDim: 128,
///     fullAttentionInterval: 4
/// ))
/// ```
public struct Qwen35HybridTransformer: ModelComponent {

    /// Configuration for Qwen 3.5 hybrid architecture.
    public struct Config: Sendable {

        // MARK: - Core Dimensions

        public let hiddenSize: Int
        public let hiddenLayers: Int
        public let intermediateSize: Int
        public let vocabularySize: Int
        public let normEps: Float

        // MARK: - Full Attention Parameters

        public let attentionHeads: Int
        public let kvHeads: Int
        public let headDimension: Int
        public let ropeTheta: Float
        public let ropeScaling: RoPEScaling?
        public let partialRotaryFactor: Float

        // MARK: - DeltaNet Parameters

        public let linearKeyHeads: Int
        public let linearValueHeads: Int
        public let linearKeyHeadDim: Int
        public let linearValueHeadDim: Int
        public let convKernelSize: Int

        // MARK: - Layer Routing

        public let fullAttentionInterval: Int

        // MARK: - Output

        public let tieWordEmbeddings: Bool

        // MARK: - Activation

        public let activation: ActivationKind
        public let gating: GatingKind

        // MARK: - Computed

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
            normEps: Float = 1e-5,
            attentionHeads: Int,
            kvHeads: Int,
            headDimension: Int,
            ropeTheta: Float = 10_000_000.0,
            ropeScaling: RoPEScaling? = nil,
            partialRotaryFactor: Float = 0.25,
            linearKeyHeads: Int,
            linearValueHeads: Int? = nil,
            linearKeyHeadDim: Int = 128,
            linearValueHeadDim: Int? = nil,
            convKernelSize: Int = 4,
            fullAttentionInterval: Int = 4,
            tieWordEmbeddings: Bool = true,
            activation: ActivationKind = .silu,
            gating: GatingKind = .swiglu
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
            self.linearValueHeads = linearValueHeads ?? linearKeyHeads
            self.linearKeyHeadDim = linearKeyHeadDim
            self.linearValueHeadDim = linearValueHeadDim ?? linearKeyHeadDim
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
                // Full attention layer with QK norm, partial RoPE, output gate
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
                // DeltaNet layer with linear attention
                Residual {
                    RMSNorm(dimension: config.hiddenSize, epsilon: config.normEps)
                    StateSpace(
                        hiddenSize: config.hiddenSize,
                        stateSize: config.linearKeyHeadDim,
                        variant: "gated_deltanet"
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
