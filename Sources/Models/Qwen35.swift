import SwiftLM

/// Qwen 3.5 hybrid Gated DeltaNet + Full Attention transformer.
///
/// Architecture pattern: every `fullAttentionInterval`-th layer uses
/// full multi-head attention; all other layers use Gated DeltaNet
/// (linear attention with O(1) per-token state update).
///
/// ```swift
/// let qwen = Qwen35(config: .init(
///     hiddenSize: 1024,
///     hiddenLayers: 24,
///     intermediateSize: 3584,
///     vocabularySize: 248320,
///     attentionHeads: 8,
///     kvHeads: 2,
///     headDim: 256,
///     linearKeyHeads: 16,
///     linearValueHeads: 16,
///     linearKeyHeadDim: 128,
///     linearValueHeadDim: 128
/// ))
/// let graph = try qwen.makeModelGraph()
/// ```
public struct Qwen35: LanguageModel {

    /// Architecture configuration for Qwen 3.5 hybrid models.
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
        public let headDim: Int

        // MARK: - RoPE

        public let ropeTheta: Float
        public let ropeScaling: RoPEScaling?
        public let partialRotaryFactor: Float

        // MARK: - DeltaNet Parameters

        public let linearKeyHeads: Int
        public let linearValueHeads: Int
        public let linearKeyHeadDim: Int
        public let linearValueHeadDim: Int
        public let convKernelSize: Int

        // MARK: - Hybrid Routing

        public let fullAttentionInterval: Int

        // MARK: - Output

        public let tieWordEmbeddings: Bool

        // MARK: - Computed

        /// RoPE dimension for partial rotation (25% of head_dim by default).
        public var ropePartialDim: Int {
            Int(Float(headDim) * partialRotaryFactor)
        }

        /// Number of hybrid groups (each = interval-1 DeltaNet + 1 FullAttention).
        public var hybridGroupCount: Int {
            hiddenLayers / fullAttentionInterval
        }

        /// Number of DeltaNet layers per hybrid group.
        public var deltaNetLayersPerGroup: Int {
            fullAttentionInterval - 1
        }

        public init(
            hiddenSize: Int,
            hiddenLayers: Int,
            intermediateSize: Int,
            vocabularySize: Int,
            normEps: Float = 1e-6,
            attentionHeads: Int,
            kvHeads: Int,
            headDim: Int = 256,
            ropeTheta: Float = 10_000_000.0,
            ropeScaling: RoPEScaling? = nil,
            partialRotaryFactor: Float = 0.25,
            linearKeyHeads: Int,
            linearValueHeads: Int,
            linearKeyHeadDim: Int = 128,
            linearValueHeadDim: Int = 128,
            convKernelSize: Int = 4,
            fullAttentionInterval: Int = 4,
            tieWordEmbeddings: Bool = false
        ) {
            precondition(
                hiddenLayers % fullAttentionInterval == 0,
                "hiddenLayers must be divisible by fullAttentionInterval"
            )
            self.hiddenSize = hiddenSize
            self.hiddenLayers = hiddenLayers
            self.intermediateSize = intermediateSize
            self.vocabularySize = vocabularySize
            self.normEps = normEps
            self.attentionHeads = attentionHeads
            self.kvHeads = kvHeads
            self.headDim = headDim
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
        }
    }

    public let config: Config

    public init(config: Config) {
        self.config = config
    }

    @ModelComponentBuilder
    public var body: some ModelComponent {
        TokenEmbedding(vocabSize: config.vocabularySize, embeddingSize: config.hiddenSize)

        // Hybrid groups: [DeltaNet x (interval-1), FullAttention x 1]
        Repeat(count: config.hybridGroupCount, label: "hybrid_groups") {

            // DeltaNet layers (linear attention, O(1) per token)
            Repeat(count: config.deltaNetLayersPerGroup, label: "deltanet_layers") {
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
                    MLP(inputSize: config.hiddenSize, intermediateSize: config.intermediateSize)
                }
            }

            // Full attention layer (standard GQA with partial RoPE)
            Residual {
                RMSNorm(dimension: config.hiddenSize, epsilon: config.normEps)
                Attention(
                    hiddenSize: config.hiddenSize,
                    headCount: config.attentionHeads,
                    kvHeadCount: config.kvHeads,
                    headDimension: config.headDim,
                    rope: RoPEAttributes(
                        dimension: config.ropePartialDim,
                        base: config.ropeTheta,
                        scaling: config.ropeScaling
                    ),
                    qkNorm: .rmsNorm
                )
            }
            Residual {
                RMSNorm(dimension: config.hiddenSize, epsilon: config.normEps)
                MLP(inputSize: config.hiddenSize, intermediateSize: config.intermediateSize)
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

// MARK: - Preset Configurations

extension Qwen35.Config {

    /// Qwen 3.5 0.8B
    public static let qwen35_0_8B = Qwen35.Config(
        hiddenSize: 1024,
        hiddenLayers: 24,
        intermediateSize: 3584,
        vocabularySize: 248320,
        attentionHeads: 8,
        kvHeads: 2,
        headDim: 256,
        linearKeyHeads: 16,
        linearValueHeads: 16,
        linearKeyHeadDim: 128,
        linearValueHeadDim: 128
    )
}
