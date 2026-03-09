import SwiftLM

/// Standard pre-norm decoder-only transformer.
///
/// Covers architectures: Llama, Qwen 2/2.5, Mistral, Phi, StarCoder2.
/// MoE variants (Mixtral) are supported via `Config.moe`.
///
/// ```swift
/// let llama = Transformer(config: .init(
///     hiddenSize: 4096,
///     hiddenLayers: 32,
///     intermediateSize: 11008,
///     attentionHeads: 32,
///     kvHeads: 8,
///     vocabularySize: 32000
/// ))
/// let graph = try llama.makeModelGraph()
/// ```
public struct Transformer: LanguageModel {

    /// Architecture configuration for standard transformers.
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

        public let normEps: Float

        // MARK: - Activation & Gating

        public let activation: ActivationKind
        public let gating: GatingKind

        // MARK: - Bias

        public let attentionBias: Bool
        public let mlpBias: Bool

        // MARK: - RoPE

        public let ropeTheta: Float
        public let ropeScaling: RoPEScaling?

        // MARK: - MoE (optional)

        public let moe: MoEConfig?

        // MARK: - Output

        public let tieWordEmbeddings: Bool

        // MARK: - Computed

        public var resolvedHeadDimension: Int {
            headDimension ?? (hiddenSize / attentionHeads)
        }

        public var isMoE: Bool { moe != nil }

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
            moe: MoEConfig? = nil,
            tieWordEmbeddings: Bool = true
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
            self.moe = moe
            self.tieWordEmbeddings = tieWordEmbeddings
        }
    }

    /// MoE sub-configuration.
    public struct MoEConfig: Sendable {

        public let expertCount: Int
        public let expertsPerToken: Int
        public let gateKind: MoEGateKind

        public init(
            expertCount: Int,
            expertsPerToken: Int,
            gateKind: MoEGateKind = .topK
        ) {
            self.expertCount = expertCount
            self.expertsPerToken = expertsPerToken
            self.gateKind = gateKind
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
                RMSNorm(dimension: config.hiddenSize, epsilon: config.normEps)
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
                    )
                )
            }
            Residual {
                RMSNorm(dimension: config.hiddenSize, epsilon: config.normEps)
                if let moe = config.moe {
                    MoE(
                        expertCount: moe.expertCount,
                        expertsPerToken: moe.expertsPerToken,
                        gateKind: moe.gateKind,
                        expertInputSize: config.hiddenSize,
                        expertIntermediateSize: config.intermediateSize,
                        expertActivation: config.activation,
                        expertGating: config.gating,
                        expertBias: config.mlpBias
                    )
                } else {
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

        RMSNorm(dimension: config.hiddenSize, epsilon: config.normEps)
        OutputHead(
            inputSize: config.hiddenSize,
            vocabSize: config.vocabularySize,
            tiedToEmbedding: config.tieWordEmbeddings
        )
    }
}
