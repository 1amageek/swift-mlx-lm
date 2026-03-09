import SwiftLM

/// LFM2 / LFM2.5 hybrid convolution + attention transformer.
///
/// Architecture: Double-gated LIV convolution layers interleaved with
/// standard GQA attention layers, following an explicit per-layer pattern.
/// Conv layers use depthwise causal Conv1d with double gating (B*x, C*conv_out).
/// Attention layers use GQA with QK RMSNorm and RoPE.
///
/// The layer pattern is defined as an array of `LayerSection`s, each
/// specifying a run of identical groups (conv×N + optional attention).
/// This supports arbitrary patterns across all model sizes.
///
/// MoE variants (8B-A1B, 24B-A2B) replace dense MLP with mixture-of-experts
/// in all layers beyond `numDenseLayers`.
///
/// ```swift
/// let lfm2 = LFM2(config: .lfm25_1_2B)
/// let graph = try lfm2.makeModelGraph()
/// ```
///
/// Reference: LiquidAI LFM2 / LFM2.5
public struct LFM2: ModelComponent {

    /// A section of the layer pattern: `groupCount` groups,
    /// each with `convsPerGroup` conv layers optionally followed by 1 attention layer.
    public struct LayerSection: Sendable {
        public let groupCount: Int
        public let convsPerGroup: Int
        public let hasAttention: Bool

        public init(groupCount: Int, convsPerGroup: Int, hasAttention: Bool = true) {
            self.groupCount = groupCount
            self.convsPerGroup = convsPerGroup
            self.hasAttention = hasAttention
        }

        /// Number of decoder layers this section contributes.
        public var layerCount: Int {
            groupCount * (convsPerGroup + (hasAttention ? 1 : 0))
        }
    }

    /// Architecture configuration for LFM2 / LFM2.5 models.
    public struct Config: Sendable {

        // MARK: - Core Dimensions

        public let hiddenSize: Int
        public let hiddenLayers: Int
        public let intermediateSize: Int
        public let vocabularySize: Int
        public let normEps: Float

        // MARK: - Attention

        public let attentionHeads: Int
        public let kvHeads: Int

        // MARK: - RoPE

        public let ropeTheta: Float

        // MARK: - Short Convolution

        /// Causal cache length for depthwise conv (kernel_size = convLCache + 1).
        public let convLCache: Int

        // MARK: - Layer Pattern

        /// Ordered sections defining the conv/attention pattern.
        public let sections: [LayerSection]

        // MARK: - MoE (optional)

        public let moe: MoEConfig?

        /// Number of leading layers that use dense MLP instead of MoE.
        /// Only relevant when `moe != nil`. Defaults to 0.
        public let numDenseLayers: Int

        // MARK: - Output

        public let tieWordEmbeddings: Bool

        // MARK: - Computed

        public var resolvedHeadDim: Int { hiddenSize / attentionHeads }
        public var convKernelSize: Int { convLCache + 1 }
        public var isMoE: Bool { moe != nil }

        public init(
            hiddenSize: Int,
            hiddenLayers: Int,
            intermediateSize: Int,
            vocabularySize: Int,
            normEps: Float = 1e-5,
            attentionHeads: Int,
            kvHeads: Int,
            ropeTheta: Float = 1_000_000.0,
            convLCache: Int = 3,
            sections: [LayerSection],
            moe: MoEConfig? = nil,
            numDenseLayers: Int = 0,
            tieWordEmbeddings: Bool = true
        ) {
            let total = sections.reduce(0) { $0 + $1.layerCount }
            precondition(
                total == hiddenLayers,
                "Section pattern (\(total)) must match hiddenLayers (\(hiddenLayers))"
            )
            self.hiddenSize = hiddenSize
            self.hiddenLayers = hiddenLayers
            self.intermediateSize = intermediateSize
            self.vocabularySize = vocabularySize
            self.normEps = normEps
            self.attentionHeads = attentionHeads
            self.kvHeads = kvHeads
            self.ropeTheta = ropeTheta
            self.convLCache = convLCache
            self.sections = sections
            self.moe = moe
            self.numDenseLayers = numDenseLayers
            self.tieWordEmbeddings = tieWordEmbeddings
        }
    }

    /// MoE sub-configuration for LFM2 MoE variants.
    public struct MoEConfig: Sendable {
        public let expertCount: Int
        public let expertsPerToken: Int
        public let moeIntermediateSize: Int
        public let expertBias: Bool

        public init(
            expertCount: Int,
            expertsPerToken: Int,
            moeIntermediateSize: Int,
            expertBias: Bool = true
        ) {
            self.expertCount = expertCount
            self.expertsPerToken = expertsPerToken
            self.moeIntermediateSize = moeIntermediateSize
            self.expertBias = expertBias
        }
    }

    public let config: Config

    public init(config: Config) {
        self.config = config
    }

    @ModelComponentBuilder
    public var body: some ModelComponent {
        TokenEmbedding(vocabSize: config.vocabularySize, embeddingSize: config.hiddenSize)

        for section in config.sections {
            if section.hasAttention {
                Repeat(count: section.groupCount) {
                    Repeat(count: section.convsPerGroup) {
                        LFM2ConvDecoderLayer(config: config)
                    }
                    LFM2AttnDecoderLayer(config: config)
                }
            } else {
                Repeat(count: section.groupCount) {
                    Repeat(count: section.convsPerGroup) {
                        LFM2ConvDecoderLayer(config: config)
                    }
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

// MARK: - Decoder Layers

/// Conv decoder layer: RMSNorm + ShortConv residual, then RMSNorm + FFN residual.
///
/// The short conv block is a double-gated LIV convolution:
/// in_proj(D -> 3D) -> chunk(B, C, x) -> B*x -> depthwise_conv1d -> C*conv_out -> out_proj
struct LFM2ConvDecoderLayer: ModelComponent {

    let config: LFM2.Config

    @ModelComponentBuilder
    var body: some ModelComponent {
        Residual {
            RMSNorm(dimension: config.hiddenSize, epsilon: config.normEps)
            StateSpace(
                hiddenSize: config.hiddenSize,
                stateSize: config.convLCache,
                variant: "short_conv"
            )
        }
        Residual {
            RMSNorm(dimension: config.hiddenSize, epsilon: config.normEps)
            if let moe = config.moe {
                MoE(
                    expertCount: moe.expertCount,
                    expertsPerToken: moe.expertsPerToken,
                    expertInputSize: config.hiddenSize,
                    expertIntermediateSize: moe.moeIntermediateSize,
                    expertBias: moe.expertBias
                )
            } else {
                MLP(inputSize: config.hiddenSize, intermediateSize: config.intermediateSize)
            }
        }
    }
}

/// Attention decoder layer: RMSNorm + GQA residual, then RMSNorm + FFN residual.
struct LFM2AttnDecoderLayer: ModelComponent {

    let config: LFM2.Config

    @ModelComponentBuilder
    var body: some ModelComponent {
        Residual {
            RMSNorm(dimension: config.hiddenSize, epsilon: config.normEps)
            Attention(
                hiddenSize: config.hiddenSize,
                headCount: config.attentionHeads,
                kvHeadCount: config.kvHeads,
                rope: RoPEAttributes(
                    dimension: config.resolvedHeadDim,
                    base: config.ropeTheta
                ),
                qkNorm: .rmsNorm
            )
        }
        Residual {
            RMSNorm(dimension: config.hiddenSize, epsilon: config.normEps)
            if let moe = config.moe {
                MoE(
                    expertCount: moe.expertCount,
                    expertsPerToken: moe.expertsPerToken,
                    expertInputSize: config.hiddenSize,
                    expertIntermediateSize: moe.moeIntermediateSize,
                    expertBias: moe.expertBias
                )
            } else {
                MLP(inputSize: config.hiddenSize, intermediateSize: config.intermediateSize)
            }
        }
    }
}

// MARK: - Preset Configurations

extension LFM2.Config {

    /// LFM2 350M (text backbone of LFM2-VL-450M)
    ///
    /// Pattern: (conv×2 + attn)×3, (conv + attn)×3, conv×1 = 16 layers
    public static let lfm2_350M = LFM2.Config(
        hiddenSize: 1024,
        hiddenLayers: 16,
        intermediateSize: 6656,
        vocabularySize: 65536,
        attentionHeads: 16,
        kvHeads: 8,
        sections: [
            .init(groupCount: 3, convsPerGroup: 2),
            .init(groupCount: 3, convsPerGroup: 1),
            .init(groupCount: 1, convsPerGroup: 1, hasAttention: false),
        ]
    )

    /// LFM 2.5 1.2B
    ///
    /// Pattern: (conv×2 + attn)×3, (conv + attn)×3, conv×1 = 16 layers
    public static let lfm25_1_2B = LFM2.Config(
        hiddenSize: 2048,
        hiddenLayers: 16,
        intermediateSize: 12288,
        vocabularySize: 65536,
        attentionHeads: 32,
        kvHeads: 8,
        sections: [
            .init(groupCount: 3, convsPerGroup: 2),
            .init(groupCount: 3, convsPerGroup: 1),
            .init(groupCount: 1, convsPerGroup: 1, hasAttention: false),
        ]
    )

    /// LFM2 2.6B
    ///
    /// Pattern: (conv×2 + attn)×2, (conv×3 + attn)×4, (conv×2 + attn)×2, conv×2 = 30 layers
    public static let lfm2_2_6B = LFM2.Config(
        hiddenSize: 2048,
        hiddenLayers: 30,
        intermediateSize: 10752,
        vocabularySize: 65536,
        attentionHeads: 32,
        kvHeads: 8,
        sections: [
            .init(groupCount: 2, convsPerGroup: 2),
            .init(groupCount: 4, convsPerGroup: 3),
            .init(groupCount: 2, convsPerGroup: 2),
            .init(groupCount: 1, convsPerGroup: 2, hasAttention: false),
        ]
    )

    /// LFM2 8B-A1B (MoE: 32 experts, 4 active per token)
    ///
    /// Pattern: (conv×2 + attn)×1, (conv×3 + attn)×4, (conv×2 + attn)×1, conv×2 = 24 layers
    public static let lfm2_8B_A1B = LFM2.Config(
        hiddenSize: 2048,
        hiddenLayers: 24,
        intermediateSize: 7168,
        vocabularySize: 65536,
        attentionHeads: 32,
        kvHeads: 8,
        sections: [
            .init(groupCount: 1, convsPerGroup: 2),
            .init(groupCount: 4, convsPerGroup: 3),
            .init(groupCount: 1, convsPerGroup: 2),
            .init(groupCount: 1, convsPerGroup: 2, hasAttention: false),
        ],
        moe: .init(expertCount: 32, expertsPerToken: 4, moeIntermediateSize: 1792)
    )

    /// LFM2 24B-A2B (MoE: 64 experts, 4 active per token)
    ///
    /// Pattern: (conv×2 + attn)×1, (conv×3 + attn)×9, conv×1 = 40 layers
    public static let lfm2_24B_A2B = LFM2.Config(
        hiddenSize: 2048,
        hiddenLayers: 40,
        intermediateSize: 11776,
        vocabularySize: 65536,
        attentionHeads: 32,
        kvHeads: 8,
        sections: [
            .init(groupCount: 1, convsPerGroup: 2),
            .init(groupCount: 9, convsPerGroup: 3),
            .init(groupCount: 1, convsPerGroup: 1, hasAttention: false),
        ],
        moe: .init(expertCount: 64, expertsPerToken: 4, moeIntermediateSize: 1536),
        numDenseLayers: 2
    )
}
