import LMArchitecture

/// LFM2 / LFM2.5 hybrid convolution + attention transformer.
///
/// Handles dense (LFM2, LFM2.5) and MoE (LFM2-8B-A1B, LFM2-24B-A2B) variants.
/// HFConfigDecoder normalizes config format differences:
///   - `full_attn_idxs` (LFM2 legacy) → `layer_types` (LFM2.5 / MoE)
///   - `block_ff_dim` (LFM2 legacy) → `intermediate_size` (LFM2.5 / MoE)
///
/// Architecture (from HuggingFace transformers Lfm2Model / Lfm2MoeModel):
///   embed_tokens → layers[conv/attn + MLP/MoE] → embedding_norm → lm_head
///   - No norm after embedding (embedding_norm is the FINAL norm)
///   - Each layer: operator_norm → conv/attn + residual → ffn_norm → MLP/MoE + residual
///   - ShortConv: in_proj(3×) → B*x gate → causal conv1d → C*conv_out gate → out_proj
///   - MoE: layers beyond `numDenseLayers` use sparse MoE instead of dense MLP
///
/// Reference: https://huggingface.co/collections/LiquidAI/lfm2
///            https://huggingface.co/collections/LiquidAI/lfm25
public struct LFM2: ModelComponent {

    public let config: ModelConfig
    private let convLayerIndices: Set<Int>

    public init(config: ModelConfig) throws {
        try Self.validate(config)
        self.config = config
        let layerTypes = config.layerTypes!
        guard layerTypes.count == config.layerCount else {
            throw ModelGraphBuildError.invalidConfig(
                "layer_types count (\(layerTypes.count)) != num_hidden_layers (\(config.layerCount))")
        }
        self.convLayerIndices = Set(layerTypes.indices.filter { layerTypes[$0] == "conv" })
    }

    public static func validate(_ config: ModelConfig) throws {
        guard config.layerTypes != nil else {
            throw ModelGraphBuildError.missingMetadata("layer_types required for LFM2")
        }
        guard config.convLCache != nil else {
            throw ModelGraphBuildError.missingMetadata("conv_L_cache required for LFM2")
        }
        if config.expertCount != nil {
            guard config.expertsPerToken != nil else {
                throw ModelGraphBuildError.missingMetadata("expertsPerToken required for LFM2 MoE")
            }
            guard config.moeIntermediateSize != nil else {
                throw ModelGraphBuildError.missingMetadata("moeIntermediateSize required for LFM2 MoE")
            }
        }
    }

    private var isMoE: Bool { config.expertCount != nil }
    private var convLCache: Int { config.convLCache ?? 3 }
    private var headDimension: Int { config.hiddenSize / config.attentionHeads }

    private func isMoELayer(_ layerIndex: Int) -> Bool {
        isMoE && layerIndex >= config.numDenseLayers
    }

    @ModelComponentBuilder
    public var body: some ModelComponent {
        TokenEmbedding(vocabSize: config.vocabSize, embeddingSize: config.hiddenSize)

        LayerStack(0..<config.layerCount) { layerIndex in
            if convLayerIndices.contains(layerIndex) {
                LFM2ConvDecoderLayer(
                    config: config, convLCache: convLCache,
                    useMoE: isMoELayer(layerIndex))
            } else {
                LFM2AttnDecoderLayer(
                    config: config, headDimension: headDimension,
                    useMoE: isMoELayer(layerIndex))
            }
        }

        RMSNorm(dimension: config.hiddenSize, epsilon: config.normEps)
        OutputHead(
            inputSize: config.hiddenSize,
            vocabSize: config.vocabSize,
            tiedToEmbedding: config.tiedEmbeddings
        )
    }
}

// MARK: - Feed-Forward

/// Builds either dense MLP or sparse MoE for a single decoder layer.
///
/// MoE requires `expertCount`, `expertsPerToken`, and `moeIntermediateSize` in ModelConfig.
/// Missing any of these when `useMoE` is true is a configuration error — no silent fallback.
struct LFM2FeedForward: ModelComponent {
    let config: ModelConfig
    let useMoE: Bool

    @ModelComponentBuilder
    var body: some ModelComponent {
        if useMoE {
            MoE(
                expertCount: config.expertCount!,
                expertsPerToken: config.expertsPerToken!,
                expertInputSize: config.hiddenSize,
                expertIntermediateSize: config.moeIntermediateSize!,
                expertBias: config.mlpBias
            )
        } else {
            MLP(inputSize: config.hiddenSize, intermediateSize: config.intermediateSize)
        }
    }
}

// MARK: - Decoder Layers

struct LFM2ConvDecoderLayer: ModelComponent {
    let config: ModelConfig
    let convLCache: Int
    let useMoE: Bool

    @ModelComponentBuilder
    var body: some ModelComponent {
        Residual {
            RMSNorm(dimension: config.hiddenSize, epsilon: config.normEps)
            ShortConv(hiddenSize: config.hiddenSize, kernelSize: convLCache)
        }
        Residual {
            RMSNorm(dimension: config.hiddenSize, epsilon: config.normEps)
            LFM2FeedForward(config: config, useMoE: useMoE)
        }
    }
}

struct LFM2AttnDecoderLayer: ModelComponent {
    let config: ModelConfig
    let headDimension: Int
    let useMoE: Bool

    @ModelComponentBuilder
    var body: some ModelComponent {
        Residual {
            RMSNorm(dimension: config.hiddenSize, epsilon: config.normEps)
            Attention(
                hiddenSize: config.hiddenSize,
                headCount: config.attentionHeads,
                kvHeadCount: config.kvHeads,
                headDimension: headDimension,
                rope: RoPEAttributes(
                    dimension: headDimension,
                    base: config.ropeTheta
                ),
                qkNorm: .rmsNorm
            )
        }
        Residual {
            RMSNorm(dimension: config.hiddenSize, epsilon: config.normEps)
            LFM2FeedForward(config: config, useMoE: useMoE)
        }
    }
}
