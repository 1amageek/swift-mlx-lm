import LMArchitecture

/// LFM2 / LFM2.5 hybrid convolution + attention transformer.
///
/// Uses `LayerStack` + `if` to express heterogeneous layer schedules.
/// Each layer is either conv or attention, determined by `layerTypes` at build time.
///
/// Reference: LiquidAI LFM2 / LFM2.5
public struct LFM2: ModelComponent {

    public let config: ModelConfig
    private let convLayerIndices: Set<Int>

    public init(config: ModelConfig) throws {
        self.config = config
        guard let layerTypes = config.layerTypes else {
            throw ModelGraphBuildError.missingMetadata("layer_types required for LFM2")
        }
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
    }

    private var convLCache: Int { config.convLCache ?? 3 }
    private var headDimension: Int { config.hiddenSize / config.attentionHeads }

    @ModelComponentBuilder
    public var body: some ModelComponent {
        TokenEmbedding(vocabSize: config.vocabSize, embeddingSize: config.hiddenSize)
        RMSNorm(dimension: config.hiddenSize, epsilon: config.normEps)

        LayerStack(0..<config.layerCount) { layerIndex in
            if convLayerIndices.contains(layerIndex) {
                LFM2ConvDecoderLayer(config: config, convLCache: convLCache)
            } else {
                LFM2AttnDecoderLayer(config: config, headDimension: headDimension)
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

struct LFM2ConvDecoderLayer: ModelComponent {
    let config: ModelConfig
    let convLCache: Int

    @ModelComponentBuilder
    var body: some ModelComponent {
        Residual {
            RMSNorm(dimension: config.hiddenSize, epsilon: config.normEps)
            ShortConv(hiddenSize: config.hiddenSize, kernelSize: convLCache)
        }
        Residual {
            RMSNorm(dimension: config.hiddenSize, epsilon: config.normEps)
            MLP(inputSize: config.hiddenSize, intermediateSize: config.intermediateSize)
        }
    }
}

struct LFM2AttnDecoderLayer: ModelComponent {
    let config: ModelConfig
    let headDimension: Int

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
            MLP(inputSize: config.hiddenSize, intermediateSize: config.intermediateSize)
        }
    }
}
