import LMArchitecture

/// Gemma 3 text backbone with Gemma-style sandwich norms and mixed attention schedule.
///
/// This declaration covers both causal Gemma 3 text models and bidirectional
/// encoder-style variants such as EmbeddingGemma. The configuration contract is:
/// - `layer_types` selects `sliding_attention` or `full_attention` per layer
/// - sliding layers may use a distinct local RoPE base
/// - bidirectional variants disable causal masking and keep full-sequence
///   attention masks while preserving the local/full RoPE schedule
public struct Gemma3Text: ModelComponent {

    public let config: ModelConfig

    public init(config: ModelConfig) throws {
        try Self.validate(config)
        self.config = config
    }

    public static func validate(_ config: ModelConfig) throws {
        guard let layerTypes = config.layerTypes else {
            throw ModelGraphBuildError.missingMetadata("layer_types required for gemma3_text")
        }
        guard layerTypes.count == config.layerCount else {
            throw ModelGraphBuildError.invalidConfig(
                "layer_types count (\(layerTypes.count)) != num_hidden_layers (\(config.layerCount))"
            )
        }
        let unsupportedTypes = Set(layerTypes).subtracting(["sliding_attention", "full_attention"])
        guard unsupportedTypes.isEmpty else {
            throw ModelGraphBuildError.invalidConfig(
                "Unsupported Gemma3Text layer_types: \(unsupportedTypes.sorted())"
            )
        }
        if config.expertCount != nil || config.expertsPerToken != nil {
            throw ModelGraphBuildError.invalidConfig(
                "Gemma3Text MoE blocks are not yet modeled in ModelDeclarations"
            )
        }
    }

    @ModelComponentBuilder
    public var body: some ModelComponent {
        Gemma3TextBackbone(validatedConfig: config)
        OutputHead(
            inputSize: config.hiddenSize,
            vocabSize: config.vocabSize,
            tiedToEmbedding: config.tiedEmbeddings
        )
    }
}

/// Gemma 3 text backbone without the language-model output head.
///
/// Embedding runtimes can compose this declaration with task-specific pooling
/// and projection layers in higher modules without leaking those concerns into
/// the shared model declaration.
public struct Gemma3TextBackbone: ModelComponent {
    public let config: ModelConfig

    public init(config: ModelConfig) throws {
        try Gemma3Text.validate(config)
        self.config = config
    }

    init(validatedConfig: ModelConfig) {
        self.config = validatedConfig
    }

    @ModelComponentBuilder
    public var body: some ModelComponent {
        TokenEmbedding(
            vocabSize: config.vocabSize,
            embeddingSize: config.hiddenSize,
            embeddingScale: Float(config.hiddenSize).squareRoot()
        )

        ForEach(0..<config.layerCount) { layerIndex in
            Gemma3TextDecoderLayer(config: config, layerIndex: layerIndex)
        }

        RMSNorm(dimension: config.hiddenSize, epsilon: config.normEps, weightBias: 1)
    }
}

struct Gemma3TextDecoderLayer: ModelComponent {

    let config: ModelConfig
    let layerIndex: Int

    private var layerType: String {
        config.layerTypes![layerIndex]
    }

    private var isFullAttention: Bool {
        layerType == "full_attention"
    }

    private var ropeBase: Float {
        if isFullAttention {
            return config.fullAttentionRopeTheta ?? config.ropeTheta
        }
        return config.localAttentionRopeTheta ?? config.ropeTheta
    }

    private var attentionScale: Float? {
        guard let queryPreAttentionScalar = config.queryPreAttentionScalar else {
            return nil
        }
        return 1.0 / queryPreAttentionScalar.squareRoot()
    }

    private var attentionWindow: AttentionWindow? {
        guard isFullAttention == false,
              let slidingWindow = config.slidingWindow else {
            return nil
        }
        if config.useBidirectionalAttention {
            // EmbeddingGemma keeps the sliding/full RoPE schedule, but the
            // bidirectional embedding path uses a full attention mask on every
            // layer instead of applying a local sliding window.
            return nil
        }
        return AttentionWindow(left: slidingWindow, right: 0)
    }

    @ModelComponentBuilder
    var body: some ModelComponent {
        Residual {
            RMSNorm(dimension: config.hiddenSize, epsilon: config.normEps, weightBias: 1)
            Attention(
                hiddenSize: config.hiddenSize,
                headCount: config.attentionHeads,
                kvHeadCount: config.kvHeads,
                headDimension: config.headDim,
                attentionScale: attentionScale,
                bias: config.attentionBias,
                causal: !config.useBidirectionalAttention,
                rope: RoPEAttributes(
                    dimension: config.ropeDimension,
                    base: ropeBase,
                    scaling: isFullAttention ? config.fullAttentionRoPEScaling : config.ropeScaling
                ),
                qkNorm: .rmsNormUnitOffset,
                window: attentionWindow
            )
            RMSNorm(dimension: config.hiddenSize, epsilon: config.normEps, weightBias: 1)
        }

        Residual {
            RMSNorm(dimension: config.hiddenSize, epsilon: config.normEps, weightBias: 1)
            MLP(
                inputSize: config.hiddenSize,
                intermediateSize: config.intermediateSize,
                activation: .custom("gelu_pytorch_tanh"),
                gating: .none,
                bias: config.mlpBias
            )
            RMSNorm(dimension: config.hiddenSize, epsilon: config.normEps, weightBias: 1)
        }
    }
}
