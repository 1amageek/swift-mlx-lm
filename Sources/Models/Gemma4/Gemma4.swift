import LMArchitecture

/// Gemma 4 text backbone with sliding/full attention routing and per-layer inputs.
///
/// This declaration models the Gemma 4 text stack used by multimodal bundles such
/// as `google/gemma-4-E2B-it`. It captures three Gemma 4 specific contracts:
/// - per-layer schedule via `layerTypes`
/// - full-attention layers with separate head dimension and RoPE parameters
/// - token-conditioned per-layer input residual after the feed-forward block
///
/// Gemma 4 MoE blocks are intentionally rejected for now rather than modeled
/// incorrectly.
public struct Gemma4: ModelComponent {

    public let config: ModelConfig

    public init(config: ModelConfig) throws {
        try Self.validate(config)
        self.config = config
    }

    public static func validate(_ config: ModelConfig) throws {
        guard let layerTypes = config.layerTypes else {
            throw ModelGraphBuildError.missingMetadata("layer_types required for Gemma4")
        }
        guard layerTypes.count == config.layerCount else {
            throw ModelGraphBuildError.invalidConfig(
                "layer_types count (\(layerTypes.count)) != num_hidden_layers (\(config.layerCount))"
            )
        }
        guard config.hiddenSizePerLayerInput != nil else {
            throw ModelGraphBuildError.missingMetadata(
                "hidden_size_per_layer_input required for Gemma4"
            )
        }
        guard config.vocabSizePerLayerInput != nil else {
            throw ModelGraphBuildError.missingMetadata(
                "vocab_size_per_layer_input required for Gemma4"
            )
        }
        guard config.globalHeadDim != nil else {
            throw ModelGraphBuildError.missingMetadata("global_head_dim required for Gemma4")
        }
        guard config.numKVSharedLayers != nil else {
            throw ModelGraphBuildError.missingMetadata("num_kv_shared_layers required for Gemma4")
        }
        if layerTypes.contains("full_attention") {
            guard config.fullAttentionRopeTheta != nil else {
                throw ModelGraphBuildError.missingMetadata(
                    "full_attention rope_theta required for Gemma4"
                )
            }
            guard config.fullAttentionPartialRotaryFactor != nil else {
                throw ModelGraphBuildError.missingMetadata(
                    "full_attention partial_rotary_factor required for Gemma4"
                )
            }
        }
        let unsupportedTypes = Set(layerTypes).subtracting(["sliding_attention", "full_attention"])
        guard unsupportedTypes.isEmpty else {
            throw ModelGraphBuildError.invalidConfig(
                "Unsupported Gemma4 layer_types: \(unsupportedTypes.sorted())"
            )
        }
        if config.expertCount != nil || config.expertsPerToken != nil {
            throw ModelGraphBuildError.invalidConfig(
                "Gemma4 MoE blocks are not yet modeled in ModelDeclarations"
            )
        }
    }

    @ModelComponentBuilder
    public var body: some ModelComponent {
        TokenEmbedding(
            vocabSize: config.vocabSize,
            embeddingSize: config.hiddenSize,
            embeddingScale: Float(config.hiddenSize).squareRoot()
        )

        // Gemma4 layers are normalized as repeating(count: 1) blocks so
        // ParameterResolver and MetalCompiler can substitute the concrete
        // layer index during lowering.
        ForEach(0..<config.layerCount) { layerIndex in
            Gemma4DecoderLayer(config: config, layerIndex: layerIndex)
        }

        RMSNorm(dimension: config.hiddenSize, epsilon: config.normEps)
        OutputHead(
            inputSize: config.hiddenSize,
            vocabSize: config.vocabSize,
            tiedToEmbedding: config.tiedEmbeddings
        )
    }
}
