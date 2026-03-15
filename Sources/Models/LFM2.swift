import SwiftLM

/// LFM2 / LFM2.5 hybrid convolution + attention transformer.
///
/// Architecture: Double-gated LIV convolution layers interleaved with
/// standard GQA attention layers, following an explicit per-layer pattern.
/// Conv layers use depthwise causal Conv1d with double gating (B*x, C*conv_out).
/// Attention layers use GQA with QK RMSNorm and RoPE.
///
/// The layer pattern is derived from `ModelConfig.layerTypes`, which provides
/// a flat array of layer type strings (e.g., ["conv", "conv", "full_attention", ...]).
///
/// MoE variants (8B-A1B, 24B-A2B) replace dense MLP with mixture-of-experts
/// in all layers beyond `numDenseLayers`.
///
/// Accepts `ModelConfig` directly. Required fields:
/// - `layerTypes` (per-layer type schedule)
/// - `convLCache` (causal cache length for depthwise conv)
///
/// Call `LFM2.validate(_:)` before constructing to ensure all required fields are present.
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

    /// Per-layer descriptor expanded from sections and `numDenseLayers`.
    ///
    /// Each entry carries the layer type (conv vs attention) and whether
    /// MoE replaces the dense MLP. This is the single source of truth
    /// for layer-level FFN variant selection.
    public struct LayerDescriptor: Sendable {

        /// Whether this layer uses short convolution (true) or full attention (false).
        public let isConvolution: Bool

        /// Whether this layer uses MoE (true) or dense MLP (false).
        public let useMoE: Bool

        public init(isConvolution: Bool, useMoE: Bool) {
            self.isConvolution = isConvolution
            self.useMoE = useMoE
        }
    }

    public let config: ModelConfig

    /// Precomputed sections derived from `config.layerTypes`.
    private let sections: [LayerSection]

    public init(config: ModelConfig) throws {
        self.config = config
        guard let layerTypes = config.layerTypes else {
            throw ModelGraphBuildError.missingMetadata("layer_types required for LFM2")
        }
        guard layerTypes.count == config.layerCount else {
            throw ModelGraphBuildError.invalidConfig(
                "layer_types count (\(layerTypes.count)) != num_hidden_layers (\(config.layerCount))")
        }
        self.sections = try LFM2.buildSections(from: layerTypes)
    }

    /// Validate that config has all required fields for LFM2.
    ///
    /// Call this before constructing an `LFM2` instance to get clear error messages
    /// for missing required fields.
    public static func validate(_ config: ModelConfig) throws {
        guard config.layerTypes != nil else {
            throw ModelGraphBuildError.missingMetadata("layer_types required for LFM2")
        }
        guard config.convLCache != nil else {
            throw ModelGraphBuildError.missingMetadata("conv_L_cache required for LFM2")
        }
    }

    // MARK: - Computed Helpers

    /// Causal cache length (kernel_size = convLCache + 1).
    private var convLCache: Int {
        config.convLCache ?? 3
    }

    /// Head dimension derived from hiddenSize / attentionHeads.
    private var resolvedHeadDim: Int {
        config.hiddenSize / config.attentionHeads
    }

    /// Whether this model uses MoE.
    private var isMoE: Bool {
        config.expertCount != nil
    }

    /// Expand sections into a flat list of layer descriptors.
    private var layerDescriptors: [LayerDescriptor] {
        var result: [LayerDescriptor] = []
        for section in sections {
            for _ in 0..<section.groupCount {
                for _ in 0..<section.convsPerGroup {
                    result.append(LayerDescriptor(
                        isConvolution: true,
                        useMoE: isMoE && result.count >= config.numDenseLayers
                    ))
                }
                if section.hasAttention {
                    result.append(LayerDescriptor(
                        isConvolution: false,
                        useMoE: isMoE && result.count >= config.numDenseLayers
                    ))
                }
            }
        }
        return result
    }

    @ModelComponentBuilder
    public var body: some ModelComponent {
        TokenEmbedding(vocabSize: config.vocabSize, embeddingSize: config.hiddenSize)

        LayerStack(layerDescriptors) { layer in
            if layer.isConvolution {
                LFM2ConvDecoderLayer(config: config, convLCache: convLCache, useMoE: layer.useMoE)
            } else {
                LFM2AttnDecoderLayer(config: config, resolvedHeadDim: resolvedHeadDim, useMoE: layer.useMoE)
            }
        }

        RMSNorm(dimension: config.hiddenSize, epsilon: config.normEps)
        OutputHead(
            inputSize: config.hiddenSize,
            vocabSize: config.vocabSize,
            tiedToEmbedding: config.tiedEmbeddings
        )
    }

    // MARK: - Section Builder

    /// Convert flat layer_types array to LFM2.LayerSection pattern.
    ///
    /// Groups consecutive layers into sections where each section is:
    /// (conv x convsPerGroup + attention?) x groupCount
    public static func buildSections(from layerTypes: [String]) throws -> [LayerSection] {
        var sections: [LayerSection] = []
        var i = 0

        while i < layerTypes.count {
            // Count consecutive conv layers
            var convCount = 0
            let groupStart = i
            while i < layerTypes.count && layerTypes[i] == "conv" {
                convCount += 1
                i += 1
            }

            // Check if followed by attention
            let hasAttention = i < layerTypes.count && layerTypes[i] == "full_attention"
            if hasAttention { i += 1 }

            if convCount == 0 && !hasAttention {
                throw ModelGraphBuildError.invalidConfig(
                    "Unexpected layer_type '\(layerTypes[groupStart])' at index \(groupStart)")
            }

            // Try to find repeating groups with the same pattern
            let patternLength = convCount + (hasAttention ? 1 : 0)
            var groupCount = 1

            while i + patternLength <= layerTypes.count {
                var matches = true
                var j = 0
                while j < convCount && (i + j) < layerTypes.count {
                    if layerTypes[i + j] != "conv" { matches = false; break }
                    j += 1
                }
                if matches && hasAttention {
                    if (i + j) >= layerTypes.count || layerTypes[i + j] != "full_attention" {
                        matches = false
                    }
                }
                if j < convCount { matches = false }

                if matches {
                    groupCount += 1
                    i += patternLength
                } else {
                    break
                }
            }

            sections.append(LayerSection(
                groupCount: groupCount,
                convsPerGroup: convCount,
                hasAttention: hasAttention
            ))
        }

        return sections
    }
}

// MARK: - Decoder Layers

/// Conv decoder layer: RMSNorm + ShortConv residual, then RMSNorm + FFN residual.
///
/// The short conv block is a double-gated LIV convolution:
/// in_proj(D -> 3D) -> chunk(B, C, x) -> B*x -> depthwise_conv1d -> C*conv_out -> out_proj
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
            if useMoE, let expertCount = config.expertCount, let expertsPerToken = config.expertsPerToken {
                MoE(
                    expertCount: expertCount,
                    expertsPerToken: expertsPerToken,
                    expertInputSize: config.hiddenSize,
                    expertIntermediateSize: config.moeIntermediateSize ?? config.intermediateSize,
                    expertBias: config.mlpBias
                )
            } else {
                MLP(inputSize: config.hiddenSize, intermediateSize: config.intermediateSize)
            }
        }
    }
}

/// Attention decoder layer: RMSNorm + GQA residual, then RMSNorm + FFN residual.
struct LFM2AttnDecoderLayer: ModelComponent {

    let config: ModelConfig
    let resolvedHeadDim: Int
    let useMoE: Bool

    @ModelComponentBuilder
    var body: some ModelComponent {
        Residual {
            RMSNorm(dimension: config.hiddenSize, epsilon: config.normEps)
            Attention(
                hiddenSize: config.hiddenSize,
                headCount: config.attentionHeads,
                kvHeadCount: config.kvHeads,
                rope: RoPEAttributes(
                    dimension: resolvedHeadDim,
                    base: config.ropeTheta
                ),
                qkNorm: .rmsNorm
            )
        }
        Residual {
            RMSNorm(dimension: config.hiddenSize, epsilon: config.normEps)
            if useMoE, let expertCount = config.expertCount, let expertsPerToken = config.expertsPerToken {
                MoE(
                    expertCount: expertCount,
                    expertsPerToken: expertsPerToken,
                    expertInputSize: config.hiddenSize,
                    expertIntermediateSize: config.moeIntermediateSize ?? config.intermediateSize,
                    expertBias: config.mlpBias
                )
            } else {
                MLP(inputSize: config.hiddenSize, intermediateSize: config.intermediateSize)
            }
        }
    }
}
