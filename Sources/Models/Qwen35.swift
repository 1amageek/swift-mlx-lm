import LMArchitecture

/// Qwen 3.5 hybrid DeltaNet + Full Attention model.
///
/// Every `fullAttentionInterval`-th layer uses full multi-head attention;
/// all other layers use Gated DeltaNet (linear attention with O(1) per-token
/// state update).
///
/// Accepts `ModelConfig` directly. Required fields:
/// - `ssmNumHeads`, `ssmKeyHeadDim`, `ssmValueHeadDim` (DeltaNet heads)
/// - `partialRotaryFactor` (partial RoPE)
/// - `fullAttentionInterval` (hybrid routing)
/// - `convKernelSize` (DeltaNet conv kernel)
///
/// Call `Qwen35.validate(_:)` before constructing to ensure all required fields are present.
public struct Qwen35: ModelComponent {

    public let config: ModelConfig

    public init(config: ModelConfig) {
        self.config = config
    }

    /// Validate that config has all required fields for Qwen3.5.
    ///
    /// Call this before constructing a `Qwen35` instance to get clear error messages
    /// for missing required fields.
    public static func validate(_ config: ModelConfig) throws {
        guard config.ssmNumHeads != nil else {
            throw ModelGraphBuildError.missingMetadata("linear_num_value_heads required for Qwen3.5")
        }
        guard config.ssmKeyHeadDim != nil else {
            throw ModelGraphBuildError.missingMetadata("linear_key_head_dim required for Qwen3.5")
        }
        guard config.ssmValueHeadDim != nil else {
            throw ModelGraphBuildError.missingMetadata("linear_value_head_dim required for Qwen3.5")
        }
        guard config.partialRotaryFactor != nil else {
            throw ModelGraphBuildError.missingMetadata("partial_rotary_factor required for Qwen3.5")
        }
        guard config.fullAttentionInterval != nil else {
            throw ModelGraphBuildError.missingMetadata("full_attention_interval required for Qwen3.5")
        }
        guard config.convKernelSize != nil else {
            throw ModelGraphBuildError.missingMetadata("conv_kernel_size required for Qwen3.5")
        }
    }

    // MARK: - Computed Helpers

    /// RoPE dimension for partial rotation.
    private var ropePartialDim: Int {
        if let factor = config.partialRotaryFactor {
            return Int(Float(config.headDim) * factor)
        }
        return config.ropeDimension
    }

    /// Flat layer schedule: true = full attention, false = DeltaNet.
    /// Every `fullAttentionInterval`-th layer (1-indexed) is full attention.
    private var layerSchedule: [Bool] {
        guard let interval = config.fullAttentionInterval, interval > 0 else {
            return Array(repeating: true, count: config.layerCount)
        }
        return (0..<config.layerCount).map { i in (i + 1) % interval == 0 }
    }

    @ModelComponentBuilder
    public var body: some ModelComponent {
        TokenEmbedding(vocabSize: config.vocabSize, embeddingSize: config.hiddenSize)

        LayerStack(layerSchedule) { isFullAttention in
            if isFullAttention {
                Qwen35AttnDecoderLayer(config: config, ropePartialDim: ropePartialDim)
            } else {
                Qwen35DeltaNetDecoderLayer(config: config)
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

// MARK: - Decoder Layers

/// DeltaNet decoder layer: RMSNorm + Gated DeltaNet residual, then RMSNorm + MLP residual.
///
/// Linear attention with O(1) per-token state update.
struct Qwen35DeltaNetDecoderLayer: ModelComponent {

    let config: ModelConfig

    @ModelComponentBuilder
    var body: some ModelComponent {
        Residual {
            RMSNorm(dimension: config.hiddenSize, epsilon: config.normEps)
            DeltaNet(
                hiddenSize: config.hiddenSize,
                numHeads: config.ssmNumHeads ?? config.attentionHeads,
                groupCount: config.ssmGroupCount ?? config.ssmNumHeads ?? config.attentionHeads,
                keyHeadDim: config.ssmKeyHeadDim ?? 128,
                valueHeadDim: config.ssmValueHeadDim ?? 128,
                convKernelSize: config.convKernelSize ?? 1,
                variant: .gated
            )
        }
        Residual {
            RMSNorm(dimension: config.hiddenSize, epsilon: config.normEps)
            MLP(inputSize: config.hiddenSize, intermediateSize: config.intermediateSize)
        }
    }
}

/// Full attention decoder layer: RMSNorm + GQA residual, then RMSNorm + MLP residual.
///
/// Standard GQA with partial RoPE (M-RoPE for VLM, standard for text-only),
/// QK RMSNorm, and sigmoid output gate packed in Q projection.
struct Qwen35AttnDecoderLayer: ModelComponent {

    let config: ModelConfig
    let ropePartialDim: Int

    @ModelComponentBuilder
    var body: some ModelComponent {
        Residual {
            RMSNorm(dimension: config.hiddenSize, epsilon: config.normEps)
            Attention(
                hiddenSize: config.hiddenSize,
                headCount: config.attentionHeads,
                kvHeadCount: config.kvHeads,
                headDimension: config.headDim,
                rope: RoPEAttributes(
                    dimension: ropePartialDim,
                    base: config.ropeTheta,
                    scaling: config.ropeScaling,
                    mropeAxes: config.mropeAxes
                ),
                qkNorm: .rmsNorm,
                outputGate: .sigmoidPackedInQProj
            )
        }
        Residual {
            RMSNorm(dimension: config.hiddenSize, epsilon: config.normEps)
            MLP(inputSize: config.hiddenSize, intermediateSize: config.intermediateSize)
        }
    }
}
