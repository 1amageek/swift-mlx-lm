/// All configuration values needed for graph construction.
///
/// Format-agnostic container populated from HuggingFace config.json.
/// The `IRGraphAssembler` uses only the fields relevant to the detected architecture.
public struct ModelConfig: Sendable {

    // MARK: Core Dimensions

    public let hiddenSize: Int
    public let layerCount: Int
    public let intermediateSize: Int
    public let vocabSize: Int

    // MARK: Attention

    public let attentionHeads: Int
    public let kvHeads: Int
    public let headDim: Int
    public let attentionBias: Bool
    public let mlpBias: Bool

    // MARK: Normalization

    public let normEps: Float
    public let normKind: NormKind

    // MARK: RoPE

    public let ropeTheta: Float
    public let ropeDimension: Int
    public let ropeScaling: RoPEScaling?

    // MARK: Output

    public let tiedEmbeddings: Bool

    // MARK: MoE (optional)

    public let expertCount: Int?
    public let expertsPerToken: Int?
    /// Per-expert intermediate size (when different from `intermediateSize`).
    /// Used by LFM2 MoE variants where expert FFN width differs from dense FFN width.
    public let moeIntermediateSize: Int?

    // MARK: Shared-Norm Parallel Attention/MLP (optional)

    public let qkNorm: Bool

    // MARK: Hybrid State-Space / Attention (optional)

    public let fullAttentionInterval: Int?
    /// Number of DeltaNet recurrence heads.
    public let ssmNumHeads: Int?
    /// Number of DeltaNet key-state groups.
    public let ssmGroupCount: Int?
    /// Per-head key dimension (dk). May differ from valueHeadDim for asymmetric DeltaNet.
    public let ssmKeyHeadDim: Int?
    /// Per-head value dimension (dv).
    public let ssmValueHeadDim: Int?
    public let convKernelSize: Int?
    /// Causal cache length for short convolution (kernel_size = convLCache + 1).
    /// Used by hybrid conv-attention architectures (LFM2 family).
    public let convLCache: Int?
    public let partialRotaryFactor: Float?

    // MARK: Sliding Window

    public let slidingWindow: Int?

    // MARK: Per-Layer Schedule

    /// Per-layer type schedule (e.g., ["linear_attention", ..., "full_attention"]).
    /// Nil when all layers are the same type (standard transformer, MoE, etc.).
    /// When present, takes precedence over `fullAttentionInterval`.
    public let layerTypes: [String]?

    // MARK: Gemma 4 Hybrid Text (optional)

    /// Hidden width of Gemma 4 per-layer input embeddings.
    public let hiddenSizePerLayerInput: Int?

    /// Vocabulary size of Gemma 4 per-layer input embeddings.
    public let vocabSizePerLayerInput: Int?

    /// Head dimension used by Gemma 4 full-attention layers.
    public let globalHeadDim: Int?

    /// KV head count used by Gemma 4 full-attention layers when different from `kvHeads`.
    public let globalKVHeads: Int?

    /// Number of trailing Gemma 4 layers that share KV cache state.
    public let numKVSharedLayers: Int?

    /// Whether Gemma 4 doubles MLP width on KV-shared layers.
    public let useDoubleWideMLP: Bool

    /// Whether Gemma 4 full-attention layers use alternative K=V attention.
    public let attentionKEqualsV: Bool

    /// RoPE theta for Gemma 4 full-attention layers.
    public let fullAttentionRopeTheta: Float?

    /// Partial rotary factor for Gemma 4 full-attention layers.
    public let fullAttentionPartialRotaryFactor: Float?

    /// RoPE scaling metadata for Gemma 4 full-attention layers.
    public let fullAttentionRoPEScaling: RoPEScaling?

    /// Optional final output logit soft-capping applied before sampling.
    public let finalLogitSoftcapping: Float?

    // MARK: Dense/MoE Layer Boundary

    /// Number of leading layers that use dense MLP instead of MoE.
    /// Only meaningful when `expertCount != nil`. Defaults to 0.
    public let numDenseLayers: Int

    // MARK: M-RoPE (VLM only)

    /// Multi-axis RoPE configuration for VLM. Nil for text-only models.
    public let mropeAxes: MRoPEAxes?

    /// Normalization layer kind.
    public enum NormKind: Sendable, Equatable {
        case rmsNorm
        case layerNorm
    }

    /// Return a copy with M-RoPE axes set.
    func withMRoPEAxes(_ axes: MRoPEAxes?) -> ModelConfig {
        ModelConfig(
            hiddenSize: hiddenSize, layerCount: layerCount,
            intermediateSize: intermediateSize, vocabSize: vocabSize,
            attentionHeads: attentionHeads, kvHeads: kvHeads, headDim: headDim,
            attentionBias: attentionBias, mlpBias: mlpBias,
            normEps: normEps, normKind: normKind,
            ropeTheta: ropeTheta, ropeDimension: ropeDimension,
            ropeScaling: ropeScaling, tiedEmbeddings: tiedEmbeddings,
            expertCount: expertCount, expertsPerToken: expertsPerToken,
            moeIntermediateSize: moeIntermediateSize,
            qkNorm: qkNorm,
            fullAttentionInterval: fullAttentionInterval,
            ssmNumHeads: ssmNumHeads, ssmGroupCount: ssmGroupCount,
            ssmKeyHeadDim: ssmKeyHeadDim,
            ssmValueHeadDim: ssmValueHeadDim,
            convKernelSize: convKernelSize, convLCache: convLCache,
            partialRotaryFactor: partialRotaryFactor,
            slidingWindow: slidingWindow, layerTypes: layerTypes,
            hiddenSizePerLayerInput: hiddenSizePerLayerInput,
            vocabSizePerLayerInput: vocabSizePerLayerInput,
            globalHeadDim: globalHeadDim,
            globalKVHeads: globalKVHeads,
            numKVSharedLayers: numKVSharedLayers,
            useDoubleWideMLP: useDoubleWideMLP,
            attentionKEqualsV: attentionKEqualsV,
            fullAttentionRopeTheta: fullAttentionRopeTheta,
            fullAttentionPartialRotaryFactor: fullAttentionPartialRotaryFactor,
            fullAttentionRoPEScaling: fullAttentionRoPEScaling,
            finalLogitSoftcapping: finalLogitSoftcapping,
            numDenseLayers: numDenseLayers,
            mropeAxes: axes
        )
    }

    public init(
        hiddenSize: Int,
        layerCount: Int,
        intermediateSize: Int,
        vocabSize: Int,
        attentionHeads: Int,
        kvHeads: Int,
        headDim: Int,
        attentionBias: Bool,
        mlpBias: Bool,
        normEps: Float,
        normKind: NormKind,
        ropeTheta: Float,
        ropeDimension: Int,
        ropeScaling: RoPEScaling?,
        tiedEmbeddings: Bool,
        expertCount: Int?,
        expertsPerToken: Int?,
        moeIntermediateSize: Int? = nil,
        qkNorm: Bool,
        fullAttentionInterval: Int?,
        ssmNumHeads: Int?,
        ssmGroupCount: Int? = nil,
        ssmKeyHeadDim: Int?,
        ssmValueHeadDim: Int?,
        convKernelSize: Int?,
        convLCache: Int? = nil,
        partialRotaryFactor: Float?,
        slidingWindow: Int?,
        layerTypes: [String]? = nil,
        hiddenSizePerLayerInput: Int? = nil,
        vocabSizePerLayerInput: Int? = nil,
        globalHeadDim: Int? = nil,
        globalKVHeads: Int? = nil,
        numKVSharedLayers: Int? = nil,
        useDoubleWideMLP: Bool = false,
        attentionKEqualsV: Bool = false,
        fullAttentionRopeTheta: Float? = nil,
        fullAttentionPartialRotaryFactor: Float? = nil,
        fullAttentionRoPEScaling: RoPEScaling? = nil,
        finalLogitSoftcapping: Float? = nil,
        numDenseLayers: Int = 0,
        mropeAxes: MRoPEAxes? = nil
    ) {
        self.hiddenSize = hiddenSize
        self.layerCount = layerCount
        self.intermediateSize = intermediateSize
        self.vocabSize = vocabSize
        self.attentionHeads = attentionHeads
        self.kvHeads = kvHeads
        self.headDim = headDim
        self.attentionBias = attentionBias
        self.mlpBias = mlpBias
        self.normEps = normEps
        self.normKind = normKind
        self.ropeTheta = ropeTheta
        self.ropeDimension = ropeDimension
        self.ropeScaling = ropeScaling
        self.tiedEmbeddings = tiedEmbeddings
        self.expertCount = expertCount
        self.expertsPerToken = expertsPerToken
        self.moeIntermediateSize = moeIntermediateSize
        self.qkNorm = qkNorm
        self.fullAttentionInterval = fullAttentionInterval
        self.ssmNumHeads = ssmNumHeads
        self.ssmGroupCount = ssmGroupCount
        self.ssmKeyHeadDim = ssmKeyHeadDim
        self.ssmValueHeadDim = ssmValueHeadDim
        self.convKernelSize = convKernelSize
        self.convLCache = convLCache
        self.partialRotaryFactor = partialRotaryFactor
        self.slidingWindow = slidingWindow
        self.layerTypes = layerTypes
        self.hiddenSizePerLayerInput = hiddenSizePerLayerInput
        self.vocabSizePerLayerInput = vocabSizePerLayerInput
        self.globalHeadDim = globalHeadDim
        self.globalKVHeads = globalKVHeads
        self.numKVSharedLayers = numKVSharedLayers
        self.useDoubleWideMLP = useDoubleWideMLP
        self.attentionKEqualsV = attentionKEqualsV
        self.fullAttentionRopeTheta = fullAttentionRopeTheta
        self.fullAttentionPartialRotaryFactor = fullAttentionPartialRotaryFactor
        self.fullAttentionRoPEScaling = fullAttentionRoPEScaling
        self.finalLogitSoftcapping = finalLogitSoftcapping
        self.numDenseLayers = numDenseLayers
        self.mropeAxes = mropeAxes
    }
}

/// Errors encountered during model graph construction.
public enum ModelGraphBuildError: Error, Sendable, CustomStringConvertible {

    /// A required configuration field is missing.
    case missingMetadata(String)

    /// Configuration values are invalid or inconsistent.
    case invalidConfig(String)

    public var description: String {
        switch self {
        case .missingMetadata(let key):
            return "Missing required configuration: \(key)"
        case .invalidConfig(let msg):
            return "Invalid model configuration: \(msg)"
        }
    }
}
