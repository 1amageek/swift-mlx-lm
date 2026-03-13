import SwiftLM

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
    public let partialRotaryFactor: Float?

    // MARK: Sliding Window

    public let slidingWindow: Int?

    // MARK: Per-Layer Schedule

    /// Per-layer type schedule (e.g., ["linear_attention", ..., "full_attention"]).
    /// Nil when all layers are the same type (standard transformer, MoE, etc.).
    /// When present, takes precedence over `fullAttentionInterval`.
    public let layerTypes: [String]?

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
            qkNorm: qkNorm,
            fullAttentionInterval: fullAttentionInterval,
            ssmNumHeads: ssmNumHeads, ssmGroupCount: ssmGroupCount,
            ssmKeyHeadDim: ssmKeyHeadDim,
            ssmValueHeadDim: ssmValueHeadDim,
            convKernelSize: convKernelSize, partialRotaryFactor: partialRotaryFactor,
            slidingWindow: slidingWindow, layerTypes: layerTypes,
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
        qkNorm: Bool,
        fullAttentionInterval: Int?,
        ssmNumHeads: Int?,
        ssmGroupCount: Int? = nil,
        ssmKeyHeadDim: Int?,
        ssmValueHeadDim: Int?,
        convKernelSize: Int?,
        partialRotaryFactor: Float?,
        slidingWindow: Int?,
        layerTypes: [String]? = nil,
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
        self.qkNorm = qkNorm
        self.fullAttentionInterval = fullAttentionInterval
        self.ssmNumHeads = ssmNumHeads
        self.ssmGroupCount = ssmGroupCount
        self.ssmKeyHeadDim = ssmKeyHeadDim
        self.ssmValueHeadDim = ssmValueHeadDim
        self.convKernelSize = convKernelSize
        self.partialRotaryFactor = partialRotaryFactor
        self.slidingWindow = slidingWindow
        self.layerTypes = layerTypes
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
