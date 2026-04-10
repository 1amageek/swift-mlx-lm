/// Attributes for a multi-head attention node.
///
/// This is a semantic unit representing the full attention operation:
/// Q/K/V projections, scaled dot-product attention, and output projection.
public struct AttentionAttributes: OperationAttributes, Codable, Equatable {

    /// Hidden dimension of the model.
    public let hiddenSize: Int

    /// Number of query attention heads.
    public let headCount: Int

    /// Number of key/value attention heads (for GQA/MQA).
    public let kvHeadCount: Int

    /// Dimension of each attention head.
    public let headDimension: Int

    /// Optional override for attention score scaling.
    ///
    /// When nil, the executor uses the backend default `1 / sqrt(headDimension)`.
    public let attentionScale: Float?

    /// Whether projections include bias terms.
    public let bias: Bool

    /// Whether attention is causal (autoregressive masking).
    public let causal: Bool

    /// RoPE configuration. Presence indicates RoPE usage.
    public let rope: RoPEAttributes?

    /// QK normalization strategy.
    public let qkNorm: QKNormKind?

    /// Optional value normalization strategy.
    public let valueNorm: AttentionValueNormKind?

    /// How the raw V tensor is sourced before value normalization.
    public let valueProjectionSource: AttentionValueProjectionSource

    /// Sliding window attention configuration.
    public let window: AttentionWindow?

    /// Optional hint for the executor about preferred implementation.
    public let implementationHint: AttentionImplementationHint?

    /// Optional output gate applied after SDPA + output projection.
    public let outputGate: AttentionGateKind?

    /// Optional source layer index whose KV cache should be reused.
    ///
    /// Used by Gemma 4 KV-shared layers, which only compute queries and
    /// attend against the last non-shared layer of the same attention type.
    public let sharedKeyValueSourceLayerIndex: Int?

    public init(
        hiddenSize: Int,
        headCount: Int,
        kvHeadCount: Int,
        headDimension: Int,
        attentionScale: Float? = nil,
        bias: Bool = false,
        causal: Bool = true,
        rope: RoPEAttributes? = nil,
        qkNorm: QKNormKind? = nil,
        valueNorm: AttentionValueNormKind? = nil,
        valueProjectionSource: AttentionValueProjectionSource = .dedicatedProjection,
        window: AttentionWindow? = nil,
        implementationHint: AttentionImplementationHint? = nil,
        outputGate: AttentionGateKind? = nil,
        sharedKeyValueSourceLayerIndex: Int? = nil
    ) {
        self.hiddenSize = hiddenSize
        self.headCount = headCount
        self.kvHeadCount = kvHeadCount
        self.headDimension = headDimension
        self.attentionScale = attentionScale
        self.bias = bias
        self.causal = causal
        self.rope = rope
        self.qkNorm = qkNorm
        self.valueNorm = valueNorm
        self.valueProjectionSource = valueProjectionSource
        self.window = window
        self.implementationHint = implementationHint
        self.outputGate = outputGate
        self.sharedKeyValueSourceLayerIndex = sharedKeyValueSourceLayerIndex
    }
}

/// QK normalization strategy applied before attention score computation.
public enum QKNormKind: Codable, Equatable, Sendable {
    case none
    case rmsNorm
    case rmsNormUnitOffset
    case layerNorm
    case custom(String)
}

/// Sliding window attention configuration.
public struct AttentionWindow: Codable, Equatable, Sendable {

    /// Number of tokens to attend to on the left.
    public let left: Int?

    /// Number of tokens to attend to on the right.
    public let right: Int?

    public init(left: Int? = nil, right: Int? = nil) {
        self.left = left
        self.right = right
    }
}

/// Hint for the executor about preferred attention implementation.
///
/// This does not affect model semantics or canonical equivalence.
public enum AttentionImplementationHint: Codable, Equatable, Sendable {
    case unspecified
    case fused
    case unfused
    case pagedKVPreferred
    case custom(String)
}

/// Output gate kind for gated attention mechanisms.
///
/// Some architectures (e.g., Qwen 3.5 full attention layers) apply a
/// sigmoid gate to the attention output. The gate values are packed
/// into the Q projection output (2x dimension).
public enum AttentionGateKind: Codable, Equatable, Sendable {

    /// Sigmoid gate packed in Q projection output.
    ///
    /// The Q projection outputs `2 * headCount * headDimension` values.
    /// The first half is queries, the second half is gate values.
    /// Gate is applied as: `output = sigmoid(gate) * attn_output`.
    case sigmoidPackedInQProj
}

/// Optional normalization applied to V projections before attention.
public enum AttentionValueNormKind: Codable, Equatable, Sendable {
    /// Per-head RMS normalization without a learned scale.
    case rmsNormNoScale
}

/// How the raw V path is produced before value normalization.
public enum AttentionValueProjectionSource: Codable, Equatable, Sendable {
    /// Use a dedicated V projection.
    case dedicatedProjection
    /// Reuse the raw K projection for the V path.
    case keyProjection
}
