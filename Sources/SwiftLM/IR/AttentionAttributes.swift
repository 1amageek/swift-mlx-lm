/// Attributes for a multi-head attention node.
///
/// This is a semantic unit representing the full attention operation:
/// Q/K/V projections, scaled dot-product attention, and output projection.
/// The compiler lowers this into individual operations in the LoweredGraph.
public struct AttentionAttributes: Codable, Equatable, Sendable {

    /// Hidden dimension of the model.
    public let hiddenSize: Int

    /// Number of query attention heads.
    public let headCount: Int

    /// Number of key/value attention heads (for GQA/MQA).
    public let kvHeadCount: Int

    /// Dimension of each attention head.
    public let headDimension: Int

    /// Whether projections include bias terms.
    public let bias: Bool

    /// Whether attention is causal (autoregressive masking).
    public let causal: Bool

    /// RoPE configuration. Presence indicates RoPE usage.
    public let rope: RoPEAttributes?

    /// QK normalization strategy.
    public let qkNorm: QKNormKind?

    /// Sliding window attention configuration.
    public let window: AttentionWindow?

    /// Optional hint for the executor about preferred implementation.
    public let implementationHint: AttentionImplementationHint?

    public init(
        hiddenSize: Int,
        headCount: Int,
        kvHeadCount: Int,
        headDimension: Int,
        bias: Bool = false,
        causal: Bool = true,
        rope: RoPEAttributes? = nil,
        qkNorm: QKNormKind? = nil,
        window: AttentionWindow? = nil,
        implementationHint: AttentionImplementationHint? = nil
    ) {
        self.hiddenSize = hiddenSize
        self.headCount = headCount
        self.kvHeadCount = kvHeadCount
        self.headDimension = headDimension
        self.bias = bias
        self.causal = causal
        self.rope = rope
        self.qkNorm = qkNorm
        self.window = window
        self.implementationHint = implementationHint
    }
}

/// QK normalization strategy applied before attention score computation.
public enum QKNormKind: Codable, Equatable, Sendable {
    case none
    case rmsNorm
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
