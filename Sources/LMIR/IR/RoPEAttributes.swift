/// Attributes for a rotary position embedding node.
public struct RoPEAttributes: OperationAttributes, Codable, Equatable {

    /// Dimensionality of the rotary embedding.
    public let dimension: Int

    /// Base frequency (theta) for rotation computation.
    public let base: Float

    /// Optional scaling configuration for extended context.
    public let scaling: RoPEScaling?

    /// Optional multi-axis RoPE configuration for vision-language models.
    ///
    /// When `nil`, standard 1-axis RoPE is used (text-only models).
    /// When set, the rotary dimensions are partitioned across multiple
    /// positional axes (temporal, height, width) for M-RoPE.
    public let mropeAxes: MRoPEAxes?

    public init(
        dimension: Int,
        base: Float = 10_000.0,
        scaling: RoPEScaling? = nil,
        mropeAxes: MRoPEAxes? = nil
    ) {
        self.dimension = dimension
        self.base = base
        self.scaling = scaling
        self.mropeAxes = mropeAxes
    }
}

/// Scaling configuration for RoPE extended context.
public struct RoPEScaling: Codable, Equatable, Sendable {

    /// Kind of scaling applied.
    public let kind: RoPEScalingKind

    /// Scaling factor.
    public let factor: Float

    /// Original max positions before scaling.
    public let originalMaxPositions: Int?

    public init(
        kind: RoPEScalingKind,
        factor: Float,
        originalMaxPositions: Int? = nil
    ) {
        self.kind = kind
        self.factor = factor
        self.originalMaxPositions = originalMaxPositions
    }
}

/// Kind of RoPE scaling.
public enum RoPEScalingKind: Codable, Equatable, Sendable {
    case linear
    case dynamic
    case yarn
    case custom(String)
}
