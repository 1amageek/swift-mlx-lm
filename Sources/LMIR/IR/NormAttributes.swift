/// Attributes for an RMS normalization node.
public struct RMSNormAttributes: OperationAttributes, Codable, Equatable {

    /// Dimension of the normalized input.
    public let dimension: Int

    /// Epsilon value for numerical stability.
    public let epsilon: Float

    /// Optional additive bias applied to the learned scale before multiplication.
    ///
    /// Some families, such as Qwen 3.5 text RMSNorm, store zero-centered
    /// weights and apply `1 + weight` at runtime.
    public let weightBias: Float

    public init(dimension: Int, epsilon: Float = 1e-6, weightBias: Float = 0) {
        self.dimension = dimension
        self.epsilon = epsilon
        self.weightBias = weightBias
    }
}

/// Attributes for a layer normalization node.
public struct LayerNormAttributes: OperationAttributes, Codable, Equatable {

    /// Dimension of the normalized input.
    public let dimension: Int

    /// Epsilon value for numerical stability.
    public let epsilon: Float

    /// Whether learnable affine parameters (scale/bias) are used.
    public let affine: Bool

    public init(dimension: Int, epsilon: Float = 1e-5, affine: Bool = true) {
        self.dimension = dimension
        self.epsilon = epsilon
        self.affine = affine
    }
}
