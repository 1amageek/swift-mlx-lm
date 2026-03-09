/// Attributes for an RMS normalization node.
public struct RMSNormAttributes: Codable, Equatable, Sendable {

    /// Dimension of the normalized input.
    public let dimension: Int

    /// Epsilon value for numerical stability.
    public let epsilon: Float

    public init(dimension: Int, epsilon: Float = 1e-6) {
        self.dimension = dimension
        self.epsilon = epsilon
    }
}

/// Attributes for a layer normalization node.
public struct LayerNormAttributes: Codable, Equatable, Sendable {

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
