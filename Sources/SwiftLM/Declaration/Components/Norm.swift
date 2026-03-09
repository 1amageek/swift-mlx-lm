/// RMS normalization component.
///
/// ```swift
/// RMSNorm(dimension: 4096)
/// ```
public struct RMSNorm: PrimitiveModelComponent {

    public let dimension: Int
    public let epsilon: Float

    public init(dimension: Int, epsilon: Float = 1e-6) {
        precondition(dimension > 0, "dimension must be positive")
        precondition(epsilon > 0, "epsilon must be positive")
        self.dimension = dimension
        self.epsilon = epsilon
    }

    public func makeDeclaration() -> ModelDeclaration {
        .primitive(.rmsNorm(RMSNormAttributes(dimension: dimension, epsilon: epsilon)))
    }
}

/// Layer normalization component.
///
/// ```swift
/// LayerNorm(dimension: 4096)
/// ```
public struct LayerNorm: PrimitiveModelComponent {

    public let dimension: Int
    public let epsilon: Float
    public let affine: Bool

    public init(dimension: Int, epsilon: Float = 1e-5, affine: Bool = true) {
        precondition(dimension > 0, "dimension must be positive")
        precondition(epsilon > 0, "epsilon must be positive")
        self.dimension = dimension
        self.epsilon = epsilon
        self.affine = affine
    }

    public func makeDeclaration() -> ModelDeclaration {
        .primitive(.layerNorm(LayerNormAttributes(
            dimension: dimension, epsilon: epsilon, affine: affine
        )))
    }
}
