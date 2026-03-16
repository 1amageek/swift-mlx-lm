/// RMS normalization component.
///
/// ```swift
/// RMSNorm(dimension: 4096)
/// ```
public struct RMSNorm: ModelComponent {

    public typealias Body = Never

    public let dimension: Int
    public let epsilon: Float

    public init(dimension: Int, epsilon: Float = 1e-6) {
        precondition(dimension > 0, "dimension must be positive")
        precondition(epsilon > 0, "epsilon must be positive")
        self.dimension = dimension
        self.epsilon = epsilon
    }
}

extension RMSNorm: PrimitiveComponent {

    package var operationKind: OperationKind {
        .primitive(RMSNormAttributes(dimension: dimension, epsilon: epsilon))
    }

    package var operationSignature: OperationSignature {
        OperationSignature(operandArity: .exact(1), resultArity: .exact(1))
    }
}

/// Layer normalization component.
///
/// ```swift
/// LayerNorm(dimension: 4096)
/// ```
public struct LayerNorm: ModelComponent {

    public typealias Body = Never

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
}

extension LayerNorm: PrimitiveComponent {

    package var operationKind: OperationKind {
        .primitive(LayerNormAttributes(dimension: dimension, epsilon: epsilon, affine: affine))
    }

    package var operationSignature: OperationSignature {
        OperationSignature(operandArity: .exact(1), resultArity: .exact(1))
    }
}
