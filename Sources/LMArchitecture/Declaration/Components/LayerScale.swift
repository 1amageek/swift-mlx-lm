/// Learned scalar multiply applied to the current hidden state.
public struct LayerScale: ModelComponent {

    public typealias Body = Never

    public let dimension: Int

    public init(dimension: Int) {
        precondition(dimension > 0, "dimension must be positive")
        self.dimension = dimension
    }
}

extension LayerScale: PrimitiveComponent {

    package var operationKind: OperationKind {
        .primitive(LayerScaleAttributes(dimension: dimension))
    }

    package var operationSignature: OperationSignature {
        OperationSignature(operandArity: .exact(1), resultArity: .exact(1))
    }
}
