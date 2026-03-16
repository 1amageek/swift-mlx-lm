/// Positional embedding component.
///
/// Adds position information to token vectors.
///
/// ```swift
/// PositionalEmbedding(maxPositions: 2048, embeddingSize: 4096, kind: .sinusoidal)
/// ```
public struct PositionalEmbedding: ModelComponent {

    public typealias Body = Never

    public let maxPositions: Int
    public let embeddingSize: Int
    public let kind: PositionalEmbeddingKind

    public init(
        maxPositions: Int,
        embeddingSize: Int,
        kind: PositionalEmbeddingKind
    ) {
        precondition(maxPositions > 0, "maxPositions must be positive")
        precondition(embeddingSize > 0, "embeddingSize must be positive")
        self.maxPositions = maxPositions
        self.embeddingSize = embeddingSize
        self.kind = kind
    }
}

extension PositionalEmbedding: PrimitiveComponent {

    package var operationKind: OperationKind {
        .primitive(PositionalEmbeddingAttributes(
            maxPositions: maxPositions,
            embeddingSize: embeddingSize,
            kind: kind
        ))
    }

    package var operationSignature: OperationSignature {
        OperationSignature(operandArity: .exact(1), resultArity: .exact(1))
    }
}
