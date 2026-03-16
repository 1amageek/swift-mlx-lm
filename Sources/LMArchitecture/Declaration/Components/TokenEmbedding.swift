/// Token embedding component.
///
/// Maps discrete token IDs to dense embedding vectors.
///
/// ```swift
/// TokenEmbedding(vocabSize: 32000, embeddingSize: 4096)
/// ```
public struct TokenEmbedding: ModelComponent {

    public typealias Body = Never

    public let vocabSize: Int
    public let embeddingSize: Int
    public let dtypeHint: DTypeHint?

    public init(
        vocabSize: Int,
        embeddingSize: Int,
        dtypeHint: DTypeHint? = nil
    ) {
        precondition(vocabSize > 0, "vocabSize must be positive")
        precondition(embeddingSize > 0, "embeddingSize must be positive")
        self.vocabSize = vocabSize
        self.embeddingSize = embeddingSize
        self.dtypeHint = dtypeHint
    }
}

extension TokenEmbedding: PrimitiveComponent {

    package var operationKind: OperationKind {
        .primitive(TokenEmbeddingAttributes(
            vocabSize: vocabSize,
            embeddingSize: embeddingSize,
            dtypeHint: dtypeHint
        ))
    }

    package var operationSignature: OperationSignature {
        OperationSignature(operandArity: .exact(0), resultArity: .exact(1))
    }
}
