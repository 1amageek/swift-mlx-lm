/// Token embedding component.
///
/// Maps discrete token IDs to dense embedding vectors.
///
/// ```swift
/// TokenEmbedding(vocabSize: 32000, embeddingSize: 4096)
/// ```
public struct TokenEmbedding: PrimitiveModelComponent {

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

    public func makeDeclaration() -> ModelDeclaration {
        .primitive(.tokenEmbedding(TokenEmbeddingAttributes(
            vocabSize: vocabSize,
            embeddingSize: embeddingSize,
            dtypeHint: dtypeHint
        )))
    }
}
