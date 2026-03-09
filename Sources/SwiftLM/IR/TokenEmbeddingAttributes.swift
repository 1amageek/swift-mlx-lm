/// Attributes for a token embedding node.
///
/// Maps discrete token IDs to dense vectors.
public struct TokenEmbeddingAttributes: Codable, Equatable, Sendable {

    /// Size of the vocabulary (number of distinct tokens).
    public let vocabSize: Int

    /// Dimensionality of each embedding vector.
    public let embeddingSize: Int

    /// Optional dtype hint for the embedding table.
    public let dtypeHint: DTypeHint?

    public init(
        vocabSize: Int,
        embeddingSize: Int,
        dtypeHint: DTypeHint? = nil
    ) {
        self.vocabSize = vocabSize
        self.embeddingSize = embeddingSize
        self.dtypeHint = dtypeHint
    }
}
