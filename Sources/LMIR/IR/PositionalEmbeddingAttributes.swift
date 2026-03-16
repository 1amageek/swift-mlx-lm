/// Attributes for a positional embedding node.
public struct PositionalEmbeddingAttributes: OperationAttributes, Codable, Equatable, Sendable {

    /// Maximum number of positions supported.
    public let maxPositions: Int

    /// Dimensionality of each positional vector.
    public let embeddingSize: Int

    /// Kind of positional embedding.
    public let kind: PositionalEmbeddingKind

    public init(
        maxPositions: Int,
        embeddingSize: Int,
        kind: PositionalEmbeddingKind
    ) {
        self.maxPositions = maxPositions
        self.embeddingSize = embeddingSize
        self.kind = kind
    }
}

/// Kind of positional embedding.
public enum PositionalEmbeddingKind: Codable, Equatable, Sendable {
    case learnedAbsolute
    case sinusoidal
}
