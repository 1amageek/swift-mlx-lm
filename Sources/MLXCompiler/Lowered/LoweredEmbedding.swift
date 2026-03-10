@preconcurrency import MLX

/// Lowered token embedding lookup.
///
/// Simple table lookup: `table[tokenIDs]`. The embedding table is always
/// stored as a dense `MLXArray` (embeddings are not quantized at the storage level).
public struct LoweredEmbedding: @unchecked Sendable {

    /// Embedding weight table of shape `[vocabSize, embeddingDim]`.
    public let table: MLXArray

    public init(table: MLXArray) {
        self.table = table
    }

    /// Look up embeddings for the given token IDs.
    public func apply(_ tokenIDs: MLXArray) -> MLXArray {
        table[tokenIDs]
    }
}
