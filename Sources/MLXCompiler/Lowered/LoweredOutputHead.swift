@preconcurrency import MLX

/// Lowered output projection (language model head).
///
/// For tied output heads, the projection uses the embedding table weight.
/// The `isTied` flag is informational — the kernel already contains the
/// correct weight at compile time.
public struct LoweredOutputHead: @unchecked Sendable {

    /// The output projection (may use embedding table weight when tied).
    public let projection: LoweredProjection

    /// Whether this head is tied to the embedding table.
    public let isTied: Bool

    public init(projection: LoweredProjection, isTied: Bool) {
        self.projection = projection
        self.isTied = isTied
    }

    /// Apply the output projection to get logits.
    public func apply(_ x: MLXArray) -> MLXArray {
        projection.apply(x)
    }
}
