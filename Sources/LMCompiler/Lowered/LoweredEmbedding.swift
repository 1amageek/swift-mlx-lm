@preconcurrency import MLX

/// Lowered token embedding lookup.
///
/// Supports both dense and quantized embedding tables. For quantized embeddings,
/// row-level dequantization is applied after index lookup (same as `DirectQuantizedEmbedding`
/// in the standard path).
public struct LoweredEmbedding: @unchecked Sendable {

    /// Storage variant determined at compile time.
    public enum Storage: @unchecked Sendable {
        /// Dense F16 embedding table — simple index lookup.
        case dense(MLXArray)
        /// Affine-quantized embedding table — index + dequantize.
        case quantized(AffineQuantizedTensor)
    }

    public let storage: Storage

    /// Initialize from a dense embedding table.
    public init(table: MLXArray) {
        self.storage = .dense(table)
    }

    /// Initialize from an affine-quantized embedding table.
    public init(quantized: AffineQuantizedTensor) {
        self.storage = .quantized(quantized)
    }

    /// Look up embeddings for the given token IDs.
    public func apply(_ tokenIDs: MLXArray) -> MLXArray {
        switch storage {
        case .dense(let table):
            return table[tokenIDs]

        case .quantized(let qt):
            let shape = tokenIDs.shape
            let indices = tokenIDs.flattened()
            let out = dequantized(
                qt.packedWeight[indices],
                scales: qt.scales[indices],
                biases: qt.zeroBiases[indices],
                groupSize: qt.groupSize,
                bits: qt.bits,
                mode: .affine
            )
            return out.reshaped(shape + [-1])
        }
    }
}
