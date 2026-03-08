/// Descriptor for a single tensor stored in a GGUF file.
///
/// Contains the tensor's name, shape, quantization type, and byte offset
/// within the tensor data section. The actual tensor data is not loaded
/// by the parser — it can be accessed later via the offset.
public struct GGUFTensorInfo: Sendable, Equatable {
    /// Tensor name (e.g., "blk.0.attn_q.weight").
    public let name: String

    /// Tensor dimensions, innermost first.
    /// For a 2D weight matrix [out_features, in_features],
    /// dimensions is [in_features, out_features].
    public let dimensions: [Int]

    /// Quantization type of the stored data.
    public let quantizationType: GGUFQuantizationType

    /// Byte offset from the start of the tensor data section.
    public let offset: UInt64

    public init(
        name: String,
        dimensions: [Int],
        quantizationType: GGUFQuantizationType,
        offset: UInt64
    ) {
        self.name = name
        self.dimensions = dimensions
        self.quantizationType = quantizationType
        self.offset = offset
    }

    /// Total number of elements in this tensor.
    public var elementCount: Int {
        dimensions.reduce(1, *)
    }

    /// Total size of the tensor data in bytes.
    public var dataSize: Int {
        let blocks = (elementCount + quantizationType.elementsPerBlock - 1) / quantizationType.elementsPerBlock
        return blocks * quantizationType.blockSize
    }
}
