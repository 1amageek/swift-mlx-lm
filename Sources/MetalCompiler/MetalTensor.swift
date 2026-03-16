import Metal

/// A tensor residing in a Metal buffer. MLX-independent.
///
/// Represents weight data loaded from safetensors, directly addressable
/// by Metal kernels via `setBuffer(buffer, offset: offset, index: N)`.
///
/// Supports both dense and quantized representations.
/// For quantized tensors, scales and zeros are stored at known offsets
/// within the same (or related) MTLBuffer.
public struct MetalTensor: @unchecked Sendable {

    /// The MTLBuffer containing this tensor's data.
    public let buffer: MTLBuffer

    /// Byte offset within the buffer.
    public let offset: Int

    /// Tensor shape (e.g., [outFeatures, inFeatures] for a weight matrix).
    public let shape: [Int]

    /// Data type of the tensor elements.
    public let dtype: MetalTensorDType

    public init(buffer: MTLBuffer, offset: Int, shape: [Int], dtype: MetalTensorDType) {
        self.buffer = buffer
        self.offset = offset
        self.shape = shape
        self.dtype = dtype
    }

    /// Total number of logical elements.
    public var elementCount: Int {
        shape.reduce(1, *)
    }

    /// Byte size of one element (for dense types).
    /// For quantized types, use `dtype.bytesPerElement`.
    public var elementSize: Int {
        dtype.bytesPerElement
    }
}

/// Data type for MetalTensor elements.
public enum MetalTensorDType: Sendable, Equatable {
    /// IEEE 754 half-precision float (2 bytes).
    case float16
    /// Brain floating point (2 bytes).
    case bfloat16
    /// IEEE 754 single-precision float (4 bytes).
    case float32

    /// Affine quantized: packed integers with per-group scales and zeros.
    /// The kernel dequantizes on-the-fly: `val = packed_nibble * scale + zero`
    case quantized(QuantizationDescriptor)

    /// Bytes per logical element.
    /// For quantized types, returns the average bytes per element.
    public var bytesPerElement: Int {
        switch self {
        case .float16, .bfloat16: return 2
        case .float32: return 4
        case .quantized(let q): return max(1, q.bits / 8)  // approximate
        }
    }

    /// Whether this type requires a quantized GEMV kernel.
    public var isQuantized: Bool {
        if case .quantized = self { return true }
        return false
    }
}

/// Describes the quantization format for a weight tensor.
///
/// Matches the MLX affine quantization format used by mlx-community models:
/// - `packedWeight`: uint32 values, each containing `32/bits` quantized elements
/// - `scales`: float16, shape [N, numGroups]
/// - `zeros`: float16, shape [N, numGroups]
/// - Dequant formula: `val = packed_nibble * scale + zero`
public struct QuantizationDescriptor: Sendable, Equatable {
    /// Bits per quantized element (2, 3, 4, 5, 6, or 8).
    public let bits: Int

    /// Number of elements per quantization group (32, 64, 128).
    public let groupSize: Int

    /// The scales tensor for this weight.
    public let scales: MetalTensorRef

    /// The zero-point biases tensor for this weight.
    public let zeros: MetalTensorRef

    public init(bits: Int, groupSize: Int, scales: MetalTensorRef, zeros: MetalTensorRef) {
        self.bits = bits
        self.groupSize = groupSize
        self.scales = scales
        self.zeros = zeros
    }
}

/// A lightweight reference to a tensor region in a Metal buffer.
/// Used for quantization metadata (scales, zeros) that may reside
/// in the same or a different MTLBuffer as the packed weight.
public struct MetalTensorRef: @unchecked Sendable, Equatable {
    public let buffer: MTLBuffer
    public let offset: Int

    public init(buffer: MTLBuffer, offset: Int) {
        self.buffer = buffer
        self.offset = offset
    }

    public static func == (lhs: MetalTensorRef, rhs: MetalTensorRef) -> Bool {
        lhs.buffer === rhs.buffer && lhs.offset == rhs.offset
    }
}
