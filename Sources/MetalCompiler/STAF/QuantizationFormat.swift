/// Declares a quantization block format for GPU execution.
///
/// Each concrete struct represents one quantization scheme.
/// The compiler reads protocol properties to select the correct GEMV kernel.
/// Adding a new quantization format = adding a new struct.
public protocol QuantizationFormat: Sendable {
    /// STAF scheme identifier (maps to `quant_scheme_id` in file).
    var schemeIdentifier: QuantizationSchemeIdentifier { get }
    /// MSL block struct name (e.g., "BlockQ4Affine").
    var blockStructName: String { get }
    /// GEMV kernel function name for this format (decode, single token).
    var gemvKernelName: String { get }
    /// GEMM kernel function name for this format (prefill, sequence).
    func gemmKernelName(bufferPrecision: BufferPrecision) -> String
    /// Number of weights per block.
    var weightsPerBlock: Int { get }
    /// Byte size of one block.
    var bytesPerBlock: Int { get }
    /// Bit width of quantized values.
    var bits: Int { get }
    /// Group size (weights per scale value).
    var groupSize: Int { get }
}

// MARK: - Dense Formats

public struct Float16Format: QuantizationFormat {
    public var schemeIdentifier: QuantizationSchemeIdentifier { .fp16RowMajor }
    public var blockStructName: String { "" }
    public var gemvKernelName: String { "gemv" }
    public func gemmKernelName(bufferPrecision: BufferPrecision) -> String {
        bufferPrecision == .float32 ? "gemm_f32s" : "gemm"
    }
    public var weightsPerBlock: Int { 1 }
    public var bytesPerBlock: Int { 2 }
    public var bits: Int { 16 }
    public var groupSize: Int { 1 }

    public init() {}
}

public struct BFloat16Format: QuantizationFormat {
    public var schemeIdentifier: QuantizationSchemeIdentifier { .bf16RowMajor }
    public var blockStructName: String { "" }
    public var gemvKernelName: String { "gemv_bf16" }
    public func gemmKernelName(bufferPrecision: BufferPrecision) -> String {
        bufferPrecision == .float32 ? "gemm_bf16_f32s" : "gemm_bf16"
    }
    public var weightsPerBlock: Int { 1 }
    public var bytesPerBlock: Int { 2 }
    public var bits: Int { 16 }
    public var groupSize: Int { 1 }

    public init() {}
}

public struct Float32Format: QuantizationFormat {
    public var schemeIdentifier: QuantizationSchemeIdentifier { .fp32RowMajor }
    public var blockStructName: String { "" }
    public var gemvKernelName: String { "gemv_f32" }
    public func gemmKernelName(bufferPrecision: BufferPrecision) -> String {
        "gemm_f32s"
    }
    public var weightsPerBlock: Int { 1 }
    public var bytesPerBlock: Int { 4 }
    public var bits: Int { 32 }
    public var groupSize: Int { 1 }

    public init() {}
}

// MARK: - INT4 Affine Formats

/// 4-bit affine quantization with group size 64.
///
/// Block layout (interleaved):
/// ```
/// ┌──────────┬──────────┬──────────────────────────┐
/// │scale (2B)│ zero (2B)│ packed quants (32B)       │
/// └──────────┴──────────┴──────────────────────────┘
/// 36 bytes per block, 64 weights
/// ```
///
/// Dequantization: `w = scale * q + zero`
/// where q is 4-bit unsigned integer (0..15)
public struct AffineQ4Group64Format: QuantizationFormat {
    public var schemeIdentifier: QuantizationSchemeIdentifier { .q4Group64ScaleF16 }
    public var blockStructName: String { "BlockQ4Affine64" }
    public var gemvKernelName: String { "gemv_q4_g64" }
    public func gemmKernelName(bufferPrecision: BufferPrecision) -> String {
        "gemv_q4_g64"
    }
    public var weightsPerBlock: Int { 64 }
    public var bytesPerBlock: Int { 4 + 32 }  // scale(2) + zero(2) + 64*4bit/8
    public var bits: Int { 4 }
    public var groupSize: Int { 64 }

    public init() {}
}

/// 4-bit affine quantization with group size 128.
///
/// Block layout:
/// ```
/// ┌──────────┬──────────┬──────────────────────────┐
/// │scale (2B)│ zero (2B)│ packed quants (64B)       │
/// └──────────┴──────────┴──────────────────────────┘
/// 68 bytes per block, 128 weights
/// ```
public struct AffineQ4Group128Format: QuantizationFormat {
    public var schemeIdentifier: QuantizationSchemeIdentifier { .q4Group128ScaleF16 }
    public var blockStructName: String { "BlockQ4Affine128" }
    public var gemvKernelName: String { "gemv_q4_g128" }
    public func gemmKernelName(bufferPrecision: BufferPrecision) -> String {
        "gemv_q4_g128"
    }
    public var weightsPerBlock: Int { 128 }
    public var bytesPerBlock: Int { 4 + 64 }
    public var bits: Int { 4 }
    public var groupSize: Int { 128 }

    public init() {}
}

// MARK: - INT8 Affine Formats

public struct AffineQ8Group32Format: QuantizationFormat {
    public var schemeIdentifier: QuantizationSchemeIdentifier { .q8Group32ScaleF16 }
    public var blockStructName: String { "BlockQ8Affine32" }
    public var gemvKernelName: String { "gemv_q8_g32" }
    public func gemmKernelName(bufferPrecision: BufferPrecision) -> String {
        bufferPrecision == .float32 ? "gemm_q8_g32_f32s" : "gemm_q8_g32"
    }
    public var weightsPerBlock: Int { 32 }
    public var bytesPerBlock: Int { 4 + 32 }  // scale(2) + zero(2) + 32 bytes
    public var bits: Int { 8 }
    public var groupSize: Int { 32 }

    public init() {}
}

public struct AffineQ8Group64Format: QuantizationFormat {
    public var schemeIdentifier: QuantizationSchemeIdentifier { .q8Group64ScaleF16 }
    public var blockStructName: String { "BlockQ8Affine64" }
    public var gemvKernelName: String { "gemv_q8_g64" }
    public func gemmKernelName(bufferPrecision: BufferPrecision) -> String {
        bufferPrecision == .float32 ? "gemm_q8_g64_f32s" : "gemm_q8_g64"
    }
    public var weightsPerBlock: Int { 64 }
    public var bytesPerBlock: Int { 4 + 64 }
    public var bits: Int { 8 }
    public var groupSize: Int { 64 }

    public init() {}
}

// MARK: - Format Registry

/// Resolves a `QuantizationSchemeIdentifier` to its corresponding `QuantizationFormat`.
public enum QuantizationFormatRegistry {

    /// All known formats, keyed by scheme identifier.
    public static func format(
        for identifier: QuantizationSchemeIdentifier
    ) -> (any QuantizationFormat)? {
        switch identifier {
        case .fp16RowMajor: return Float16Format()
        case .bf16RowMajor: return BFloat16Format()
        case .fp32RowMajor: return Float32Format()
        case .q4Group64ScaleF16: return AffineQ4Group64Format()
        case .q4Group128ScaleF16: return AffineQ4Group128Format()
        case .q8Group32ScaleF16: return AffineQ8Group32Format()
        case .q8Group64ScaleF16: return AffineQ8Group64Format()
        case .passthrough: return Float16Format()
        default: return nil
        }
    }

    /// Determine the best quantization format for an MLX quantized tensor.
    public static func formatForMLXQuantization(
        bits: Int, groupSize: Int
    ) -> (any QuantizationFormat)? {
        switch (bits, groupSize) {
        case (4, 64): return AffineQ4Group64Format()
        case (4, 128): return AffineQ4Group128Format()
        case (8, 32): return AffineQ8Group32Format()
        case (8, 64): return AffineQ8Group64Format()
        default: return nil
        }
    }
}
