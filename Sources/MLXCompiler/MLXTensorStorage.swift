import MLX

/// Provenance tag for quantized tensors — diagnostic use only.
///
/// Records how the quantized representation was created, without importing
/// GGUFParser (which lives in the MLXLM module).
public enum QuantizationOrigin: String, Sendable {
    case ggufQ4_0, ggufQ4_1, ggufQ4_K, ggufQ5_K, ggufQ6_K, ggufQ8_0, ggufQ8_1
    case ggufQ5_0, ggufQ5_1, ggufQ8_K, ggufQ2_K, ggufQ3_K, ggufTQ2_0
    case ggufIQ4_NL, ggufIQ4_XS
    case ggufIQ2_XXS, ggufIQ2_XS, ggufIQ2_S
    case ggufIQ3_XXS, ggufIQ3_S
    case ggufIQ1_S, ggufIQ1_M
    case ggufTQ1_0
    case mlxQuantized, unknown
}

/// Pre-packed affine quantized tensor ready for `quantizedMatmul`.
///
/// Stores the exact arrays produced by GGUF direct packing (or MLX `quantize()`).
/// These are backend payload — `scales` and `zeroBiases` are NOT semantic model
/// parameters (they describe the quantization encoding, not learned biases).
public struct AffineQuantizedTensor: @unchecked Sendable {

    /// Logical weight shape before packing: `[outFeatures, inFeatures]`.
    public let logicalShape: [Int]

    /// Packed weight data (UInt32).
    public let packedWeight: MLXArray

    /// Per-group quantization scales.
    public let scales: MLXArray

    /// Per-group quantization zero-point biases.
    public let zeroBiases: MLXArray

    /// Number of elements per quantization group (16, 32, or 64).
    public let groupSize: Int

    /// Quantization bit width (2, 3, 4, 5, 6, or 8).
    public let bits: Int

    /// How this quantized tensor was created.
    public let origin: QuantizationOrigin

    public init(
        logicalShape: [Int],
        packedWeight: MLXArray,
        scales: MLXArray,
        zeroBiases: MLXArray,
        groupSize: Int,
        bits: Int,
        origin: QuantizationOrigin
    ) {
        self.logicalShape = logicalShape
        self.packedWeight = packedWeight
        self.scales = scales
        self.zeroBiases = zeroBiases
        self.groupSize = groupSize
        self.bits = bits
        self.origin = origin
    }
}

/// Backend-native weight representation that preserves quantization metadata.
///
/// Unlike `MLXWeightStore` which stores bare `MLXArray` (losing quantization info),
/// `MLXTensorStorage` carries the full representation needed for compile-time
/// kernel selection (`matmul` vs `quantizedMatmul`).
public enum MLXTensorStorage: @unchecked Sendable {

    /// Dense (unquantized) weight as a single MLXArray.
    case dense(MLXArray)

    /// Affine-quantized weight with pre-packed data for `quantizedMatmul`.
    case affineQuantized(AffineQuantizedTensor)
}
