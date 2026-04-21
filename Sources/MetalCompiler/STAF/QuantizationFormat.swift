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

    // MARK: - Phase 1: unified format classification

    /// Whether this format uses block-packed quantization.
    /// `false` for dense formats (fp16 / bf16 / fp32).
    var isQuantized: Bool { get }

    /// Whether the format is bit-aligned — supports direct shift/mask dequant (2/4/8/16/32 bits).
    /// Non-aligned formats (3/5/6 bits) require group-level expansion.
    var isAligned: Bool { get }

    /// MSL type used for the weight buffer parameter.
    ///
    /// Dense: "half" / "uint16_t" / "float".
    /// Quantized: "uchar" (block-packed byte buffer; the kernel reinterpret-casts to block structs).
    var bufferElementType: String { get }

    // MARK: - Phase 1: MSL generation hooks

    /// MSL source fragment declaring the block struct and inline dequant helpers.
    ///
    /// - Empty for dense formats (no block packing).
    /// - For quantized formats, returns the `struct BlockQxAffineN { ... }` definition
    ///   plus any helper functions used by `perWeightReadExpression`.
    /// - Compiler is expected to deduplicate by `schemeIdentifier` when assembling a program.
    var mslDeclarations: String { get }

    /// Aligned-format per-weight read expression.
    ///
    /// The caller (unified aligned GEMV scaffold) already captured `scale` and `zero`
    /// as local `float` variables for the current block. This hook returns a single
    /// MSL expression that evaluates to the dequantized float weight at
    /// `blocksVar[weightIndexVar]`.
    ///
    /// - `blocksVar`: name of the packed-quant array within the current block
    ///   (for dense, the flat weight array; for Q4/Q8, e.g. "blk.qs").
    /// - `weightIndexVar`: name of the weight index variable within `blocksVar`.
    ///
    /// Returns `nil` for non-aligned formats (Q3/Q5/Q6) — those must use `emitGroupDequant`.
    func perWeightReadExpression(
        blocksVar: String,
        weightIndexVar: String
    ) -> String?

    /// Non-aligned format per-group dequant statement.
    ///
    /// Emits MSL statements that populate `outputArrayVar[0..<groupSize]` with dequantized
    /// floats for block `blockIndexVar`. `blocksVar` names the typed block array pointer.
    ///
    /// Returns `nil` for aligned formats — those use `perWeightReadExpression` instead.
    func emitGroupDequant(
        blocksVar: String,
        blockIndexVar: String,
        outputArrayVar: String
    ) -> String?
}

// MARK: - Protocol default implementations

public extension QuantizationFormat {
    /// Default: aligned iff bits is one of {2, 4, 8, 16, 32}.
    /// Non-aligned formats (Q3=3, Q5=5, Q6=6) override and return `false` explicitly,
    /// but the default covers them via this branch too.
    var isAligned: Bool {
        switch bits {
        case 2, 4, 8, 16, 32: return true
        default: return false
        }
    }

    /// Default: no group-level dequant. Aligned formats only use per-weight reads.
    func emitGroupDequant(
        blocksVar: String,
        blockIndexVar: String,
        outputArrayVar: String
    ) -> String? { nil }

    /// Default: no per-weight read. Non-aligned formats only use group dequant.
    func perWeightReadExpression(
        blocksVar: String,
        weightIndexVar: String
    ) -> String? { nil }
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

    public var isQuantized: Bool { false }
    public var bufferElementType: String { "half" }
    public var mslDeclarations: String { "" }

    public func perWeightReadExpression(
        blocksVar: String,
        weightIndexVar: String
    ) -> String? {
        "float(\(blocksVar)[\(weightIndexVar)])"
    }

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

    public var isQuantized: Bool { false }
    public var bufferElementType: String { "uint16_t" }
    public var mslDeclarations: String { "" }

    public func perWeightReadExpression(
        blocksVar: String,
        weightIndexVar: String
    ) -> String? {
        "bf16_to_float(\(blocksVar)[\(weightIndexVar)])"
    }

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

    public var isQuantized: Bool { false }
    public var bufferElementType: String { "float" }
    public var mslDeclarations: String { "" }

    public func perWeightReadExpression(
        blocksVar: String,
        weightIndexVar: String
    ) -> String? {
        "(\(blocksVar)[\(weightIndexVar)])"
    }

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
        // Multi-row GEMM kernels (sequence-aware). Decode-only `gemv_q4_g64`
        // must NOT be used for prefill (seqLen>1) — it ignores gid.y.
        bufferPrecision == .float32 ? "gemm_q4_g64_f32s" : "gemm_q4_g64"
    }
    public var weightsPerBlock: Int { 64 }
    public var bytesPerBlock: Int { 4 + 32 }  // scale(2) + zero(2) + 64*4bit/8
    public var bits: Int { 4 }
    public var groupSize: Int { 64 }

    public var isQuantized: Bool { true }
    public var bufferElementType: String { "uchar" }
    public var mslDeclarations: String { Q4AffineMSL.blockStruct(name: blockStructName, weightsPerBlock: weightsPerBlock) }

    public func perWeightReadExpression(
        blocksVar: String,
        weightIndexVar: String
    ) -> String? {
        Q4AffineMSL.perWeightExpression(blocksVar: blocksVar, weightIndexVar: weightIndexVar)
    }

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
        // Multi-row GEMM kernels (sequence-aware). Decode-only `gemv_q4_g128`
        // must NOT be used for prefill (seqLen>1) — it ignores gid.y.
        bufferPrecision == .float32 ? "gemm_q4_g128_f32s" : "gemm_q4_g128"
    }
    public var weightsPerBlock: Int { 128 }
    public var bytesPerBlock: Int { 4 + 64 }
    public var bits: Int { 4 }
    public var groupSize: Int { 128 }

    public var isQuantized: Bool { true }
    public var bufferElementType: String { "uchar" }
    public var mslDeclarations: String { Q4AffineMSL.blockStruct(name: blockStructName, weightsPerBlock: weightsPerBlock) }

    public func perWeightReadExpression(
        blocksVar: String,
        weightIndexVar: String
    ) -> String? {
        Q4AffineMSL.perWeightExpression(blocksVar: blocksVar, weightIndexVar: weightIndexVar)
    }

    public init() {}
}

// MARK: - INT2 Affine Formats

/// 2-bit affine quantization with group size 16.
///
/// Block layout (4 weights per byte):
/// ```
/// ┌──────────┬──────────┬───────────────────┐
/// │scale (2B)│ zero (2B)│ packed quants (4B)│
/// └──────────┴──────────┴───────────────────┘
/// 8 bytes per block, 16 weights
/// ```
///
/// Dequantization: `w = scale * q + zero` where q ∈ [0, 3].
///
/// - Note: catalog wiring (GEMV/GEMM dispatch) is deferred to Phase 5
///   (WeightFormat enum migration). The format struct and unified GEMV
///   generator are available via the protocol registry for Phase 5 to
///   hook up without further protocol changes.
public struct AffineQ2Group16Format: QuantizationFormat {
    public var schemeIdentifier: QuantizationSchemeIdentifier { .q2Group16ScaleF16 }
    public var blockStructName: String { "BlockQ2Affine16" }
    public var gemvKernelName: String { "gemv_q2_g16" }
    public func gemmKernelName(bufferPrecision: BufferPrecision) -> String {
        bufferPrecision == .float32 ? "gemm_q2_g16_f32s" : "gemm_q2_g16"
    }
    public var weightsPerBlock: Int { 16 }
    public var bytesPerBlock: Int { 4 + 4 }  // scale(2) + zero(2) + 16*2bit/8
    public var bits: Int { 2 }
    public var groupSize: Int { 16 }

    public var isQuantized: Bool { true }
    public var bufferElementType: String { "uchar" }
    public var mslDeclarations: String { Q2AffineMSL.blockStruct(name: blockStructName, weightsPerBlock: weightsPerBlock) }

    public func perWeightReadExpression(
        blocksVar: String,
        weightIndexVar: String
    ) -> String? {
        Q2AffineMSL.perWeightExpression(blocksVar: blocksVar, weightIndexVar: weightIndexVar)
    }

    public init() {}
}

/// 2-bit affine quantization with group size 32.
///
/// Block layout (4 weights per byte):
/// ```
/// ┌──────────┬──────────┬───────────────────┐
/// │scale (2B)│ zero (2B)│ packed quants (8B)│
/// └──────────┴──────────┴───────────────────┘
/// 12 bytes per block, 32 weights
/// ```
public struct AffineQ2Group32Format: QuantizationFormat {
    public var schemeIdentifier: QuantizationSchemeIdentifier { .q2Group32ScaleF16 }
    public var blockStructName: String { "BlockQ2Affine32" }
    public var gemvKernelName: String { "gemv_q2_g32" }
    public func gemmKernelName(bufferPrecision: BufferPrecision) -> String {
        bufferPrecision == .float32 ? "gemm_q2_g32_f32s" : "gemm_q2_g32"
    }
    public var weightsPerBlock: Int { 32 }
    public var bytesPerBlock: Int { 4 + 8 }
    public var bits: Int { 2 }
    public var groupSize: Int { 32 }

    public var isQuantized: Bool { true }
    public var bufferElementType: String { "uchar" }
    public var mslDeclarations: String { Q2AffineMSL.blockStruct(name: blockStructName, weightsPerBlock: weightsPerBlock) }

    public func perWeightReadExpression(
        blocksVar: String,
        weightIndexVar: String
    ) -> String? {
        Q2AffineMSL.perWeightExpression(blocksVar: blocksVar, weightIndexVar: weightIndexVar)
    }

    public init() {}
}

// MARK: - 3-bit Affine Formats (non-aligned)

/// 3-bit affine quantization with group size 16.
///
/// Non-aligned packing (MLX `extract_bits<3>`): 8 weights share 3 bytes.
/// Block layout:
/// ```
/// ┌──────────┬──────────┬───────────────────┐
/// │scale (2B)│ zero (2B)│ packed quants (6B)│
/// └──────────┴──────────┴───────────────────┘
/// 10 bytes per block, 16 weights (2 packs of 8)
/// ```
public struct AffineQ3Group16Format: QuantizationFormat {
    public var schemeIdentifier: QuantizationSchemeIdentifier { .q3Group16ScaleF16 }
    public var blockStructName: String { "BlockQ3Affine16" }
    public var gemvKernelName: String { "gemv_q3_g16" }
    public func gemmKernelName(bufferPrecision: BufferPrecision) -> String {
        bufferPrecision == .float32 ? "gemm_q3_g16_f32s" : "gemm_q3_g16"
    }
    public var weightsPerBlock: Int { 16 }
    public var bytesPerBlock: Int { 4 + 6 }
    public var bits: Int { 3 }
    public var groupSize: Int { 16 }

    public var isQuantized: Bool { true }
    public var bufferElementType: String { "uchar" }
    public var mslDeclarations: String {
        Q3AffineMSL.blockStruct(name: blockStructName, weightsPerBlock: weightsPerBlock)
    }

    public func emitGroupDequant(
        blocksVar: String,
        blockIndexVar: String,
        outputArrayVar: String
    ) -> String? {
        Q3AffineMSL.groupDequant(
            blocksVar: blocksVar,
            outputArrayVar: outputArrayVar,
            weightsPerBlock: weightsPerBlock
        )
    }

    public init() {}
}

/// 3-bit affine quantization with group size 32.
public struct AffineQ3Group32Format: QuantizationFormat {
    public var schemeIdentifier: QuantizationSchemeIdentifier { .q3Group32ScaleF16 }
    public var blockStructName: String { "BlockQ3Affine32" }
    public var gemvKernelName: String { "gemv_q3_g32" }
    public func gemmKernelName(bufferPrecision: BufferPrecision) -> String {
        bufferPrecision == .float32 ? "gemm_q3_g32_f32s" : "gemm_q3_g32"
    }
    public var weightsPerBlock: Int { 32 }
    public var bytesPerBlock: Int { 4 + 12 }
    public var bits: Int { 3 }
    public var groupSize: Int { 32 }

    public var isQuantized: Bool { true }
    public var bufferElementType: String { "uchar" }
    public var mslDeclarations: String {
        Q3AffineMSL.blockStruct(name: blockStructName, weightsPerBlock: weightsPerBlock)
    }

    public func emitGroupDequant(
        blocksVar: String,
        blockIndexVar: String,
        outputArrayVar: String
    ) -> String? {
        Q3AffineMSL.groupDequant(
            blocksVar: blocksVar,
            outputArrayVar: outputArrayVar,
            weightsPerBlock: weightsPerBlock
        )
    }

    public init() {}
}

// MARK: - 5-bit Affine Formats (non-aligned)

/// 5-bit affine quantization with group size 32.
///
/// Non-aligned packing (MLX `extract_bits<5>`): 8 weights share 5 bytes.
/// Block layout:
/// ```
/// ┌──────────┬──────────┬────────────────────┐
/// │scale (2B)│ zero (2B)│ packed quants (20B)│
/// └──────────┴──────────┴────────────────────┘
/// 24 bytes per block, 32 weights (4 packs of 8)
/// ```
public struct AffineQ5Group32Format: QuantizationFormat {
    public var schemeIdentifier: QuantizationSchemeIdentifier { .q5Group32ScaleF16 }
    public var blockStructName: String { "BlockQ5Affine32" }
    public var gemvKernelName: String { "gemv_q5_g32" }
    public func gemmKernelName(bufferPrecision: BufferPrecision) -> String {
        bufferPrecision == .float32 ? "gemm_q5_g32_f32s" : "gemm_q5_g32"
    }
    public var weightsPerBlock: Int { 32 }
    public var bytesPerBlock: Int { 4 + 20 }
    public var bits: Int { 5 }
    public var groupSize: Int { 32 }

    public var isQuantized: Bool { true }
    public var bufferElementType: String { "uchar" }
    public var mslDeclarations: String {
        Q5AffineMSL.blockStruct(name: blockStructName, weightsPerBlock: weightsPerBlock)
    }

    public func emitGroupDequant(
        blocksVar: String,
        blockIndexVar: String,
        outputArrayVar: String
    ) -> String? {
        Q5AffineMSL.groupDequant(
            blocksVar: blocksVar,
            outputArrayVar: outputArrayVar,
            weightsPerBlock: weightsPerBlock
        )
    }

    public init() {}
}

/// 5-bit affine quantization with group size 64.
public struct AffineQ5Group64Format: QuantizationFormat {
    public var schemeIdentifier: QuantizationSchemeIdentifier { .q5Group64ScaleF16 }
    public var blockStructName: String { "BlockQ5Affine64" }
    public var gemvKernelName: String { "gemv_q5_g64" }
    public func gemmKernelName(bufferPrecision: BufferPrecision) -> String {
        bufferPrecision == .float32 ? "gemm_q5_g64_f32s" : "gemm_q5_g64"
    }
    public var weightsPerBlock: Int { 64 }
    public var bytesPerBlock: Int { 4 + 40 }
    public var bits: Int { 5 }
    public var groupSize: Int { 64 }

    public var isQuantized: Bool { true }
    public var bufferElementType: String { "uchar" }
    public var mslDeclarations: String {
        Q5AffineMSL.blockStruct(name: blockStructName, weightsPerBlock: weightsPerBlock)
    }

    public func emitGroupDequant(
        blocksVar: String,
        blockIndexVar: String,
        outputArrayVar: String
    ) -> String? {
        Q5AffineMSL.groupDequant(
            blocksVar: blocksVar,
            outputArrayVar: outputArrayVar,
            weightsPerBlock: weightsPerBlock
        )
    }

    public init() {}
}

// MARK: - 6-bit Affine Formats (non-aligned)

/// 6-bit affine quantization with group size 16.
///
/// Non-aligned packing: 4 weights share 3 bytes.
/// ```
/// ┌──────────┬──────────┬────────────────────┐
/// │scale (2B)│ zero (2B)│ packed quants (12B)│
/// └──────────┴──────────┴────────────────────┘
/// 16 bytes per block, 16 weights
/// ```
/// MLX packing (see mlx/backend/cpu/quantized.cpp `extract_bits<6>`):
/// - w[0] = b0 & 0x3f
/// - w[1] = ((b0 >> 6) & 0x03) | ((b1 & 0x0f) << 2)
/// - w[2] = ((b1 >> 4) & 0x0f) | ((b2 & 0x03) << 4)
/// - w[3] = b2 >> 2
public struct AffineQ6Group16Format: QuantizationFormat {
    public var schemeIdentifier: QuantizationSchemeIdentifier { .q6Group16ScaleF16 }
    public var blockStructName: String { "BlockQ6Affine16" }
    public var gemvKernelName: String { "gemv_q6_g16" }
    public func gemmKernelName(bufferPrecision: BufferPrecision) -> String {
        bufferPrecision == .float32 ? "gemm_q6_g16_f32s" : "gemm_q6_g16"
    }
    public var weightsPerBlock: Int { 16 }
    public var bytesPerBlock: Int { 4 + 12 }
    public var bits: Int { 6 }
    public var groupSize: Int { 16 }

    public var isQuantized: Bool { true }
    public var bufferElementType: String { "uchar" }
    public var mslDeclarations: String {
        Q6AffineMSL.blockStruct(name: blockStructName, weightsPerBlock: weightsPerBlock)
    }

    public func emitGroupDequant(
        blocksVar: String,
        blockIndexVar: String,
        outputArrayVar: String
    ) -> String? {
        Q6AffineMSL.groupDequant(
            blocksVar: blocksVar,
            outputArrayVar: outputArrayVar,
            weightsPerBlock: weightsPerBlock
        )
    }

    public init() {}
}

/// 6-bit affine quantization with group size 32.
///
/// Non-aligned packing: 4 weights share 3 bytes, 8 groups of 4 per block.
public struct AffineQ6Group32Format: QuantizationFormat {
    public var schemeIdentifier: QuantizationSchemeIdentifier { .q6Group32ScaleF16 }
    public var blockStructName: String { "BlockQ6Affine32" }
    public var gemvKernelName: String { "gemv_q6_g32" }
    public func gemmKernelName(bufferPrecision: BufferPrecision) -> String {
        bufferPrecision == .float32 ? "gemm_q6_g32_f32s" : "gemm_q6_g32"
    }
    public var weightsPerBlock: Int { 32 }
    public var bytesPerBlock: Int { 4 + 24 }
    public var bits: Int { 6 }
    public var groupSize: Int { 32 }

    public var isQuantized: Bool { true }
    public var bufferElementType: String { "uchar" }
    public var mslDeclarations: String {
        Q6AffineMSL.blockStruct(name: blockStructName, weightsPerBlock: weightsPerBlock)
    }

    public func emitGroupDequant(
        blocksVar: String,
        blockIndexVar: String,
        outputArrayVar: String
    ) -> String? {
        Q6AffineMSL.groupDequant(
            blocksVar: blocksVar,
            outputArrayVar: outputArrayVar,
            weightsPerBlock: weightsPerBlock
        )
    }

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

    public var isQuantized: Bool { true }
    public var bufferElementType: String { "uchar" }
    public var mslDeclarations: String { Q8AffineMSL.blockStruct(name: blockStructName, weightsPerBlock: weightsPerBlock) }

    public func perWeightReadExpression(
        blocksVar: String,
        weightIndexVar: String
    ) -> String? {
        Q8AffineMSL.perWeightExpression(blocksVar: blocksVar, weightIndexVar: weightIndexVar)
    }

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

    public var isQuantized: Bool { true }
    public var bufferElementType: String { "uchar" }
    public var mslDeclarations: String { Q8AffineMSL.blockStruct(name: blockStructName, weightsPerBlock: weightsPerBlock) }

    public func perWeightReadExpression(
        blocksVar: String,
        weightIndexVar: String
    ) -> String? {
        Q8AffineMSL.perWeightExpression(blocksVar: blocksVar, weightIndexVar: weightIndexVar)
    }

    public init() {}
}

/// 8-bit affine quantization with group size 128.
public struct AffineQ8Group128Format: QuantizationFormat {
    public var schemeIdentifier: QuantizationSchemeIdentifier { .q8Group128ScaleF16 }
    public var blockStructName: String { "BlockQ8Affine128" }
    public var gemvKernelName: String { "gemv_q8_g128" }
    public func gemmKernelName(bufferPrecision: BufferPrecision) -> String {
        bufferPrecision == .float32 ? "gemm_q8_g128_f32s" : "gemm_q8_g128"
    }
    public var weightsPerBlock: Int { 128 }
    public var bytesPerBlock: Int { 4 + 128 }
    public var bits: Int { 8 }
    public var groupSize: Int { 128 }

    public var isQuantized: Bool { true }
    public var bufferElementType: String { "uchar" }
    public var mslDeclarations: String { Q8AffineMSL.blockStruct(name: blockStructName, weightsPerBlock: weightsPerBlock) }

    public func perWeightReadExpression(
        blocksVar: String,
        weightIndexVar: String
    ) -> String? {
        Q8AffineMSL.perWeightExpression(blocksVar: blocksVar, weightIndexVar: weightIndexVar)
    }

    public init() {}
}

// MARK: - MSL Fragment Helpers

/// MSL source fragments for Q2 affine block formats.
///
/// 2-bit packing: 4 weights per byte, weight `k` occupies bits `(k & 3) * 2 ..< (k & 3) * 2 + 2`
/// of byte `k >> 2`.
enum Q2AffineMSL {
    /// Block struct definition. `weightsPerBlock` controls the packed array size
    /// (weightsPerBlock / 4 bytes since each byte holds 4 2-bit quants).
    static func blockStruct(name: String, weightsPerBlock: Int) -> String {
        let packedBytes = weightsPerBlock / 4
        return """
        struct \(name) {
            half scale;
            half zero;
            uchar qs[\(packedBytes)];
        };
        """
    }

    /// Per-weight dequant expression assuming `scale` and `zero` are captured as
    /// `float` locals by the caller.
    static func perWeightExpression(blocksVar: String, weightIndexVar: String) -> String {
        "(scale * float((\(blocksVar)[(\(weightIndexVar)) >> 2] >> (((\(weightIndexVar)) & 3) * 2)) & 0x3) + zero)"
    }
}

/// MSL source fragments for Q4 affine block formats.
///
/// These helpers centralize the block struct and per-weight dequant expression
/// used by the unified generator (Phase 2+). Existing hand-written Q4 kernels
/// in `MetalSourceGenerator+Quantized.swift` are not affected.
enum Q4AffineMSL {
    /// Block struct definition. `weightsPerBlock` controls the packed array size
    /// (weightsPerBlock / 2 bytes since each byte holds 2 nibbles).
    static func blockStruct(name: String, weightsPerBlock: Int) -> String {
        let packedBytes = weightsPerBlock / 2
        return """
        struct \(name) {
            half scale;
            half zero;
            uchar qs[\(packedBytes)];
        };
        """
    }

    /// Per-weight dequant expression assuming `scale` and `zero` are captured as
    /// `float` locals by the caller. Reads weight `k` from packed nibbles:
    /// even k → low nibble, odd k → high nibble.
    static func perWeightExpression(blocksVar: String, weightIndexVar: String) -> String {
        "(scale * float((\(blocksVar)[(\(weightIndexVar)) >> 1] >> (((\(weightIndexVar)) & 1) * 4)) & 0xF) + zero)"
    }
}

/// MSL source fragments for Q3 affine block formats.
///
/// Non-aligned packing (see MLX `mlx/backend/cpu/quantized.cpp` `extract_bits<3>`):
/// 8 weights share 3 bytes.
/// - w[0] = b0 & 0x07
/// - w[1] = (b0 >> 3) & 0x07
/// - w[2] = ((b0 >> 6) & 0x03) | ((b1 & 0x01) << 2)
/// - w[3] = (b1 >> 1) & 0x07
/// - w[4] = (b1 >> 4) & 0x07
/// - w[5] = ((b1 >> 7) & 0x01) | ((b2 & 0x03) << 1)
/// - w[6] = (b2 >> 2) & 0x07
/// - w[7] = (b2 >> 5) & 0x07
enum Q3AffineMSL {
    /// Block struct definition. Packed bytes = (weightsPerBlock / 8) * 3.
    static func blockStruct(name: String, weightsPerBlock: Int) -> String {
        let packedBytes = (weightsPerBlock / 8) * 3
        return """
        struct \(name) {
            half scale;
            half zero;
            uchar qs[\(packedBytes)];
        };
        """
    }

    /// Emits a loop that expands one block's `weightsPerBlock` weights into
    /// `outputArrayVar[0..<weightsPerBlock]` as float. Assumes `scale` and
    /// `zero` are captured as `float` locals by the caller.
    static func groupDequant(
        blocksVar: String,
        outputArrayVar: String,
        weightsPerBlock: Int
    ) -> String {
        let numGroups = weightsPerBlock / 8
        return """
        for (uint g = 0; g < \(numGroups); g++) {
                            uchar b0 = \(blocksVar)[g * 3 + 0];
                            uchar b1 = \(blocksVar)[g * 3 + 1];
                            uchar b2 = \(blocksVar)[g * 3 + 2];
                            \(outputArrayVar)[g * 8 + 0] = scale * float(b0 & 0x07) + zero;
                            \(outputArrayVar)[g * 8 + 1] = scale * float((b0 >> 3) & 0x07) + zero;
                            \(outputArrayVar)[g * 8 + 2] = scale * float(((b0 >> 6) & 0x03) | ((b1 & 0x01) << 2)) + zero;
                            \(outputArrayVar)[g * 8 + 3] = scale * float((b1 >> 1) & 0x07) + zero;
                            \(outputArrayVar)[g * 8 + 4] = scale * float((b1 >> 4) & 0x07) + zero;
                            \(outputArrayVar)[g * 8 + 5] = scale * float(((b1 >> 7) & 0x01) | ((b2 & 0x03) << 1)) + zero;
                            \(outputArrayVar)[g * 8 + 6] = scale * float((b2 >> 2) & 0x07) + zero;
                            \(outputArrayVar)[g * 8 + 7] = scale * float((b2 >> 5) & 0x07) + zero;
                        }
        """
    }
}

/// MSL source fragments for Q5 affine block formats.
///
/// Non-aligned packing (see MLX `mlx/backend/cpu/quantized.cpp` `extract_bits<5>`):
/// 8 weights share 5 bytes.
/// - w[0] = b0 & 0x1f
/// - w[1] = ((b0 >> 5) & 0x07) | ((b1 & 0x03) << 3)
/// - w[2] = (b1 >> 2) & 0x1f
/// - w[3] = ((b1 >> 7) & 0x01) | ((b2 & 0x0f) << 1)
/// - w[4] = ((b2 >> 4) & 0x0f) | ((b3 & 0x01) << 4)
/// - w[5] = (b3 >> 1) & 0x1f
/// - w[6] = ((b3 >> 6) & 0x03) | ((b4 & 0x07) << 2)
/// - w[7] = (b4 >> 3) & 0x1f
enum Q5AffineMSL {
    /// Block struct definition. Packed bytes = (weightsPerBlock / 8) * 5.
    static func blockStruct(name: String, weightsPerBlock: Int) -> String {
        let packedBytes = (weightsPerBlock / 8) * 5
        return """
        struct \(name) {
            half scale;
            half zero;
            uchar qs[\(packedBytes)];
        };
        """
    }

    /// Emits a loop that expands one block's `weightsPerBlock` weights into
    /// `outputArrayVar[0..<weightsPerBlock]` as float. Assumes `scale` and
    /// `zero` are captured as `float` locals by the caller.
    static func groupDequant(
        blocksVar: String,
        outputArrayVar: String,
        weightsPerBlock: Int
    ) -> String {
        let numGroups = weightsPerBlock / 8
        return """
        for (uint g = 0; g < \(numGroups); g++) {
                            uchar b0 = \(blocksVar)[g * 5 + 0];
                            uchar b1 = \(blocksVar)[g * 5 + 1];
                            uchar b2 = \(blocksVar)[g * 5 + 2];
                            uchar b3 = \(blocksVar)[g * 5 + 3];
                            uchar b4 = \(blocksVar)[g * 5 + 4];
                            \(outputArrayVar)[g * 8 + 0] = scale * float(b0 & 0x1f) + zero;
                            \(outputArrayVar)[g * 8 + 1] = scale * float(((b0 >> 5) & 0x07) | ((b1 & 0x03) << 3)) + zero;
                            \(outputArrayVar)[g * 8 + 2] = scale * float((b1 >> 2) & 0x1f) + zero;
                            \(outputArrayVar)[g * 8 + 3] = scale * float(((b1 >> 7) & 0x01) | ((b2 & 0x0f) << 1)) + zero;
                            \(outputArrayVar)[g * 8 + 4] = scale * float(((b2 >> 4) & 0x0f) | ((b3 & 0x01) << 4)) + zero;
                            \(outputArrayVar)[g * 8 + 5] = scale * float((b3 >> 1) & 0x1f) + zero;
                            \(outputArrayVar)[g * 8 + 6] = scale * float(((b3 >> 6) & 0x03) | ((b4 & 0x07) << 2)) + zero;
                            \(outputArrayVar)[g * 8 + 7] = scale * float((b4 >> 3) & 0x1f) + zero;
                        }
        """
    }
}

/// MSL source fragments for Q6 affine block formats.
///
/// Non-aligned packing: 4 weights share 3 bytes (see MLX
/// `mlx/backend/cpu/quantized.cpp` `extract_bits<6>`). This helper emits a
/// group-level dequant statement that populates a thread-local float array
/// for the unified GEMV kernel.
enum Q6AffineMSL {
    /// Block struct definition. Packed bytes = (weightsPerBlock / 4) * 3.
    static func blockStruct(name: String, weightsPerBlock: Int) -> String {
        let packedBytes = (weightsPerBlock / 4) * 3
        return """
        struct \(name) {
            half scale;
            half zero;
            uchar qs[\(packedBytes)];
        };
        """
    }

    /// Emits a loop that expands one block's `weightsPerBlock` weights into
    /// `outputArrayVar[0..<weightsPerBlock]` as float. Assumes `scale` and
    /// `zero` are captured as `float` locals by the caller.
    static func groupDequant(
        blocksVar: String,
        outputArrayVar: String,
        weightsPerBlock: Int
    ) -> String {
        let numGroups = weightsPerBlock / 4
        return """
        for (uint g = 0; g < \(numGroups); g++) {
                            uchar b0 = \(blocksVar)[g * 3 + 0];
                            uchar b1 = \(blocksVar)[g * 3 + 1];
                            uchar b2 = \(blocksVar)[g * 3 + 2];
                            \(outputArrayVar)[g * 4 + 0] = scale * float(b0 & 0x3f) + zero;
                            \(outputArrayVar)[g * 4 + 1] = scale * float(((b0 >> 6) & 0x03) | ((b1 & 0x0f) << 2)) + zero;
                            \(outputArrayVar)[g * 4 + 2] = scale * float(((b1 >> 4) & 0x0f) | ((b2 & 0x03) << 4)) + zero;
                            \(outputArrayVar)[g * 4 + 3] = scale * float(b2 >> 2) + zero;
                        }
        """
    }
}

/// MSL source fragments for Q8 affine block formats.
enum Q8AffineMSL {
    /// Block struct definition. One byte per quant.
    static func blockStruct(name: String, weightsPerBlock: Int) -> String {
        """
        struct \(name) {
            half scale;
            half zero;
            uchar qs[\(weightsPerBlock)];
        };
        """
    }

    /// Per-weight dequant expression assuming `scale` and `zero` are captured as
    /// `float` locals by the caller.
    static func perWeightExpression(blocksVar: String, weightIndexVar: String) -> String {
        "(scale * float(\(blocksVar)[\(weightIndexVar)]) + zero)"
    }
}

// MARK: - MLX Quantization Hint

/// Explicit MLX quantization metadata read from `config.json`.
///
/// Safetensors shape alone cannot disambiguate some format pairs
/// (e.g. Q4G64 vs Q8G32 both satisfy `bits × group_size = 256`), so the
/// planner requires this hint whenever quantized companions
/// (`.scales` / `.biases`) are present. Silent defaults are forbidden
/// — missing-but-required hints surface as explicit errors.
public struct MLXQuantizationHint: Sendable, Hashable {
    public let bits: Int
    public let groupSize: Int

    public init(bits: Int, groupSize: Int) {
        self.bits = bits
        self.groupSize = groupSize
    }
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
        case .q2Group16ScaleF16: return AffineQ2Group16Format()
        case .q2Group32ScaleF16: return AffineQ2Group32Format()
        case .q3Group16ScaleF16: return AffineQ3Group16Format()
        case .q3Group32ScaleF16: return AffineQ3Group32Format()
        case .q4Group64ScaleF16: return AffineQ4Group64Format()
        case .q4Group128ScaleF16: return AffineQ4Group128Format()
        // Q4G128Zero (0x42) uses the same 68-byte block layout as Q4G128. Alias
        // to the same struct to resolve the orphan scheme ID without behavioral
        // divergence — both decode to the same packed nibble representation.
        case .q4Group128ScaleF16Zero: return AffineQ4Group128Format()
        case .q5Group32ScaleF16: return AffineQ5Group32Format()
        case .q5Group64ScaleF16: return AffineQ5Group64Format()
        case .q6Group16ScaleF16: return AffineQ6Group16Format()
        case .q6Group32ScaleF16: return AffineQ6Group32Format()
        case .q8Group32ScaleF16: return AffineQ8Group32Format()
        case .q8Group64ScaleF16: return AffineQ8Group64Format()
        case .q8Group128ScaleF16: return AffineQ8Group128Format()
        case .passthrough: return Float16Format()
        default: return nil
        }
    }

    /// Determine the best quantization format for an MLX quantized tensor.
    public static func formatForMLXQuantization(
        bits: Int, groupSize: Int
    ) -> (any QuantizationFormat)? {
        switch (bits, groupSize) {
        case (2, 16): return AffineQ2Group16Format()
        case (2, 32): return AffineQ2Group32Format()
        case (3, 16): return AffineQ3Group16Format()
        case (3, 32): return AffineQ3Group32Format()
        case (4, 64): return AffineQ4Group64Format()
        case (4, 128): return AffineQ4Group128Format()
        case (5, 32): return AffineQ5Group32Format()
        case (5, 64): return AffineQ5Group64Format()
        case (6, 16): return AffineQ6Group16Format()
        case (6, 32): return AffineQ6Group32Format()
        case (8, 32): return AffineQ8Group32Format()
        case (8, 64): return AffineQ8Group64Format()
        case (8, 128): return AffineQ8Group128Format()
        default: return nil
        }
    }
}
