/// Descriptor for a direct quantized GEMM kernel (register-resident weight unpacking).
///
/// Formats that implement a hand-tuned GEMM kernel that reads packed weights directly
/// return a non-nil `DirectQuantizedGEMM` from `directGEMMKernel()` /
/// `batchedGEMMKernel(count:)`. Formats without such a kernel return nil, and the
/// compiler routes through the dequant→BF16→MPP GEMM pipeline instead.
///
/// This is a **capability declaration**, not an error fallback: both paths are
/// correctness-preserving. The direct kernel is strictly a performance optimization.
public struct DirectQuantizedGEMM: Sendable {
    public let kernelName: String
    public let threadgroupMemoryLength: Int

    public init(kernelName: String, threadgroupMemoryLength: Int) {
        self.kernelName = kernelName
        self.threadgroupMemoryLength = threadgroupMemoryLength
    }
}

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

    /// Per-weight read expression.
    ///
    /// The caller (unified GEMV / dequant scaffold) already captured `scale`
    /// and `zero` as local `float` variables for the current block. This hook
    /// returns a single MSL expression that evaluates to the dequantized float
    /// weight at `blocksVar[weightIndexVar]`.
    ///
    /// - `blocksVar`: name of the packed-quant array within the current block
    ///   (for dense, the flat weight array; for Q4/Q8, e.g. "blk.qs").
    /// - `weightIndexVar`: name of the weight index variable within `blocksVar`.
    ///
    /// All quantized formats (aligned and non-aligned) must implement this.
    /// Non-aligned formats (Q3/Q5/Q6) return a ternary-chain expression that
    /// the Metal compiler flattens into predicated selection.
    func perWeightReadExpression(
        blocksVar: String,
        weightIndexVar: String
    ) -> String?

    // MARK: - Prefill dispatch capabilities

    /// Name of the dequant-to-BFloat16 kernel for this format, used by the
    /// prefill dequant→BF16→MPP GEMM pipeline.
    ///
    /// Returns nil for dense formats (no dequant needed). For quantized formats
    /// the default implementation returns the unified
    /// `dequant_q{bits}_g{group}_bf16` name; the same symbol is emitted by
    /// `MetalKernelSourceCatalog` regardless of whether the source is produced
    /// by the hand-tuned Q4 generator or the unified generator.
    var dequantToBFloatKernelName: String? { get }

    /// Direct prefill GEMM kernel that reads packed weights in registers,
    /// bypassing the two-step dequant→AMX pipeline.
    ///
    /// Returns nil when the format has no hand-tuned direct GEMM kernel; the
    /// compiler will route through the dequant→BF16→MPP GEMM pipeline instead.
    /// Both paths are correctness-preserving; the direct kernel is strictly a
    /// performance optimization that must match the prefill sequence-aware
    /// buffer signature (see `DirectQuantizedGEMM`).
    func directGEMMKernel() -> DirectQuantizedGEMM?

    /// Batched prefill GEMM kernel that shares a single input across `count`
    /// projections (Q/K/V, gate+up etc.), bypassing the dequant→AMX pipeline.
    ///
    /// Returns nil when no batched direct kernel exists for this (format, count)
    /// combination; the compiler will fall through to per-projection direct
    /// GEMM or the dequant→BF16 MPP path.
    func batchedGEMMKernel(count: Int) -> DirectQuantizedGEMM?

    /// Generate the MSL source for the direct prefill GEMM kernel declared by
    /// `directGEMMKernel()`. Must return non-nil whenever `directGEMMKernel()`
    /// does; returns nil otherwise so the compiler routes through the
    /// dequant→BF16→MPP pipeline.
    func directGEMMKernelSource(
        name: String,
        bufferPrecision: BufferPrecision
    ) -> String?

    /// Generate the MSL source for the batched direct prefill GEMM kernel
    /// declared by `batchedGEMMKernel(count:)`. Must return non-nil whenever
    /// `batchedGEMMKernel(count:)` does for the same `count`.
    func batchedGEMMKernelSource(
        name: String,
        count: Int,
        bufferPrecision: BufferPrecision
    ) -> String?

    // MARK: - Embedding lookup capabilities

    /// Format token used to disambiguate embedding-lookup kernel names.
    /// Dense fp16 returns "" (base name is plain `embedding_lookup`), dense
    /// bf16/fp32 return "bf16"/"fp32", quantized formats return
    /// "q{bits}_g{groupSize}".
    var embeddingLookupToken: String { get }

    /// Generate the MSL source for the embedding-lookup kernel on this format.
    /// Dense formats delegate to `generateEmbeddingLookup`; quantized formats
    /// delegate to the matching hand-tuned or unified generator.
    func embeddingLookupKernelSource(
        name: String,
        bufferPrecision: BufferPrecision,
        isSequence: Bool,
        embeddingScale: Float?
    ) -> String
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

    /// Default: no per-weight read. Formats must override to provide one.
    func perWeightReadExpression(
        blocksVar: String,
        weightIndexVar: String
    ) -> String? { nil }

    /// Default: quantized formats use the unified `dequant_q{bits}_g{group}_bf16`
    /// symbol (matches both the hand-tuned Q4 generator output and the unified
    /// generator output). Dense formats return nil — no dequant step is needed.
    var dequantToBFloatKernelName: String? {
        guard isQuantized else { return nil }
        return "dequant_q\(bits)_g\(groupSize)_bf16"
    }

    /// Default: no direct GEMM kernel. Formats with a hand-tuned kernel
    /// (Q4G64 / Q4G128 / Q8G32 / Q8G64) override this.
    func directGEMMKernel() -> DirectQuantizedGEMM? { nil }

    /// Default: no batched direct GEMM kernel. Q4G64 / Q4G128 override this.
    func batchedGEMMKernel(count: Int) -> DirectQuantizedGEMM? { nil }

    /// Default: no direct GEMM source. Formats that override `directGEMMKernel()`
    /// also override this to produce the matching MSL.
    func directGEMMKernelSource(
        name: String,
        bufferPrecision: BufferPrecision
    ) -> String? { nil }

    /// Default: no batched direct GEMM source. Formats that override
    /// `batchedGEMMKernel(count:)` also override this.
    func batchedGEMMKernelSource(
        name: String,
        count: Int,
        bufferPrecision: BufferPrecision
    ) -> String? { nil }

    /// Default embedding-lookup token derivation:
    /// * Quantized: `q{bits}_g{groupSize}`
    /// * Dense bf16 / fp32: `bf16` / `fp32`
    /// * Dense fp16: empty (the base kernel name has no token)
    var embeddingLookupToken: String {
        if isQuantized { return "q\(bits)_g\(groupSize)" }
        if isBFloat16 { return "bf16" }
        if isFloat32 { return "fp32" }
        return ""
    }

    /// Default: dense formats use `generateEmbeddingLookup`; quantized formats
    /// pick the appropriate generator based on their bit-width.
    func embeddingLookupKernelSource(
        name: String,
        bufferPrecision: BufferPrecision,
        isSequence: Bool,
        embeddingScale: Float?
    ) -> String {
        if isQuantized {
            switch bits {
            case 4:
                return MetalSourceGenerator.generateQuantizedEmbeddingLookupQ4(
                    name: name,
                    bufferPrecision: bufferPrecision,
                    groupSize: groupSize,
                    isSequence: isSequence,
                    embeddingScale: embeddingScale
                )
            case 8:
                return MetalSourceGenerator.generateQuantizedEmbeddingLookupQ8(
                    name: name,
                    bufferPrecision: bufferPrecision,
                    groupSize: groupSize,
                    isSequence: isSequence,
                    embeddingScale: embeddingScale
                )
            default:
                return MetalSourceGenerator.generateUnifiedQuantizedEmbeddingLookup(
                    name: name,
                    format: self,
                    bufferPrecision: bufferPrecision,
                    isSequence: isSequence,
                    embeddingScale: embeddingScale
                )
            }
        }
        return MetalSourceGenerator.generateEmbeddingLookup(
            name: name,
            bufferPrecision: bufferPrecision,
            weightFormat: self,
            isSequence: isSequence,
            embeddingScale: embeddingScale
        )
    }

    // MARK: - Polymorphic queries

    /// True when the format is dense fp16 (`.fp16RowMajor`).
    var isFloat16: Bool { schemeIdentifier == .fp16RowMajor }

    /// True when the format is dense bf16 (`.bf16RowMajor`).
    var isBFloat16: Bool { schemeIdentifier == .bf16RowMajor }

    /// True when the format is dense fp32 (`.fp32RowMajor`).
    var isFloat32: Bool { schemeIdentifier == .fp32RowMajor }
}

// MARK: - Dot-syntax factories for `any QuantizationFormat`

public extension QuantizationFormat where Self == Float16Format {
    static var float16: Float16Format { Float16Format() }
}

public extension QuantizationFormat where Self == BFloat16Format {
    static var bfloat16: BFloat16Format { BFloat16Format() }
}

public extension QuantizationFormat where Self == Float32Format {
    static var float32: Float32Format { Float32Format() }
}

/// Equality for `any QuantizationFormat` existentials: compare by scheme identifier.
public func == (lhs: any QuantizationFormat, rhs: any QuantizationFormat) -> Bool {
    lhs.schemeIdentifier == rhs.schemeIdentifier
}

public func != (lhs: any QuantizationFormat, rhs: any QuantizationFormat) -> Bool {
    lhs.schemeIdentifier != rhs.schemeIdentifier
}

public func == (lhs: (any QuantizationFormat)?, rhs: any QuantizationFormat) -> Bool {
    guard let lhs else { return false }
    return lhs.schemeIdentifier == rhs.schemeIdentifier
}

public func == (lhs: any QuantizationFormat, rhs: (any QuantizationFormat)?) -> Bool {
    guard let rhs else { return false }
    return lhs.schemeIdentifier == rhs.schemeIdentifier
}

public func != (lhs: (any QuantizationFormat)?, rhs: any QuantizationFormat) -> Bool {
    !(lhs == rhs)
}

public func != (lhs: any QuantizationFormat, rhs: (any QuantizationFormat)?) -> Bool {
    !(lhs == rhs)
}

public extension QuantizationFormat {

    // MARK: - Legacy WeightFormat compatibility

    /// MSL type name used for the raw weight buffer parameter.
    /// Alias for `bufferElementType` to match the legacy enum API.
    var bufferType: String { bufferElementType }

    /// Per-element weight byte stride for raw dense buffers.
    /// Quantized formats always use `uchar`-addressed packed bytes (1 B stride).
    var storageByteSize: Int {
        if isQuantized { return MemoryLayout<UInt8>.stride }
        switch bits {
        case 16: return MemoryLayout<UInt16>.stride
        case 32: return MemoryLayout<Float>.stride
        default: return MemoryLayout<UInt8>.stride
        }
    }

    /// MSL expression that reads one dense weight and yields a float.
    ///
    /// Only valid for dense formats (fp16 / bf16 / fp32). Block-packed quantized
    /// formats cannot produce a single-element read expression because they require
    /// per-block scale/zero lookup; callers must route quantized weights through a
    /// dedicated quantized kernel. The protocol returns the bare scalar read
    /// expression because the caller already indexes into the dense buffer.
    func readExpression(_ expr: String) -> String {
        if isQuantized {
            fatalError("QuantizationFormat.readExpression called on quantized format \(schemeIdentifier); quantized weights must be routed through a dedicated quantized kernel, not a dense GEMV/GEMM template.")
        }
        switch bits {
        case 16:
            // Float16 stores as `half`; BFloat16 stores as `uint16_t` and needs
            // an explicit bf16→float helper.
            return bufferElementType == "half" ? "float(\(expr))" : "bf16_to_float(\(expr))"
        case 32:
            return "(\(expr))"
        default:
            fatalError("QuantizationFormat.readExpression: unsupported dense bit width \(bits) for \(schemeIdentifier)")
        }
    }
}

// MARK: - Threadgroup memory helper

/// Thread-group memory length used by all hand-tuned direct quantized GEMM kernels.
///
/// All current direct kernels (`gemm_q{4,8}_g{group}_f32s`) allocate
/// `max(groupSize * 2, 256) * sizeof(float)` bytes of TG memory, where the
/// `groupSize * 2` term accounts for scale + zero cached per group and 256 is
/// the lower bound tile width.
@inlinable
public func directQuantizedGEMMThreadgroupMemoryLength(groupSize: Int) -> Int {
    max(groupSize * 2, 256) * MemoryLayout<Float>.size
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

    public func directGEMMKernel() -> DirectQuantizedGEMM? {
        DirectQuantizedGEMM(
            kernelName: "gemm_q4_g64_f32s",
            threadgroupMemoryLength: directQuantizedGEMMThreadgroupMemoryLength(groupSize: groupSize)
        )
    }

    public func batchedGEMMKernel(count: Int) -> DirectQuantizedGEMM? {
        DirectQuantizedGEMM(
            kernelName: "batched_gemm_q4_g64_\(count)",
            threadgroupMemoryLength: directQuantizedGEMMThreadgroupMemoryLength(groupSize: groupSize)
        )
    }

    public func directGEMMKernelSource(
        name: String,
        bufferPrecision: BufferPrecision
    ) -> String? {
        MetalSourceGenerator.generateQuantizedGEMM_Q4(
            name: name, bufferPrecision: bufferPrecision, groupSize: groupSize)
    }

    public func batchedGEMMKernelSource(
        name: String,
        count: Int,
        bufferPrecision: BufferPrecision
    ) -> String? {
        switch count {
        case 2:
            return MetalSourceGenerator.generateBatchedQuantizedGEMM_Q4_2(
                name: name, bufferPrecision: bufferPrecision, groupSize: groupSize)
        case 3:
            return MetalSourceGenerator.generateBatchedQuantizedGEMM_Q4_3(
                name: name, bufferPrecision: bufferPrecision, groupSize: groupSize)
        default:
            return nil
        }
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

    public func directGEMMKernel() -> DirectQuantizedGEMM? {
        DirectQuantizedGEMM(
            kernelName: "gemm_q4_g128_f32s",
            threadgroupMemoryLength: directQuantizedGEMMThreadgroupMemoryLength(groupSize: groupSize)
        )
    }

    public func batchedGEMMKernel(count: Int) -> DirectQuantizedGEMM? {
        DirectQuantizedGEMM(
            kernelName: "batched_gemm_q4_g128_\(count)",
            threadgroupMemoryLength: directQuantizedGEMMThreadgroupMemoryLength(groupSize: groupSize)
        )
    }

    public func directGEMMKernelSource(
        name: String,
        bufferPrecision: BufferPrecision
    ) -> String? {
        MetalSourceGenerator.generateQuantizedGEMM_Q4(
            name: name, bufferPrecision: bufferPrecision, groupSize: groupSize)
    }

    public func batchedGEMMKernelSource(
        name: String,
        count: Int,
        bufferPrecision: BufferPrecision
    ) -> String? {
        switch count {
        case 2:
            return MetalSourceGenerator.generateBatchedQuantizedGEMM_Q4_2(
                name: name, bufferPrecision: bufferPrecision, groupSize: groupSize)
        case 3:
            return MetalSourceGenerator.generateBatchedQuantizedGEMM_Q4_3(
                name: name, bufferPrecision: bufferPrecision, groupSize: groupSize)
        default:
            return nil
        }
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

    public func perWeightReadExpression(
        blocksVar: String,
        weightIndexVar: String
    ) -> String? {
        Q3AffineMSL.perWeightExpression(blocksVar: blocksVar, weightIndexVar: weightIndexVar)
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

    public func perWeightReadExpression(
        blocksVar: String,
        weightIndexVar: String
    ) -> String? {
        Q3AffineMSL.perWeightExpression(blocksVar: blocksVar, weightIndexVar: weightIndexVar)
    }

    public init() {}
}

/// 3-bit affine quantization with group size 64.
public struct AffineQ3Group64Format: QuantizationFormat {
    public var schemeIdentifier: QuantizationSchemeIdentifier { .q3Group64ScaleF16 }
    public var blockStructName: String { "BlockQ3Affine64" }
    public var gemvKernelName: String { "gemv_q3_g64" }
    public func gemmKernelName(bufferPrecision: BufferPrecision) -> String {
        bufferPrecision == .float32 ? "gemm_q3_g64_f32s" : "gemm_q3_g64"
    }
    public var weightsPerBlock: Int { 64 }
    public var bytesPerBlock: Int { 4 + 24 }
    public var bits: Int { 3 }
    public var groupSize: Int { 64 }

    public var isQuantized: Bool { true }
    public var bufferElementType: String { "uchar" }
    public var mslDeclarations: String {
        Q3AffineMSL.blockStruct(name: blockStructName, weightsPerBlock: weightsPerBlock)
    }

    public func perWeightReadExpression(
        blocksVar: String,
        weightIndexVar: String
    ) -> String? {
        Q3AffineMSL.perWeightExpression(blocksVar: blocksVar, weightIndexVar: weightIndexVar)
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

    public func perWeightReadExpression(
        blocksVar: String,
        weightIndexVar: String
    ) -> String? {
        Q5AffineMSL.perWeightExpression(blocksVar: blocksVar, weightIndexVar: weightIndexVar)
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

    public func perWeightReadExpression(
        blocksVar: String,
        weightIndexVar: String
    ) -> String? {
        Q5AffineMSL.perWeightExpression(blocksVar: blocksVar, weightIndexVar: weightIndexVar)
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

    public func perWeightReadExpression(
        blocksVar: String,
        weightIndexVar: String
    ) -> String? {
        Q6AffineMSL.perWeightExpression(blocksVar: blocksVar, weightIndexVar: weightIndexVar)
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

    public func perWeightReadExpression(
        blocksVar: String,
        weightIndexVar: String
    ) -> String? {
        Q6AffineMSL.perWeightExpression(blocksVar: blocksVar, weightIndexVar: weightIndexVar)
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

    public func directGEMMKernel() -> DirectQuantizedGEMM? {
        DirectQuantizedGEMM(
            kernelName: "gemm_q8_g32_f32s",
            threadgroupMemoryLength: directQuantizedGEMMThreadgroupMemoryLength(groupSize: groupSize)
        )
    }

    public func directGEMMKernelSource(
        name: String,
        bufferPrecision: BufferPrecision
    ) -> String? {
        MetalSourceGenerator.generateQuantizedGEMM_Q8(
            name: name, bufferPrecision: bufferPrecision, groupSize: groupSize)
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

    public func directGEMMKernel() -> DirectQuantizedGEMM? {
        DirectQuantizedGEMM(
            kernelName: "gemm_q8_g64_f32s",
            threadgroupMemoryLength: directQuantizedGEMMThreadgroupMemoryLength(groupSize: groupSize)
        )
    }

    public func directGEMMKernelSource(
        name: String,
        bufferPrecision: BufferPrecision
    ) -> String? {
        MetalSourceGenerator.generateQuantizedGEMM_Q8(
            name: name, bufferPrecision: bufferPrecision, groupSize: groupSize)
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

    /// Per-weight dequant expression assuming `scale` and `zero` are captured
    /// as `float` locals by the caller. 8 weights share 3 bytes — group
    /// `g = k/8` lives at bytes `blocksVar[g*3 .. g*3+2]`, and slot
    /// `s = k%8` selects one of eight bit patterns.
    static func perWeightExpression(blocksVar: String, weightIndexVar: String) -> String {
        let k = "(\(weightIndexVar))"
        let qs = blocksVar
        let base = "((\(k)) >> 3) * 3"
        let slot = "((\(k)) & 7)"
        let b0 = "\(qs)[\(base) + 0]"
        let b1 = "\(qs)[\(base) + 1]"
        let b2 = "\(qs)[\(base) + 2]"
        return "(scale * float(" +
            "\(slot) == 0 ? (\(b0) & 0x07) : " +
            "\(slot) == 1 ? ((\(b0) >> 3) & 0x07) : " +
            "\(slot) == 2 ? (((\(b0) >> 6) & 0x03) | ((\(b1) & 0x01) << 2)) : " +
            "\(slot) == 3 ? ((\(b1) >> 1) & 0x07) : " +
            "\(slot) == 4 ? ((\(b1) >> 4) & 0x07) : " +
            "\(slot) == 5 ? (((\(b1) >> 7) & 0x01) | ((\(b2) & 0x03) << 1)) : " +
            "\(slot) == 6 ? ((\(b2) >> 2) & 0x07) : " +
            "((\(b2) >> 5) & 0x07)" +
            ") + zero)"
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

    /// Per-weight dequant expression assuming `scale` and `zero` are captured
    /// as `float` locals by the caller. 8 weights share 5 bytes — group
    /// `g = k/8` lives at bytes `blocksVar[g*5 .. g*5+4]`, and slot
    /// `s = k%8` selects one of eight bit patterns.
    static func perWeightExpression(blocksVar: String, weightIndexVar: String) -> String {
        let k = "(\(weightIndexVar))"
        let qs = blocksVar
        let base = "((\(k)) >> 3) * 5"
        let slot = "((\(k)) & 7)"
        let b0 = "\(qs)[\(base) + 0]"
        let b1 = "\(qs)[\(base) + 1]"
        let b2 = "\(qs)[\(base) + 2]"
        let b3 = "\(qs)[\(base) + 3]"
        let b4 = "\(qs)[\(base) + 4]"
        return "(scale * float(" +
            "\(slot) == 0 ? (\(b0) & 0x1f) : " +
            "\(slot) == 1 ? (((\(b0) >> 5) & 0x07) | ((\(b1) & 0x03) << 3)) : " +
            "\(slot) == 2 ? ((\(b1) >> 2) & 0x1f) : " +
            "\(slot) == 3 ? (((\(b1) >> 7) & 0x01) | ((\(b2) & 0x0f) << 1)) : " +
            "\(slot) == 4 ? (((\(b2) >> 4) & 0x0f) | ((\(b3) & 0x01) << 4)) : " +
            "\(slot) == 5 ? ((\(b3) >> 1) & 0x1f) : " +
            "\(slot) == 6 ? (((\(b3) >> 6) & 0x03) | ((\(b4) & 0x07) << 2)) : " +
            "((\(b4) >> 3) & 0x1f)" +
            ") + zero)"
    }

}

/// MSL source fragments for Q6 affine block formats.
///
/// Non-aligned packing: 4 weights share 3 bytes (see MLX
/// `mlx/backend/cpu/quantized.cpp` `extract_bits<6>`).
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

    /// Per-weight dequant expression assuming `scale` and `zero` are captured
    /// as `float` locals by the caller. Reads weight `k` from the packed
    /// stream at `blocksVar`: 4 weights share 3 bytes, so group `g = k/4`
    /// lives at bytes `blocksVar[g*3 .. g*3+2]`, and slot `s = k%4` selects
    /// one of four bit patterns. The expression is a single ternary chain so
    /// the Metal compiler can CSE shared sub-expressions and emit tight
    /// branch-free selection in the common case (k varies by simdgroup
    /// thread).
    static func perWeightExpression(blocksVar: String, weightIndexVar: String) -> String {
        let k = "(\(weightIndexVar))"
        let qs = blocksVar
        let base = "((\(k)) >> 2) * 3"
        let slot = "((\(k)) & 3)"
        let b0 = "\(qs)[\(base) + 0]"
        let b1 = "\(qs)[\(base) + 1]"
        let b2 = "\(qs)[\(base) + 2]"
        return "(scale * float(" +
            "\(slot) == 0 ? (\(b0) & 0x3f) : " +
            "\(slot) == 1 ? (((\(b0) >> 6) & 0x03) | ((\(b1) & 0x0f) << 2)) : " +
            "\(slot) == 2 ? (((\(b1) >> 4) & 0x0f) | ((\(b2) & 0x03) << 4)) : " +
            "(\(b2) >> 2)" +
            ") + zero)"
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
        case .q3Group64ScaleF16: return AffineQ3Group64Format()
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
        case (3, 64): return AffineQ3Group64Format()
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
