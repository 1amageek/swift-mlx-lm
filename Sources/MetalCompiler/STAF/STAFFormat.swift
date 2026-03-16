import Foundation

// MARK: - STAF File Format Constants

/// SafeTensor Accelerated Format — GPU-ready executable cache.
///
/// STAF is a zero-copy GPU weight format derived from safetensors.
/// Weights are stored in interleaved quantization blocks that Metal
/// kernels read directly. The safetensors files remain source of truth.
public enum STAF {

    /// Magic bytes: "STAF" (0x53544146)
    public static let magic: UInt32 = 0x46415453  // little-endian "STAF"

    /// File header size in bytes (packed, no alignment padding).
    public static let headerSize: Int = 64

    /// Section table entry size in bytes (packed, no alignment padding).
    public static let sectionEntrySize: Int = 128

    /// Payload alignment (must match page size for bytesNoCopy).
    public static let payloadAlignment: Int = 4096

    /// Per-tensor alignment within payload.
    public static let tensorAlignment: Int = 256

    /// Maximum number of shape dimensions.
    public static let maximumDimensions: Int = 8
}

// MARK: - File Header (64 bytes)

/// STAF file header. Fixed 64-byte structure at offset 0.
///
/// Cache invalidation uses file metadata (mtime), not content hashing.
/// STAF is a regenerable cache — if invalid, delete and reconvert.
public struct STAFHeader: Sendable {
    /// Magic: "STAF" (0x53544146 little-endian)
    public var magic: UInt32
    /// Reserved for future use (zero).
    public var reserved0: (
        UInt8, UInt8, UInt8, UInt8,
        UInt8, UInt8, UInt8, UInt8,
        UInt8, UInt8, UInt8, UInt8,
        UInt8, UInt8, UInt8, UInt8,
        UInt8, UInt8, UInt8, UInt8,
        UInt8, UInt8, UInt8, UInt8,
        UInt8, UInt8, UInt8, UInt8,
        UInt8, UInt8, UInt8, UInt8,
        UInt8, UInt8, UInt8, UInt8
    )
    /// Total number of tensors.
    public var sectionCount: UInt32
    /// Byte offset of the section table.
    public var sectionTableOffset: UInt32
    /// Byte offset of the string table.
    public var stringTableOffset: UInt32
    /// Byte size of the string table.
    public var stringTableSize: UInt32
    /// Reserved for future use (zero).
    public var reserved1: (UInt8, UInt8, UInt8, UInt8, UInt8, UInt8, UInt8, UInt8)
}

// MARK: - Section Table Entry (128 bytes)

/// Metadata for a single tensor in the STAF file.
public struct STAFSectionEntry: Sendable {
    /// Byte offset of tensor name in string table.
    public var nameOffset: UInt32
    /// Byte length of tensor name (excluding null terminator).
    public var nameLength: UInt32
    /// Quantization scheme identifier.
    public var quantizationSchemeIdentifier: UInt8
    /// Semantic role hint for runtime.
    public var semanticRole: UInt8
    /// Original dtype from safetensors.
    public var originalDType: UInt8
    /// Number of dimensions (max 8).
    public var dimensionCount: UInt8
    /// Shape (up to 8 dimensions, unused set to 0).
    public var shape: (UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32)
    /// Byte offset of payload from file start.
    public var payloadOffset: UInt64
    /// Byte size of payload.
    public var payloadSize: UInt64
    /// Alignment applied to this tensor's payload.
    public var alignment: UInt32
    /// Quantization block size (number of weights per block).
    public var blockSize: UInt32
    /// Quantization group size (number of weights per scale group).
    public var groupSize: UInt32
    /// CRC-32 checksum of payload.
    public var checksum: UInt32
    /// Index of source safetensors shard (0 for single file).
    public var shardIndex: UInt32
    /// Reserved for future use.
    public var reserved: (
        UInt8, UInt8, UInt8, UInt8, UInt8, UInt8, UInt8, UInt8,
        UInt8, UInt8, UInt8, UInt8, UInt8, UInt8, UInt8, UInt8,
        UInt8, UInt8, UInt8, UInt8, UInt8, UInt8, UInt8, UInt8,
        UInt8, UInt8, UInt8, UInt8, UInt8, UInt8, UInt8, UInt8,
        UInt8, UInt8, UInt8, UInt8, UInt8, UInt8, UInt8, UInt8,
        UInt8, UInt8, UInt8, UInt8, UInt8, UInt8, UInt8, UInt8
    )
}

// MARK: - Quantization Scheme Identifier

/// Identifies the quantization format and block layout of a tensor's payload.
///
/// The runtime uses this to select the correct GEMV kernel.
/// Each ID fully specifies: bit width, group size, scale type, and block layout.
public enum QuantizationSchemeIdentifier: UInt8, Sendable, CaseIterable {

    // Dense formats
    case fp16RowMajor       = 0x00
    case bf16RowMajor       = 0x01
    case fp32RowMajor       = 0x02

    // INT8
    case q8Group32ScaleF16  = 0x10
    case q8Group64ScaleF16  = 0x11
    case q8Group128ScaleF16 = 0x12

    // INT6
    case q6Group16ScaleF16  = 0x20
    case q6Group32ScaleF16  = 0x21

    // INT5
    case q5Group32ScaleF16  = 0x30
    case q5Group64ScaleF16  = 0x31

    // INT4
    case q4Group64ScaleF16  = 0x40
    case q4Group128ScaleF16 = 0x41
    case q4Group128ScaleF16Zero = 0x42

    // INT3
    case q3Group16ScaleF16  = 0x50
    case q3Group32ScaleF16  = 0x51

    // INT2
    case q2Group16ScaleF16  = 0x60
    case q2Group32ScaleF16  = 0x61

    // Passthrough (unknown tensor, stored as FP16)
    case passthrough        = 0xFF
}

// MARK: - Semantic Role

/// Hint for runtime to identify tensor purpose without name parsing.
public enum SemanticRole: UInt8, Sendable {
    case unknown           = 0x00
    case tokenEmbedding    = 0x01
    case attentionQuery    = 0x02
    case attentionKey      = 0x03
    case attentionValue    = 0x04
    case attentionOutput   = 0x05
    case mlpGate           = 0x06
    case mlpUp             = 0x07
    case mlpDown           = 0x08
    case normWeight        = 0x09
    case languageModelHead = 0x0A
    case moeExpertGate     = 0x0B
    case moeExpertUp       = 0x0C
    case moeExpertDown     = 0x0D
    case moeRouter         = 0x0E
    case other             = 0xFF
}

// MARK: - Original DType

/// Records the dtype of the source safetensors tensor.
public enum OriginalDType: UInt8, Sendable {
    case float32  = 0x00
    case float16  = 0x01
    case bfloat16 = 0x02
    case int32    = 0x03
    case int16    = 0x04
    case int8     = 0x05
    case int4     = 0x06
    case unknown  = 0xFF
}
