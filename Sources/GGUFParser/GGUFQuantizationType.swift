/// Quantization type for GGUF tensor data.
///
/// Values match the GGML type enum used in the GGUF specification.
public enum GGUFQuantizationType: UInt32, Sendable, CaseIterable {
    case f32 = 0
    case f16 = 1
    case q4_0 = 2
    case q4_1 = 3
    case q5_0 = 6
    case q5_1 = 7
    case q8_0 = 8
    case q8_1 = 9
    case q2_K = 10
    case q3_K = 11
    case q4_K = 12
    case q5_K = 13
    case q6_K = 14
    case q8_K = 15
    case iq2_XXS = 16
    case iq2_XS = 17
    case iq3_XXS = 18
    case iq1_S = 19
    case iq4_NL = 20
    case iq3_S = 21
    case iq2_S = 22
    case iq4_XS = 23
    case i8 = 24
    case i16 = 25
    case i32 = 26
    case i64 = 27
    case f64 = 28
    case iq1_M = 29
    case bf16 = 30
    case tq1_0 = 34
    case tq2_0 = 35

    /// Number of bytes per block for this quantization type.
    public var blockSize: Int {
        switch self {
        case .f32: return 4
        case .f16, .bf16: return 2
        case .f64: return 8
        case .i8: return 1
        case .i16: return 2
        case .i32: return 4
        case .i64: return 8
        case .q4_0: return 18      // 32 elements: 2 (f16 scale) + 16 (4-bit values)
        case .q4_1: return 20      // 32 elements: 2 (f16 scale) + 2 (f16 min) + 16
        case .q5_0: return 22      // 32 elements: 2 + 4 (high bits) + 16
        case .q5_1: return 24      // 32 elements: 2 + 2 + 4 + 16
        case .q8_0: return 34      // 32 elements: 2 (f16 scale) + 32 (int8 values)
        case .q8_1: return 36      // 32 elements: 4 (f32 scale) + 32
        case .q2_K: return 84      // 256 elements: scales(16) + qs(64) + d(2) + dmin(2)
        case .q3_K: return 110     // 256 elements: hmask(32) + qs(64) + scales(12) + d(2)
        case .q4_K: return 144     // 256 elements
        case .q5_K: return 176     // 256 elements
        case .q6_K: return 210     // 256 elements
        case .q8_K: return 292     // 256 elements
        case .iq2_XXS: return 66
        case .iq2_XS: return 74
        case .iq2_S: return 82
        case .iq3_XXS: return 98
        case .iq3_S: return 110
        case .iq1_S: return 50
        case .iq1_M: return 56
        case .iq4_NL: return 18
        case .iq4_XS: return 136    // 256 elements: d(2) + scales_h(2) + scales_l(64) + qs(64) + sign(4)
        case .tq1_0: return 54
        case .tq2_0: return 66     // 256 elements: qs(64) + d(2)
        }
    }

    /// Number of elements per block.
    public var elementsPerBlock: Int {
        switch self {
        case .f32, .f16, .bf16, .f64, .i8, .i16, .i32, .i64:
            return 1
        case .q4_0, .q4_1, .q5_0, .q5_1, .q8_0, .q8_1, .iq4_NL:
            return 32
        case .q2_K, .q3_K, .q4_K, .q5_K, .q6_K, .q8_K:
            return 256
        case .iq2_XXS, .iq2_XS, .iq2_S, .iq3_XXS, .iq3_S, .iq1_S, .iq1_M:
            return 256
        case .iq4_XS:
            return 256
        case .tq1_0:
            return 256
        case .tq2_0:
            return 256
        }
    }

    /// Whether this is an unquantized (float/int) type.
    public var isUnquantized: Bool {
        switch self {
        case .f32, .f16, .bf16, .f64, .i8, .i16, .i32, .i64:
            return true
        default:
            return false
        }
    }
}
