/// A 16-bit brain floating-point value.
///
/// BFloat16 uses the same exponent range as Float32 (8-bit exponent)
/// with reduced precision (7-bit mantissa). This makes it ideal for
/// neural network weights where dynamic range matters more than precision.
///
/// ```
/// BFloat16: [sign:1][exponent:8][mantissa:7]
/// Float16:  [sign:1][exponent:5][mantissa:10]
/// Float32:  [sign:1][exponent:8][mantissa:23]
/// ```
///
/// BFloat16 is the upper 16 bits of a Float32, so conversion is a simple shift.
public struct BFloat16: Sendable, Equatable, Hashable {

    /// The raw bit pattern stored as UInt16.
    public var bitPattern: UInt16

    /// Create a BFloat16 from its raw bit pattern.
    public init(bitPattern: UInt16) {
        self.bitPattern = bitPattern
    }

    /// Convert a Float32 value to BFloat16 with round-to-nearest-even.
    public init(_ value: Float) {
        let bits = value.bitPattern
        // Round to nearest even: add rounding bias based on LSB of BF16 result
        let lsb = (bits >> 16) & 1
        let roundingBias = 0x7FFF &+ lsb
        let rounded = bits &+ roundingBias
        self.bitPattern = UInt16(rounded >> 16)
    }

    /// Convert a Float16 value to BFloat16 via Float32.
    public init(_ value: Float16) {
        self.init(Float(value))
    }

    // MARK: - Conversion to Other Types

    /// The Float32 representation (exact — BF16 is the upper 16 bits of Float32).
    public var floatValue: Float {
        Float(bitPattern: UInt32(bitPattern) << 16)
    }

    /// The Float16 representation (may lose precision if outside Float16 range).
    public var float16Value: Float16 {
        Float16(floatValue)
    }

    // MARK: - Special Values

    public static let zero = BFloat16(bitPattern: 0x0000)
    public static let one = BFloat16(bitPattern: 0x3F80)
    public static let nan = BFloat16(bitPattern: 0x7FC0)
    public static let infinity = BFloat16(bitPattern: 0x7F80)
    public static let negativeInfinity = BFloat16(bitPattern: 0xFF80)

    // MARK: - Classification

    public var isNaN: Bool {
        (bitPattern & 0x7F80) == 0x7F80 && (bitPattern & 0x007F) != 0
    }

    public var isInfinite: Bool {
        (bitPattern & 0x7FFF) == 0x7F80
    }

    public var isZero: Bool {
        (bitPattern & 0x7FFF) == 0
    }

    public var isFinite: Bool {
        (bitPattern & 0x7F80) != 0x7F80
    }

    public var isNegative: Bool {
        (bitPattern & 0x8000) != 0
    }
}

// MARK: - Literals

extension BFloat16: ExpressibleByFloatLiteral {
    public init(floatLiteral value: Float) {
        self.init(value)
    }
}

extension BFloat16: ExpressibleByIntegerLiteral {
    public init(integerLiteral value: Int) {
        self.init(Float(value))
    }
}

// MARK: - Comparable

extension BFloat16: Comparable {
    public static func < (lhs: BFloat16, rhs: BFloat16) -> Bool {
        lhs.floatValue < rhs.floatValue
    }
}

// MARK: - CustomStringConvertible

extension BFloat16: CustomStringConvertible {
    public var description: String {
        floatValue.description
    }
}

// MARK: - Float Interop

extension Float {
    /// Create a Float32 from a BFloat16 value (exact).
    public init(_ value: BFloat16) {
        self = Float(bitPattern: UInt32(value.bitPattern) << 16)
    }
}

extension Float16 {
    /// Create a Float16 from a BFloat16 value (may lose precision).
    public init(_ value: BFloat16) {
        self = Float16(Float(value))
    }
}
