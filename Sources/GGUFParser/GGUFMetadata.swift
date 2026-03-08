/// Type-safe representation of a GGUF metadata value.
///
/// GGUF metadata supports a fixed set of value types.
/// This enum provides exhaustive, type-safe access without `Any` casts.
public enum GGUFMetadataValue: Sendable, Equatable {
    case uint8(UInt8)
    case int8(Int8)
    case uint16(UInt16)
    case int16(Int16)
    case uint32(UInt32)
    case int32(Int32)
    case float32(Float)
    case bool(Bool)
    case string(String)
    case uint64(UInt64)
    case int64(Int64)
    case float64(Double)
    case array([GGUFMetadataValue])

    // MARK: - Convenience Accessors

    public var stringValue: String? {
        if case .string(let v) = self { return v }
        return nil
    }

    public var uint32Value: UInt32? {
        if case .uint32(let v) = self { return v }
        return nil
    }

    public var int32Value: Int32? {
        if case .int32(let v) = self { return v }
        return nil
    }

    public var uint64Value: UInt64? {
        if case .uint64(let v) = self { return v }
        return nil
    }

    public var int64Value: Int64? {
        if case .int64(let v) = self { return v }
        return nil
    }

    public var float32Value: Float? {
        if case .float32(let v) = self { return v }
        return nil
    }

    public var float64Value: Double? {
        if case .float64(let v) = self { return v }
        return nil
    }

    public var boolValue: Bool? {
        if case .bool(let v) = self { return v }
        return nil
    }

    public var arrayValue: [GGUFMetadataValue]? {
        if case .array(let v) = self { return v }
        return nil
    }

    /// Interpret the value as an integer, coercing from any integer type.
    public var intValue: Int? {
        switch self {
        case .uint8(let v): return Int(v)
        case .int8(let v): return Int(v)
        case .uint16(let v): return Int(v)
        case .int16(let v): return Int(v)
        case .uint32(let v): return Int(v)
        case .int32(let v): return Int(v)
        case .uint64(let v): return Int(exactly: v)
        case .int64(let v): return Int(exactly: v)
        default: return nil
        }
    }

    /// Interpret the value as a floating-point number.
    public var doubleValue: Double? {
        switch self {
        case .float32(let v): return Double(v)
        case .float64(let v): return v
        default: return nil
        }
    }
}

/// Metadata value type identifier used in the GGUF binary format.
public enum GGUFMetadataValueType: UInt32, Sendable {
    case uint8 = 0
    case int8 = 1
    case uint16 = 2
    case int16 = 3
    case uint32 = 4
    case int32 = 5
    case float32 = 6
    case bool = 7
    case string = 8
    case array = 9
    case uint64 = 10
    case int64 = 11
    case float64 = 12
}
