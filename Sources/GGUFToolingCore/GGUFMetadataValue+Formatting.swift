import GGUFParser

public extension GGUFMetadataValue {
    var ggufType: GGUFMetadataValueType {
        switch self {
        case .uint8:
            return .uint8
        case .int8:
            return .int8
        case .uint16:
            return .uint16
        case .int16:
            return .int16
        case .uint32:
            return .uint32
        case .int32:
            return .int32
        case .float32:
            return .float32
        case .bool:
            return .bool
        case .string:
            return .string
        case .uint64:
            return .uint64
        case .int64:
            return .int64
        case .float64:
            return .float64
        case .array:
            return .array
        }
    }

    var displayString: String {
        switch self {
        case .uint8(let value):
            return String(value)
        case .int8(let value):
            return String(value)
        case .uint16(let value):
            return String(value)
        case .int16(let value):
            return String(value)
        case .uint32(let value):
            return String(value)
        case .int32(let value):
            return String(value)
        case .float32(let value):
            return String(value)
        case .bool(let value):
            return value ? "true" : "false"
        case .string(let value):
            return value
        case .uint64(let value):
            return String(value)
        case .int64(let value):
            return String(value)
        case .float64(let value):
            return String(value)
        case .array(let values):
            return "[" + values.map(\.displayString).joined(separator: ", ") + "]"
        }
    }
}
