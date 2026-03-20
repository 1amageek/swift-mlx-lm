import Foundation

public enum STAFMetadataValue: Sendable, Equatable {
    case bool(Bool)
    case uint32(UInt32)
    case uint64(UInt64)
    case float32(Float)
    case float64(Double)
    case string(String)
}

enum STAFMetadataValueType: UInt8, Sendable {
    case bool = 0x01
    case uint32 = 0x02
    case uint64 = 0x03
    case float32 = 0x04
    case float64 = 0x05
    case string = 0x06
}

extension STAFMetadataValue {
    var valueType: STAFMetadataValueType {
        switch self {
        case .bool:
            return .bool
        case .uint32:
            return .uint32
        case .uint64:
            return .uint64
        case .float32:
            return .float32
        case .float64:
            return .float64
        case .string:
            return .string
        }
    }
}
