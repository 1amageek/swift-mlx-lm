import Foundation

/// Errors that can occur when parsing a GGUF file.
public enum GGUFError: Error, LocalizedError {
    case invalidMagic(UInt32)
    case unsupportedVersion(UInt32)
    case unexpectedEndOfData(context: String)
    case invalidMetadataValueType(UInt32)
    case invalidQuantizationType(UInt32)
    case metadataKeyNotFound(String)
    case metadataTypeMismatch(key: String, expected: String, actual: String)

    public var errorDescription: String? {
        switch self {
        case .invalidMagic(let magic):
            return "Invalid GGUF magic number: 0x\(String(magic, radix: 16)). Expected 0x46554747 ('GGUF')."
        case .unsupportedVersion(let version):
            return "Unsupported GGUF version: \(version). Supported versions: 2, 3."
        case .unexpectedEndOfData(let context):
            return "Unexpected end of data while reading \(context)."
        case .invalidMetadataValueType(let raw):
            return "Invalid metadata value type: \(raw)."
        case .invalidQuantizationType(let raw):
            return "Invalid quantization type: \(raw)."
        case .metadataKeyNotFound(let key):
            return "Required metadata key not found: '\(key)'."
        case .metadataTypeMismatch(let key, let expected, let actual):
            return "Metadata type mismatch for '\(key)': expected \(expected), got \(actual)."
        }
    }
}
