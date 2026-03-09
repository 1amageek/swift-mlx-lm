/// Hint for preferred tensor data type.
///
/// This is a declarative hint, not a runtime enforcement.
/// The compiler/executor decides the actual dtype based on backend capabilities.
public enum DTypeHint: String, Codable, Equatable, Sendable {
    case float16
    case bfloat16
    case float32
    case int8
    case int4
}
