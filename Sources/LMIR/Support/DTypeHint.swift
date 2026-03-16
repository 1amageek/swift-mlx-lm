/// Hint for preferred tensor data type.
///
/// This is a declarative hint, not a runtime enforcement.
/// The compiler/executor decides the actual dtype based on backend capabilities.
///
/// Integer cases correspond to quantized representations where the bit width
/// matches MLX `quantizedMM` supported widths (2, 3, 4, 5, 6, 8).
public enum DTypeHint: String, Codable, Equatable, Sendable {
    case float16
    case bfloat16
    case float32
    case int2
    case int3
    case int4
    case int5
    case int6
    case int8
}
