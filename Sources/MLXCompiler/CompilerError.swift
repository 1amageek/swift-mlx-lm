import SwiftLM

/// Errors produced by the MLX compiler and executor.
public enum CompilerError: Error, Sendable {

    /// A required weight was not found in the weight store.
    case missingWeight(ParameterSlot)

    /// The weight storage type was not MLXArray.
    case invalidWeightStorage(ParameterSlot, String)

    /// An operation kind is not supported by the executor.
    case unsupportedOperation(String)

    /// A state-space variant is not implemented.
    case unsupportedVariant(String)

    /// The model graph has an invalid structure.
    case invalidGraphStructure(String)

    /// A runtime execution error occurred.
    case executionError(String)
}

extension CompilerError: CustomStringConvertible {

    public var description: String {
        switch self {
        case .missingWeight(let slot):
            return "Missing weight: path=\(slot.path.components), role=\(slot.role)"
        case .invalidWeightStorage(let slot, let detail):
            return "Invalid weight storage at \(slot.path.components): \(detail)"
        case .unsupportedOperation(let name):
            return "Unsupported operation: \(name)"
        case .unsupportedVariant(let variant):
            return "Unsupported state-space variant: \(variant)"
        case .invalidGraphStructure(let detail):
            return "Invalid graph structure: \(detail)"
        case .executionError(let detail):
            return "Execution error: \(detail)"
        }
    }
}
