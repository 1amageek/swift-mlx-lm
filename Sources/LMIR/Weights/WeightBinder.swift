/// Binds raw checkpoint tensors to semantic parameter slots in the model graph.
///
/// The binder maps checkpoint naming conventions (e.g., GGUF tensor names)
/// into semantic parameter slots defined by the model graph.
/// This is where structure-dependent meaning becomes explicit.
public protocol WeightBinder: Sendable {

    /// Bind raw tensors to the parameter slots expected by the model graph.
    ///
    /// - Parameters:
    ///   - raw: Raw weights with checkpoint-native tensor names.
    ///   - graph: The model graph defining expected parameter slots.
    /// - Returns: Weights bound to semantic parameter slots.
    /// - Throws: If required tensors are missing, shapes are incompatible,
    ///   or unexpected tensors are present (depending on policy).
    func bind(_ raw: RawWeights, to graph: ModelGraph) throws -> BoundWeights
}
