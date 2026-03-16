/// A semantic address for a parameter within a model graph.
///
/// Parameter slots identify where a weight tensor belongs in the model structure.
/// The binder maps raw checkpoint tensor names into parameter slots.
///
/// Uses `StructuralPath` for stable addressing that survives canonicalization.
public struct ParameterSlot: Hashable, Codable, Sendable {

    /// Stable structural path to the parameter location.
    public let path: StructuralPath

    /// Semantic role of the parameter.
    public let role: ParameterRole

    public init(path: StructuralPath, role: ParameterRole) {
        self.path = path
        self.role = role
    }
}

/// Semantic role of a parameter tensor.
public enum ParameterRole: Hashable, Codable, Sendable {
    case weight
    case bias
    case scale
    case embeddingTable
    case outputProjection
    case custom(String)
}
