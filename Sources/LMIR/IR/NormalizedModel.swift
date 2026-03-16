/// The result of normalizing a `ModelDeclaration`.
///
/// Bundles the semantic `ModelGraph` with its diagnostic `ModelGraphMetadata`.
/// The graph contains only semantic information; labels and other diagnostics
/// are in the metadata sidecar.
///
/// ```swift
/// let normalized = try normalize(model.body.makeDeclaration())
/// let graph = normalized.graph           // for compilation
/// let metadata = normalized.metadata     // for diagnostics
/// let canonical = canonicalize(graph)    // for equivalence comparison
/// ```
public struct NormalizedModel: Sendable {

    /// The semantic graph (pure structural information).
    public let graph: ModelGraph

    /// Diagnostic metadata (labels, source locations).
    public let metadata: ModelGraphMetadata

    public init(graph: ModelGraph, metadata: ModelGraphMetadata = ModelGraphMetadata()) {
        self.graph = graph
        self.metadata = metadata
    }
}
