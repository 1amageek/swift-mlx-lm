// MARK: - Structure-only Operations

extension ModelComponent {

    /// Produce the normalized (structurally closed) semantic IR for this component.
    ///
    /// Returns the `NormalizedModel` containing both the semantic graph
    /// and diagnostic metadata. The graph is well-formed but NOT
    /// canonicalized. For equivalence comparison, pass `result.graph`
    /// through `canonicalize(_:)`.
    public func makeNormalizedModel() throws -> NormalizedModel {
        try normalize(self)
    }

    /// Convenience: produce just the semantic graph (discarding metadata).
    public func makeModelGraph() throws -> ModelGraph {
        try normalize(self).graph
    }
}

