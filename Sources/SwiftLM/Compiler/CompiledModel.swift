/// A validated, bound, executable model package.
///
/// Produced by `ModelCompiler` from a `ModelGraph` and `BoundWeights`.
/// Contains everything needed for an `Executor` to run inference.
public struct CompiledModel: Sendable {

    /// The original semantic graph (preserved for diagnostics and identity).
    public let semanticGraph: ModelGraph

    /// The lowered execution graph.
    public let loweredGraph: LoweredGraph

    /// Weights bound to semantic parameter slots.
    public let weights: BoundWeights

    /// Runtime execution plan produced by the compiler.
    public let runtimePlan: RuntimePlan

    public init(
        semanticGraph: ModelGraph,
        loweredGraph: LoweredGraph,
        weights: BoundWeights,
        runtimePlan: RuntimePlan
    ) {
        self.semanticGraph = semanticGraph
        self.loweredGraph = loweredGraph
        self.weights = weights
        self.runtimePlan = runtimePlan
    }
}

/// Runtime execution plan produced by the compiler.
///
/// Contains backend-specific scheduling, memory layout, and optimization
/// decisions. The concrete implementation depends on the backend.
public struct RuntimePlan: Sendable {

    /// Opaque backend-specific plan data.
    public let data: any Sendable

    public init(data: any Sendable) {
        self.data = data
    }
}
