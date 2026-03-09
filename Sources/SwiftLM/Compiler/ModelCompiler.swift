/// Transforms a declarative model definition into an executable compiled model.
///
/// The compiler is the semantic boundary where declaration becomes executable.
/// It performs:
/// - Canonicalization
/// - Validation
/// - Binding verification
/// - Semantic-to-operational lowering (ModelGraph → LoweredGraph)
/// - Graph optimization
/// - Runtime plan generation
public protocol ModelCompiler: Sendable {

    /// The compiled artifact type produced by this compiler.
    associatedtype Compiled: Sendable

    /// Compile a model graph with bound weights into an executable form.
    func compile(
        graph: ModelGraph,
        weights: BoundWeights
    ) throws -> Compiled
}
