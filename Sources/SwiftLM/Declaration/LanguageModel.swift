/// A declarative language model definition.
///
/// `LanguageModel` combines a structural declaration (`body`) with a weight
/// declaration (`weights`). It is the top-level user-facing abstraction in SwiftLM.
///
/// `LanguageModel` is NOT:
/// - a file loader
/// - a checkpoint artifact
/// - a stateful runtime module
/// - a forward-pass executor
///
/// `LanguageModel` IS:
/// - an architecture declaration + weight declaration
///
/// ```swift
/// struct Qwen3_5: LanguageModel {
///     var weights: some WeightsSpec {
///         WeightsDeclaration.gguf(location: "qwen3.5.gguf")
///     }
///
///     var body: some ModelComponent {
///         TokenEmbedding(vocabSize: 151936, embeddingSize: 4096)
///         Repeat(count: 32) {
///             TransformerBlock(
///                 hiddenSize: 4096,
///                 headCount: 32,
///                 kvHeadCount: 8,
///                 intermediateSize: 11008
///             )
///         }
///         RMSNorm(dimension: 4096, epsilon: 1e-6)
///         OutputHead(inputSize: 4096, vocabSize: 151936)
///     }
/// }
/// ```
public protocol LanguageModel: Sendable {

    /// The type of weight specification for this model.
    associatedtype Weights: WeightsSpec

    /// The type of structural body for this model.
    associatedtype Body: ModelComponent

    /// Declarative weight specification.
    @WeightsBuilder var weights: Weights { get }

    /// Declarative structural body.
    @ModelComponentBuilder var body: Body { get }
}

extension LanguageModel {

    /// Produce the open declaration tree for this model's body.
    public func makeModelDeclaration() -> ModelDeclaration {
        body.makeDeclaration()
    }

    /// Produce the weight declaration for this model.
    public func makeWeightsDeclaration() -> WeightsDeclaration {
        weights.makeDeclaration()
    }

    /// Produce the normalized (structurally closed) semantic IR for this model.
    ///
    /// Returns the `NormalizedModel` containing both the semantic graph
    /// and diagnostic metadata. The graph is well-formed but NOT
    /// canonicalized. For equivalence comparison, pass `result.graph`
    /// through `canonicalize(_:)`.
    public func makeNormalizedModel() throws -> NormalizedModel {
        try normalize(body.makeDeclaration())
    }

    /// Convenience: produce just the semantic graph (discarding metadata).
    public func makeModelGraph() throws -> ModelGraph {
        try normalize(body.makeDeclaration()).graph
    }
}
