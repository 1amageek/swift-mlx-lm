/// A declarative language model architecture definition.
///
/// `LanguageModel` declares ONLY the structural topology of a model.
/// Weights are a separate concern, attached later via the `.weights(_:)` modifier.
///
/// `LanguageModel` is NOT:
/// - a file loader
/// - a checkpoint artifact
/// - a stateful runtime module
/// - a forward-pass executor
/// - a weight declaration
///
/// `LanguageModel` IS:
/// - a pure architecture declaration
///
/// ```swift
/// struct Qwen3_5: LanguageModel {
///     var body: some ModelComponent {
///         TokenEmbedding(vocabSize: 151936, embeddingSize: 4096)
///         Repeat(count: 32) {
///             Residual {
///                 RMSNorm(dimension: 4096)
///                 Attention(hiddenSize: 4096, headCount: 32, kvHeadCount: 8)
///             }
///             Residual {
///                 RMSNorm(dimension: 4096)
///                 MLP(inputSize: 4096, intermediateSize: 11008)
///             }
///         }
///         RMSNorm(dimension: 4096, epsilon: 1e-6)
///         OutputHead(inputSize: 4096, vocabSize: 151936)
///     }
/// }
///
/// // Weights are attached externally:
/// let weighted = Qwen3_5().weights(.gguf(location: "qwen3.5.gguf"))
/// ```
public protocol LanguageModel: Sendable {

    /// The type of structural body for this model.
    associatedtype Body: ModelComponent

    /// Declarative structural body.
    @ModelComponentBuilder var body: Body { get }
}

// MARK: - Structure-only Operations

extension LanguageModel {

    /// Produce the open declaration tree for this model's body.
    public func makeModelDeclaration() -> ModelDeclaration {
        body.makeDeclaration()
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

// MARK: - Weight Modifier

extension LanguageModel {

    /// Attach a weight declaration to this model.
    ///
    /// Returns a `WeightedModel` that bundles the architecture and weight source.
    /// The model itself is unchanged — weights are an external annotation.
    ///
    /// ```swift
    /// let model = Qwen35(config: .qwen35_0_8B)
    /// let weighted = model.weights(.gguf(location: "model.gguf"))
    /// ```
    public func weights(_ declaration: WeightsDeclaration) -> WeightedModel<Self> {
        WeightedModel(model: self, weightsDeclaration: declaration)
    }

    /// Attach a composed weight declaration using a builder.
    ///
    /// ```swift
    /// let weighted = model.weights {
    ///     WeightsDeclaration.gguf(location: "base.gguf")
    ///     WeightsDeclaration.safetensors(directory: "adapter/", indexFile: nil)
    /// }
    /// ```
    public func weights(
        @WeightsBuilder _ builder: () -> WeightsDeclaration
    ) -> WeightedModel<Self> {
        WeightedModel(model: self, weightsDeclaration: builder())
    }
}

/// A model bundled with a weight declaration.
///
/// `WeightedModel` is produced by the `.weights(_:)` modifier on `LanguageModel`.
/// It carries both the structural graph and the weight source, ready for
/// resolution and compilation.
///
/// `WeightedModel` is NOT a `LanguageModel`. This is intentional — it represents
/// a different concept: a structure-plus-weights bundle, not a pure structure.
///
/// ```swift
/// let weighted = Qwen35(config: .qwen35_0_8B)
///     .weights(.gguf(location: "model.gguf"))
///
/// let graph = try weighted.makeModelGraph()
/// let weightsDecl = weighted.weightsDeclaration
/// ```
public struct WeightedModel<M: LanguageModel>: Sendable {

    /// The underlying model (structure only).
    public let model: M

    /// The weight declaration attached to this model.
    public let weightsDeclaration: WeightsDeclaration

    public init(model: M, weightsDeclaration: WeightsDeclaration) {
        self.model = model
        self.weightsDeclaration = weightsDeclaration
    }

    /// Produce the open declaration tree for the model's body.
    public func makeModelDeclaration() -> ModelDeclaration {
        model.makeModelDeclaration()
    }

    /// Produce the normalized semantic IR for the model.
    public func makeNormalizedModel() throws -> NormalizedModel {
        try model.makeNormalizedModel()
    }

    /// Produce just the semantic graph.
    public func makeModelGraph() throws -> ModelGraph {
        try model.makeModelGraph()
    }

    /// Replace the weight declaration with a different one.
    public func weights(_ declaration: WeightsDeclaration) -> WeightedModel<M> {
        WeightedModel(model: model, weightsDeclaration: declaration)
    }

    /// Replace the weight declaration using a builder.
    ///
    /// ```swift
    /// let updated = weighted.weights {
    ///     WeightsDeclaration.gguf(location: "new-base.gguf")
    ///     WeightsDeclaration.safetensors(directory: "adapter/", indexFile: nil)
    /// }
    /// ```
    public func weights(
        @WeightsBuilder _ builder: () -> WeightsDeclaration
    ) -> WeightedModel<M> {
        WeightedModel(model: model, weightsDeclaration: builder())
    }
}
