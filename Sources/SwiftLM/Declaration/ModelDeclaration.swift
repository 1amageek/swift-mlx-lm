/// Open declaration tree produced by `ModelComponent.makeDeclaration()`.
///
/// `ModelDeclaration` is the intermediate representation between the user-facing
/// DSL and the closed semantic IR (`ModelGraph`). It captures the full
/// compositional structure of a model declaration without assigning IDs
/// or flattening regions.
///
/// All front-ends (Swift DSL, GGUF restorer, JSON restorer) produce
/// `ModelDeclaration` values. The `SemanticNormalizer` then converts them
/// into a canonical, region-bearing `ModelGraph`.
///
/// ```swift
/// let declaration: ModelDeclaration = .sequence([
///     .primitive(.tokenEmbedding(TokenEmbeddingAttributes(vocabSize: 32000, embeddingSize: 4096))),
///     .repeating(count: 32, label: "layers", body: .sequence([
///         .residual(strategy: .add, body: .sequence([
///             .primitive(.rmsNorm(RMSNormAttributes(dimension: 4096))),
///             .primitive(.attention(AttentionAttributes(hiddenSize: 4096, headCount: 32, kvHeadCount: 8, headDimension: 128))),
///         ])),
///     ])),
///     .primitive(.rmsNorm(RMSNormAttributes(dimension: 4096))),
///     .primitive(.outputHead(OutputHeadAttributes(inputSize: 4096, vocabSize: 32000))),
/// ])
/// ```
public indirect enum ModelDeclaration: Sendable, Equatable {

    /// A single semantic primitive (leaf node).
    case primitive(PrimitiveDeclaration)

    /// Sequential composition of declarations (flattened into a `Region` during normalization).
    case sequence([ModelDeclaration])

    /// Residual skip connection wrapping a body declaration.
    case residual(strategy: ResidualStrategy, body: ModelDeclaration)

    /// Parallel branches sharing the same input, merged by the given strategy.
    case parallel(merge: ParallelMergeStrategy, branches: [ModelDeclaration])

    /// Repeated block applied `count` times.
    case repeating(count: Int, label: String?, body: ModelDeclaration)

    /// Label for debugging and diagnostics (stripped during normalization).
    case labeled(String, ModelDeclaration)
}

/// Closed set of primitive semantic declarations.
///
/// Each case represents a meaningful architectural unit at the semantic level.
/// The vocabulary matches `OperationKind`'s primitive cases exactly.
public enum PrimitiveDeclaration: Sendable, Equatable {

    /// Token embedding: maps token IDs to dense vectors.
    case tokenEmbedding(TokenEmbeddingAttributes)

    /// Positional embedding: adds position information to token vectors.
    case positionalEmbedding(PositionalEmbeddingAttributes)

    /// Rotary position embedding: applies rotation-based position encoding.
    case rope(RoPEAttributes)

    /// Multi-head attention: Q/K/V projections, scaled dot-product, output projection.
    case attention(AttentionAttributes)

    /// Feed-forward network: gate/up/down projections with activation.
    case mlp(MLPAttributes)

    /// Mixture-of-Experts: routes tokens to expert MLPs via gating.
    case moe(MoEAttributes)

    /// RMS normalization.
    case rmsNorm(RMSNormAttributes)

    /// Layer normalization.
    case layerNorm(LayerNormAttributes)

    /// Linear projection.
    case linear(LinearAttributes)

    /// Output head: projects hidden states to vocabulary logits.
    case outputHead(OutputHeadAttributes)

    /// State-space model block (Mamba, DeltaNet, etc.).
    case stateSpace(StateSpaceAttributes)

    /// Custom operation (escape hatch).
    case custom(CustomNodeAttributes)
}

// MARK: - ModelComponent Conformance

extension ModelDeclaration: ModelComponent {

    public func makeDeclaration() -> ModelDeclaration { self }
}
