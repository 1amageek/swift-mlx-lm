/// A declarative structural building block for model topology.
///
/// `ModelComponent` is the user-facing abstraction for defining model structure,
/// analogous to SwiftUI's `View` protocol. It is open for extension — users can
/// define custom composite components that compose existing ones.
///
/// `ModelComponent` produces a `ModelDeclaration` value — a pure, context-free
/// description of the model's structure. No graph is built during declaration;
/// the declaration tree is converted to a canonical `ModelGraph` by the
/// `SemanticNormalizer` in a separate phase.
///
/// ```swift
/// struct TransformerBlock: CompositeModelComponent {
///     let hiddenSize: Int
///     let headCount: Int
///     let kvHeadCount: Int
///     let intermediateSize: Int
///
///     var body: some ModelComponent {
///         Residual {
///             RMSNorm(dimension: hiddenSize)
///             Attention(hiddenSize: hiddenSize, headCount: headCount, kvHeadCount: kvHeadCount)
///         }
///         Residual {
///             RMSNorm(dimension: hiddenSize)
///             MLP(inputSize: hiddenSize, intermediateSize: intermediateSize)
///         }
///     }
/// }
/// ```
public protocol ModelComponent: Sendable {

    /// Produce a pure declaration value describing this component's structure.
    ///
    /// This method is context-free: it does not mutate any builder state.
    /// The returned `ModelDeclaration` is an open tree that is later
    /// normalized into a closed, canonical `ModelGraph`.
    func makeDeclaration() -> ModelDeclaration
}

/// A primitive model component that directly produces a semantic declaration.
///
/// Primitive components are leaf nodes in the declaration tree. They correspond
/// to `PrimitiveDeclaration` cases (attention, MLP, normalization, etc.).
public protocol PrimitiveModelComponent: ModelComponent {}

/// A composite model component that defines its structure via a `body`.
///
/// Composite components are user-defined compositions of other components.
/// Their `makeDeclaration()` is automatically derived from `body`.
public protocol CompositeModelComponent: ModelComponent {

    /// The type of the composed body.
    associatedtype Body: ModelComponent

    /// Declarative structural body composed from other components.
    @ModelComponentBuilder var body: Body { get }
}

extension CompositeModelComponent {

    public func makeDeclaration() -> ModelDeclaration {
        body.makeDeclaration()
    }
}
