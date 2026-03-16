/// A declarative structural building block for model topology.
///
/// `ModelComponent` is the user-facing abstraction for defining model structure,
/// analogous to SwiftUI's `View` protocol. It is open for extension — users can
/// define custom composite components that compose existing ones.
///
/// **Primitive components** (MLP, Attention, etc.) set `Body = Never`.
/// **Composite components** define `body` to compose other components —
/// normalization recurses into `body` automatically.
///
/// ```swift
/// struct TransformerBlock: ModelComponent {
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

    /// The type of the composed body. Primitive components use `Never`.
    associatedtype Body: ModelComponent

    /// Declarative structural body composed from other components.
    ///
    /// Primitive components (where `Body == Never`) get a default
    /// implementation that traps — they are handled directly by
    /// the normalizer via `PrimitiveComponent`.
    @ModelComponentBuilder var body: Body { get }
}

// MARK: - Primitive Default

extension ModelComponent where Body == Never {

    /// Primitive components have no body — accessing it is a programming error.
    public var body: Never {
        fatalError("\(type(of: self)) is a primitive ModelComponent and has no body")
    }
}

// MARK: - Never Conformance

extension Never: ModelComponent {
    public typealias Body = Never
}
