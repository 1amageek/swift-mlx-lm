/// Wraps a conditional choice between two model components.
///
/// Produced by `ModelComponentBuilder.buildEither` when using `if/else`
/// in a builder block. Exactly one branch is active at a time.
public struct ConditionalComponent<First: ModelComponent, Second: ModelComponent>: ModelComponent {

    public typealias Body = Never

    /// Which branch is active.
    package enum Storage: Sendable {
        case first(First)
        case second(Second)
    }

    package let storage: Storage

    package init(storage: Storage) {
        self.storage = storage
    }
}
