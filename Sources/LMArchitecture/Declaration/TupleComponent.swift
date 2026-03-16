/// Sequential composition of model components using type-safe parameter packs.
///
/// `TupleComponent` is the multi-child return type of `ModelComponentBuilder`,
/// analogous to SwiftUI's `TupleView`. It holds a typed tuple of child
/// components that are evaluated sequentially during normalization.
///
/// Users never construct `TupleComponent` directly — it is produced by
/// the result builder and hidden behind `some ModelComponent` opaque types.
public struct TupleComponent<each C: ModelComponent>: ModelComponent {

    public typealias Body = Never

    /// Child components in sequential order, stored as a typed tuple.
    package let value: (repeat each C)

    /// Create a tuple component from a parameter pack of components.
    package init(_ value: repeat each C) {
        self.value = (repeat each value)
    }
}
