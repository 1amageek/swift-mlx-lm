/// Dynamic sequence of model components for runtime-determined compositions.
///
/// `ArrayComponent` is the return type of `ModelComponentBuilder.buildArray`,
/// enabling `for...in` loops in component builders. Each child is evaluated
/// sequentially during normalization.
///
/// Users never construct `ArrayComponent` directly — it is produced by
/// the result builder and hidden behind `some ModelComponent` opaque types.
public struct ArrayComponent: ModelComponent {

    public typealias Body = Never

    /// Child components in sequential order.
    package let children: [any ModelComponent]

    package init(children: [any ModelComponent]) {
        self.children = children
    }
}
