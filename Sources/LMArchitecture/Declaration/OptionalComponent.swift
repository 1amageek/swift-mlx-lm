/// Wraps an optional model component for conditional inclusion.
///
/// Produced by `ModelComponentBuilder.buildOptional` when using `if` without
/// `else` in a builder block. When `content` is `nil`, the component acts
/// as a pass-through (identity) during normalization.
public struct OptionalComponent<Content: ModelComponent>: ModelComponent {

    public typealias Body = Never

    /// The wrapped component, or `nil` if the condition was false.
    package let content: Content?

    package init(content: Content?) {
        self.content = content
    }
}
