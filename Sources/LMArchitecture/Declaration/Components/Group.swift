/// Labeled grouping component.
///
/// Groups child components and optionally attaches a label for debugging
/// and diagnostics. Does NOT introduce a distinct semantic operation in the
/// `ModelGraph` — labels are stripped during normalization and the children
/// are flattened into the enclosing region.
///
/// The `@ModelComponentBuilder` already provides implicit sequential
/// composition. Use `Group` only when you need a labeled group.
///
/// ```swift
/// Group(label: "transformer_block") {
///     RMSNorm(dimension: 4096)
///     Attention(hiddenSize: 4096, headCount: 32, kvHeadCount: 8)
/// }
/// ```
public struct Group<Content: ModelComponent>: ModelComponent {

    public typealias Body = Never

    public let label: String?
    public let content: Content

    public init(
        label: String? = nil,
        @ModelComponentBuilder content: () -> Content
    ) {
        self.label = label
        self.content = content()
    }
}
