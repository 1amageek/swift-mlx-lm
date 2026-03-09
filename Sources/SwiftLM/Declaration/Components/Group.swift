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
public struct Group: ModelComponent {

    public let label: String?
    public let body: ModelDeclaration

    public init(
        label: String? = nil,
        @ModelComponentBuilder content: () -> ModelDeclaration
    ) {
        self.label = label
        self.body = content()
    }

    public func makeDeclaration() -> ModelDeclaration {
        if let label {
            return .labeled(label, body)
        }
        return body
    }
}
