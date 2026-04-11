/// Per-request controls for whether template-defined reasoning content is
/// included in user-visible output.
public struct ReasoningOptions: Sendable, Equatable {
    /// Controls how template-defined reasoning content is surfaced to callers.
    public var visibility: Visibility

    public init(visibility: Visibility = .hidden) {
        self.visibility = visibility
    }

    public enum Visibility: Sendable, Equatable {
        case hidden
        case inline
        case separate
    }

    public static let hidden = ReasoningOptions(visibility: .hidden)
    public static let inline = ReasoningOptions(visibility: .inline)
    public static let separate = ReasoningOptions(visibility: .separate)
}
