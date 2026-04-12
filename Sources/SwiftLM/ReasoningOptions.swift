/// Per-request controls for whether template-defined reasoning content is
/// included in user-visible output.
///
/// `ReasoningOptions` affects streamed output visibility only. Use
/// ``PromptPreparationOptions`` for prompt-time thinking control.
public struct ReasoningOptions: Sendable, Equatable {
    /// Controls how template-defined reasoning content is surfaced to callers.
    public var visibility: Visibility

    public init(visibility: Visibility = .hidden) {
        self.visibility = visibility
    }

    /// Controls whether reasoning content is hidden, left inline, or emitted
    /// as separate stream events.
    public enum Visibility: Sendable, Equatable {
        case hidden
        case inline
        case separate
    }

    /// Hide reasoning content from user-visible output.
    public static let hidden = ReasoningOptions(visibility: .hidden)
    /// Leave reasoning content inline with visible text.
    public static let inline = ReasoningOptions(visibility: .inline)
    /// Emit reasoning content as separate ``GenerationEvent/reasoning(_:)`` events.
    public static let separate = ReasoningOptions(visibility: .separate)
}
