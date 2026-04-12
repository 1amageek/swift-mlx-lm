/// Prompt/render-time configuration applied before token generation begins.
///
/// `PromptPreparationOptions` only affects prompt construction. Use
/// ``GenerationParameters`` and ``ReasoningOptions`` for output-time behavior.
public struct PromptPreparationOptions: Sendable, Equatable {
    /// Enables model thinking for chat templates that expose an `enable_thinking` variable.
    ///
    /// This is prompt-time configuration, not output visibility policy.
    public var isThinkingEnabled: Bool
    /// Additional variables forwarded only to prompt-template rendering.
    ///
    /// Keep template variables here rather than in generation parameters so
    /// prompt rendering and output behavior stay separated.
    public var templateVariables: [String: PromptTemplateValue]

    public init(
        isThinkingEnabled: Bool = false,
        templateVariables: [String: PromptTemplateValue] = [:]
    ) {
        self.isThinkingEnabled = isThinkingEnabled
        self.templateVariables = templateVariables
    }
}

/// Value type used for prompt-template variables.
public enum PromptTemplateValue: Sendable, Equatable {
    case boolean(Bool)
    case int(Int)
    case double(Double)
    case string(String)
}
