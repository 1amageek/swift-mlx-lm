/// Prompt/render-time configuration applied before token generation begins.
public struct PromptPreparationOptions: Sendable, Equatable {
    /// Enables model thinking for chat templates that expose an `enable_thinking` variable.
    public var isThinkingEnabled: Bool
    /// Additional variables forwarded only to prompt-template rendering.
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
