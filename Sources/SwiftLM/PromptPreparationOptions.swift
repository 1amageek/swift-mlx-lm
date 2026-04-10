/// Prompt/render-time configuration applied before token generation begins.
public struct PromptPreparationOptions: Sendable, Equatable {
    /// Enables model thinking for chat templates that expose an `enable_thinking` variable.
    public var thinkingEnabled: Bool
    /// Additional variables forwarded only to prompt-template rendering.
    public var templateVariables: [String: PromptTemplateValue]

    public init(
        thinkingEnabled: Bool = false,
        templateVariables: [String: PromptTemplateValue] = [:]
    ) {
        self.thinkingEnabled = thinkingEnabled
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
