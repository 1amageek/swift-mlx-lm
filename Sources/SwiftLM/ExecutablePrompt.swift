/// Runtime-executable prompt shape.
///
/// `ExecutablePrompt` is the validated execution contract consumed by
/// `LanguageModelContext` and the Metal runtime. Text tokens remain the semantic
/// source of truth, while multimodal execution details live in `VisualContext`.
public struct ExecutablePrompt: Sendable {
    public var tokenIDs: [Int]
    public var attentionMask: [Int]?
    public var visualContext: VisualContext?
    var gemma4PromptContext: Gemma4PromptContext?

    public init(
        tokenIDs: [Int],
        attentionMask: [Int]? = nil,
        visualContext: VisualContext? = nil
    ) {
        self.tokenIDs = tokenIDs
        self.attentionMask = attentionMask
        self.visualContext = visualContext
        self.gemma4PromptContext = nil
    }

    init(
        tokenIDs: [Int],
        attentionMask: [Int]? = nil,
        visualContext: VisualContext? = nil,
        gemma4PromptContext: Gemma4PromptContext?
    ) {
        self.tokenIDs = tokenIDs
        self.attentionMask = attentionMask
        self.visualContext = visualContext
        self.gemma4PromptContext = gemma4PromptContext
    }
}
