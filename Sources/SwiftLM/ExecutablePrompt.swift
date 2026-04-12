/// Runtime-executable prompt shape.
///
/// `ExecutablePrompt` is the validated execution contract consumed by
/// `LanguageModelContext` and the Metal runtime. Text tokens remain the semantic
/// source of truth, while multimodal execution details live in `VisualContext`.
///
/// Most applications should not construct this type directly. Prefer
/// ``LanguageModelContainer/generate(_:parameters:)`` for one-shot generation,
/// or build an `ExecutablePrompt` only when you need explicit prompt staging.
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

    public init(
        preparedPrompt: PreparedPrompt,
        using context: LanguageModelContext
    ) throws {
        self = try context.executablePrompt(for: preparedPrompt)
    }

    public init(
        preparedPrompt: PreparedPrompt,
        using container: LanguageModelContainer
    ) throws {
        self = try container.prototypeContext.executablePrompt(for: preparedPrompt)
    }
}
