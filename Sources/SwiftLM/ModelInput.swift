/// User-facing input for generation.
///
/// `ModelInput` is the primary public input type for `SwiftLM`.
/// It can represent plain text prompts and chat-style prompts, and it is
/// intentionally shaped so multimodal content can be introduced without
/// replacing the public API again.
public struct ModelInput: Sendable {
    /// The prompt payload.
    public var prompt: Prompt

    public init(_ prompt: String) {
        self.prompt = .text(prompt)
    }

    public init(prompt: String) {
        self.prompt = .text(prompt)
    }

    public init(chat: [InputMessage]) {
        self.prompt = .chat(chat)
    }

    /// Prompt representation.
    public enum Prompt: Sendable {
        case text(String)
        case chat([InputMessage])
    }
}
