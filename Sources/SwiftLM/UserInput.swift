/// User-facing input for text generation.
public struct UserInput: Sendable {
    /// The text prompt or chat messages.
    public var prompt: Prompt

    public init(prompt: String) {
        self.prompt = .text(prompt)
    }

    public init(chat: [ChatMessage]) {
        self.prompt = .chat(chat)
    }

    /// Prompt representation.
    public enum Prompt: Sendable {
        case text(String)
        case chat([ChatMessage])
    }
}

/// A single message in a chat conversation.
public struct ChatMessage: Sendable {
    public var role: Role
    public var content: String

    public init(role: Role, content: String) {
        self.role = role
        self.content = content
    }

    public static func system(_ content: String) -> ChatMessage {
        ChatMessage(role: .system, content: content)
    }

    public static func user(_ content: String) -> ChatMessage {
        ChatMessage(role: .user, content: content)
    }

    public static func assistant(_ content: String) -> ChatMessage {
        ChatMessage(role: .assistant, content: content)
    }

    public enum Role: String, Sendable {
        case user
        case assistant
        case system
        case tool
    }
}
