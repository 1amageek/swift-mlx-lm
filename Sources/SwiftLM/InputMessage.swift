/// A single message in a chat-style prompt.
public struct InputMessage: Sendable {
    public var role: Role
    public var content: [Content]

    public init(role: Role, content: String) {
        self.role = role
        self.content = [.text(content)]
    }

    public init(role: Role, content: [Content]) {
        self.role = role
        self.content = content
    }

    public static func system(_ content: String) -> InputMessage {
        InputMessage(role: .system, content: content)
    }

    public static func system(_ content: [Content]) -> InputMessage {
        InputMessage(role: .system, content: content)
    }

    public static func user(_ content: String) -> InputMessage {
        InputMessage(role: .user, content: content)
    }

    public static func user(_ content: [Content]) -> InputMessage {
        InputMessage(role: .user, content: content)
    }

    public static func assistant(_ content: String) -> InputMessage {
        InputMessage(role: .assistant, content: content)
    }

    public static func assistant(_ content: [Content]) -> InputMessage {
        InputMessage(role: .assistant, content: content)
    }

    public static func tool(_ content: String) -> InputMessage {
        InputMessage(role: .tool, content: content)
    }

    public static func tool(_ content: [Content]) -> InputMessage {
        InputMessage(role: .tool, content: content)
    }

    var containsImageContent: Bool {
        content.contains { item in
            if case .image = item {
                return true
            }
            return false
        }
    }

    var containsVideoContent: Bool {
        content.contains { item in
            if case .video = item {
                return true
            }
            return false
        }
    }

    var containsVisualContent: Bool {
        containsImageContent || containsVideoContent
    }

    var textContent: String {
        content.compactMap { item in
            if case .text(let text) = item {
                return text
            }
            return nil
        }
        .joined()
    }

    public enum Content: Sendable {
        case text(String)
        case image(InputImage)
        case video(InputVideo)
    }

    public enum Role: String, Sendable {
        case user
        case assistant
        case system
        case tool
    }
}
