/// User-facing input for text generation.
public struct UserInput: Sendable {

    /// Chat messages forming the conversation.
    public var chat: [Chat.Message]

    /// Tool specifications (JSON schema dictionaries).
    public var tools: [ToolSpec]?

    /// Additional values for the chat template rendering context (e.g. enable_thinking).
    public var additionalContext: [String: any Sendable]?

    /// Processing options for media.
    public var processing: MediaProcessing

    /// Images collected from chat messages.
    public var images: [InputImage] {
        chat.flatMap(\.images)
    }

    /// Videos collected from chat messages.
    public var videos: [InputVideo] {
        chat.flatMap(\.videos)
    }

    public init(
        chat: [Chat.Message],
        tools: [ToolSpec]? = nil,
        additionalContext: [String: any Sendable]? = nil,
        processing: MediaProcessing = MediaProcessing()
    ) {
        self.chat = chat
        self.tools = tools
        self.additionalContext = additionalContext
        self.processing = processing
    }

    /// Convenience initializer for a single user message.
    public init(prompt: String) {
        self.chat = [.user(prompt)]
        self.tools = nil
        self.additionalContext = nil
        self.processing = MediaProcessing()
    }
}
