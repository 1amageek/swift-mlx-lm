/// Chat message types for constructing prompts.
public enum Chat {

    /// A single message in a conversation.
    public struct Message: Sendable {
        public var role: Role
        public var content: String
        public var images: [InputImage]
        public var videos: [InputVideo]

        public init(
            role: Role,
            content: String,
            images: [InputImage] = [],
            videos: [InputVideo] = []
        ) {
            self.role = role
            self.content = content
            self.images = images
            self.videos = videos
        }

        public static func system(
            _ content: String,
            images: [InputImage] = [],
            videos: [InputVideo] = []
        ) -> Self {
            Message(role: .system, content: content, images: images, videos: videos)
        }

        public static func user(
            _ content: String,
            images: [InputImage] = [],
            videos: [InputVideo] = []
        ) -> Self {
            Message(role: .user, content: content, images: images, videos: videos)
        }

        public static func assistant(
            _ content: String,
            images: [InputImage] = [],
            videos: [InputVideo] = []
        ) -> Self {
            Message(role: .assistant, content: content, images: images, videos: videos)
        }

        public static func tool(_ content: String) -> Self {
            Message(role: .tool, content: content)
        }

        public enum Role: String, Sendable, Codable {
            case user
            case assistant
            case system
            case tool
        }
    }
}
