/// User-facing input for text embeddings.
///
/// `TextEmbeddingInput` is the primary public input type for text embedding
/// APIs. It keeps the input text and bundle-defined prompt selection together
/// so higher-level code can pass embedding requests as values instead of
/// parallel arguments.
public struct TextEmbeddingInput: Sendable, Equatable {
    public var text: String
    public var promptName: String?

    public init(_ text: String, promptName: String? = nil) {
        self.text = text
        self.promptName = promptName
    }

    public init(text: String, promptName: String? = nil) {
        self.text = text
        self.promptName = promptName
    }
}
