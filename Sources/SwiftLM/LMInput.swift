/// Tokenized input for inference.
///
/// MLX-free: tokens are plain `[Int]` arrays, not MLXArray.
/// The Metal backend converts to MTLBuffer internally.
public struct LMInput: Sendable {
    /// Tokenized text input.
    public var text: Text

    public init(tokens: [Int]) {
        self.text = Text(tokens: tokens)
    }

    public init(text: Text) {
        self.text = text
    }

    /// Tokenized text representation.
    public struct Text: Sendable {
        /// Token IDs.
        public var tokens: [Int]
        /// Optional attention mask.
        public var mask: [Int]?

        public init(tokens: [Int], mask: [Int]? = nil) {
            self.tokens = tokens
            self.mask = mask
        }
    }
}
