/// Protocol for text tokenization.
///
/// Encodes text to token IDs and decodes token IDs back to text.
public protocol Tokenizer: Sendable {
    /// Encode text into token IDs.
    func encode(text: String) -> [Int]

    /// Decode token IDs back into text.
    func decode(tokens: [Int]) -> String

    /// Beginning-of-sequence token ID.
    var bosTokenID: Int? { get }

    /// End-of-sequence token ID.
    var eosTokenID: Int? { get }

    /// Total vocabulary size.
    var vocabularySize: Int { get }

    /// Convert a single token ID to its string representation.
    func tokenToString(_ id: Int) -> String?
}
