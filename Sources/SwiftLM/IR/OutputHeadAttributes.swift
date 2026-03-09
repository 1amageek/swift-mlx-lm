/// Attributes for an output head (language model head) node.
///
/// Projects hidden states back to vocabulary logits.
public struct OutputHeadAttributes: Codable, Equatable, Sendable {

    /// Input dimension (hidden size).
    public let inputSize: Int

    /// Vocabulary size (output dimension).
    public let vocabSize: Int

    /// Whether the output projection shares weights with the token embedding.
    public let tiedToEmbedding: Bool

    /// Whether a bias term is included.
    public let bias: Bool

    public init(
        inputSize: Int,
        vocabSize: Int,
        tiedToEmbedding: Bool = true,
        bias: Bool = false
    ) {
        self.inputSize = inputSize
        self.vocabSize = vocabSize
        self.tiedToEmbedding = tiedToEmbedding
        self.bias = bias
    }
}
