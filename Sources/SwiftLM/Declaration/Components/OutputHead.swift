/// Output head (language model head) component.
///
/// Projects hidden states to vocabulary logits.
///
/// ```swift
/// OutputHead(inputSize: 4096, vocabSize: 32000, tiedToEmbedding: true)
/// ```
public struct OutputHead: PrimitiveModelComponent {

    public let inputSize: Int
    public let vocabSize: Int
    public let tiedToEmbedding: Bool
    public let bias: Bool

    public init(
        inputSize: Int,
        vocabSize: Int,
        tiedToEmbedding: Bool = true,
        bias: Bool = false
    ) {
        precondition(inputSize > 0, "inputSize must be positive")
        precondition(vocabSize > 0, "vocabSize must be positive")
        self.inputSize = inputSize
        self.vocabSize = vocabSize
        self.tiedToEmbedding = tiedToEmbedding
        self.bias = bias
    }

    public func makeDeclaration() -> ModelDeclaration {
        .primitive(.outputHead(OutputHeadAttributes(
            inputSize: inputSize,
            vocabSize: vocabSize,
            tiedToEmbedding: tiedToEmbedding,
            bias: bias
        )))
    }
}
