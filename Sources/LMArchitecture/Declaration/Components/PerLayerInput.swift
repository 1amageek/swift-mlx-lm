/// Gemma 4 style per-layer input residual component.
///
/// This is a semantic primitive for the token-conditioned residual used
/// after the feed-forward block in Gemma 4 text layers.
public struct PerLayerInput: ModelComponent {

    public typealias Body = Never

    public let hiddenSize: Int
    public let perLayerInputSize: Int
    public let vocabSize: Int
    public let activation: ActivationKind

    public init(
        hiddenSize: Int,
        perLayerInputSize: Int,
        vocabSize: Int,
        activation: ActivationKind = .custom("gelu_pytorch_tanh")
    ) {
        precondition(hiddenSize > 0, "hiddenSize must be positive")
        precondition(perLayerInputSize > 0, "perLayerInputSize must be positive")
        precondition(vocabSize > 0, "vocabSize must be positive")
        self.hiddenSize = hiddenSize
        self.perLayerInputSize = perLayerInputSize
        self.vocabSize = vocabSize
        self.activation = activation
    }
}

extension PerLayerInput: PrimitiveComponent {

    package var operationKind: OperationKind {
        .primitive(PerLayerInputAttributes(
            hiddenSize: hiddenSize,
            perLayerInputSize: perLayerInputSize,
            vocabSize: vocabSize,
            activation: activation
        ))
    }

    package var operationSignature: OperationSignature {
        OperationSignature(operandArity: .exact(1), resultArity: .exact(1))
    }
}
