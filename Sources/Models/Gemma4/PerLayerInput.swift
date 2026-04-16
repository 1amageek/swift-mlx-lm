import LMArchitecture

/// Gemma 4 style per-layer input residual component.
///
/// This is a semantic primitive for the token-conditioned residual used
/// after the feed-forward block in Gemma 4 text layers.
public struct PerLayerInput: ModelComponent {

    public typealias Attributes = PerLayerInputAttributes

    public let hiddenSize: Int
    public let perLayerInputSize: Int
    public let vocabSize: Int
    public let activation: ActivationKind
    public let normWeightBias: Float

    public init(
        hiddenSize: Int,
        perLayerInputSize: Int,
        vocabSize: Int,
        activation: ActivationKind = .custom("gelu_pytorch_tanh"),
        normWeightBias: Float = 0
    ) {
        precondition(hiddenSize > 0, "hiddenSize must be positive")
        precondition(perLayerInputSize > 0, "perLayerInputSize must be positive")
        precondition(vocabSize > 0, "vocabSize must be positive")
        self.hiddenSize = hiddenSize
        self.perLayerInputSize = perLayerInputSize
        self.vocabSize = vocabSize
        self.activation = activation
        self.normWeightBias = normWeightBias
    }
}

extension PerLayerInput {

    public var attributes: PerLayerInputAttributes {
        PerLayerInputAttributes(
            hiddenSize: hiddenSize,
            perLayerInputSize: perLayerInputSize,
            vocabSize: vocabSize,
            activation: activation,
            normWeightBias: normWeightBias
        )
    }
}
