/// Attributes for a Gemma 4 style per-layer input residual.
///
/// This semantic node represents the token-conditioned residual applied
/// after the feed-forward block in Gemma 4 text layers:
/// current hidden -> gate -> token-derived per-layer embedding modulation
/// -> projection -> norm -> residual add.
public struct PerLayerInputAttributes: OperationAttributes, Codable, Equatable {

    /// Model hidden size.
    public let hiddenSize: Int

    /// Width of the per-layer input embedding space.
    public let perLayerInputSize: Int

    /// Vocabulary size of the per-layer input embedding table.
    public let vocabSize: Int

    /// Activation used by the input gate.
    public let activation: ActivationKind

    public init(
        hiddenSize: Int,
        perLayerInputSize: Int,
        vocabSize: Int,
        activation: ActivationKind = .custom("gelu_pytorch_tanh")
    ) {
        self.hiddenSize = hiddenSize
        self.perLayerInputSize = perLayerInputSize
        self.vocabSize = vocabSize
        self.activation = activation
    }
}
