/// Attributes for a feed-forward MLP node.
///
/// Represents the full MLP block as a semantic unit, including
/// gate/up/down projections and activation function.
public struct MLPAttributes: OperationAttributes, Codable, Equatable {

    /// Input dimension (hidden size of the model).
    public let inputSize: Int

    /// Output dimension (typically same as input for residual blocks).
    public let outputSize: Int

    /// Intermediate dimension (expansion factor applied).
    public let intermediateSize: Int

    /// Activation function applied in the MLP.
    public let activation: ActivationKind

    /// Gating strategy for the MLP.
    public let gating: GatingKind

    /// Whether projections include bias terms.
    public let bias: Bool

    public init(
        inputSize: Int,
        outputSize: Int,
        intermediateSize: Int,
        activation: ActivationKind = .silu,
        gating: GatingKind = .swiglu,
        bias: Bool = false
    ) {
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.intermediateSize = intermediateSize
        self.activation = activation
        self.gating = gating
        self.bias = bias
    }
}

/// Activation function kind.
public enum ActivationKind: Codable, Equatable, Sendable {
    case gelu
    case silu
    case swish
    case relu
    case custom(String)
}

/// Gating strategy for MLP blocks.
public enum GatingKind: Codable, Equatable, Sendable {
    case none
    case glu
    case geglu
    case swiglu
    case custom(String)
}
