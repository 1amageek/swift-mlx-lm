/// Feed-forward network component.
///
/// ```swift
/// MLP(inputSize: 4096, intermediateSize: 11008)
/// ```
public struct MLP: ModelComponent {

    public typealias Body = Never

    public let inputSize: Int
    public let outputSize: Int
    public let intermediateSize: Int
    public let activation: ActivationKind
    public let gating: GatingKind
    public let bias: Bool

    public init(
        inputSize: Int,
        outputSize: Int? = nil,
        intermediateSize: Int,
        activation: ActivationKind = .silu,
        gating: GatingKind = .swiglu,
        bias: Bool = false
    ) {
        let resolvedOutputSize = outputSize ?? inputSize
        precondition(inputSize > 0, "inputSize must be positive")
        precondition(resolvedOutputSize > 0, "outputSize must be positive")
        precondition(intermediateSize > 0, "intermediateSize must be positive")
        self.inputSize = inputSize
        self.outputSize = resolvedOutputSize
        self.intermediateSize = intermediateSize
        self.activation = activation
        self.gating = gating
        self.bias = bias
    }
}

extension MLP: PrimitiveComponent {

    package var operationKind: OperationKind {
        .primitive(MLPAttributes(
            inputSize: inputSize,
            outputSize: outputSize,
            intermediateSize: intermediateSize,
            activation: activation,
            gating: gating,
            bias: bias
        ))
    }

    package var operationSignature: OperationSignature {
        OperationSignature(operandArity: .exact(1), resultArity: .exact(1))
    }
}
