/// Linear projection component.
///
/// Available for cases where higher-level components (Attention, MLP)
/// are not applicable.
///
/// ```swift
/// Linear(inputSize: 4096, outputSize: 32000)
/// ```
public struct Linear: ModelComponent {

    public typealias Body = Never

    public let inputSize: Int
    public let outputSize: Int
    public let bias: Bool

    public init(inputSize: Int, outputSize: Int, bias: Bool = false) {
        precondition(inputSize > 0, "inputSize must be positive")
        precondition(outputSize > 0, "outputSize must be positive")
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.bias = bias
    }
}

extension Linear: PrimitiveComponent {

    package var operationKind: OperationKind {
        .primitive(LinearAttributes(inputSize: inputSize, outputSize: outputSize, bias: bias))
    }

    package var operationSignature: OperationSignature {
        OperationSignature(operandArity: .exact(1), resultArity: .exact(1))
    }
}
