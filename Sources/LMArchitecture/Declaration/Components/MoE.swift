/// Mixture-of-Experts component.
///
/// Routes tokens to a subset of expert MLPs via a gating mechanism.
///
/// ```swift
/// MoE(
///     expertCount: 8,
///     expertsPerToken: 2,
///     expertInputSize: 4096,
///     expertIntermediateSize: 14336
/// )
/// ```
public struct MoE: ModelComponent {

    public typealias Body = Never

    public let expertCount: Int
    public let expertsPerToken: Int
    public let gateKind: MoEGateKind
    public let expertInputSize: Int
    public let expertOutputSize: Int
    public let expertIntermediateSize: Int
    public let expertActivation: ActivationKind
    public let expertGating: GatingKind
    public let expertBias: Bool

    public init(
        expertCount: Int,
        expertsPerToken: Int,
        gateKind: MoEGateKind = .topK,
        expertInputSize: Int,
        expertOutputSize: Int? = nil,
        expertIntermediateSize: Int,
        expertActivation: ActivationKind = .silu,
        expertGating: GatingKind = .swiglu,
        expertBias: Bool = false
    ) {
        precondition(expertCount > 0, "expertCount must be positive")
        precondition(expertsPerToken > 0, "expertsPerToken must be positive")
        precondition(expertsPerToken <= expertCount, "expertsPerToken must not exceed expertCount")
        let resolvedExpertOutputSize = expertOutputSize ?? expertInputSize
        precondition(expertInputSize > 0, "expertInputSize must be positive")
        precondition(resolvedExpertOutputSize > 0, "expertOutputSize must be positive")
        precondition(expertIntermediateSize > 0, "expertIntermediateSize must be positive")
        self.expertCount = expertCount
        self.expertsPerToken = expertsPerToken
        self.gateKind = gateKind
        self.expertInputSize = expertInputSize
        self.expertOutputSize = resolvedExpertOutputSize
        self.expertIntermediateSize = expertIntermediateSize
        self.expertActivation = expertActivation
        self.expertGating = expertGating
        self.expertBias = expertBias
    }
}

extension MoE: PrimitiveComponent {

    package var operationKind: OperationKind {
        .primitive(MoEAttributes(
            expertCount: expertCount,
            expertsPerToken: expertsPerToken,
            gateKind: gateKind,
            expertMLP: MLPAttributes(
                inputSize: expertInputSize,
                outputSize: expertOutputSize,
                intermediateSize: expertIntermediateSize,
                activation: expertActivation,
                gating: expertGating,
                bias: expertBias
            )
        ))
    }

    package var operationSignature: OperationSignature {
        OperationSignature(operandArity: .exact(1), resultArity: .exact(1))
    }
}
