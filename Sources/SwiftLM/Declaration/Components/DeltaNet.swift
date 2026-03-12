/// DeltaNet state-space family component.
///
/// This is a semantic wrapper over the generic `stateSpace` IR node so model
/// declarations can name the paper-level family explicitly while still lowering
/// to the same optimized kernel path.
public struct DeltaNet: ModelComponent {

    public typealias Body = Never

    public enum Variant: String, Sendable, Equatable {
        case standard = "deltanet"
        case gated = "gated_deltanet"
    }

    public let hiddenSize: Int
    public let stateSize: Int
    public let variant: Variant

    public init(
        hiddenSize: Int,
        stateSize: Int,
        variant: Variant = .standard
    ) {
        precondition(hiddenSize > 0, "hiddenSize must be positive")
        precondition(stateSize > 0, "stateSize must be positive")
        self.hiddenSize = hiddenSize
        self.stateSize = stateSize
        self.variant = variant
    }
}

extension DeltaNet: PrimitiveComponent {

    package var operationKind: OperationKind {
        .stateSpace(StateSpaceAttributes(
            hiddenSize: hiddenSize,
            stateSize: stateSize,
            variant: variant.rawValue
        ))
    }

    package var operationSignature: OperationSignature {
        OperationSignature(operandArity: .exact(1), resultArity: .exact(1))
    }
}
