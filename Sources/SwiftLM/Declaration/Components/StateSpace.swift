/// State-space model component.
///
/// Represents SSM variants such as Mamba, DeltaNet, or similar
/// recurrent/selective-state-space architectures.
///
/// ```swift
/// StateSpace(hiddenSize: 4096, stateSize: 16, variant: "mamba")
/// ```
public struct StateSpace: PrimitiveModelComponent {

    public let hiddenSize: Int
    public let stateSize: Int
    public let variant: String

    public init(hiddenSize: Int, stateSize: Int, variant: String) {
        precondition(hiddenSize > 0, "hiddenSize must be positive")
        precondition(stateSize > 0, "stateSize must be positive")
        self.hiddenSize = hiddenSize
        self.stateSize = stateSize
        self.variant = variant
    }

    public func makeDeclaration() -> ModelDeclaration {
        .primitive(.stateSpace(StateSpaceAttributes(
            hiddenSize: hiddenSize,
            stateSize: stateSize,
            variant: variant
        )))
    }
}
