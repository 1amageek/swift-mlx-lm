/// Attributes for a state-space model node.
///
/// Represents SSM variants such as Mamba, DeltaNet, or similar
/// recurrent/selective-state-space architectures.
public struct StateSpaceAttributes: Codable, Equatable, Sendable {

    /// Hidden dimension of the state-space block.
    public let hiddenSize: Int

    /// State dimension (latent state size).
    public let stateSize: Int

    /// SSM variant identifier (e.g., "mamba", "deltanet").
    public let variant: String

    public init(hiddenSize: Int, stateSize: Int, variant: String) {
        self.hiddenSize = hiddenSize
        self.stateSize = stateSize
        self.variant = variant
    }
}
