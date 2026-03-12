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

    /// Required computation precision for the recurrence loop.
    ///
    /// State-space models accumulate state over thousands of timesteps.
    /// Operations like `exp()`, `softplus()`, and outer-product accumulation
    /// require higher precision than the default Float16 to avoid overflow
    /// (Float16 max ≈ 65504, `exp(x)` overflows for x > 11) and accumulated
    /// rounding error.
    ///
    /// This is a **structural declaration** — the precision requirement is
    /// intrinsic to the recurrence math, not an implementation detail.
    /// The compiler reads this at compile time and resolves dtype transitions
    /// at operation boundaries (analogous to kernel selection in `LoweredProjection`).
    ///
    /// Projections (matmul/quantizedMM) produce activations in their native
    /// output dtype. The compiler inserts dtype casts at the boundary between
    /// projections and the recurrence, and casts back before the output projection.
    public let computeDType: ComputeDType

    /// Computation precision for state-space recurrence.
    public enum ComputeDType: String, Codable, Equatable, Sendable {
        /// 32-bit floating point. Required for DeltaNet and similar SSMs
        /// where sequential state accumulation overflows Float16.
        case float32
        /// 16-bit floating point. Sufficient for SSMs without long-range
        /// state accumulation (reserved for future variants).
        case float16
    }

    public init(
        hiddenSize: Int, stateSize: Int, variant: String,
        computeDType: ComputeDType = .float32
    ) {
        self.hiddenSize = hiddenSize
        self.stateSize = stateSize
        self.variant = variant
        self.computeDType = computeDType
    }
}
