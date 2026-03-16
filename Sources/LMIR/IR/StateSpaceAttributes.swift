/// Attributes for a state-space model node.
///
/// Represents SSM variants such as Mamba, DeltaNet, or similar
/// recurrent/selective-state-space architectures.
///
/// ## DeltaNet dimensions
///
/// The state matrix S is described by two structural axes:
///
/// - `numHeads`: number of value heads
/// - `groupCount`: number of recurrent key/query heads
///
/// Symmetric models use `groupCount == numHeads`, yielding a state layout of
/// `[B, numHeads, keyHeadDim, valueHeadDim]`.
///
/// Asymmetric DeltaNet variants may use fewer key heads than value heads.
/// In that case:
///
/// - `query` / `key` are projected to `groupCount` heads
/// - `value`, `beta`, and `decay` use `numHeads`
/// - `query` / `key` are expanded to value-head count before the recurrence
/// - the recurrent state still has layout `[B, numHeads, keyHeadDim, valueHeadDim]`
///
/// Key and value dimensions can differ (asymmetric DeltaNet). For example:
///
/// - Qwen3.5-0.8B: numHeads=16, groupCount=16, dk=128, dv=128 (symmetric)
/// - Qwen3.5-4B: numHeads=32, groupCount=16, dk=128, dv=128 (asymmetric head counts)
///
/// All dimensions come from GGUF metadata, not tensor shape inference.
public struct StateSpaceAttributes: OperationAttributes, Codable, Equatable {

    /// Hidden dimension of the state-space block.
    public let hiddenSize: Int

    /// Number of recurrence heads. Each head maintains an independent
    /// state matrix of shape `[keyHeadDim, valueHeadDim]`.
    public let numHeads: Int

    /// Number of recurrent key/query heads.
    ///
    /// Symmetric DeltaNet uses one key head per value head.
    /// Asymmetric variants may use fewer key heads and expand them to match the
    /// value-head count before the recurrence update.
    public let groupCount: Int

    /// Per-head key/query dimension (dk). Controls the "memory address" space.
    public let keyHeadDim: Int

    /// Per-head value dimension (dv). Controls the "memory content" width.
    public let valueHeadDim: Int

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
        hiddenSize: Int, numHeads: Int, groupCount: Int? = nil, keyHeadDim: Int, valueHeadDim: Int,
        variant: String, computeDType: ComputeDType = .float32
    ) {
        self.hiddenSize = hiddenSize
        self.numHeads = numHeads
        self.groupCount = groupCount ?? numHeads
        self.keyHeadDim = keyHeadDim
        self.valueHeadDim = valueHeadDim
        self.variant = variant
        self.computeDType = computeDType
    }
}
