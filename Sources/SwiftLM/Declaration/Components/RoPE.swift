/// Standalone rotary position embedding component.
///
/// Applies rotation-based position encoding as a separate operation.
/// For most architectures, RoPE is embedded within `Attention` via
/// `AttentionAttributes.rope`. Use this component only when RoPE
/// is applied independently of attention.
///
/// ```swift
/// RoPE(dimension: 128, base: 500_000.0)
/// ```
public struct RoPE: PrimitiveModelComponent {

    public let dimension: Int
    public let base: Float
    public let scaling: RoPEScaling?

    public init(
        dimension: Int,
        base: Float = 10_000.0,
        scaling: RoPEScaling? = nil
    ) {
        precondition(dimension > 0, "dimension must be positive")
        precondition(base > 0, "base must be positive")
        self.dimension = dimension
        self.base = base
        self.scaling = scaling
    }

    public func makeDeclaration() -> ModelDeclaration {
        .primitive(.rope(RoPEAttributes(
            dimension: dimension,
            base: base,
            scaling: scaling
        )))
    }
}
