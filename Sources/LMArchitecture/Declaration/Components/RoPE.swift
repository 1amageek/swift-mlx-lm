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
public struct RoPE: ModelComponent {

    public typealias Body = Never

    public let dimension: Int
    public let base: Float
    public let scaling: RoPEScaling?
    public let mropeAxes: MRoPEAxes?

    public init(
        dimension: Int,
        base: Float = 10_000.0,
        scaling: RoPEScaling? = nil,
        mropeAxes: MRoPEAxes? = nil
    ) {
        precondition(dimension > 0, "dimension must be positive")
        precondition(base > 0, "base must be positive")
        self.dimension = dimension
        self.base = base
        self.scaling = scaling
        self.mropeAxes = mropeAxes
    }
}

extension RoPE: PrimitiveComponent {

    package var operationKind: OperationKind {
        .primitive(RoPEAttributes(
            dimension: dimension,
            base: base,
            scaling: scaling,
            mropeAxes: mropeAxes
        ))
    }

    package var operationSignature: OperationSignature {
        OperationSignature(operandArity: .exact(1), resultArity: .exact(1))
    }
}
