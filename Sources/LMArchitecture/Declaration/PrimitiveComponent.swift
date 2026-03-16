/// Package-internal protocol for primitive model components.
///
/// Primitive components (MLP, Attention, RMSNorm, etc.) conform to this
/// protocol to provide their `OperationKind` and `OperationSignature`
/// for normalization. This replaces the `PrimitiveDeclaration` enum —
/// each component type IS the declaration.
///
/// The normalizer checks `component as? any PrimitiveComponent` to
/// handle all primitive types uniformly via dynamic dispatch.
package protocol PrimitiveComponent: ModelComponent where Body == Never {

    /// The IR operation kind produced by this component.
    var operationKind: OperationKind { get }

    /// The arity signature for this component.
    var operationSignature: OperationSignature { get }
}
