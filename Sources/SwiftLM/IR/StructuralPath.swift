/// Stable structural address for an operation or value within a `ModelGraph`.
///
/// Unlike `OperationKey` and `ValueID`, a `StructuralPath` is invariant
/// across normalization and canonicalization. It identifies a location by
/// its position in the region hierarchy.
///
/// ```swift
/// // "Root region, 2nd operation, body region, 1st operation"
/// let path = StructuralPath(components: [.operation(1), .regionBody, .operation(0)])
///
/// // "Root region, 1st operation, 2nd operand"
/// let operandPath = StructuralPath(components: [.operation(0), .operand(1)])
///
/// // "Root region, parameter 0"
/// let paramPath = StructuralPath(components: [.parameter(0)])
/// ```
public struct StructuralPath: Hashable, Codable, Sendable {

    /// Path components from root to target.
    public let components: [StructuralPathComponent]

    public init(components: [StructuralPathComponent] = []) {
        self.components = components
    }

    /// Append a component to this path.
    public func appending(_ component: StructuralPathComponent) -> StructuralPath {
        StructuralPath(components: components + [component])
    }
}

/// A single step in a structural path.
public enum StructuralPathComponent: Hashable, Codable, Sendable {

    /// The nth operation within a region.
    case operation(Int)

    /// The body region of a structural operation (residual, repeating).
    case regionBody

    /// The nth branch of a parallel operation.
    case regionBranch(Int)

    /// The nth formal parameter of a region.
    case parameter(Int)

    /// The nth operand of an operation.
    case operand(Int)

    /// The nth result of an operation or region.
    case result(Int)

    /// A named field within an operation (e.g., "q_proj").
    case field(String)

    /// A numeric index (e.g., expert index in MoE).
    case index(Int)
}
