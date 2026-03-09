/// Stable identifier for a node within a LoweredGraph.
public typealias LoweredNodeID = Int

/// Operator-level operations produced by lowering semantic nodes.
///
/// These represent individual computational steps that a backend can execute.
/// Different backends may produce different legal lowerings from the same
/// semantic ModelGraph.
public enum LoweredOp: Codable, Equatable, Sendable {
    case gather
    case matmul
    case add
    case mul
    case reshape
    case transpose
    case split
    case concat
    case ropeApply
    case softmax
    case sdpa
    case activation(ActivationKind)
    case rmsNorm
    case layerNorm
    case custom(String)
}

/// A single node in the LoweredGraph.
public struct LoweredNode: Codable, Equatable, Sendable {

    /// Unique identifier within the lowered graph.
    public let id: LoweredNodeID

    /// Operator to execute.
    public let op: LoweredOp

    /// IDs of nodes whose outputs feed into this node.
    public let inputs: [LoweredNodeID]

    /// Operator-specific attributes for this lowered node.
    public let attributes: LoweredAttributes

    public init(
        id: LoweredNodeID,
        op: LoweredOp,
        inputs: [LoweredNodeID] = [],
        attributes: LoweredAttributes = LoweredAttributes()
    ) {
        self.id = id
        self.op = op
        self.inputs = inputs
        self.attributes = attributes
    }
}

/// Operator-specific attributes for a lowered node.
///
/// Carries parameter values needed by the operator at execution time
/// (dimensions, epsilon, activation kind, etc.).
public struct LoweredAttributes: Codable, Equatable, Sendable {

    /// Arbitrary key-value pairs for operator configuration.
    public let values: [String: LoweredAttributeValue]

    public init(values: [String: LoweredAttributeValue] = [:]) {
        self.values = values
    }
}

/// A single attribute value in lowered node attributes.
public enum LoweredAttributeValue: Codable, Equatable, Sendable {
    case int(Int)
    case float(Float)
    case bool(Bool)
    case string(String)
    case ints([Int])
    case floats([Float])
}

/// Execution-oriented IR produced by the compiler from a ModelGraph.
///
/// LoweredGraph expands semantic units (attention, MLP) into individual
/// operator-level operations. It is backend-neutral but much closer to
/// runtime execution than ModelGraph.
///
/// Canonical model identity is NOT defined on LoweredGraph.
/// Use ModelGraph for equivalence checks.
public struct LoweredGraph: Codable, Equatable, Sendable {

    /// All nodes in the lowered graph.
    public var nodes: [LoweredNode]

    /// IDs of the output nodes.
    public var outputs: [LoweredNodeID]

    public init(nodes: [LoweredNode] = [], outputs: [LoweredNodeID] = []) {
        self.nodes = nodes
        self.outputs = outputs
    }
}
