/// A value produced by an `Operation` or `Region` parameter within a `ModelGraph`.
///
/// `ValueID` is a graph-local address, reassigned during normalization and
/// canonicalization. It is NOT a stable identity — for stable addressing,
/// use `StructuralPath`.
public struct ValueID: Hashable, Codable, Sendable {
    public let rawValue: Int
    public init(rawValue: Int) { self.rawValue = rawValue }
}

/// A formal input value declared by a `Region`.
///
/// Region parameters define the values that flow into the region from
/// the enclosing scope. Each parameter corresponds to one of the
/// parent operation's operands.
public struct RegionParameter: Hashable, Codable, Sendable {
    public let id: ValueID
    public init(id: ValueID) { self.id = id }
}

/// A value consumed by an `Operation` as input.
///
/// References a `ValueID` produced by a prior operation or region parameter.
public struct Operand: Hashable, Codable, Sendable {
    public let value: ValueID
    public init(value: ValueID) { self.value = value }
}

/// A value produced by an `Operation` as output.
///
/// Each operation result introduces a new `ValueID` into the enclosing scope
/// that downstream operations can reference via `Operand`.
public struct OperationResult: Hashable, Codable, Sendable {
    public let id: ValueID
    public init(id: ValueID) { self.id = id }
}

/// A reference to a value yielded as a `Region` output.
///
/// Region results declare which values flow out of the region back
/// to the enclosing scope.
public struct ValueUse: Hashable, Codable, Sendable {
    public let value: ValueID
    public init(value: ValueID) { self.value = value }
}

/// Graph-local key for an operation within a `ModelGraph`.
///
/// Reassigned during canonicalization. For stable addressing,
/// use `StructuralPath`.
public struct OperationKey: Hashable, Codable, Sendable {
    public let rawValue: Int
    public init(rawValue: Int) { self.rawValue = rawValue }
}

/// Opaque attributes for a primitive computation node.
///
/// IR does not interpret these — backend compilers read them via their own protocols.
public protocol OperationAttributes: Sendable {}

/// Canonical, region-bearing, value-explicit structural representation
/// of a language model.
///
/// `ModelGraph` is the semantic ground truth of SwiftLM. It is a
/// **hierarchical semantic SSA IR** with:
///
/// - **Regions**: sequential blocks of `Operation`s with explicit
///   parameters (inputs) and results (outputs).
/// - **Explicit value flow**: each `Operation` declares its operands
///   (consumed `ValueID`s) and results (produced `ValueID`s).
///   Values are defined once (SSA-style) and referenced by downstream
///   operations via `Operand`.
/// - **Region-bearing operations**: structural operations (`residual`,
///   `parallel`, `repeating`) contain nested `Region`s with explicit
///   parameter/result interfaces. Nested regions are scope-isolated —
///   parent values are NOT visible; they must be passed explicitly via
///   the structural operation's operands → region parameters interface.
/// - **Multi-input / multi-result generality**: the IR supports
///   arbitrary operand and result arity. Current DSL front-ends exercise
///   unary flow, but the IR is not constrained to unary.
///
/// This design is inspired by MLIR's region-bearing operations and SSA
/// value semantics. Canonical equivalence is defined on this IR via
/// `canonicalize(_:)`.
///
/// `ModelGraph` contains ONLY semantic information. Diagnostic metadata
/// (labels, source locations) is stored separately in `ModelGraphMetadata`.
public struct ModelGraph: Sendable {

    /// The top-level region containing all operations.
    public let rootRegion: Region

    public init(rootRegion: Region) {
        self.rootRegion = rootRegion
    }
}

/// A sequential block of operations with explicit inputs and outputs.
///
/// A `Region` is a scope: it declares its own `parameters` (input values)
/// and `results` (output values). Operations within consume and produce
/// values through `operands` and `results`.
///
/// Parameters and results support arbitrary arity (multi-input / multi-result).
/// The root region has no parameters (top-level entry point).
/// Sub-regions (inside residual, parallel, repeating) receive their inputs
/// through parameters mapped from the parent operation's operands.
public struct Region: Sendable {

    /// Formal input values to this region.
    public let parameters: [RegionParameter]

    /// Operations in this region, in execution order.
    public let operations: [Operation]

    /// Output values yielded by this region.
    public let results: [ValueUse]

    public init(
        parameters: [RegionParameter] = [],
        operations: [Operation] = [],
        results: [ValueUse] = []
    ) {
        self.parameters = parameters
        self.operations = operations
        self.results = results
    }
}

/// A static reference to a named weight tensor.
///
/// `ParameterBinding` connects an operation's parameter slot (e.g., "q_proj")
/// to a concrete tensor name in the weight store (e.g.,
/// "model.layers.5.self_attn.q_proj.weight").
///
/// This is model-structural information, not backend-specific.
/// Any backend (Metal, TPU) reads the same bindings.
public struct ParameterBinding: Sendable, Hashable {

    /// Role of this parameter within the operation (e.g., "q_proj", "scale").
    public let role: String

    /// Tensor name in the weight store (e.g., safetensors / STAF key).
    public let tensorName: String

    public init(role: String, tensorName: String) {
        self.role = role
        self.tensorName = tensorName
    }
}

/// A single operation in a `ModelGraph`.
///
/// Each operation explicitly declares three axes:
/// - `operands`: dynamic data flow (activations from prior operations)
/// - `results`: values produced for downstream operations
/// - `parameterBindings`: static data references (weight tensor names)
///
/// Structural operations additionally contain nested `Region`s.
public struct Operation: Sendable {

    /// Graph-local key. Reassigned during canonicalization.
    public let key: OperationKey

    /// Semantic kind and attributes of this operation.
    public let kind: OperationKind

    /// Values consumed by this operation (dynamic data flow).
    public let operands: [Operand]

    /// Values produced by this operation.
    public let results: [OperationResult]

    /// Static weight references resolved at graph construction time.
    ///
    /// Each binding maps a parameter role (e.g., "q_proj") to a tensor name
    /// in the weight store (e.g., "model.layers.5.self_attn.q_proj.weight").
    /// The compiler reads these directly — no external mapping needed.
    public let parameterBindings: [ParameterBinding]

    public init(
        key: OperationKey,
        kind: OperationKind,
        operands: [Operand] = [],
        results: [OperationResult] = [],
        parameterBindings: [ParameterBinding] = []
    ) {
        self.key = key
        self.kind = kind
        self.operands = operands
        self.results = results
        self.parameterBindings = parameterBindings
    }
}

/// Operation kind: either a primitive computation or a structural connector.
///
/// Primitive operations hold opaque attributes — the IR does not interpret them.
/// Backend compilers (MetalCompiler, etc.) read the attributes via their own protocols.
///
/// Structural operations define how computations connect (control flow).
///
/// ## Structural Operation Contracts
///
/// Each structural operation imposes arity constraints on its regions:
///
/// - **residual**: `body.parameters.count == operands.count`,
///   `body.results.count == results.count`, result arity == operand arity
/// - **parallel**: all `branches[i].parameters.count == operands.count`,
///   all branch result arities equal, `results.count == branch result arity`
///   (concrete strategies are tensor-level; `.custom` has no arity constraint)
/// - **repeating**: `body.parameters.count == body.results.count == operands.count == results.count`
///   (loop-carried tuple shape must match)
/// - **conditional**: `then.parameters.count == else.parameters.count == operands.count`,
///   `then.results.count == else.results.count == results.count`
///
/// These contracts are enforced by `GraphValidator`.
public indirect enum OperationKind: Sendable {

    // MARK: - Primitive (opaque — IR doesn't know what computation this is)

    /// A leaf computation node with opaque attributes.
    /// The IR only knows this node's operands and results (value flow).
    /// What it computes is determined by the backend via protocol conformance.
    case primitive(any OperationAttributes)

    // MARK: - Structural (region-bearing — IR knows the connection pattern)

    /// Residual connection: output = combine(input, f(input)).
    case residual(strategy: ResidualStrategy, body: Region)

    /// Parallel branches: executes branches on same input, merges results.
    case parallel(merge: ParallelMergeStrategy, branches: [Region])

    /// Repeat: stacks identical blocks a fixed number of times.
    case repeating(count: Int, body: Region)

    /// Compile-time conditional: selects between two bodies based on static condition.
    case conditional(condition: ConditionKind, then: Region, else: Region)
}

/// Compile-time condition for conditional operations.
public enum ConditionKind: Sendable, Codable, Equatable {
    /// Branch based on iteration index within a repeating block.
    case layerIndices(Set<Int>)
}
