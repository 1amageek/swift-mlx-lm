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
public struct ModelGraph: Codable, Equatable, Sendable {

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
public struct Region: Codable, Equatable, Sendable {

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

/// A single operation in a `ModelGraph`.
///
/// Each operation explicitly declares:
/// - `operands`: values it consumes from the enclosing scope (multi-input capable)
/// - `results`: values it produces for downstream operations (multi-result capable)
///
/// Structural operations additionally contain nested `Region`s.
public struct Operation: Codable, Equatable, Sendable {

    /// Graph-local key. Reassigned during canonicalization.
    public let key: OperationKey

    /// Semantic kind and attributes of this operation.
    public let kind: OperationKind

    /// Values consumed by this operation, visible at this point in the region
    /// (region parameters and prior operation results).
    public let operands: [Operand]

    /// Values produced by this operation.
    public let results: [OperationResult]

    public init(
        key: OperationKey,
        kind: OperationKind,
        operands: [Operand] = [],
        results: [OperationResult] = []
    ) {
        self.key = key
        self.kind = kind
        self.operands = operands
        self.results = results
    }
}

/// Closed set of semantic operation kinds in the `ModelGraph`.
///
/// Primitive cases represent leaf computational units.
/// Structural cases (residual, parallel, repeating) are region-bearing:
/// they contain nested `Region`s that capture sub-structure.
///
/// This enum contains ONLY semantic information. Diagnostic metadata
/// (labels) is stored separately in `ModelGraphMetadata`.
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
///
/// These contracts are enforced by `GraphValidator`.
public indirect enum OperationKind: Codable, Equatable, Sendable {

    // MARK: - Primitive Operations

    /// Token embedding: maps token IDs to dense vectors.
    case tokenEmbedding(TokenEmbeddingAttributes)

    /// Positional embedding: adds position information to token vectors.
    case positionalEmbedding(PositionalEmbeddingAttributes)

    /// Rotary position embedding: applies rotation-based position encoding.
    case rope(RoPEAttributes)

    /// Multi-head attention: Q/K/V projections, scaled dot-product, output projection.
    case attention(AttentionAttributes)

    /// Feed-forward network: gate/up/down projections with activation.
    case mlp(MLPAttributes)

    /// Mixture-of-Experts: routes tokens to expert MLPs via gating.
    case moe(MoEAttributes)

    /// RMS normalization.
    case rmsNorm(RMSNormAttributes)

    /// Layer normalization.
    case layerNorm(LayerNormAttributes)

    /// Linear projection.
    case linear(LinearAttributes)

    /// Output head: projects hidden states to vocabulary logits.
    case outputHead(OutputHeadAttributes)

    /// State-space model block (Mamba, DeltaNet, etc.).
    case stateSpace(StateSpaceAttributes)

    // MARK: - Structural Operations (region-bearing)

    /// Residual connection: output = input + f(input).
    /// The `body` region contains the transformation branch.
    case residual(strategy: ResidualStrategy, body: Region)

    /// Parallel branches: executes branches on same input, merges results.
    /// Each element in `branches` is an independent computation path.
    case parallel(merge: ParallelMergeStrategy, branches: [Region])

    /// Repeat: stacks identical blocks a fixed number of times.
    /// The `body` region is the repeated block.
    case repeating(count: Int, body: Region)

    // MARK: - Escape Hatch

    /// Custom operation: intentionally constrained escape hatch.
    case custom(CustomNodeAttributes)
}
