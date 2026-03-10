/// Validates structural well-formedness of a `ModelGraph`.
///
/// `GraphValidator` checks IR-level contracts independent of any
/// particular front-end (DSL) or profile. These are invariants that
/// every well-formed `ModelGraph` must satisfy:
///
/// - All value uses reference values defined within the same region scope
///   (SSA dominance with explicit region interfaces).
/// - Structural operation contracts hold (arity matching between
///   operands, results, and nested region parameters/results).
/// - Parallel merge strategy result arity contracts are satisfied.
///
/// **Scope model**: Nested regions are validated in isolation from their
/// parent scope. A region's visible values are ONLY its own parameters
/// and the results of prior operations within that region. Parent scope
/// values are NOT visible — they must be explicitly passed via the
/// structural operation's operands → region parameters interface.
///
/// For front-end-specific validation (e.g., "current DSL primitives
/// are unary"), use a separate profile validator.
public enum GraphValidator {

    /// Validate structural well-formedness of a model graph.
    ///
    /// - Throws: `GraphValidationError` if any structural contract is violated.
    public static func validate(_ graph: ModelGraph) throws {
        try validateRegion(graph.rootRegion)
    }
}

// MARK: - Errors

/// Errors from graph structural validation.
public enum GraphValidationError: Error, Sendable {

    /// An operand references a value that is not defined in scope.
    case undefinedValue(ValueID, inOperation: OperationKey)

    /// A region result references a value that is not defined in scope.
    case undefinedRegionResult(ValueID)

    /// Residual body parameter arity does not match operand arity.
    case residualArityMismatch(
        operandCount: Int,
        bodyParameterCount: Int,
        operationKey: OperationKey
    )

    /// Residual body result arity does not match operation result arity.
    case residualResultArityMismatch(
        resultCount: Int,
        bodyResultCount: Int,
        operationKey: OperationKey
    )

    /// Parallel branch parameter arity does not match operand arity.
    case parallelBranchArityMismatch(
        branchIndex: Int,
        operandCount: Int,
        branchParameterCount: Int,
        operationKey: OperationKey
    )

    /// Parallel branch result arity violates merge strategy contract.
    case parallelMergeArityMismatch(
        strategy: ParallelMergeStrategy,
        branchResultCounts: [Int],
        operationResultCount: Int,
        operationKey: OperationKey
    )

    /// Repeating body parameter arity does not match operand arity.
    case repeatingArityMismatch(
        operandCount: Int,
        bodyParameterCount: Int,
        operationKey: OperationKey
    )

    /// Repeating body result arity does not match body parameter arity.
    case repeatingLoopArityMismatch(
        bodyParameterCount: Int,
        bodyResultCount: Int,
        operationKey: OperationKey
    )

    /// Repeating operation result arity does not match operand arity.
    case repeatingResultArityMismatch(
        operandCount: Int,
        resultCount: Int,
        operationKey: OperationKey
    )
}

// MARK: - Internal

private typealias ValueSet = Set<ValueID>

/// Validate a region in isolation.
///
/// The region's visible scope starts with ONLY its own parameters.
/// Each operation's results are added to the scope as they are processed.
/// Parent scope values are NOT visible — they must be passed explicitly
/// through the structural operation's operands → region parameters interface.
private func validateRegion(_ region: Region) throws {
    var defined = ValueSet()

    // Region parameters are the only entry values.
    for param in region.parameters {
        defined.insert(param.id)
    }

    // Validate each operation in sequence.
    for op in region.operations {
        try validateOperation(op, defined: &defined)
    }

    // Validate region results reference values defined within this region.
    for result in region.results {
        guard defined.contains(result.value) else {
            throw GraphValidationError.undefinedRegionResult(result.value)
        }
    }
}

/// Validate a single operation and its nested regions.
private func validateOperation(
    _ op: Operation,
    defined: inout ValueSet
) throws {
    // Check all operands reference values visible at this point in the region.
    for operand in op.operands {
        guard defined.contains(operand.value) else {
            throw GraphValidationError.undefinedValue(operand.value, inOperation: op.key)
        }
    }

    // Validate structural operation contracts and nested regions.
    try validateStructuralContract(op)

    // Validate nested regions in isolation (their own scope, not parent scope).
    switch op.kind {
    case .residual(_, let body):
        try validateRegion(body)

    case .parallel(_, let branches):
        for branch in branches {
            try validateRegion(branch)
        }

    case .repeating(_, let body):
        try validateRegion(body)

    case .layerStack(let layers):
        for layer in layers {
            try validateRegion(layer)
        }

    default:
        break
    }

    // Operation results introduce values into the enclosing region's scope.
    for result in op.results {
        defined.insert(result.id)
    }
}

/// Validate arity contracts for structural (region-bearing) operations.
private func validateStructuralContract(_ op: Operation) throws {
    switch op.kind {
    case .residual(_, let body):
        // body.parameters.count == operands.count
        if body.parameters.count != op.operands.count {
            throw GraphValidationError.residualArityMismatch(
                operandCount: op.operands.count,
                bodyParameterCount: body.parameters.count,
                operationKey: op.key
            )
        }
        // body.results.count == results.count
        if body.results.count != op.results.count {
            throw GraphValidationError.residualResultArityMismatch(
                resultCount: op.results.count,
                bodyResultCount: body.results.count,
                operationKey: op.key
            )
        }

    case .parallel(let strategy, let branches):
        // All branches: parameters.count == operands.count
        for (i, branch) in branches.enumerated() {
            if branch.parameters.count != op.operands.count {
                throw GraphValidationError.parallelBranchArityMismatch(
                    branchIndex: i,
                    operandCount: op.operands.count,
                    branchParameterCount: branch.parameters.count,
                    operationKey: op.key
                )
            }
        }

        // Merge strategy result arity contract
        let branchResultCounts = branches.map(\.results.count)
        try validateMergeArity(
            strategy: strategy,
            branchResultCounts: branchResultCounts,
            operationResultCount: op.results.count,
            operationKey: op.key
        )

    case .repeating(_, let body):
        // body.parameters.count == operands.count
        if body.parameters.count != op.operands.count {
            throw GraphValidationError.repeatingArityMismatch(
                operandCount: op.operands.count,
                bodyParameterCount: body.parameters.count,
                operationKey: op.key
            )
        }
        // body.results.count == body.parameters.count (loop-carried)
        if body.results.count != body.parameters.count {
            throw GraphValidationError.repeatingLoopArityMismatch(
                bodyParameterCount: body.parameters.count,
                bodyResultCount: body.results.count,
                operationKey: op.key
            )
        }
        // results.count == operands.count
        if op.results.count != op.operands.count {
            throw GraphValidationError.repeatingResultArityMismatch(
                operandCount: op.operands.count,
                resultCount: op.results.count,
                operationKey: op.key
            )
        }

    case .layerStack(let layers):
        // Each layer: parameters.count == operands.count
        for layer in layers {
            if layer.parameters.count != op.operands.count {
                throw GraphValidationError.repeatingArityMismatch(
                    operandCount: op.operands.count,
                    bodyParameterCount: layer.parameters.count,
                    operationKey: op.key
                )
            }
            // Each layer: results.count == parameters.count (loop-carried)
            if layer.results.count != layer.parameters.count {
                throw GraphValidationError.repeatingLoopArityMismatch(
                    bodyParameterCount: layer.parameters.count,
                    bodyResultCount: layer.results.count,
                    operationKey: op.key
                )
            }
        }
        // results.count == operands.count
        if op.results.count != op.operands.count {
            throw GraphValidationError.repeatingResultArityMismatch(
                operandCount: op.operands.count,
                resultCount: op.results.count,
                operationKey: op.key
            )
        }

    default:
        break
    }
}

/// Validate parallel merge strategy result arity.
///
/// All concrete strategies (`.add`, `.concat`, `.stack`) are tensor-level
/// operations that preserve value-flow arity:
///
/// - All branches must have equal result count.
/// - `op.results.count` must equal that count.
///
/// `.custom`: No arity constraint at the IR level.
private func validateMergeArity(
    strategy: ParallelMergeStrategy,
    branchResultCounts: [Int],
    operationResultCount: Int,
    operationKey: OperationKey
) throws {
    switch strategy {
    case .add, .concat, .stack:
        // All concrete strategies: tensor-level combination,
        // value-flow arity preserved.
        guard let first = branchResultCounts.first else { return }
        let allEqual = branchResultCounts.allSatisfy { $0 == first }
        let matchesOp = operationResultCount == first
        if !allEqual || !matchesOp {
            throw GraphValidationError.parallelMergeArityMismatch(
                strategy: strategy,
                branchResultCounts: branchResultCounts,
                operationResultCount: operationResultCount,
                operationKey: operationKey
            )
        }

    case .visionMerge:
        // Vision merge requires exactly 2 branches (text + vision)
        // with equal result arity.
        guard let first = branchResultCounts.first else { return }
        let allEqual = branchResultCounts.allSatisfy { $0 == first }
        let matchesOp = operationResultCount == first
        if branchResultCounts.count != 2 || !allEqual || !matchesOp {
            throw GraphValidationError.parallelMergeArityMismatch(
                strategy: strategy,
                branchResultCounts: branchResultCounts,
                operationResultCount: operationResultCount,
                operationKey: operationKey
            )
        }

    case .custom:
        // No arity constraint for custom merge strategies.
        break
    }
}
