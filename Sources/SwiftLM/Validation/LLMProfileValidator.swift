/// Validates that a `ModelGraph` conforms to the Core LLM Profile.
///
/// While `GraphValidator` checks structural well-formedness of any
/// `ModelGraph` (SSA dominance, region arity contracts), `LLMProfileValidator`
/// checks domain-specific constraints for autoregressive language models:
///
/// - The root region is non-empty and produces exactly one result.
/// - Primitive operations respect their `OperationSignature` operand arity.
/// - `tokenEmbedding` is a source (zero operands).
///
/// This separation follows the principle that the IR is general-purpose,
/// while profile validators enforce domain constraints.
///
/// ```swift
/// let graph = try model.makeModelGraph()
/// try GraphValidator.validate(graph)       // structural well-formedness
/// try LLMProfileValidator.validate(graph)  // LLM domain constraints
/// ```
public enum LLMProfileValidator {

    /// Validate that a model graph conforms to the Core LLM Profile.
    ///
    /// - Throws: `LLMProfileError` if any domain constraint is violated.
    public static func validate(_ graph: ModelGraph) throws {
        let root = graph.rootRegion

        // Root must contain at least one operation.
        guard !root.operations.isEmpty else {
            throw LLMProfileError.emptyModel
        }

        // Root must produce exactly one result (single hidden-state stream).
        guard root.results.count == 1 else {
            throw LLMProfileError.expectedSingleResult(
                got: root.results.count
            )
        }

        try validateRegionPrimitiveArity(root)
    }
}

// MARK: - Errors

/// Errors from LLM profile validation.
public enum LLMProfileError: Error, Sendable {

    /// The root region contains no operations.
    case emptyModel

    /// The root region does not produce exactly one result.
    case expectedSingleResult(got: Int)

    /// A primitive operation's operand count does not match its signature.
    case operandArityMismatch(
        expected: Arity,
        actual: Int,
        operationKey: OperationKey
    )
}

// MARK: - Internal

/// Walk a region and validate primitive operand arity against signatures.
private func validateRegionPrimitiveArity(_ region: Region) throws {
    for op in region.operations {
        // Check primitive operand arity.
        if let primitiveDecl = primitiveDeclaration(from: op.kind) {
            let (_, signature) = primitiveInfo(from: primitiveDecl)
            switch signature.operandArity {
            case .exact(let expected):
                if op.operands.count != expected {
                    throw LLMProfileError.operandArityMismatch(
                        expected: signature.operandArity,
                        actual: op.operands.count,
                        operationKey: op.key
                    )
                }
            case .variadic:
                break
            }
        }

        // Recurse into nested regions.
        switch op.kind {
        case .residual(_, let body):
            try validateRegionPrimitiveArity(body)
        case .parallel(_, let branches):
            for branch in branches {
                try validateRegionPrimitiveArity(branch)
            }
        case .repeating(_, let body):
            try validateRegionPrimitiveArity(body)
        default:
            break
        }
    }
}

/// Extract a `PrimitiveDeclaration` from an `OperationKind`, if primitive.
private func primitiveDeclaration(from kind: OperationKind) -> PrimitiveDeclaration? {
    switch kind {
    case .tokenEmbedding(let a): return .tokenEmbedding(a)
    case .positionalEmbedding(let a): return .positionalEmbedding(a)
    case .rope(let a): return .rope(a)
    case .attention(let a): return .attention(a)
    case .mlp(let a): return .mlp(a)
    case .moe(let a): return .moe(a)
    case .rmsNorm(let a): return .rmsNorm(a)
    case .layerNorm(let a): return .layerNorm(a)
    case .linear(let a): return .linear(a)
    case .outputHead(let a): return .outputHead(a)
    case .stateSpace(let a): return .stateSpace(a)
    case .custom(let a): return .custom(a)
    case .residual, .parallel, .repeating: return nil
    }
}
