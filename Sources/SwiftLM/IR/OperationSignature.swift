/// Declares the operand and result arity of an operation kind.
///
/// `OperationSignature` separates the arity contract from the normalizer
/// implementation. Each `PrimitiveDeclaration` / `OperationKind` has a
/// well-defined signature that the normalizer consults when generating
/// `ValueID`s.
///
/// Current DSL primitives are all unary (1 operand, 1 result), but the
/// IR supports arbitrary arity. Future semantic nodes (e.g., router with
/// auxiliary loss, multimodal split) can declare multi-result signatures
/// without changing the normalizer.
public struct OperationSignature: Sendable, Equatable {

    /// Expected operand arity.
    public let operandArity: Arity

    /// Expected result arity.
    public let resultArity: Arity

    public init(operandArity: Arity, resultArity: Arity) {
        self.operandArity = operandArity
        self.resultArity = resultArity
    }
}

/// Arity specification for operand or result count.
public enum Arity: Sendable, Equatable {

    /// Exactly N values.
    case exact(Int)

    /// Variable number of values (determined at normalization time).
    case variadic
}

// MARK: - Primitive Signatures

/// Returns the operation kind and arity signature for a primitive declaration.
///
/// This is the single source of truth for primitive semantic contracts.
/// The normalizer uses `resultArity` to generate the correct number
/// of result `ValueID`s. `operandArity` is the semantic truth about
/// what each primitive expects; enforcement is deferred to profile
/// validation (not the normalizer, which remains tolerant).
public func primitiveInfo(
    from primitive: PrimitiveDeclaration
) -> (kind: OperationKind, signature: OperationSignature) {
    let kind: OperationKind
    let signature: OperationSignature

    switch primitive {
    case .tokenEmbedding(let a):
        // Source operation: produces embeddings from weight parameters,
        // does not consume upstream hidden state.
        kind = .tokenEmbedding(a)
        signature = OperationSignature(operandArity: .exact(0), resultArity: .exact(1))

    case .positionalEmbedding(let a):
        kind = .positionalEmbedding(a)
        signature = OperationSignature(operandArity: .exact(1), resultArity: .exact(1))

    case .rope(let a):
        kind = .rope(a)
        signature = OperationSignature(operandArity: .exact(1), resultArity: .exact(1))

    case .attention(let a):
        kind = .attention(a)
        signature = OperationSignature(operandArity: .exact(1), resultArity: .exact(1))

    case .mlp(let a):
        kind = .mlp(a)
        signature = OperationSignature(operandArity: .exact(1), resultArity: .exact(1))

    case .moe(let a):
        kind = .moe(a)
        signature = OperationSignature(operandArity: .exact(1), resultArity: .exact(1))

    case .rmsNorm(let a):
        kind = .rmsNorm(a)
        signature = OperationSignature(operandArity: .exact(1), resultArity: .exact(1))

    case .layerNorm(let a):
        kind = .layerNorm(a)
        signature = OperationSignature(operandArity: .exact(1), resultArity: .exact(1))

    case .linear(let a):
        kind = .linear(a)
        signature = OperationSignature(operandArity: .exact(1), resultArity: .exact(1))

    case .outputHead(let a):
        kind = .outputHead(a)
        signature = OperationSignature(operandArity: .exact(1), resultArity: .exact(1))

    case .stateSpace(let a):
        kind = .stateSpace(a)
        signature = OperationSignature(operandArity: .exact(1), resultArity: .exact(1))

    case .custom(let a):
        // Escape hatch: no arity constraint.
        kind = .custom(a)
        signature = OperationSignature(operandArity: .variadic, resultArity: .variadic)
    }

    return (kind, signature)
}

/// Resolves an `Arity` to a concrete count given an upstream tuple size.
///
/// - `.exact(n)` always returns `n`.
/// - `.variadic` falls back to `fallback` (typically the upstream count).
public func resolveArity(_ arity: Arity, fallback: Int) -> Int {
    switch arity {
    case .exact(let n): return n
    case .variadic: return fallback
    }
}
