/// Declares the operand and result arity of an operation kind.
///
/// `OperationSignature` separates the arity contract from the normalizer
/// implementation. Each `OperationKind` has a well-defined signature that
/// validators consult when checking contracts.
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

/// Returns the arity signature for a primitive operation kind.
///
/// Structural operations (residual, parallel, repeating) return `nil`.
///
/// This is the single source of truth for primitive semantic contracts.
/// The normalizer uses `resultArity` to generate the correct number
/// of result `ValueID`s. `operandArity` is the semantic truth about
/// what each primitive expects; enforcement is deferred to profile
/// validation (not the normalizer, which remains tolerant).
public func primitiveSignature(from kind: OperationKind) -> OperationSignature? {
    switch kind {
    case .tokenEmbedding:
        return OperationSignature(operandArity: .exact(0), resultArity: .exact(1))
    case .positionalEmbedding:
        return OperationSignature(operandArity: .exact(1), resultArity: .exact(1))
    case .rope:
        return OperationSignature(operandArity: .exact(1), resultArity: .exact(1))
    case .attention:
        return OperationSignature(operandArity: .exact(1), resultArity: .exact(1))
    case .mlp:
        return OperationSignature(operandArity: .exact(1), resultArity: .exact(1))
    case .moe:
        return OperationSignature(operandArity: .exact(1), resultArity: .exact(1))
    case .rmsNorm:
        return OperationSignature(operandArity: .exact(1), resultArity: .exact(1))
    case .layerNorm:
        return OperationSignature(operandArity: .exact(1), resultArity: .exact(1))
    case .linear:
        return OperationSignature(operandArity: .exact(1), resultArity: .exact(1))
    case .outputHead:
        return OperationSignature(operandArity: .exact(1), resultArity: .exact(1))
    case .stateSpace:
        return OperationSignature(operandArity: .exact(1), resultArity: .exact(1))
    case .custom:
        return OperationSignature(operandArity: .variadic, resultArity: .variadic)
    case .residual, .parallel, .repeating, .layerStack:
        return nil
    }
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
