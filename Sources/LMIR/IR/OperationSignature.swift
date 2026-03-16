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
/// Structural operations (residual, parallel, repeating, conditional) return `nil`.
/// Primitive operations return a default unary signature (1 operand, 1 result).
///
/// Backend compilers may override this default by consulting the concrete
/// `OperationAttributes` type for operation-specific arity.
public func primitiveSignature(from kind: OperationKind) -> OperationSignature? {
    switch kind {
    case .primitive(let attrs):
        if attrs is TokenEmbeddingAttributes {
            return OperationSignature(operandArity: .exact(0), resultArity: .exact(1))
        }
        // Default: unary primitive (1 operand, 1 result)
        return OperationSignature(operandArity: .exact(1), resultArity: .exact(1))
    case .residual, .parallel, .repeating, .conditional:
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
