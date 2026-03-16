/// Canonicalizes a `ModelGraph` for structural equivalence comparison.
///
/// Canonicalization normalizes attribute defaults (strips implementation hints)
/// and reassigns `OperationKey`s and `ValueID`s in a deterministic depth-first
/// order so that structurally equivalent graphs produce `Equatable`-equal results.
///
/// Canonicalization preserves multi-value arity: parameter/operand/result
/// tuples retain their element count; only the `ValueID` numbering changes.
///
/// Canonicalization operates ONLY on the semantic `ModelGraph`, not on
/// `ModelGraphMetadata`. Metadata is not part of canonical identity.
///
/// ```swift
/// let a = try normalize(modelA.body.makeDeclaration()).graph
/// let b = try normalize(modelB.body.makeDeclaration()).graph
/// // Canonical graphs have deterministic ValueID/OperationKey assignment
/// // for structural equivalence comparison.
/// ```
public func canonicalize(_ graph: ModelGraph) -> ModelGraph {
    var ctx = CanonContext()
    let canonRoot = canonicalizeRegion(graph.rootRegion, ctx: &ctx)
    return ModelGraph(rootRegion: canonRoot)
}

// MARK: - Context

private struct CanonContext {
    var nextKey: Int = 0
    var nextValue: Int = 0
    var valueMap: [ValueID: ValueID] = [:]

    mutating func mapValue(_ old: ValueID) -> ValueID {
        if let mapped = valueMap[old] { return mapped }
        let new = ValueID(rawValue: nextValue)
        nextValue += 1
        valueMap[old] = new
        return new
    }

    mutating func freshKey() -> OperationKey {
        defer { nextKey += 1 }
        return OperationKey(rawValue: nextKey)
    }
}

// MARK: - Region / Operation

private func canonicalizeRegion(_ region: Region, ctx: inout CanonContext) -> Region {
    let newParams = region.parameters.map { RegionParameter(id: ctx.mapValue($0.id)) }
    let newOps = region.operations.map { canonicalizeOperation($0, ctx: &ctx) }
    let newResults = region.results.map { ValueUse(value: ctx.mapValue($0.value)) }
    return Region(parameters: newParams, operations: newOps, results: newResults)
}

private func canonicalizeOperation(_ op: Operation, ctx: inout CanonContext) -> Operation {
    let newKey = ctx.freshKey()
    let newOperands = op.operands.map { Operand(value: ctx.mapValue($0.value)) }
    let newKind = canonicalizeKind(op.kind, ctx: &ctx)
    let newResults = op.results.map { OperationResult(id: ctx.mapValue($0.id)) }
    return Operation(key: newKey, kind: newKind, operands: newOperands, results: newResults)
}

// MARK: - Kind Normalization

private func canonicalizeKind(_ kind: OperationKind, ctx: inout CanonContext) -> OperationKind {
    switch kind {
    case .primitive(let attrs):
        return .primitive(canonicalizePrimitiveAttributes(attrs))

    case .residual(let strategy, let body):
        return .residual(strategy: strategy, body: canonicalizeRegion(body, ctx: &ctx))

    case .parallel(let merge, let branches):
        return .parallel(merge: merge, branches: branches.map { canonicalizeRegion($0, ctx: &ctx) })

    case .repeating(let count, let body):
        return .repeating(count: count, body: canonicalizeRegion(body, ctx: &ctx))

    case .conditional(let condition, let thenRegion, let elseRegion):
        return .conditional(
            condition: condition,
            then: canonicalizeRegion(thenRegion, ctx: &ctx),
            else: canonicalizeRegion(elseRegion, ctx: &ctx)
        )
    }
}

private func canonicalizePrimitiveAttributes(_ attrs: any OperationAttributes) -> any OperationAttributes {
    switch attrs {
    case let a as AttentionAttributes:
        return AttentionAttributes(
            hiddenSize: a.hiddenSize,
            headCount: a.headCount,
            kvHeadCount: a.kvHeadCount,
            headDimension: a.headDimension,
            bias: a.bias,
            causal: a.causal,
            rope: a.rope.map { canonicalizeRoPE($0) },
            qkNorm: a.qkNorm,
            window: a.window,
            implementationHint: nil
        )
    case let a as RoPEAttributes:
        return canonicalizeRoPE(a)
    case let a as MLPAttributes:
        return MLPAttributes(
            inputSize: a.inputSize,
            outputSize: a.outputSize,
            intermediateSize: a.intermediateSize,
            activation: a.activation,
            gating: a.gating,
            bias: a.bias
        )
    case let a as RMSNormAttributes:
        return RMSNormAttributes(dimension: a.dimension, epsilon: a.epsilon)
    case let a as LayerNormAttributes:
        return LayerNormAttributes(dimension: a.dimension, epsilon: a.epsilon, affine: a.affine)
    default:
        return attrs
    }
}

private func canonicalizeRoPE(_ attrs: RoPEAttributes) -> RoPEAttributes {
    RoPEAttributes(
        dimension: attrs.dimension,
        base: attrs.base,
        scaling: attrs.scaling,
        mropeAxes: attrs.mropeAxes
    )
}
