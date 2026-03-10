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
/// assert(canonicalize(a) == canonicalize(b))
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
    case .attention(let attrs):
        return .attention(AttentionAttributes(
            hiddenSize: attrs.hiddenSize,
            headCount: attrs.headCount,
            kvHeadCount: attrs.kvHeadCount,
            headDimension: attrs.headDimension,
            bias: attrs.bias,
            causal: attrs.causal,
            rope: attrs.rope.map { canonicalizeRoPE($0) },
            qkNorm: attrs.qkNorm,
            window: attrs.window,
            implementationHint: nil
        ))

    case .rope(let attrs):
        return .rope(canonicalizeRoPE(attrs))

    case .mlp(let attrs):
        return .mlp(MLPAttributes(
            inputSize: attrs.inputSize,
            outputSize: attrs.outputSize,
            intermediateSize: attrs.intermediateSize,
            activation: attrs.activation,
            gating: attrs.gating,
            bias: attrs.bias
        ))

    case .rmsNorm(let attrs):
        return .rmsNorm(RMSNormAttributes(dimension: attrs.dimension, epsilon: attrs.epsilon))

    case .layerNorm(let attrs):
        return .layerNorm(LayerNormAttributes(dimension: attrs.dimension, epsilon: attrs.epsilon, affine: attrs.affine))

    case .residual(let strategy, let body):
        return .residual(strategy: strategy, body: canonicalizeRegion(body, ctx: &ctx))

    case .parallel(let merge, let branches):
        return .parallel(merge: merge, branches: branches.map { canonicalizeRegion($0, ctx: &ctx) })

    case .repeating(let count, let body):
        return .repeating(count: count, body: canonicalizeRegion(body, ctx: &ctx))

    case .layerStack(let layers):
        return .layerStack(layers: layers.map { canonicalizeRegion($0, ctx: &ctx) })

    default:
        return kind
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
