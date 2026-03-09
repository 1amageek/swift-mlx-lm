/// Converts an open `ModelDeclaration` tree into a `NormalizedModel`.
///
/// **Normalization** performs structural closure with explicit value flow:
/// - Flattens nested `.sequence` declarations into flat `Region` operation lists.
/// - Strips `.labeled` annotations into `ModelGraphMetadata`.
/// - Converts `.residual`, `.parallel`, `.repeating` into region-bearing operations
///   with explicit parameters and results.
/// - Assigns `OperationKey`s and `ValueID`s in depth-first order.
///
/// The normalizer operates in the **general multi-value IR** form. Each
/// declaration step receives an `upstream: [ValueID]` tuple and returns
/// a `NormalizedRegionFragment` with `results: [ValueID]`. The current
/// DSL front-end only exercises unary flow, but the normalizer itself
/// is not constrained to unary.
///
/// The resulting `ModelGraph` is structurally closed and value-explicit, but NOT
/// yet in canonical form for equivalence comparison. Use `canonicalize(_:)`
/// after normalization when identity comparison is needed.
///
/// ```text
/// ModelDeclaration --normalize()--> NormalizedModel
///     .graph:    ModelGraph (value-explicit, region-bearing, multi-value capable)
///     .metadata: ModelGraphMetadata (labels, diagnostics)
/// ```
public func normalize(_ declaration: ModelDeclaration) throws -> NormalizedModel {
    var ctx = NormalizationContext()
    let fragment = try normalizeDecl(declaration, upstream: [], ctx: &ctx)
    let results = fragment.results.map { ValueUse(value: $0) }
    let rootRegion = Region(parameters: [], operations: fragment.operations, results: results)

    let graph = ModelGraph(rootRegion: rootRegion)
    let metadata = buildMetadata(graph: graph, keyLabels: ctx.keyLabels)
    return NormalizedModel(graph: graph, metadata: metadata)
}

// MARK: - Fragment

/// Intermediate result of normalizing a declaration subtree.
///
/// Contains the flattened operations and the result value tuple
/// produced by the subtree. Used internally by the normalizer.
public struct NormalizedRegionFragment: Sendable {

    /// Operations produced by this subtree, in execution order.
    public let operations: [Operation]

    /// Result value tuple produced by this subtree.
    /// Empty if the subtree produces no values.
    public let results: [ValueID]

    public init(operations: [Operation] = [], results: [ValueID] = []) {
        self.operations = operations
        self.results = results
    }
}

// MARK: - Context

private struct NormalizationContext {
    var nextValue: Int = 0
    var nextKey: Int = 0
    var keyLabels: [Int: String] = [:]

    mutating func freshValue() -> ValueID {
        defer { nextValue += 1 }
        return ValueID(rawValue: nextValue)
    }

    mutating func freshKey() -> OperationKey {
        defer { nextKey += 1 }
        return OperationKey(rawValue: nextKey)
    }
}

// MARK: - Core Normalization

/// Normalize a declaration into operations and result values.
///
/// - Parameters:
///   - declaration: The declaration subtree to normalize.
///   - upstream: The value tuple flowing into this subtree from preceding operations.
///   - ctx: Normalization context for fresh ID generation.
/// - Returns: A fragment containing operations and result value IDs.
private func normalizeDecl(
    _ declaration: ModelDeclaration,
    upstream: [ValueID],
    ctx: inout NormalizationContext
) throws -> NormalizedRegionFragment {
    switch declaration {
    case .primitive(let prim):
        let (kind, signature) = primitiveInfo(from: prim)
        let key = ctx.freshKey()
        // The normalizer is intentionally tolerant with respect to operand arity.
        // Semantic operand contracts (signature.operandArity) are enforced by
        // LLMProfileValidator, not here.
        let operands = upstream.map { Operand(value: $0) }
        let resultCount = resolveArity(signature.resultArity, fallback: upstream.count)
        let resultValues = (0..<resultCount).map { _ in ctx.freshValue() }
        let op = Operation(
            key: key,
            kind: kind,
            operands: operands,
            results: resultValues.map { OperationResult(id: $0) }
        )
        return NormalizedRegionFragment(operations: [op], results: resultValues)

    case .sequence(let children):
        // Empty sequence is identity: passes upstream through unchanged.
        if children.isEmpty {
            return NormalizedRegionFragment(operations: [], results: upstream)
        }
        var ops: [Operation] = []
        var current = upstream
        for child in children {
            let fragment = try normalizeDecl(child, upstream: current, ctx: &ctx)
            ops.append(contentsOf: fragment.operations)
            current = fragment.results
        }
        return NormalizedRegionFragment(operations: ops, results: current)

    case .residual(let strategy, let body):
        let bodyParams = upstream.map { _ in ctx.freshValue() }
        let bodyFragment = try normalizeDecl(body, upstream: bodyParams, ctx: &ctx)
        let bodyRegion = Region(
            parameters: bodyParams.map { RegionParameter(id: $0) },
            operations: bodyFragment.operations,
            results: bodyFragment.results.map { ValueUse(value: $0) }
        )

        let key = ctx.freshKey()
        let resultValues = upstream.map { _ in ctx.freshValue() }
        let operands = upstream.map { Operand(value: $0) }
        let op = Operation(
            key: key,
            kind: .residual(strategy: strategy, body: bodyRegion),
            operands: operands,
            results: resultValues.map { OperationResult(id: $0) }
        )
        return NormalizedRegionFragment(operations: [op], results: resultValues)

    case .parallel(let merge, let branches):
        var branchRegions: [Region] = []
        for branch in branches {
            let branchParams = upstream.map { _ in ctx.freshValue() }
            let branchFragment = try normalizeDecl(branch, upstream: branchParams, ctx: &ctx)
            branchRegions.append(Region(
                parameters: branchParams.map { RegionParameter(id: $0) },
                operations: branchFragment.operations,
                results: branchFragment.results.map { ValueUse(value: $0) }
            ))
        }

        let key = ctx.freshKey()
        // Result arity derived from branch results.
        // All concrete merge strategies are tensor-level operations
        // that preserve value-flow arity.
        let branchResultArity = branchRegions.first?.results.count ?? 0
        let resultValues = (0..<branchResultArity).map { _ in ctx.freshValue() }
        let operands = upstream.map { Operand(value: $0) }
        let op = Operation(
            key: key,
            kind: .parallel(merge: merge, branches: branchRegions),
            operands: operands,
            results: resultValues.map { OperationResult(id: $0) }
        )
        return NormalizedRegionFragment(operations: [op], results: resultValues)

    case .repeating(let count, let label, let body):
        let bodyParams = upstream.map { _ in ctx.freshValue() }
        let bodyFragment = try normalizeDecl(body, upstream: bodyParams, ctx: &ctx)
        let bodyRegion = Region(
            parameters: bodyParams.map { RegionParameter(id: $0) },
            operations: bodyFragment.operations,
            results: bodyFragment.results.map { ValueUse(value: $0) }
        )

        let key = ctx.freshKey()
        let resultValues = upstream.map { _ in ctx.freshValue() }
        let operands = upstream.map { Operand(value: $0) }
        let op = Operation(
            key: key,
            kind: .repeating(count: count, body: bodyRegion),
            operands: operands,
            results: resultValues.map { OperationResult(id: $0) }
        )

        if let label {
            ctx.keyLabels[key.rawValue] = label
        }

        return NormalizedRegionFragment(operations: [op], results: resultValues)

    case .labeled(let label, let inner):
        let fragment = try normalizeDecl(inner, upstream: upstream, ctx: &ctx)
        // Attach label to the first operation produced (if any)
        if let firstOp = fragment.operations.first {
            ctx.keyLabels[firstOp.key.rawValue] = label
        }
        return fragment
    }
}

// MARK: - Metadata Construction

/// Walk the graph to build StructuralPath-based metadata from key-based labels.
private func buildMetadata(
    graph: ModelGraph,
    keyLabels: [Int: String]
) -> ModelGraphMetadata {
    guard !keyLabels.isEmpty else { return ModelGraphMetadata() }

    var entries: [AnnotationEntry] = []
    walkRegion(graph.rootRegion, basePath: StructuralPath(), keyLabels: keyLabels, entries: &entries)
    return ModelGraphMetadata(annotations: entries)
}

private func walkRegion(
    _ region: Region,
    basePath: StructuralPath,
    keyLabels: [Int: String],
    entries: inout [AnnotationEntry]
) {
    for (i, op) in region.operations.enumerated() {
        let opPath = basePath.appending(.operation(i))

        if let label = keyLabels[op.key.rawValue] {
            entries.append(AnnotationEntry(
                path: opPath,
                annotation: OperationAnnotation(label: label)
            ))
        }

        switch op.kind {
        case .residual(_, let body):
            walkRegion(body, basePath: opPath.appending(.regionBody), keyLabels: keyLabels, entries: &entries)
        case .parallel(_, let branches):
            for (j, branch) in branches.enumerated() {
                walkRegion(branch, basePath: opPath.appending(.regionBranch(j)), keyLabels: keyLabels, entries: &entries)
            }
        case .repeating(_, let body):
            walkRegion(body, basePath: opPath.appending(.regionBody), keyLabels: keyLabels, entries: &entries)
        default:
            break
        }
    }
}

