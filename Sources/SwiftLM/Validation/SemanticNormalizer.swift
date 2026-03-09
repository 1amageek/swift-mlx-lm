/// Converts a `ModelComponent` tree into a `NormalizedModel`.
///
/// **Normalization** performs structural closure with explicit value flow:
/// - Flattens `TupleComponent` children into flat `Region` operation lists.
/// - Strips `Group` labels into `ModelGraphMetadata`.
/// - Converts `Residual`, `Parallel`, `Repeat` into region-bearing operations
///   with explicit parameters and results.
/// - Assigns `OperationKey`s and `ValueID`s in depth-first order.
///
/// The normalizer operates in the **general multi-value IR** form. Each
/// component step receives an `upstream: [ValueID]` tuple and returns
/// a `NormalizedRegionFragment` with `results: [ValueID]`. The current
/// DSL front-end only exercises unary flow, but the normalizer itself
/// is not constrained to unary.
///
/// The resulting `ModelGraph` is structurally closed and value-explicit, but NOT
/// yet in canonical form for equivalence comparison. Use `canonicalize(_:)`
/// after normalization when identity comparison is needed.
///
/// ```text
/// ModelComponent --normalize()--> NormalizedModel
///     .graph:    ModelGraph (value-explicit, region-bearing, multi-value capable)
///     .metadata: ModelGraphMetadata (labels, diagnostics)
/// ```
public func normalize(_ component: some ModelComponent) throws -> NormalizedModel {
    var ctx = NormalizationContext()
    let fragment = try normalizeComponent(component, upstream: [], ctx: &ctx)
    let results = fragment.results.map { ValueUse(value: $0) }
    let rootRegion = Region(parameters: [], operations: fragment.operations, results: results)

    let graph = ModelGraph(rootRegion: rootRegion)
    let metadata = buildMetadata(graph: graph, keyLabels: ctx.keyLabels)
    return NormalizedModel(graph: graph, metadata: metadata)
}

// MARK: - Fragment

/// Intermediate result of normalizing a component subtree.
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

// MARK: - Built-in Component Protocol

/// File-private protocol for built-in structural components.
///
/// Each built-in component (`TupleComponent`, `Residual`, `Parallel`, etc.)
/// conforms to this protocol to provide normalization logic. The normalizer
/// dispatches via `as? any _BuiltinComponent` before falling through to
/// primitive or user-defined composite handling.
///
/// This replaces `as?` checks on concrete generic types, which don't work
/// because generic type parameters are unknown at the cast site.
private protocol _BuiltinComponent: ModelComponent {
    func _normalize(
        upstream: [ValueID],
        ctx: inout NormalizationContext
    ) throws -> NormalizedRegionFragment
}

/// File-private protocol for extracting parallel branches from tuple types.
///
/// `TupleComponent` conforms to this so that `Parallel._normalize` can
/// decompose a tuple into independent branch regions.
private protocol _BranchExtractable {
    func _asBranches(
        upstream: [ValueID],
        ctx: inout NormalizationContext
    ) throws -> [Region]
}

// MARK: - Core Normalization

/// Normalize a component into operations and result values.
///
/// Dispatches on built-in component types via `_BuiltinComponent` protocol,
/// then checks for primitive components, and finally falls through to
/// user-defined composites (recursing into `body`).
private func normalizeComponent<C: ModelComponent>(
    _ component: C,
    upstream: [ValueID],
    ctx: inout NormalizationContext
) throws -> NormalizedRegionFragment {
    // 1. Built-in structural components (TupleComponent, Residual, etc.)
    if let builtin = component as? any _BuiltinComponent {
        return try builtin._normalize(upstream: upstream, ctx: &ctx)
    }

    // 2. Primitive operations (MLP, Attention, RMSNorm, etc.)
    if let primitive = component as? any PrimitiveComponent {
        return try normalizePrimitive(primitive, upstream: upstream, ctx: &ctx)
    }

    // 3. User-defined composite — recurse into body
    return try normalizeComponent(component.body, upstream: upstream, ctx: &ctx)
}

// MARK: - Primitive

private func normalizePrimitive(
    _ primitive: any PrimitiveComponent,
    upstream: [ValueID],
    ctx: inout NormalizationContext
) throws -> NormalizedRegionFragment {
    let kind = primitive.operationKind
    let signature = primitive.operationSignature
    let key = ctx.freshKey()
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
}

// MARK: - Built-in Conformances: TupleComponent

extension TupleComponent: _BuiltinComponent {
    fileprivate func _normalize(
        upstream: [ValueID],
        ctx: inout NormalizationContext
    ) throws -> NormalizedRegionFragment {
        var ops: [Operation] = []
        var current = upstream
        for child in repeat each value {
            let fragment = try normalizeComponent(child, upstream: current, ctx: &ctx)
            ops.append(contentsOf: fragment.operations)
            current = fragment.results
        }
        return NormalizedRegionFragment(operations: ops, results: current)
    }
}

extension TupleComponent: _BranchExtractable {
    fileprivate func _asBranches(
        upstream: [ValueID],
        ctx: inout NormalizationContext
    ) throws -> [Region] {
        var regions: [Region] = []
        for child in repeat each value {
            let params = upstream.map { _ in ctx.freshValue() }
            let fragment = try normalizeComponent(child, upstream: params, ctx: &ctx)
            regions.append(Region(
                parameters: params.map { RegionParameter(id: $0) },
                operations: fragment.operations,
                results: fragment.results.map { ValueUse(value: $0) }
            ))
        }
        return regions
    }
}

// MARK: - Built-in Conformances: ArrayComponent

extension ArrayComponent: _BuiltinComponent {
    fileprivate func _normalize(
        upstream: [ValueID],
        ctx: inout NormalizationContext
    ) throws -> NormalizedRegionFragment {
        var ops: [Operation] = []
        var current = upstream
        for child in children {
            let fragment = try normalizeComponent(child, upstream: current, ctx: &ctx)
            ops.append(contentsOf: fragment.operations)
            current = fragment.results
        }
        return NormalizedRegionFragment(operations: ops, results: current)
    }
}

// MARK: - Built-in Conformances: OptionalComponent

extension OptionalComponent: _BuiltinComponent {
    fileprivate func _normalize(
        upstream: [ValueID],
        ctx: inout NormalizationContext
    ) throws -> NormalizedRegionFragment {
        if let content = content {
            return try normalizeComponent(content, upstream: upstream, ctx: &ctx)
        } else {
            return NormalizedRegionFragment(operations: [], results: upstream)
        }
    }
}

// MARK: - Built-in Conformances: ConditionalComponent

extension ConditionalComponent: _BuiltinComponent {
    fileprivate func _normalize(
        upstream: [ValueID],
        ctx: inout NormalizationContext
    ) throws -> NormalizedRegionFragment {
        switch storage {
        case .first(let first):
            return try normalizeComponent(first, upstream: upstream, ctx: &ctx)
        case .second(let second):
            return try normalizeComponent(second, upstream: upstream, ctx: &ctx)
        }
    }
}

// MARK: - Built-in Conformances: Residual

extension Residual: _BuiltinComponent {
    fileprivate func _normalize(
        upstream: [ValueID],
        ctx: inout NormalizationContext
    ) throws -> NormalizedRegionFragment {
        let bodyParams = upstream.map { _ in ctx.freshValue() }
        let bodyFragment = try normalizeComponent(
            content, upstream: bodyParams, ctx: &ctx
        )
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
    }
}

// MARK: - Built-in Conformances: Parallel

extension Parallel: _BuiltinComponent {
    fileprivate func _normalize(
        upstream: [ValueID],
        ctx: inout NormalizationContext
    ) throws -> NormalizedRegionFragment {
        // Extract branches: TupleComponent → each child is a branch;
        // single component → one branch.
        let branchRegions: [Region]
        if let extractable = content as? any _BranchExtractable {
            branchRegions = try extractable._asBranches(upstream: upstream, ctx: &ctx)
        } else {
            // Single component = single branch
            let params = upstream.map { _ in ctx.freshValue() }
            let fragment = try normalizeComponent(content, upstream: params, ctx: &ctx)
            branchRegions = [Region(
                parameters: params.map { RegionParameter(id: $0) },
                operations: fragment.operations,
                results: fragment.results.map { ValueUse(value: $0) }
            )]
        }

        let key = ctx.freshKey()
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
    }
}

// MARK: - Built-in Conformances: Repeat

extension Repeat: _BuiltinComponent {
    fileprivate func _normalize(
        upstream: [ValueID],
        ctx: inout NormalizationContext
    ) throws -> NormalizedRegionFragment {
        let bodyParams = upstream.map { _ in ctx.freshValue() }
        let bodyFragment = try normalizeComponent(
            content, upstream: bodyParams, ctx: &ctx
        )
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

        if let label = label {
            ctx.keyLabels[key.rawValue] = label
        }

        return NormalizedRegionFragment(operations: [op], results: resultValues)
    }
}

// MARK: - Built-in Conformances: Group

extension Group: _BuiltinComponent {
    fileprivate func _normalize(
        upstream: [ValueID],
        ctx: inout NormalizationContext
    ) throws -> NormalizedRegionFragment {
        let fragment = try normalizeComponent(
            content, upstream: upstream, ctx: &ctx
        )
        if let label = label, let firstOp = fragment.operations.first {
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
