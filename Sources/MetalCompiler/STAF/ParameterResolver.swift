import LMIR

/// Resolves parameter bindings for a ModelGraph by walking the IR
/// and delegating tensor-name construction to a `WeightNamingConvention`.
///
/// The resolver walks the graph, tracks layer indices from repeating blocks,
/// and invokes the convention for each primitive operation. It owns no
/// family-specific knowledge; that lives with the model declarations.
///
/// ## Module Layout
///
/// - `LMIR` defines the `WeightNamingConvention` protocol and
///   `WeightNamingScope` enum (backend-independent IR concern).
/// - `MetalCompiler` (this module) provides the IR-walking driver
///   `ParameterResolver` but ships no built-in conventions. Family
///   naming is not a compiler concern.
/// - `ModelDeclarations` (`Sources/Models/<Family>/`) provides the
///   concrete conformances — `LlamaFamilyNaming`, `Gemma4FamilyNaming`,
///   `LFM2FamilyNaming`, etc. — alongside the corresponding
///   `ModelComponent` declarations. Building a graph from any of these
///   declarations already requires `ModelDeclarations`, so needing
///   the matching convention from the same module is not an extra
///   dependency.
///
/// ```swift
/// import MetalCompiler
/// import ModelDeclarations        // source of Transformer + LlamaFamilyNaming
///
/// let graph = try ModelGraph(Transformer(config: config))
/// let resolved = ParameterResolver().resolve(
///     graph: graph,
///     convention: LlamaFamilyNaming()
/// )
/// ```
///
/// User-defined models provide their own `WeightNamingConvention`
/// conformance; no change to `MetalCompiler` is required.
public struct ParameterResolver: Sendable {

    public init() {}

    /// Resolve all parameter bindings in a ModelGraph using the given convention.
    ///
    /// Returns a new ModelGraph with `parameterBindings` populated on every
    /// primitive operation that has weight requirements.
    public func resolve(
        graph: ModelGraph,
        convention: any WeightNamingConvention
    ) -> ModelGraph {
        let resolvedRegion = resolveRegion(
            graph.rootRegion,
            convention: convention,
            scope: .root,
            residualIndex: 0
        )
        return ModelGraph(rootRegion: resolvedRegion)
    }

    // MARK: - Region Walk

    private func resolveRegion(
        _ region: Region,
        convention: any WeightNamingConvention,
        scope: WeightNamingScope,
        residualIndex: Int
    ) -> Region {
        var operations: [Operation] = []
        var currentResidualIndex = residualIndex
        // Track layer index for flat-expanded LayerStack.
        // Each decoder layer = 2 residual blocks (norm+op, norm+mlp).
        // Count residual blocks in root scope and divide by 2 to get layer index.
        var residualCount = 0

        // Track norm position within the current scope for sandwich norm disambiguation.
        // Resets per-region so each Residual body starts from normIndex 0.
        var normCounter = 0

        for operation in region.operations {
            var effectiveScope = scope
            // In root scope, assign layer index from residual block count.
            // Each pair of residuals = one decoder layer.
            if case .root = scope, case .residual = operation.kind {
                effectiveScope = .layer(index: residualCount / 2)
                residualCount += 1
            }

            let resolved = resolveOperation(
                operation,
                convention: convention,
                scope: effectiveScope,
                residualIndex: &currentResidualIndex,
                normIndex: normCounter
            )
            operations.append(resolved)

            // Increment norm counter for each norm primitive encountered in this region.
            if case .primitive(let attrs) = operation.kind,
               attrs is RMSNormAttributes || attrs is LayerNormAttributes {
                normCounter += 1
            }
        }

        return Region(
            parameters: region.parameters,
            operations: operations,
            results: region.results
        )
    }

    private func resolveOperation(
        _ operation: Operation,
        convention: any WeightNamingConvention,
        scope: WeightNamingScope,
        residualIndex: inout Int,
        normIndex: Int = 0
    ) -> Operation {
        switch operation.kind {

        case .primitive(let attributes):
            let bindings = convention.bindings(
                for: attributes,
                scope: scope,
                residualIndex: residualIndex,
                normIndex: normIndex
            )
            return Operation(
                key: operation.key,
                kind: operation.kind,
                operands: operation.operands,
                results: operation.results,
                parameterBindings: bindings
            )

        case .residual(let strategy, let body):
            let savedIndex = residualIndex
            // Each layer has 2 residual blocks: index 0 = operator/attention norm,
            // index 1 = ffn norm. Use modulo to get the intra-layer position.
            let resolvedBody = resolveRegion(
                body, convention: convention,
                scope: scope, residualIndex: savedIndex % 2)
            residualIndex = savedIndex + 1
            return Operation(
                key: operation.key,
                kind: .residual(strategy: strategy, body: resolvedBody),
                operands: operation.operands,
                results: operation.results,
                parameterBindings: []
            )

        case .repeating(let count, let body):
            // Use layer 0 as template. The compiler substitutes
            // .layers.0. → .layers.{iteration}. during unroll.
            let templateBody = resolveRegion(
                body, convention: convention,
                scope: .layer(index: 0),
                residualIndex: 0)
            return Operation(
                key: operation.key,
                kind: .repeating(count: count, body: templateBody),
                operands: operation.operands,
                results: operation.results,
                parameterBindings: []
            )

        case .conditional(let condition, let thenBody, let elseBody):
            let resolvedThen = resolveRegion(
                thenBody, convention: convention,
                scope: scope, residualIndex: 0)
            let resolvedElse = resolveRegion(
                elseBody, convention: convention,
                scope: scope, residualIndex: 0)
            return Operation(
                key: operation.key,
                kind: .conditional(condition: condition, then: resolvedThen, else: resolvedElse),
                operands: operation.operands,
                results: operation.results,
                parameterBindings: []
            )

        case .parallel(let merge, let branches):
            let resolvedBranches = branches.map {
                resolveRegion($0, convention: convention, scope: scope, residualIndex: 0)
            }
            return Operation(
                key: operation.key,
                kind: .parallel(merge: merge, branches: resolvedBranches),
                operands: operation.operands,
                results: operation.results,
                parameterBindings: []
            )
        }
    }
}
