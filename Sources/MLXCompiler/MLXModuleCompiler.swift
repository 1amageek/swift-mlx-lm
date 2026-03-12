@preconcurrency import MLX
import MLXNN
import SwiftLM

/// Compiles a SwiftLM `ModelGraph` + `BoundWeights` into a `GraphModel` (MLXNN Module tree).
///
/// Unlike `MLXCompiler` which produces an `MLXCompiledModel` for the tree-walking
/// `MLXExecutor`, `MLXModuleCompiler` produces a proper MLXNN `Module` tree where
/// every linear projection is a `Linear` layer annotated with `@ModuleInfo`.
///
/// This enables:
/// - **Quantization**: `MLXNN.quantize(model:)` swaps `Linear` → `QuantizedLinear`,
///   using `quantizedMM` instead of `matmul` for all projections.
/// - **LoRA**: Module tree is compatible with MLXNN's LoRA adapter system.
/// - **MLX.compile()**: The forward pass can be wrapped in `MLX.compile()` for
///   whole-model graph optimization.
///
/// Usage:
/// ```swift
/// let compiler = MLXModuleCompiler()
/// let model = try compiler.compile(
///     graph: try modelComponent.makeModelGraph(),
///     weights: boundWeights
/// )
///
/// // Quantize all Linear layers to 4-bit
/// quantize(model: model, groupSize: 64, bits: 4)
///
/// // Forward pass uses quantizedMM automatically
/// let logits = try model.forward(tokenIDs: tokens)
/// ```
public struct MLXModuleCompiler: Sendable {

    public init() {}

    /// Compile a model graph with bound weights into an MLXNN Module tree.
    public func compile(graph: ModelGraph, weights: BoundWeights) throws -> GraphModel {
        let store = try MLXWeightStore(boundWeights: weights)

        // Track embedding for tied output head resolution
        var embeddingModule: GraphEmbedding?
        var attentionModules: [GraphAttention] = []
        var deltaNetModules: [GraphDeltaNet] = []

        let modules = try compileRegion(
            graph.rootRegion,
            store: store,
            pathComponents: [],
            embeddingModule: &embeddingModule,
            attentionModules: &attentionModules,
            deltaNetModules: &deltaNetModules
        )

        let root = GraphSequential(modules)

        return GraphModel(
            root: root,
            attentionModules: attentionModules,
            deltaNetModules: deltaNetModules
        )
    }

    // MARK: - Region Compilation

    private func compileRegion(
        _ region: Region,
        store: MLXWeightStore,
        pathComponents: [StructuralPathComponent],
        embeddingModule: inout GraphEmbedding?,
        attentionModules: inout [GraphAttention],
        deltaNetModules: inout [GraphDeltaNet]
    ) throws -> [Module] {
        var modules: [Module] = []

        for (i, op) in region.operations.enumerated() {
            let opPath = pathComponents + [.operation(i)]
            let path = StructuralPath(components: opPath)

            let module = try compileOperation(
                op, store: store, pathComponents: opPath, path: path,
                embeddingModule: &embeddingModule,
                attentionModules: &attentionModules,
                deltaNetModules: &deltaNetModules
            )
            modules.append(module)
        }

        return modules
    }

    // MARK: - Operation Compilation

    private func compileOperation(
        _ op: Operation,
        store: MLXWeightStore,
        pathComponents: [StructuralPathComponent],
        path: StructuralPath,
        embeddingModule: inout GraphEmbedding?,
        attentionModules: inout [GraphAttention],
        deltaNetModules: inout [GraphDeltaNet]
    ) throws -> Module {
        switch op.kind {

        // MARK: Primitive Operations

        case .tokenEmbedding(let attrs):
            let emb = try GraphEmbedding(attrs: attrs, store: store, path: path)
            embeddingModule = emb
            return emb

        case .attention(let attrs):
            let cache = MLXKVCacheSimple()
            let attn = try GraphAttention(attrs: attrs, store: store, path: path, cache: cache)
            attentionModules.append(attn)
            return attn

        case .mlp(let attrs):
            return try GraphMLP(attrs: attrs, store: store, path: path)

        case .moe(let attrs):
            return try GraphMoE(attrs: attrs, store: store, path: path)

        case .rmsNorm(let attrs):
            return try GraphRMSNorm(attrs: attrs, store: store, path: path)

        case .layerNorm(let attrs):
            return try GraphLayerNorm(attrs: attrs, store: store, path: path)

        case .linear(let attrs):
            return try GraphLinear(attrs: attrs, store: store, path: path)

        case .outputHead(let attrs):
            if attrs.tiedToEmbedding {
                guard let emb = embeddingModule else {
                    throw CompilerError.invalidGraphStructure(
                        "Tied output head found but no token embedding was compiled.")
                }
                return GraphOutputHead(tiedTo: emb)
            }
            return try GraphOutputHead(attrs: attrs, store: store, path: path)

        case .stateSpace(let attrs):
            // Validate variant
            let variant = attrs.variant.lowercased()
            guard variant == "deltanet" || variant == "gated_deltanet"
                || variant == "gated-deltanet" || variant == "short_conv"
            else {
                throw CompilerError.unsupportedVariant(attrs.variant)
            }

            if variant == "short_conv" {
                // Short conv is compiled as a simplified DeltaNet with only conv
                // For now, use the full DeltaNet module which handles both
                let cache = MLXRecurrentCache()
                let dn = try GraphDeltaNet(attrs: attrs, store: store, path: path, cache: cache)
                deltaNetModules.append(dn)
                return dn
            }

            let cache = MLXRecurrentCache()
            let dn = try GraphDeltaNet(attrs: attrs, store: store, path: path, cache: cache)
            deltaNetModules.append(dn)
            return dn

        case .rope(let attrs):
            return GraphRoPE(attrs: attrs)

        case .positionalEmbedding(let attrs):
            return try GraphPositionalEmbedding(attrs: attrs, store: store, path: path)

        // MARK: Structural Operations

        case .residual(let strategy, let body):
            let bodyPath = pathComponents + [.regionBody]
            let bodyModules = try compileRegion(
                body, store: store, pathComponents: bodyPath,
                embeddingModule: &embeddingModule,
                attentionModules: &attentionModules,
                deltaNetModules: &deltaNetModules
            )
            return GraphResidual(body: GraphSequential(bodyModules), strategy: strategy)

        case .repeating(let count, let body):
            var iterations: [Module] = []
            iterations.reserveCapacity(count)
            for i in 0..<count {
                let iterPath = pathComponents + [.regionBody, .index(i)]
                let iterModules = try compileRegion(
                    body, store: store, pathComponents: iterPath,
                    embeddingModule: &embeddingModule,
                    attentionModules: &attentionModules,
                    deltaNetModules: &deltaNetModules
                )
                iterations.append(GraphSequential(iterModules))
            }
            return GraphRepeating(iterations: iterations)

        case .layerStack(let layers):
            var iterations: [Module] = []
            iterations.reserveCapacity(layers.count)
            for (i, layer) in layers.enumerated() {
                let iterPath = pathComponents + [.regionBody, .index(i)]
                let iterModules = try compileRegion(
                    layer, store: store, pathComponents: iterPath,
                    embeddingModule: &embeddingModule,
                    attentionModules: &attentionModules,
                    deltaNetModules: &deltaNetModules
                )
                iterations.append(GraphSequential(iterModules))
            }
            return GraphRepeating(iterations: iterations)

        case .parallel(let merge, let branches):
            var branchModules: [Module] = []
            branchModules.reserveCapacity(branches.count)
            for (i, branch) in branches.enumerated() {
                let branchPath = pathComponents + [.regionBranch(i)]
                let modules = try compileRegion(
                    branch, store: store, pathComponents: branchPath,
                    embeddingModule: &embeddingModule,
                    attentionModules: &attentionModules,
                    deltaNetModules: &deltaNetModules
                )
                branchModules.append(GraphSequential(modules))
            }
            return GraphParallel(branches: branchModules, merge: merge)

        // MARK: Unsupported

        case .custom(let attrs):
            throw CompilerError.unsupportedOperation("custom(\(attrs.domain).\(attrs.name))")
        }
    }
}
