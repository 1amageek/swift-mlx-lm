import MLX
import SwiftLM

/// Compiles a SwiftLM `ModelGraph` + `BoundWeights` into an executable `MLXCompiledModel`.
///
/// The compiler performs:
/// 1. Weight extraction (`BoundWeights` → `InferenceWeightStore`)
/// 2. Cache inference (auto-detect attention/stateSpace → path-keyed `CacheDescriptor` array)
/// 3. Embedding discovery with validation (for tied output head weights)
///
/// This is a **ModelGraph interpreter backend**. The compiler does NOT lower the
/// graph to operator-level IR. The executor interprets the semantic `ModelGraph`
/// directly, preserving MLX fused kernel optimizations
/// (MLXFast.scaledDotProductAttention, MLXFast.rmsNorm, MLXFast.RoPE).
///
/// ```swift
/// let compiler = MLXCompiler()
/// let compiled = try compiler.compile(
///     graph: try model.makeModelGraph(),
///     weights: boundWeights
/// )
/// ```
public struct MLXCompiler: Sendable {

    public init() {}

    /// Compile a model graph with bound weights into an executable model.
    public func compile(graph: ModelGraph, weights: BoundWeights) throws -> MLXCompiledModel {
        let weightStore = try InferenceWeightStore(boundWeights: weights)

        var cacheDescriptors: [CacheDescriptor] = []
        var embeddingPaths: [StructuralPath] = []
        var hasTiedOutputHead = false
        var cacheIndex = 0

        scanRegion(
            graph.rootRegion,
            pathComponents: [],
            cacheDescriptors: &cacheDescriptors,
            cacheIndex: &cacheIndex,
            embeddingPaths: &embeddingPaths,
            hasTiedOutputHead: &hasTiedOutputHead
        )

        // Validate embedding discovery
        let embeddingPath: StructuralPath?
        if embeddingPaths.count > 1 && hasTiedOutputHead {
            throw CompilerError.invalidGraphStructure(
                "Found \(embeddingPaths.count) token embeddings but model has a tied output head. "
                + "Cannot determine which embedding to tie.")
        } else if hasTiedOutputHead && embeddingPaths.isEmpty {
            throw CompilerError.invalidGraphStructure(
                "Model has a tied output head but no token embedding was found.")
        } else {
            embeddingPath = embeddingPaths.first
        }

        return MLXCompiledModel(
            graph: graph,
            weightStore: weightStore,
            cacheDescriptors: cacheDescriptors,
            embeddingPath: embeddingPath
        )
    }

    // MARK: - Graph Scanning

    /// Depth-first scan of the model graph to discover caches, embeddings,
    /// and tied output heads.
    ///
    /// Cache descriptors are keyed by `StructuralPath` so that the executor
    /// can resolve cache slots by path lookup rather than execution-order counters.
    private func scanRegion(
        _ region: Region,
        pathComponents: [StructuralPathComponent],
        cacheDescriptors: inout [CacheDescriptor],
        cacheIndex: inout Int,
        embeddingPaths: inout [StructuralPath],
        hasTiedOutputHead: inout Bool
    ) {
        for (opIndex, op) in region.operations.enumerated() {
            let opPath = pathComponents + [.operation(opIndex)]

            switch op.kind {
            case .tokenEmbedding:
                embeddingPaths.append(StructuralPath(components: opPath))

            case .outputHead(let attrs):
                if attrs.tiedToEmbedding {
                    hasTiedOutputHead = true
                }

            case .attention:
                let path = StructuralPath(components: opPath)
                cacheDescriptors.append(
                    CacheDescriptor(path: path, kind: .kv, slotIndex: cacheIndex))
                cacheIndex += 1

            case .stateSpace:
                let path = StructuralPath(components: opPath)
                cacheDescriptors.append(
                    CacheDescriptor(path: path, kind: .recurrent, slotIndex: cacheIndex))
                cacheIndex += 1

            case .residual(_, let body):
                scanRegion(
                    body,
                    pathComponents: opPath + [.regionBody],
                    cacheDescriptors: &cacheDescriptors,
                    cacheIndex: &cacheIndex,
                    embeddingPaths: &embeddingPaths,
                    hasTiedOutputHead: &hasTiedOutputHead
                )

            case .parallel(_, let branches):
                for (i, branch) in branches.enumerated() {
                    scanRegion(
                        branch,
                        pathComponents: opPath + [.regionBranch(i)],
                        cacheDescriptors: &cacheDescriptors,
                        cacheIndex: &cacheIndex,
                        embeddingPaths: &embeddingPaths,
                        hasTiedOutputHead: &hasTiedOutputHead
                    )
                }

            case .repeating(let count, let body):
                for i in 0..<count {
                    scanRegion(
                        body,
                        pathComponents: opPath + [.regionBody, .index(i)],
                        cacheDescriptors: &cacheDescriptors,
                        cacheIndex: &cacheIndex,
                        embeddingPaths: &embeddingPaths,
                        hasTiedOutputHead: &hasTiedOutputHead
                    )
                }

            case .layerStack(let layers):
                for (i, layer) in layers.enumerated() {
                    scanRegion(
                        layer,
                        pathComponents: opPath + [.regionBody, .index(i)],
                        cacheDescriptors: &cacheDescriptors,
                        cacheIndex: &cacheIndex,
                        embeddingPaths: &embeddingPaths,
                        hasTiedOutputHead: &hasTiedOutputHead
                    )
                }

            default:
                break
            }
        }
    }
}
