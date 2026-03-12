import MLX
import SwiftLM

/// A compiled model bundle ready for execution by `MLXExecutor`.
///
/// This is a **graph-interpreter artifact**: it bundles the semantic graph,
/// resolved weight store, cache descriptors, and path-based cache index
/// for direct interpretation by the executor.
///
/// The executor walks the `ModelGraph` directly at the semantic level
/// (attention, mlp, etc.) rather than lowering to operator-level instructions,
/// which preserves MLX fused kernel optimizations (SDPA, rmsNorm, RoPE).
///
/// Weight storage uses `InferenceWeightStore` which supports both dense
/// (`MLXArray`) and quantized (`AffineQuantizedTensor`) weights. The executor
/// dispatches `quantizedMatmul` or standard `matmul` based on storage type
/// via `LoweredProjection`.
///
/// ```swift
/// let compiler = MLXCompiler()
/// let compiled = try compiler.compile(graph: modelGraph, weights: boundWeights)
/// let executor = MLXExecutor(compiledModel: compiled)
/// let logits = try executor.forward(tokenIDs: tokens)
/// ```
///
/// - Note: Marked `@unchecked Sendable` because `MLXArray` (used in
///   `InferenceWeightStore`) is not declared `Sendable`. This type is only
///   used under executor-controlled synchronization with immutable
///   weight storage after compilation.
public struct MLXCompiledModel: @unchecked Sendable {

    /// The semantic model graph (ground truth for execution).
    public let graph: ModelGraph

    /// Typed weight storage supporting both dense and quantized weights.
    public let weightStore: InferenceWeightStore

    /// Cache requirements for each cacheable layer, in DFS execution order.
    public let cacheDescriptors: [CacheDescriptor]

    /// Path-based cache slot lookup.
    ///
    /// Maps `StructuralPath` of a cacheable operation to its flat slot index.
    /// The executor uses this instead of an execution-order counter,
    /// making cache addressing independent of execution path ordering.
    public let cacheSlotByPath: [StructuralPath: Int]

    /// Path to the token embedding operation (used for tied output head weights).
    public let embeddingPath: StructuralPath?

    public init(
        graph: ModelGraph,
        weightStore: InferenceWeightStore,
        cacheDescriptors: [CacheDescriptor],
        embeddingPath: StructuralPath? = nil
    ) {
        self.graph = graph
        self.weightStore = weightStore
        self.cacheDescriptors = cacheDescriptors
        self.embeddingPath = embeddingPath

        var slotMap: [StructuralPath: Int] = [:]
        slotMap.reserveCapacity(cacheDescriptors.count)
        for desc in cacheDescriptors {
            slotMap[desc.path] = desc.slotIndex
        }
        self.cacheSlotByPath = slotMap
    }
}
