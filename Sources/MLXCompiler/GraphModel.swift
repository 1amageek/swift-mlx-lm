@preconcurrency import MLX
import MLXNN
import SwiftLM

/// MLXNN Module tree compiled from a `ModelGraph`.
///
/// `GraphModel` is the entry point for the Module Compiler path.
/// Unlike `MLXExecutor` (tree-walking interpreter using `MLX.matmul`),
/// `GraphModel` is an MLXNN `Module` tree where all linear projections
/// are `Linear` layers that can be quantized to `QuantizedLinear` via
/// `MLXNN.quantize()`, enabling `quantizedMM` for inference.
///
/// Usage:
/// ```swift
/// let compiler = MLXModuleCompiler()
/// let model = try compiler.compile(graph: modelGraph, weights: boundWeights)
///
/// // Optional: quantize for faster inference
/// quantize(model: model, groupSize: 64, bits: 4)
///
/// let logits = try model.forward(tokenIDs: tokens)
/// ```
///
/// - Note: Marked `@unchecked Sendable` because MLXNN `Module` is not
///   `Sendable`. `GraphModel` is designed for serial forward calls;
///   concurrent access must be synchronized externally.
public final class GraphModel: Module, @unchecked Sendable {

    @ModuleInfo var root: GraphSequential

    /// All attention modules for cache reset.
    private var attentionModules: [GraphAttention]

    /// All state-space modules for cache reset.
    private var deltaNetModules: [GraphDeltaNet]

    init(
        root: GraphSequential,
        attentionModules: [GraphAttention],
        deltaNetModules: [GraphDeltaNet]
    ) {
        self.attentionModules = attentionModules
        self.deltaNetModules = deltaNetModules
        self._root.wrappedValue = root
    }

    // MARK: - Forward API

    /// Execute a forward pass, returning logits.
    ///
    /// KV caches persist across calls for autoregressive generation.
    /// Call `resetCaches()` to clear state between independent sequences.
    ///
    /// Accepts 1D `[L]` or 2D `[B, L]` token IDs.
    public func forward(tokenIDs: MLXArray) throws -> MLXArray {
        let needsSqueeze = tokenIDs.ndim == 1
        let input = needsSqueeze ? tokenIDs.expandedDimensions(axis: 0) : tokenIDs

        let logits = root(input)

        if needsSqueeze && logits.dim(0) == 1 {
            return logits.squeezed(axis: 0)
        }
        return logits
    }

    /// Reset all caches (call between independent sequences).
    public func resetCaches() {
        for attn in attentionModules {
            attn.kvCache = MLXKVCacheSimple()
        }
        for ssm in deltaNetModules {
            ssm.cache = MLXRecurrentCache()
        }
    }
}

// MARK: - Executor Protocol Conformance

extension GraphModel: Executor {

    public func run(
        _ model: CompiledModel,
        inputs: ModelInputs
    ) async throws -> ModelOutputs {
        guard let tokenArray = inputs.tokenIDs.storage as? MLXArray else {
            throw CompilerError.invalidWeightStorage(
                ParameterSlot(path: StructuralPath(), role: .custom("tokenIDs")),
                "Expected MLXArray for tokenIDs, got \(type(of: inputs.tokenIDs.storage))")
        }

        let logits = try forward(tokenIDs: tokenArray)

        let cachedLength = attentionModules.first?.kvCache.offset
            ?? deltaNetModules.first.map { $0.cache.offset }
            ?? 0
        let cacheState = KVCacheState(storage: cachedLength as Int, cachedLength: cachedLength)

        return ModelOutputs(
            logits: TensorData(
                shape: logits.shape.map { $0 },
                dtype: .float32,
                storage: logits),
            cache: cacheState
        )
    }
}
