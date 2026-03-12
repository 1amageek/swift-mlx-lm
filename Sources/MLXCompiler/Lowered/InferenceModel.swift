@preconcurrency import MLX
import SwiftLM

// MARK: - Inference State

/// External inference state passed in/out of inference steps.
///
/// Value type — enables functional-style inference and `MLX.compile()` tracing.
/// Each call to `prefill()` or `decode()` returns a new state.
public struct InferenceState: @unchecked Sendable {

    /// Cache state for each layer (indexed by compile-time slot index).
    public var caches: [LoweredCacheState]

    /// Next token position (total tokens processed so far).
    public var nextPosition: Int

    public init(caches: [LoweredCacheState], nextPosition: Int = 0) {
        self.caches = caches
        self.nextPosition = nextPosition
    }
}

// MARK: - Plans

/// Prefill plan — processes the full prompt sequence.
public struct PrefillPlan: @unchecked Sendable {
    public let steps: [LoweredStep]

    public init(steps: [LoweredStep]) {
        self.steps = steps
    }
}

/// Decode plan — processes one token at a time.
///
/// Steps are flattened from recursive `.residual(body:)` into linear
/// `[.saveResidual, ...body..., .addResidual]` to eliminate recursive
/// dispatch overhead for single-token decode.
public struct DecodePlan: @unchecked Sendable {
    public let steps: [FlatStep]

    public init(steps: [LoweredStep]) {
        self.steps = flattenSteps(steps)
    }
}

// MARK: - Compilation Statistics

/// Statistics about projection packing decisions made during compilation.
///
/// Use these counters to verify that the compiler is packing projections
/// as expected. Any non-zero `unpacked` count indicates a fallback to
/// individual matmuls — typically caused by mixed kernel variants or
/// incompatible quantization parameters across projections.
public struct CompilationStats: Sendable, Equatable {

    /// Number of attention layers where QKV packing succeeded.
    public var packedAttentionCount: Int = 0

    /// Number of attention layers where QKV packing failed (fallback).
    public var unpackedAttentionCount: Int = 0

    /// Number of MLP layers where gate+up packing succeeded.
    public var packedMLPCount: Int = 0

    /// Number of MLP layers where gate+up packing failed (fallback).
    public var unpackedMLPCount: Int = 0

    /// Number of MLP layers where gating is disabled (packing not applicable).
    public var ungatedMLPCount: Int = 0

    /// Number of fused sub-layers produced during step flattening.
    public var fusedSubLayerCount: Int = 0

    /// Number of unfused residual blocks (fallback).
    public var unfusedResidualCount: Int = 0

    public init() {}
}

// MARK: - Metadata

/// Metadata about the compiled inference model.
public struct InferenceMetadata: Sendable {

    /// Total number of cache slots (attention + recurrent).
    public let cacheSlotCount: Int

    /// Cache descriptors for each slot.
    public let cacheDescriptors: [CacheDescriptor]

    /// Whether the output head is tied to the embedding.
    public let hasTiedOutputHead: Bool

    /// Statistics about compiler optimizations applied.
    public let compilationStats: CompilationStats

    public init(
        cacheSlotCount: Int,
        cacheDescriptors: [CacheDescriptor],
        hasTiedOutputHead: Bool,
        compilationStats: CompilationStats = CompilationStats()
    ) {
        self.cacheSlotCount = cacheSlotCount
        self.cacheDescriptors = cacheDescriptors
        self.hasTiedOutputHead = hasTiedOutputHead
        self.compilationStats = compilationStats
    }
}

// MARK: - Lowered Inference Model

/// Fully lowered inference model with compile-time kernel selection.
///
/// All repeating blocks are unrolled, each layer has its own lowered ops
/// and compile-time resolved cache slot index. No MLXNN Module hierarchy.
///
/// Usage:
/// ```swift
/// let model: MLXLoweredInferenceModel = try compiler.compile(graph:weights:)
/// var state = model.makeState()
/// let (prefillLogits, state2) = model.prefill(tokenIDs: prompt, state: state)
/// let (decodeLogits, state3) = model.decode(tokenIDs: nextToken, state: state2)
/// ```
public struct MLXLoweredInferenceModel: @unchecked Sendable {

    /// Plan for processing the full prompt.
    public let prefill: PrefillPlan

    /// Plan for single-token decode steps.
    public let decode: DecodePlan

    /// Model metadata (cache count, tied head info).
    public let metadata: InferenceMetadata

    public init(
        prefill: PrefillPlan,
        decode: DecodePlan,
        metadata: InferenceMetadata
    ) {
        self.prefill = prefill
        self.decode = decode
        self.metadata = metadata
    }

    /// Create a fresh inference state with empty caches.
    public func makeState() -> InferenceState {
        var caches = [LoweredCacheState]()
        caches.reserveCapacity(metadata.cacheSlotCount)
        for desc in metadata.cacheDescriptors {
            switch desc.kind {
            case .kv, .rotating, .quantized:
                caches.append(.kv(LoweredKVCache()))
            case .recurrent:
                caches.append(.recurrent(LoweredRecurrentCache()))
            }
        }
        return InferenceState(caches: caches, nextPosition: 0)
    }

    /// Run prefill on the full prompt.
    ///
    /// When `embeddings` is provided, token embedding lookup is skipped and
    /// the pre-computed embeddings are used directly (VLM sequential chunk).
    ///
    /// When `positionIds` is provided (shape `[3, B, S]` for M-RoPE), per-token
    /// position encoding is used instead of the scalar `nextPosition` offset.
    ///
    /// Returns logits for the last token and the updated state.
    public func prefill(
        tokenIDs: MLXArray,
        embeddings: MLXArray? = nil,
        positionIds: MLXArray? = nil,
        state: InferenceState
    ) -> (MLXArray, InferenceState) {
        var mutableState = state
        let options = ExecutionOptions(
            embeddings: embeddings, positionIds: positionIds)
        let logits = executeSteps(
            prefill.steps, input: tokenIDs, state: &mutableState, options: options)
        // Derive sequence length from embeddings when provided (VLM vision chunk)
        let seqLen: Int
        if let embeddings {
            seqLen = embeddings.dim(1) // [B, S, D]
        } else {
            seqLen = tokenIDs.dim(tokenIDs.ndim - 1)
        }
        mutableState.nextPosition += seqLen
        return (logits, mutableState)
    }

    /// Run a single decode step.
    ///
    /// Uses the flattened step execution engine for reduced dispatch overhead.
    /// Returns logits for the next token and the updated state.
    public func decode(
        tokenIDs: MLXArray,
        positionIds: MLXArray? = nil,
        state: InferenceState
    ) -> (MLXArray, InferenceState) {
        var mutableState = state
        let options = ExecutionOptions(positionIds: positionIds)
        let logits = executeFlatSteps(
            decode.steps, input: tokenIDs, state: &mutableState, options: options)
        let seqLen = tokenIDs.dim(tokenIDs.ndim - 1)
        mutableState.nextPosition += seqLen
        return (logits, mutableState)
    }
}

// MARK: - Execution Options

/// Options passed through the execution engine for VLM support.
///
/// - `embeddings`: Pre-computed embeddings that bypass token embedding lookup.
/// - `positionIds`: Per-token M-RoPE positions `[3, B, S]` that override scalar offset.
public struct ExecutionOptions: @unchecked Sendable {
    public let embeddings: MLXArray?
    public let positionIds: MLXArray?

    public init(embeddings: MLXArray? = nil, positionIds: MLXArray? = nil) {
        self.embeddings = embeddings
        self.positionIds = positionIds
    }

    public static let `default` = ExecutionOptions()
}

// MARK: - Execution Engine

/// Execute a sequence of lowered steps.
///
/// This is the core execution loop for the lowered inference model.
/// It walks the flattened step list, dispatching each operation to its
/// `apply()` method with the external cache state.
func executeSteps(
    _ steps: [LoweredStep], input: MLXArray, state: inout InferenceState,
    options: ExecutionOptions = .default
) -> MLXArray {
    var h = input
    for step in steps {
        switch step {
        case .op(let op):
            h = executeOp(op, input: h, state: &state, options: options)

        case .residual(let body):
            h = h + executeSteps(body, input: h, state: &state, options: options)

        case .parallel(let merge, let branches):
            let results = branches.map { branch in
                executeSteps(branch, input: h, state: &state, options: options)
            }
            h = mergeResults(results, strategy: merge)
        }
    }
    return h
}

/// Execute a single lowered inference operation.
func executeOp(
    _ op: LoweredInferenceOp, input: MLXArray, state: inout InferenceState,
    options: ExecutionOptions
) -> MLXArray {
    switch op {
    case .tokenEmbedding(let emb):
        // Skip embedding lookup when pre-computed embeddings are provided
        if let embeddings = options.embeddings { return embeddings }
        return emb.apply(input)
    case .attention(let attn):
        return attn.apply(input, caches: &state.caches, positionIds: options.positionIds)
    case .mlp(let mlp):
        return mlp.apply(input)
    case .moe(let moe):
        return moe.apply(input)
    case .norm(let norm):
        return norm.apply(input)
    case .outputHead(let head):
        return head.apply(input)
    case .deltaNet(let dn):
        return dn.apply(input, caches: &state.caches)
    case .rope(let rope):
        return rope.apply(input, offset: state.nextPosition)
    case .positionalEmbedding(let posEmb):
        return posEmb.apply(input, offset: state.nextPosition)
    case .linear(let proj):
        return proj.apply(input)
    }
}

/// Merge parallel branch results according to the merge strategy.
func mergeResults(
    _ results: [MLXArray], strategy: ParallelMergeStrategy
) -> MLXArray {
    switch strategy {
    case .add:
        return results.dropFirst().reduce(results[0]) { $0 + $1 }
    case .concat:
        return concatenated(results, axis: -1)
    case .stack:
        return stacked(results, axis: 0)
    default:
        // Fallback for unsupported strategies
        return results.dropFirst().reduce(results[0]) { $0 + $1 }
    }
}
