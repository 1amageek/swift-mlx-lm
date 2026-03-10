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

// MARK: - Metadata

/// Metadata about the compiled inference model.
public struct InferenceMetadata: Sendable {

    /// Total number of cache slots (attention + recurrent).
    public let cacheSlotCount: Int

    /// Cache descriptors for each slot.
    public let cacheDescriptors: [CacheDescriptor]

    /// Whether the output head is tied to the embedding.
    public let hasTiedOutputHead: Bool

    public init(
        cacheSlotCount: Int,
        cacheDescriptors: [CacheDescriptor],
        hasTiedOutputHead: Bool
    ) {
        self.cacheSlotCount = cacheSlotCount
        self.cacheDescriptors = cacheDescriptors
        self.hasTiedOutputHead = hasTiedOutputHead
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
    /// Returns logits for the last token and the updated state.
    public func prefill(
        tokenIDs: MLXArray, state: InferenceState
    ) -> (MLXArray, InferenceState) {
        var mutableState = state
        let logits = executeSteps(prefill.steps, input: tokenIDs, state: &mutableState)
        let seqLen = tokenIDs.dim(tokenIDs.ndim - 1)
        mutableState.nextPosition += seqLen
        return (logits, mutableState)
    }

    /// Run a single decode step.
    ///
    /// Uses the flattened step execution engine for reduced dispatch overhead.
    /// Returns logits for the next token and the updated state.
    public func decode(
        tokenIDs: MLXArray, state: InferenceState
    ) -> (MLXArray, InferenceState) {
        var mutableState = state
        let logits = executeFlatSteps(decode.steps, input: tokenIDs, state: &mutableState)
        let seqLen = tokenIDs.dim(tokenIDs.ndim - 1)
        mutableState.nextPosition += seqLen
        return (logits, mutableState)
    }
}

// MARK: - Execution Engine

/// Execute a sequence of lowered steps.
///
/// This is the core execution loop for the lowered inference model.
/// It walks the flattened step list, dispatching each operation to its
/// `apply()` method with the external cache state.
func executeSteps(
    _ steps: [LoweredStep], input: MLXArray, state: inout InferenceState
) -> MLXArray {
    var h = input
    for step in steps {
        switch step {
        case .op(let op):
            h = executeOp(op, input: h, state: &state)

        case .residual(let body):
            h = h + executeSteps(body, input: h, state: &state)

        case .parallel(let merge, let branches):
            let results = branches.map { branch in
                executeSteps(branch, input: h, state: &state)
            }
            h = mergeResults(results, strategy: merge)
        }
    }
    return h
}

/// Execute a single lowered inference operation.
func executeOp(
    _ op: LoweredInferenceOp, input: MLXArray, state: inout InferenceState
) -> MLXArray {
    switch op {
    case .tokenEmbedding(let emb):
        return emb.apply(input)
    case .attention(let attn):
        return attn.apply(input, caches: &state.caches)
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
