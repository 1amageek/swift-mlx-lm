import MLX
import MLXFast
import MLXNN
import MLXCompiler
import SwiftLM

// MARK: - CompiledKVCache

/// Adapts `InferenceState` (value-type) to `KVCache` (reference-type protocol).
///
/// A single `CompiledKVCache` instance wraps the entire `InferenceState` for
/// the compiled model. All cache slots are managed internally — the `KVCache`
/// protocol interface provides just enough surface for `TokenIterator` to work.
///
/// Design: `LanguageModel.newCache()` returns `[CompiledKVCache]` (single element).
/// `callAsFunction` reads/writes through this shared reference.
final class CompiledKVCache: KVCache, @unchecked Sendable {

    /// The mutable inference state (contains all layer caches + position).
    var inferenceState: InferenceState

    init(inferenceState: InferenceState) {
        self.inferenceState = inferenceState
    }

    // MARK: - KVCache Protocol

    var offset: Int { inferenceState.nextPosition }

    var maxSize: Int? { nil }

    var isTrimmable: Bool { false }

    func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        fatalError("CompiledKVCache does not support direct KV updates")
    }

    @discardableResult
    func trim(_ n: Int) -> Int {
        0
    }

    var state: [MLXArray] {
        get {
            var arrays: [MLXArray] = []
            for cache in inferenceState.caches {
                switch cache {
                case .kv(let kv):
                    // Store keys and values (or empty placeholder if nil)
                    arrays.append(kv.keys ?? MLXArray())
                    arrays.append(kv.values ?? MLXArray())
                case .recurrent(let rc):
                    arrays.append(rc.convState ?? MLXArray())
                    arrays.append(rc.recurrentState ?? MLXArray())
                case .empty:
                    break
                }
            }
            return arrays
        }
        set {
            var idx = 0
            for i in 0..<inferenceState.caches.count {
                switch inferenceState.caches[i] {
                case .kv(var kv):
                    if idx + 1 < newValue.count {
                        kv.keys = newValue[idx].size > 0 ? newValue[idx] : nil
                        kv.values = newValue[idx + 1].size > 0 ? newValue[idx + 1] : nil
                        idx += 2
                    }
                    inferenceState.caches[i] = .kv(kv)
                case .recurrent(var rc):
                    if idx + 1 < newValue.count {
                        rc.convState = newValue[idx].size > 0 ? newValue[idx] : nil
                        rc.recurrentState = newValue[idx + 1].size > 0 ? newValue[idx + 1] : nil
                        idx += 2
                    }
                    inferenceState.caches[i] = .recurrent(rc)
                case .empty:
                    break
                }
            }
        }
    }

    var metaState: [String] {
        get {
            // Format: [nextPosition, cacheCount, type0, offset0, type1, offset1, ...]
            var meta: [String] = [
                String(inferenceState.nextPosition),
                String(inferenceState.caches.count),
            ]
            for cache in inferenceState.caches {
                switch cache {
                case .kv(let kv):
                    meta.append("kv")
                    meta.append(String(kv.offset))
                    meta.append(String(kv.step))
                case .recurrent(let rc):
                    meta.append("recurrent")
                    meta.append(String(rc.offset))
                    meta.append("0") // padding for uniform format
                case .empty:
                    meta.append("empty")
                    meta.append("0")
                    meta.append("0")
                }
            }
            return meta
        }
        set {
            guard newValue.count >= 2,
                  let nextPos = Int(newValue[0]),
                  let cacheCount = Int(newValue[1])
            else { return }

            inferenceState.nextPosition = nextPos

            // If caches already exist (e.g., state was set first), only update
            // offsets without rebuilding. This preserves KV arrays set via `state`.
            if inferenceState.caches.count == cacheCount {
                var idx = 2
                for i in 0..<cacheCount {
                    guard idx < newValue.count else { break }
                    let offset = idx + 1 < newValue.count ? (Int(newValue[idx + 1]) ?? 0) : 0
                    idx += 3

                    switch inferenceState.caches[i] {
                    case .kv(var kv):
                        kv.offset = offset
                        inferenceState.caches[i] = .kv(kv)
                    case .recurrent(var rc):
                        rc.offset = offset
                        inferenceState.caches[i] = .recurrent(rc)
                    case .empty:
                        break
                    }
                }
                return
            }

            // No existing caches — build from scratch (used by instantiateCache)
            var caches: [LoweredCacheState] = []
            var idx = 2
            for _ in 0..<cacheCount {
                guard idx < newValue.count else { break }
                let type = newValue[idx]
                let offset = idx + 1 < newValue.count ? (Int(newValue[idx + 1]) ?? 0) : 0
                let step = idx + 2 < newValue.count ? (Int(newValue[idx + 2]) ?? 256) : 256
                idx += 3

                switch type {
                case "kv":
                    var kv = LoweredKVCache(step: step)
                    kv.offset = offset
                    caches.append(.kv(kv))
                case "recurrent":
                    caches.append(.recurrent(LoweredRecurrentCache(offset: offset)))
                case "empty":
                    caches.append(.empty)
                default:
                    caches.append(.empty)
                }
            }
            inferenceState.caches = caches
        }
    }

    func innerState() -> [MLXArray] {
        var arrays: [MLXArray] = []
        for cache in inferenceState.caches {
            switch cache {
            case .kv(let kv):
                if let k = kv.keys { arrays.append(k) }
                if let v = kv.values { arrays.append(v) }
            case .recurrent(let rc):
                if let cs = rc.convState { arrays.append(cs) }
                if let rs = rc.recurrentState { arrays.append(rs) }
            case .empty:
                break
            }
        }
        return arrays
    }

    func makeMask(queryLength n: Int) -> MLXFast.ScaledDotProductAttentionMaskMode {
        n > 1 ? .causal : .none
    }
}

// MARK: - CompiledLanguageModel

/// Adapts `MLXLoweredInferenceModel` to the `LanguageModel` protocol.
///
/// This wrapper enables compiled models to be used through the standard
/// `ModelContext` → `TokenIterator` → `generate()` pipeline without changes
/// to the generation infrastructure.
///
/// The compiled model uses `InferenceState` (value-type) internally, which is
/// wrapped in a `CompiledKVCache` (reference-type) for `KVCache` protocol
/// compatibility. On each forward pass, the state is read from and written
/// back to the shared `CompiledKVCache` reference.
class CompiledLanguageModel: Module, LanguageModel, @unchecked Sendable {

    let lowered: MLXLoweredInferenceModel

    init(lowered: MLXLoweredInferenceModel) {
        self.lowered = lowered
        super.init()
    }

    // MARK: - LanguageModel Protocol

    func callAsFunction(
        _ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?
    ) -> LMOutput {
        guard let compiledCache = cache?.first as? CompiledKVCache else {
            let freshState = lowered.makeState()
            let (logits, _) = lowered.prefill(
                tokenIDs: input.tokens, state: freshState)
            return LMOutput(logits: logits)
        }

        let tokenCount = input.tokens.dim(input.tokens.ndim - 1)

        if tokenCount > 1 {
            let (logits, newState) = lowered.prefill(
                tokenIDs: input.tokens, state: compiledCache.inferenceState)
            compiledCache.inferenceState = newState
            return LMOutput(logits: logits)
        } else {
            let (logits, newState) = lowered.decode(
                tokenIDs: input.tokens, state: compiledCache.inferenceState)
            compiledCache.inferenceState = newState
            return LMOutput(logits: logits)
        }
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let input = LMInput.Text(tokens: inputs)
        return callAsFunction(input, cache: cache, state: nil).logits
    }

    func newCache(parameters: GenerateParameters?) -> [KVCache] {
        let state = lowered.makeState()
        return [CompiledKVCache(inferenceState: state)]
    }

    func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        weights
    }

    var layerCount: Int {
        lowered.metadata.cacheSlotCount
    }

    var kvHeads: [Int] {
        Array(repeating: 1, count: lowered.metadata.cacheSlotCount)
    }

    func prepare(
        _ input: LMInput, cache: [KVCache], windowSize: Int?
    ) throws -> PrepareResult {
        let tokens = input.text.tokens
        let tokenCount = tokens.dim(tokens.ndim - 1)

        guard let compiledCache = cache.first as? CompiledKVCache else {
            let output = callAsFunction(input.text, cache: cache, state: nil)
            return .logits(output)
        }

        let prefillOffset = compiledCache.offset
        let windowSize = windowSize ?? tokenCount

        if prefillOffset + windowSize < tokenCount {
            let chunk = tokens[0..., prefillOffset..<(prefillOffset + windowSize)]
            let chunkInput = LMInput.Text(tokens: chunk)
            let output = callAsFunction(chunkInput, cache: cache, state: nil)
            eval(output.logits)
            return .tokens(LMInput.Text(tokens: tokens))
        }

        let remaining = tokens[0..., prefillOffset...]
        let output = callAsFunction(
            LMInput.Text(tokens: remaining),
            cache: cache,
            state: nil
        )
        return .logits(output)
    }
}
