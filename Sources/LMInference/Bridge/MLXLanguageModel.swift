@preconcurrency import MLX
import MLXFast
import MLXNN
import LMCompiler
import SwiftLM

// MARK: - MLXInferenceKVCache

/// Adapts `InferenceState` (value-type) to `KVCache` (reference-type protocol).
///
/// A single `MLXInferenceKVCache` instance wraps the entire `InferenceState` for
/// the compiled model. All cache slots are managed internally — the `KVCache`
/// protocol interface provides just enough surface for `TokenIterator` to work.
///
/// Design: `LanguageModel.newCache()` returns `[MLXInferenceKVCache]` (single element).
/// `callAsFunction` reads/writes through this shared reference.
final class MLXInferenceKVCache: KVCache, Updatable, @unchecked Sendable {

    /// The mutable inference state (contains all layer caches + position).
    var inferenceState: InferenceState

    init(inferenceState: InferenceState) {
        self.inferenceState = inferenceState
    }

    // MARK: - KVCache Protocol

    var offset: Int { inferenceState.nextPosition }

    var maxSize: Int? { nil }

    var isTrimmable: Bool {
        // Trimmable only if all internal cache slots are KV (no recurrent/DeltaNet).
        // For hybrid models like Qwen3.5, this is false — which is correct because
        // DeltaNet recurrent state cannot discard individual token contributions.
        // PrefixCachePool handles this by reusing without trimming (trimCount == 0).
        inferenceState.caches.allSatisfy { cache in
            switch cache {
            case .kv: return true
            case .recurrent, .empty: return false
            }
        }
    }

    func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        fatalError("MLXInferenceKVCache does not support direct KV updates")
    }

    @discardableResult
    func trim(_ n: Int) -> Int {
        guard isTrimmable, n > 0 else { return 0 }

        var minTrimmed = n
        for i in 0..<inferenceState.caches.count {
            switch inferenceState.caches[i] {
            case .kv(var kv):
                let trimmed = min(n, kv.offset)
                kv.offset -= trimmed
                inferenceState.caches[i] = .kv(kv)
                minTrimmed = min(minTrimmed, trimmed)
            default:
                return 0
            }
        }
        inferenceState.nextPosition -= minTrimmed
        return minTrimmed
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

// MARK: - VLM Configuration

/// Vision-language model configuration for MLXLanguageModel.
///
/// When present, enables sequential chunk processing: text chunks use
/// token embedding lookup, vision chunks use pre-computed embeddings
/// from the external vision encoder. M-RoPE position IDs are computed
/// for the full sequence and sliced per chunk.
struct VLMConfig: @unchecked Sendable {

    /// External vision encoder: pixels + grid THW → embeddings.
    let visionEncoder: @Sendable (MLXArray, [(t: Int, h: Int, w: Int)]) -> MLXArray

    /// Token ID for `<|image_pad|>` placeholder tokens.
    let imageTokenId: Int

    /// Token ID for `<|video_pad|>` placeholder tokens (nil if unsupported).
    let videoTokenId: Int?

    /// Spatial merge factor for vision patches (typically 2).
    let spatialMergeSize: Int
}

// MARK: - MLXLanguageModel

/// Adapts `MLXInferenceModel` to the `LanguageModel` protocol.
///
/// Handles both text-only and VLM (vision-language) models through the
/// same lowered inference path. For VLM:
/// - Vision encoder runs separately (not in the IR graph)
/// - Text decoder IR is identical to text-only (with M-RoPE axes configured)
/// - Sequential chunk processing: text → vision → text, each updating KV cache
///
/// The compiled model uses `InferenceState` (value-type) internally, which is
/// wrapped in a `MLXInferenceKVCache` (reference-type) for `KVCache` protocol
/// compatibility. On each forward pass, the state is read from and written
/// back to the shared `MLXInferenceKVCache` reference.
class MLXLanguageModel: Module, LanguageModel, @unchecked Sendable {

    let lowered: MLXInferenceModel

    /// Optional VLM configuration. When set, enables vision-language processing.
    let vlmConfig: VLMConfig?

    /// Tracks the M-RoPE text position for VLM decode steps.
    /// Only used when `vlmConfig` is non-nil.
    var mropeNextPosition: Int = 0

    init(lowered: MLXInferenceModel, vlmConfig: VLMConfig? = nil) {
        self.lowered = lowered
        self.vlmConfig = vlmConfig
        super.init()
    }

    /// Whether this model has VLM capabilities.
    var isVLM: Bool { vlmConfig != nil }

    // MARK: - LanguageModel Protocol

    func callAsFunction(
        _ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?
    ) -> LMOutput {
        guard let compiledCache = cache?.first as? MLXInferenceKVCache else {
            let freshState = lowered.makeState()
            let (logits, _) = lowered.prefill(
                tokenIDs: input.tokens,
                embeddings: input.embeddings,
                positionIds: input.positionIds,
                state: freshState)
            return LMOutput(logits: logits)
        }

        let tokenCount: Int
        if let embeddings = input.embeddings {
            tokenCount = embeddings.dim(1)
        } else {
            tokenCount = input.tokens.dim(input.tokens.ndim - 1)
        }

        if tokenCount > 1 {
            // Prefill — processes a prompt chunk (text or vision embeddings)
            let (logits, newState) = lowered.prefill(
                tokenIDs: input.tokens,
                embeddings: input.embeddings,
                positionIds: input.positionIds,
                state: compiledCache.inferenceState)
            compiledCache.inferenceState = newState
            return LMOutput(logits: logits)
        } else {
            // Decode — single token via optimized lowered path
            //
            // For VLM: generate sequential M-RoPE position IDs if not provided.
            let positionIds: MLXArray?
            if vlmConfig != nil {
                positionIds = input.positionIds ?? makeSequentialPositionIds(
                    batchSize: input.tokens.dim(0), seqLen: 1,
                    startPosition: mropeNextPosition)
                mropeNextPosition += 1
            } else {
                positionIds = input.positionIds
            }

            let (logits, newState) = lowered.decode(
                tokenIDs: input.tokens,
                positionIds: positionIds,
                state: compiledCache.inferenceState)
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
        mropeNextPosition = 0
        return [MLXInferenceKVCache(inferenceState: state)]
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

    var recommendedPrefillStepSize: Int? {
        let hasRecurrent = lowered.metadata.cacheDescriptors.contains { $0.kind == .recurrent }
        return hasRecurrent ? 512 : nil
    }

    func prepare(
        _ input: LMInput, cache: [KVCache], windowSize: Int?
    ) throws -> PrepareResult {
        let tokens = input.text.tokens
        let tokenCount = tokens.dim(tokens.ndim - 1)

        guard let compiledCache = cache.first as? MLXInferenceKVCache else {
            let output = callAsFunction(input.text, cache: cache, state: nil)
            return .logits(output)
        }

        let prefillOffset = compiledCache.offset

        // VLM first prefill with image: sequential chunk processing
        if let vlmConfig, prefillOffset == 0, let image = input.image {
            let output = processVLMPrefill(
                tokens: tokens, image: image, video: input.video,
                vlmConfig: vlmConfig, compiledCache: compiledCache)
            return .logits(output)
        }

        // VLM text-only first prefill: compute M-RoPE positions
        if vlmConfig != nil, prefillOffset == 0 {
            let (positionIds, nextTextPos) = computeMRoPEPositionIds(
                inputIds: tokens, image: nil, video: nil)
            self.mropeNextPosition = nextTextPos

            let (logits, newState) = lowered.prefill(
                tokenIDs: tokens, positionIds: positionIds,
                state: compiledCache.inferenceState)
            compiledCache.inferenceState = newState
            return .logits(LMOutput(logits: logits))
        }

        // Standard text prefill (with windowing)
        let windowSize = windowSize ?? tokenCount

        if prefillOffset + windowSize < tokenCount {
            let chunk = tokens[0..., prefillOffset..<(prefillOffset + windowSize)]

            // For VLM windowed prefill, generate sequential M-RoPE positions
            let positionIds: MLXArray?
            if vlmConfig != nil {
                positionIds = makeSequentialPositionIds(
                    batchSize: chunk.dim(0), seqLen: windowSize,
                    startPosition: mropeNextPosition)
                mropeNextPosition += windowSize
            } else {
                positionIds = nil
            }

            let chunkInput = LMInput.Text(tokens: chunk, positionIds: positionIds)
            let output = callAsFunction(chunkInput, cache: cache, state: nil)
            eval(output.logits)
            return .tokens(LMInput.Text(tokens: tokens))
        }

        let remaining = tokens[0..., prefillOffset...]

        // For VLM remaining prefill, generate sequential M-RoPE positions
        let positionIds: MLXArray?
        let remainingLen = tokenCount - prefillOffset
        if vlmConfig != nil {
            positionIds = makeSequentialPositionIds(
                batchSize: remaining.dim(0), seqLen: remainingLen,
                startPosition: mropeNextPosition)
            mropeNextPosition += remainingLen
        } else {
            positionIds = nil
        }

        let output = callAsFunction(
            LMInput.Text(tokens: remaining, positionIds: positionIds),
            cache: cache,
            state: nil
        )
        return .logits(output)
    }

    // MARK: - VLM Sequential Chunk Processing

    /// Process VLM prefill with sequential chunks.
    ///
    /// Pipeline (llama.cpp-style):
    /// 1. Compute M-RoPE position IDs for the full mixed text+vision sequence
    /// 2. Run vision encoder on image pixels → vision embeddings
    /// 3. Split token sequence at vision placeholder boundaries
    /// 4. Process each chunk: text via tokenIDs, vision via embeddings
    /// 5. Each chunk updates the KV cache; return logits from the last chunk
    private func processVLMPrefill(
        tokens: MLXArray,
        image: LMInput.ProcessedImage,
        video: LMInput.ProcessedVideo?,
        vlmConfig: VLMConfig,
        compiledCache: MLXInferenceKVCache
    ) -> LMOutput {
        // 1. Compute M-RoPE position IDs for full sequence
        let (fullPositionIds, nextTextPos) = computeMRoPEPositionIds(
            inputIds: tokens, image: image, video: video)
        self.mropeNextPosition = nextTextPos

        // 2. Run vision encoder
        let gridTHW = (image.frames ?? []).map { (t: $0.t, h: $0.h, w: $0.w) }
        let visionEmbeddings = vlmConfig.visionEncoder(image.pixels, gridTHW)

        // 3. Split into chunks and process sequentially
        let chunks = splitIntoChunks(
            tokens: tokens, visionEmbeddings: visionEmbeddings,
            positionIds: fullPositionIds, vlmConfig: vlmConfig)

        var lastLogits: MLXArray!
        for chunk in chunks {
            let (logits, newState) = lowered.prefill(
                tokenIDs: chunk.tokenIDs,
                embeddings: chunk.embeddings,
                positionIds: chunk.positionIds,
                state: compiledCache.inferenceState)
            compiledCache.inferenceState = newState
            lastLogits = logits
        }

        return LMOutput(logits: lastLogits)
    }

    /// A chunk of tokens or embeddings for sequential prefill.
    private struct PrefillChunk {
        let tokenIDs: MLXArray
        let embeddings: MLXArray?
        let positionIds: MLXArray
    }

    /// Split a mixed text+vision token sequence into chunks at vision placeholder
    /// boundaries. Text chunks carry tokenIDs; vision chunks carry embeddings.
    private func splitIntoChunks(
        tokens: MLXArray,
        visionEmbeddings: MLXArray,
        positionIds: MLXArray,
        vlmConfig: VLMConfig
    ) -> [PrefillChunk] {
        let B = tokens.dim(0)
        let S = tokens.dim(1)
        let flatIds: [Int32] = tokens.reshaped(-1).asArray(Int32.self)
        let imageId = Int32(vlmConfig.imageTokenId)
        let videoId = vlmConfig.videoTokenId.map { Int32($0) }

        var chunks: [PrefillChunk] = []
        var currentStart = 0
        var visionEmbOffset = 0
        var i = 0

        while i < S {
            let tokenId = flatIds[i]
            let isVision = tokenId == imageId
                || (videoId.map { tokenId == $0 } ?? false)

            if isVision {
                // Flush preceding text chunk
                if currentStart < i {
                    chunks.append(PrefillChunk(
                        tokenIDs: tokens[0..., currentStart..<i],
                        embeddings: nil,
                        positionIds: positionIds[0..., 0..., currentStart..<i]))
                }

                // Find end of contiguous vision token run
                var j = i
                while j < S {
                    let t = flatIds[j]
                    let isV = t == imageId || (videoId.map { t == $0 } ?? false)
                    if !isV { break }
                    j += 1
                }
                let visionLen = j - i

                // Slice vision embeddings and position IDs
                let visionEmb = visionEmbeddings[
                    0..., visionEmbOffset..<(visionEmbOffset + visionLen), 0...]
                let dummyTokens = MLXArray.zeros([B, visionLen]).asType(.int32)
                chunks.append(PrefillChunk(
                    tokenIDs: dummyTokens,
                    embeddings: visionEmb,
                    positionIds: positionIds[0..., 0..., i..<j]))

                visionEmbOffset += visionLen
                currentStart = j
                i = j
            } else {
                i += 1
            }
        }

        // Flush remaining text
        if currentStart < S {
            chunks.append(PrefillChunk(
                tokenIDs: tokens[0..., currentStart..<S],
                embeddings: nil,
                positionIds: positionIds[0..., 0..., currentStart..<S]))
        }

        return chunks
    }

    // MARK: - M-RoPE Position ID Computation

    /// Compute M-RoPE 3D position IDs for mixed text+vision sequences.
    ///
    /// Text tokens: all 3 axes get the same sequential position.
    /// Vision tokens: temporal/height/width from grid layout (after spatial merge).
    ///
    /// - Returns: `(positionIds: [3, B, S], nextTextPosition: Int)`.
    private func computeMRoPEPositionIds(
        inputIds: MLXArray,
        image: LMInput.ProcessedImage?,
        video: LMInput.ProcessedVideo?
    ) -> (MLXArray, Int) {
        guard let vlmConfig else {
            fatalError("computeMRoPEPositionIds called without VLM config")
        }

        let B = inputIds.dim(0)
        let S = inputIds.dim(1)

        var temporalPos = [Int32](repeating: 0, count: B * S)
        var heightPos = [Int32](repeating: 0, count: B * S)
        var widthPos = [Int32](repeating: 0, count: B * S)

        let flatIds = inputIds.reshaped(-1)

        var currentTextPos: Int32 = 0
        var visionTokenIdx = 0
        let allGrids = (image?.frames ?? []) + (video?.frames ?? [])
        var gridIdx = 0

        for i in 0..<(B * S) {
            let tokenId: Int32 = flatIds[i].item()

            let isVisionToken = tokenId == Int32(vlmConfig.imageTokenId)
                || (vlmConfig.videoTokenId.map { tokenId == Int32($0) } ?? false)
            if isVisionToken {
                if gridIdx < allGrids.count {
                    let grid = allGrids[gridIdx]
                    let mergedH = grid.h / vlmConfig.spatialMergeSize
                    let mergedW = grid.w / vlmConfig.spatialMergeSize
                    let totalMerged = grid.t * mergedH * mergedW

                    let posInGrid = visionTokenIdx
                    let tPos = posInGrid / (mergedH * mergedW)
                    let hPos = (posInGrid % (mergedH * mergedW)) / mergedW
                    let wPos = posInGrid % mergedW

                    temporalPos[i] = currentTextPos + Int32(tPos)
                    heightPos[i] = currentTextPos + Int32(hPos)
                    widthPos[i] = currentTextPos + Int32(wPos)

                    visionTokenIdx += 1
                    if visionTokenIdx >= totalMerged {
                        currentTextPos += Int32(max(grid.t, max(mergedH, mergedW)))
                        visionTokenIdx = 0
                        gridIdx += 1
                    }
                }
            } else {
                temporalPos[i] = currentTextPos
                heightPos[i] = currentTextPos
                widthPos[i] = currentTextPos
                currentTextPos += 1
            }
        }

        let tArray = MLXArray(temporalPos).reshaped(B, S)
        let hArray = MLXArray(heightPos).reshaped(B, S)
        let wArray = MLXArray(widthPos).reshaped(B, S)

        return (stacked([tArray, hArray, wArray], axis: 0), Int(currentTextPos))
    }

    /// Create sequential M-RoPE position IDs where all 3 axes share the same
    /// sequential positions. Used for text-only decode steps after VLM prefill.
    private func makeSequentialPositionIds(
        batchSize: Int, seqLen: Int, startPosition: Int
    ) -> MLXArray {
        let positions = tiled(
            MLXArray(Int32(startPosition)..<Int32(startPosition + seqLen))
                .reshaped(1, seqLen),
            repetitions: [batchSize, 1])
        return stacked([positions, positions, positions], axis: 0)
    }
}
