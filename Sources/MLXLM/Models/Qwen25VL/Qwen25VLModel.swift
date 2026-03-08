import Foundation
import MLX
import MLXFast
import MLXNN

// MARK: - M-RoPE Attention

/// Multi-head attention with Multimodal Rotary Position Embedding (M-RoPE).
///
/// Splits head dimensions into 3 sections (temporal, height, width) and applies
/// separate rotary embeddings with distinct position IDs for each section.
class Qwen25VLAttention: Module {

    let config: Qwen25VLConfiguration.TextConfiguration
    let mropeConfig: Qwen25VLConfiguration.MRoPEConfiguration
    let numHeads: Int
    let numKVHeads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var wq: Linear
    @ModuleInfo(key: "k_proj") var wk: Linear
    @ModuleInfo(key: "v_proj") var wv: Linear
    @ModuleInfo(key: "o_proj") var wo: Linear

    init(_ config: Qwen25VLConfiguration.TextConfiguration,
         mrope: Qwen25VLConfiguration.MRoPEConfiguration) {
        self.config = config
        self.mropeConfig = mrope
        self.numHeads = config.attentionHeads
        self.numKVHeads = config.kvHeads
        self.headDim = config.headDim
        self.scale = pow(Float(headDim), -0.5)

        let dim = config.hiddenSize
        self._wq.wrappedValue = Linear(dim, numHeads * headDim, bias: true)
        self._wk.wrappedValue = Linear(dim, numKVHeads * headDim, bias: true)
        self._wv.wrappedValue = Linear(dim, numKVHeads * headDim, bias: true)
        self._wo.wrappedValue = Linear(numHeads * headDim, dim, bias: false)
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?,
        positionIds: MLXArray?
    ) -> MLXArray {
        let (B, L, _) = (x.dim(0), x.dim(1), x.dim(2))

        var queries = wq(x).reshaped(B, L, numHeads, headDim).transposed(0, 2, 1, 3)
        var keys = wk(x).reshaped(B, L, numKVHeads, headDim).transposed(0, 2, 1, 3)
        let values = wv(x).reshaped(B, L, numKVHeads, headDim).transposed(0, 2, 1, 3)

        // Apply M-RoPE: split head_dim into 3 sections, apply RoPE per section
        if let positionIds {
            queries = applyMRoPE(queries, positionIds: positionIds)
            keys = applyMRoPE(keys, positionIds: positionIds)
        } else {
            // Fallback: standard RoPE with cache offset
            let offset = cache?.offset ?? 0
            queries = MLXFast.RoPE(
                queries, dimensions: headDim, traditional: false,
                base: config.ropeTheta, scale: 1.0, offset: offset
            )
            keys = MLXFast.RoPE(
                keys, dimensions: headDim, traditional: false,
                base: config.ropeTheta, scale: 1.0, offset: offset
            )
        }

        let output = attentionWithCacheUpdate(
            queries: queries, keys: keys, values: values,
            cache: cache, scale: scale, mask: mask
        )

        return wo(output.transposed(0, 2, 1, 3).reshaped(B, L, -1))
    }

    /// Apply M-RoPE by splitting head dimensions into temporal/height/width sections.
    private func applyMRoPE(_ x: MLXArray, positionIds: MLXArray) -> MLXArray {
        // positionIds: [3, B, S] — temporal, height, width position IDs
        // x: [B, H, S, D] — queries or keys
        let sections = mropeConfig.sections  // [16, 24, 24]

        // Split head dim into 3 sections
        // Each section gets RoPE with its corresponding position IDs
        var parts = [MLXArray]()
        var dimOffset = 0

        for (i, sectionSize) in sections.enumerated() {
            let ropeDims = sectionSize * 2  // RoPE applies to pairs
            let sectionSlice = x[0..., 0..., 0..., dimOffset..<(dimOffset + ropeDims)]
            let sectionPositions = positionIds[i]  // [B, S]

            let rotated = MLXFast.RoPE(
                sectionSlice, dimensions: ropeDims, traditional: false,
                base: config.ropeTheta, scale: 1.0, offset: sectionPositions
            )
            parts.append(rotated)
            dimOffset += ropeDims
        }

        // Append remaining dims (if headDim > sum of sections * 2)
        if dimOffset < headDim {
            parts.append(x[0..., 0..., 0..., dimOffset...])
        }

        return concatenated(parts, axis: -1)
    }
}

// MARK: - MLP

/// Standard SwiGLU feed-forward for Qwen2.5-VL language model.
class Qwen25VLMLP: Module {

    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(_ config: Qwen25VLConfiguration.TextConfiguration) {
        let dim = config.hiddenSize
        let ffnDim = config.intermediateSize

        self._gateProj.wrappedValue = Linear(dim, ffnDim, bias: false)
        self._upProj.wrappedValue = Linear(dim, ffnDim, bias: false)
        self._downProj.wrappedValue = Linear(ffnDim, dim, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(silu(gateProj(x)) * upProj(x))
    }
}

// MARK: - Decoder Layer

/// A single decoder layer with M-RoPE attention and SwiGLU MLP.
class Qwen25VLDecoderLayer: Module {

    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "self_attn") var selfAttn: Qwen25VLAttention
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm
    @ModuleInfo var mlp: Qwen25VLMLP

    init(_ config: Qwen25VLConfiguration.TextConfiguration,
         mrope: Qwen25VLConfiguration.MRoPEConfiguration) {
        self._inputLayerNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.normEps)
        self._selfAttn.wrappedValue = Qwen25VLAttention(config, mrope: mrope)
        self._postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.normEps)
        self._mlp.wrappedValue = Qwen25VLMLP(config)
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?,
        positionIds: MLXArray?
    ) -> MLXArray {
        let r = selfAttn(inputLayerNorm(x), mask: mask, cache: cache, positionIds: positionIds)
        let h = x + r
        return h + mlp(postAttentionLayerNorm(h))
    }
}

// MARK: - Text Model (Inner)

/// Qwen2.5-VL text decoder (embedding + transformer layers + final norm).
class Qwen25VLTextModel: Module {

    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "layers") var layers: [Qwen25VLDecoderLayer]
    @ModuleInfo(key: "norm") var norm: RMSNorm

    let config: Qwen25VLConfiguration.TextConfiguration

    init(_ config: Qwen25VLConfiguration.TextConfiguration,
         mrope: Qwen25VLConfiguration.MRoPEConfiguration) {
        self.config = config
        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize, dimensions: config.hiddenSize
        )
        self._layers.wrappedValue = (0..<config.hiddenLayers).map {
            _ in Qwen25VLDecoderLayer(config, mrope: mrope)
        }
        self._norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.normEps)
    }

    /// Forward pass with pre-computed embeddings.
    ///
    /// When `inputEmbeddings` is provided, it is used instead of `embedTokens(inputIds)`.
    /// This allows vision tokens to be merged into the sequence before the transformer.
    func callAsFunction(
        _ inputIds: MLXArray,
        inputEmbeddings: MLXArray? = nil,
        cache: [KVCache]? = nil,
        positionIds: MLXArray? = nil
    ) -> MLXArray {
        var h = inputEmbeddings ?? embedTokens(inputIds)

        for (i, layer) in layers.enumerated() {
            let mask = createAttentionMask(h: h, cache: cache?[i])
            h = layer(h, mask: mask, cache: cache?[i], positionIds: positionIds)
        }

        return norm(h)
    }
}

// MARK: - Qwen2.5-VL Model (Top-Level VLM)

/// Qwen2.5-VL vision-language model.
///
/// Combines a ViT-based vision encoder with a Qwen2 text decoder.
/// Vision tokens are encoded and merged into the text sequence during
/// ``prepare(_:cache:windowSize:)``, after which generation proceeds as
/// standard autoregressive text generation with M-RoPE positioning.
class Qwen25VLModel: Module, VisionLanguageModel, KVCacheDimensionProvider {

    let configuration: Qwen25VLConfiguration

    @ModuleInfo(key: "visual") var visionEncoder: Qwen25VLVisionTransformer
    @ModuleInfo(key: "model") var textModel: Qwen25VLTextModel
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    /// Cached position state for generation phase (after vision is processed).
    private var nextPosition: Int = 0

    var imageTokenId: Int { configuration.imageTokenId }
    var videoTokenId: Int { configuration.videoTokenId }
    var vocabularySize: Int { configuration.text.vocabularySize }
    var layerCount: Int { textModel.layers.count }
    var kvHeads: [Int] {
        (0..<configuration.text.hiddenLayers).map { _ in configuration.text.kvHeads }
    }

    init(_ config: Qwen25VLConfiguration) {
        self.configuration = config

        self._visionEncoder.wrappedValue = Qwen25VLVisionTransformer(config.vision)
        self._textModel.wrappedValue = Qwen25VLTextModel(config.text, mrope: config.mrope)
        if !config.text.tieWordEmbeddings {
            self._lmHead.wrappedValue = Linear(
                config.text.hiddenSize, config.text.vocabularySize, bias: false
            )
        }
    }

    // MARK: - VisionLanguageModel

    func encodeVision(
        image: LMInput.ProcessedImage?,
        video: LMInput.ProcessedVideo?
    ) throws -> MLXArray? {
        if let image {
            let gridTHW = image.frames ?? [LMInput.THW(t: 1, h: 1, w: 1)]
            return visionEncoder(image.pixels, gridTHW: gridTHW)
        }
        if let video {
            let gridTHW = video.frames ?? [LMInput.THW(t: 1, h: 1, w: 1)]
            return visionEncoder(video.pixels, gridTHW: gridTHW)
        }
        return nil
    }

    // MARK: - LanguageModel

    func prepare(
        _ input: LMInput, cache: [KVCache], windowSize: Int?
    ) throws -> PrepareResult {
        let tokens = input.text.tokens
        let tokenCount = tokens.dim(tokens.ndim - 1)

        // Encode vision on first call (before any prefill chunks)
        let prefillOffset = cache.first?.offset ?? 0

        if prefillOffset == 0 {
            // First prepare call — encode vision and merge embeddings
            let visionEmbeddings = try encodeVision(image: input.image, video: input.video)

            // Compute M-RoPE position IDs
            let positionIds = computeMRoPEPositionIds(
                inputIds: tokens, image: input.image, video: input.video
            )

            // Get text embeddings and merge vision
            var embeddings = textModel.embedTokens(tokens)
            if let visionEmbeddings {
                embeddings = mergeVisionEmbeddings(
                    textEmbeddings: embeddings,
                    visionEmbeddings: visionEmbeddings,
                    inputIds: tokens
                )
            }

            // Full prefill — no chunking for VLM (vision embeddings are already merged)
            let output = forwardFromEmbeddings(
                embeddings, inputIds: tokens, cache: cache, positionIds: positionIds
            )
            nextPosition = tokenCount
            return .logits(output)
        }

        // Subsequent calls (shouldn't happen for VLM since we do full prefill)
        let remaining = tokens[0..., prefillOffset...]
        let output = callAsFunction(
            LMInput.Text(tokens: remaining),
            cache: cache,
            state: nil
        )
        return .logits(output)
    }

    func callAsFunction(
        _ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?
    ) -> LMOutput {
        // Generation phase: text-only with sequential M-RoPE positions
        let positionIds = input.positionIds ?? makeSequentialPositionIds(
            batchSize: input.tokens.dim(0),
            seqLen: input.tokens.dim(1),
            startPosition: nextPosition
        )

        let h = textModel(input.tokens, cache: cache, positionIds: positionIds)
        let logits = forwardLogits(h)

        nextPosition += input.tokens.dim(1)
        return LMOutput(logits: logits)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let h = textModel(inputs, cache: cache)
        return forwardLogits(h)
    }

    func newCache(parameters: GenerateParameters?) -> [KVCache] {
        nextPosition = 0
        let params = parameters ?? GenerateParameters()
        return createKVCaches(layerCount: layerCount, parameters: params)
    }

    func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var result = weights.filter { !$0.key.contains("rotary_emb.inv_freq") }

        // Transpose Conv3d weights from PyTorch [O, I, D, H, W] to MLX [O, D, H, W, I]
        for key in Array(result.keys) where key.contains("patch_embed.proj.weight") {
            if let w = result[key], w.ndim == 5 {
                // PyTorch: [O, I, D, H, W] → MLX: [O, D, H, W, I]
                result[key] = w.transposed(0, 2, 3, 4, 1)
            }
        }

        return result
    }

    // MARK: - Private Helpers

    private func forwardLogits(_ h: MLXArray) -> MLXArray {
        if let lmHead {
            return lmHead(h)
        }
        return textModel.embedTokens.asLinear(h)
    }

    /// Forward pass from pre-computed embeddings (for prefill with vision).
    private func forwardFromEmbeddings(
        _ embeddings: MLXArray,
        inputIds: MLXArray,
        cache: [KVCache],
        positionIds: MLXArray?
    ) -> LMOutput {
        let h = textModel(inputIds, inputEmbeddings: embeddings, cache: cache, positionIds: positionIds)
        let logits = forwardLogits(h)
        return LMOutput(logits: logits)
    }

    /// Replace placeholder token embeddings with vision encoder output.
    private func mergeVisionEmbeddings(
        textEmbeddings: MLXArray,
        visionEmbeddings: MLXArray,
        inputIds: MLXArray
    ) -> MLXArray {
        let B = textEmbeddings.dim(0)
        let S = textEmbeddings.dim(1)
        let D = textEmbeddings.dim(2)

        // Find positions of image/video tokens in the input
        let flatIds = inputIds.reshaped(-1)
        let isImage = flatIds .== MLXArray(Int32(imageTokenId))
        let isVideo = flatIds .== MLXArray(Int32(videoTokenId))
        let isVision = isImage .|| isVideo

        // Build vision index via cumulative sum: each vision position gets its index
        // into visionEmbeddings. Non-vision positions get stale indices but are masked out.
        let visionCumsum = cumsum(isVision.asType(DType.int32)) - 1

        // Clamp indices to valid range for gather
        let clampedIdx = clip(visionCumsum, min: 0, max: max(visionEmbeddings.dim(0) - 1, 0))

        // Gather: create full-size tensor with vision embeddings at every position
        // (non-vision positions will have wrong values, but we mask them out below)
        let visionGathered = visionEmbeddings[clampedIdx]  // [B*S, D]

        // Mix: use isVision mask to select between vision and text embeddings
        let flatEmbeddings = textEmbeddings.reshaped(B * S, D)
        let mask = isVision.reshaped(B * S, 1)
        let merged = which(mask, visionGathered, flatEmbeddings)

        return merged.reshaped(B, S, D)
    }

    /// Compute M-RoPE 3D position IDs for mixed text+vision sequences.
    ///
    /// Text tokens: all 3 dimensions get the same sequential position.
    /// Image tokens: temporal=constant, height/width from grid layout.
    private func computeMRoPEPositionIds(
        inputIds: MLXArray,
        image: LMInput.ProcessedImage?,
        video: LMInput.ProcessedVideo?
    ) -> MLXArray {
        let B = inputIds.dim(0)
        let S = inputIds.dim(1)
        let spatialMerge = configuration.vision.spatialMergeSize

        // Start with sequential positions for all tokens
        var temporalPos = [Int32](repeating: 0, count: B * S)
        var heightPos = [Int32](repeating: 0, count: B * S)
        var widthPos = [Int32](repeating: 0, count: B * S)

        let flatIds = inputIds.reshaped(-1)

        var currentTextPos: Int32 = 0
        var visionTokenIdx = 0

        // Collect grid info
        let imageGrids = image?.frames ?? []
        let videoGrids = video?.frames ?? []
        let allGrids = imageGrids + videoGrids

        var gridIdx = 0

        for i in 0..<(B * S) {
            let tokenId: Int32 = flatIds[i].item()

            if tokenId == Int32(imageTokenId) || tokenId == Int32(videoTokenId) {
                // Vision token — assign spatial positions
                if gridIdx < allGrids.count {
                    let grid = allGrids[gridIdx]
                    let mergedH = grid.h / spatialMerge
                    let mergedW = grid.w / spatialMerge
                    let totalMerged = grid.t * mergedH * mergedW

                    // Position within the current image/video grid
                    let posInGrid = visionTokenIdx
                    let tPos = posInGrid / (mergedH * mergedW)
                    let hPos = (posInGrid % (mergedH * mergedW)) / mergedW
                    let wPos = posInGrid % mergedW

                    temporalPos[i] = currentTextPos + Int32(tPos)
                    heightPos[i] = currentTextPos + Int32(hPos)
                    widthPos[i] = currentTextPos + Int32(wPos)

                    visionTokenIdx += 1

                    // Move to next grid when current one is exhausted
                    if visionTokenIdx >= totalMerged {
                        currentTextPos += Int32(max(grid.t, max(mergedH, mergedW)))
                        visionTokenIdx = 0
                        gridIdx += 1
                    }
                }
            } else {
                // Text token — all dimensions get same position
                temporalPos[i] = currentTextPos
                heightPos[i] = currentTextPos
                widthPos[i] = currentTextPos
                currentTextPos += 1
            }
        }

        // Shape: [3, B, S]
        let tArray = MLXArray(temporalPos).reshaped(B, S)
        let hArray = MLXArray(heightPos).reshaped(B, S)
        let wArray = MLXArray(widthPos).reshaped(B, S)

        return stacked([tArray, hArray, wArray], axis: 0)
    }

    /// Create sequential position IDs for text-only generation.
    private func makeSequentialPositionIds(
        batchSize: Int, seqLen: Int, startPosition: Int
    ) -> MLXArray {
        let positions = tiled(
            MLXArray(Int32(startPosition)..<Int32(startPosition + seqLen))
                .reshaped(1, seqLen),
            repetitions: [batchSize, 1]
        )
        // All 3 dimensions are identical for text tokens
        return stacked([positions, positions, positions], axis: 0)
    }
}

// MARK: - LoRA

extension Qwen25VLModel: LoRAModel {
    var loraLayers: [Module] {
        textModel.layers
    }
}
