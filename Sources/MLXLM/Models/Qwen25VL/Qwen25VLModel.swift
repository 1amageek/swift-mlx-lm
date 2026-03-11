import Foundation
import GGUFParser
import GGUFTokenizer
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

    /// Apply contiguous M-RoPE by splitting head dimensions into temporal/height/width sections.
    private func applyMRoPE(_ x: MLXArray, positionIds: MLXArray) -> MLXArray {
        let sections = mropeConfig.sections

        let halfDim = headDim / 2
        let freqExponents = MLXArray(stride(from: Float(0), to: Float(halfDim), by: 1.0))
        let invFreq = 1.0 / pow(MLXArray(config.ropeTheta), freqExponents / Float(halfDim))

        var cosPerAxis = [MLXArray]()
        var sinPerAxis = [MLXArray]()

        for axis in 0..<3 {
            let axisPositions = positionIds[axis]
            let freqs = expandedDimensions(axisPositions, axis: -1).asType(DType.float32)
                * invFreq.reshaped(1, 1, halfDim)
            let emb = concatenated([freqs, freqs], axis: -1)
            cosPerAxis.append(cos(emb))
            sinPerAxis.append(sin(emb))
        }

        let doubledSections = sections.map { $0 * 2 }
        var cosChunks = [MLXArray]()
        var sinChunks = [MLXArray]()
        var dimOffset = 0

        for (i, sectionDim) in doubledSections.enumerated() {
            let axisIdx = i % 3
            cosChunks.append(cosPerAxis[axisIdx][0..., 0..., dimOffset..<(dimOffset + sectionDim)])
            sinChunks.append(sinPerAxis[axisIdx][0..., 0..., dimOffset..<(dimOffset + sectionDim)])
            dimOffset += sectionDim
        }

        if dimOffset < headDim {
            let remaining = headDim - dimOffset
            let axisIdx = doubledSections.count % 3
            cosChunks.append(cosPerAxis[axisIdx][0..., 0..., dimOffset..<(dimOffset + remaining)])
            sinChunks.append(sinPerAxis[axisIdx][0..., 0..., dimOffset..<(dimOffset + remaining)])
        }

        var cosEmb = concatenated(cosChunks, axis: -1)
        var sinEmb = concatenated(sinChunks, axis: -1)
        cosEmb = cosEmb.expandedDimensions(axis: 1)
        sinEmb = sinEmb.expandedDimensions(axis: 1)

        let half = headDim / 2
        let x1 = x[0..., 0..., 0..., ..<half]
        let x2 = x[0..., 0..., 0..., half...]
        let rotateHalf = concatenated([-x2, x1], axis: -1)

        return x * cosEmb + rotateHalf * sinEmb
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

    /// Forward pass with optional embedding injection and M-RoPE position IDs.
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
/// All shared VLM orchestration (vision encoding, embedding merging,
/// M-RoPE position IDs) is provided by ``VisionLanguageModel``
/// protocol default implementations.
class Qwen25VLModel: Module, VisionLanguageModel, KVCacheDimensionProvider {

    let configuration: Qwen25VLConfiguration

    @ModuleInfo(key: "visual") var visionEncoder: Qwen25VLVisionTransformer
    @ModuleInfo(key: "model") var textModel: Qwen25VLTextModel
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    // MARK: VisionLanguageModel

    var nextPosition: Int = 0
    var imageTokenId: Int { configuration.imageTokenId }
    var videoTokenId: Int { configuration.videoTokenId }
    var spatialMergeSize: Int { configuration.vision.spatialMergeSize }

    // MARK: KVCacheDimensionProvider

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

    // MARK: - Bridge Methods

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

    func embedTokens(_ tokens: MLXArray) -> MLXArray {
        textModel.embedTokens(tokens)
    }

    func forwardTextModel(
        _ inputs: MLXArray, cache: [KVCache]?,
        inputEmbeddings: MLXArray?, positionIds: MLXArray?
    ) -> MLXArray {
        textModel(inputs, inputEmbeddings: inputEmbeddings, cache: cache, positionIds: positionIds)
    }

    func forwardLogits(_ h: MLXArray) -> MLXArray {
        if let lmHead { return lmHead(h) }
        return textModel.embedTokens.asLinear(h)
    }

    // MARK: - Model-Specific

    func newCache(parameters: GenerateParameters?) -> [KVCache] {
        nextPosition = 0
        let params = parameters ?? GenerateParameters()
        return createKVCaches(layerCount: layerCount, parameters: params)
    }

    func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var result = weights.filter { !$0.key.contains("rotary_emb.inv_freq") }

        for key in Array(result.keys) where key.contains("patch_embed.proj.weight") {
            if let w = result[key], w.ndim == 5 {
                result[key] = w.transposed(0, 2, 3, 4, 1)
            }
        }

        return result
    }
}

// MARK: - GGUF Loading

extension Qwen25VLModel: GGUFLoadableModel {

    /// Detects Qwen 2.5-VL: requires mmproj URL and must NOT have DeltaNet SSM tensors.
    package static func canLoad(from file: GGUFFile, context: GGUFLoadContext) -> Bool {
        context.mmprojURL != nil
            && !file.tensors.contains { $0.name == "blk.0.ssm_beta.weight" }
    }

    package static func load(
        from file: GGUFFile, context: GGUFLoadContext
    ) throws -> GGUFLoadResult {
        guard let mmprojURL = context.mmprojURL else {
            throw GGUFLoadError.missingMetadata("mmproj file required for VLM")
        }

        // Build text configuration from GGUF metadata
        let embed = try file.require(.embeddingLength)
        let blocks = try file.require(.blockCount)
        let heads = try file.require(.headCount)
        let ffn = file[.feedForwardLength] ?? (embed * 4)
        let kv = file[.headCountKV] ?? heads
        let normEps = file[.attentionLayerNormRMSEpsilon] ?? 1e-5
        let ropeTheta = file[.ropeFreqBase] ?? 10_000.0
        let vocabSize = file.vocabularySize ?? 0
        let tieWordEmbeddings = detectTieWordEmbeddings(from: file)

        let textConfig = Qwen25VLConfiguration.TextConfiguration(
            hiddenSize: embed,
            hiddenLayers: blocks,
            intermediateSize: ffn,
            attentionHeads: heads,
            kvHeads: kv,
            vocabularySize: vocabSize,
            normEps: normEps,
            ropeTheta: ropeTheta,
            maxPositionEmbeddings: file[.contextLength],
            tieWordEmbeddings: tieWordEmbeddings
        )

        // Load vision encoder from mmproj
        let visionLoader = GGUFVisionLoader()
        let (loadedVisionEncoder, loadedVisionConfig) = try visionLoader.load(url: mmprojURL)
        var visionConfig = loadedVisionConfig
        visionConfig.outHiddenSize = textConfig.hiddenSize

        // Resolve vision token IDs
        let tokenizer = context.tokenizer
        guard let imageTokenId = tokenizer.tokenID(for: "<|image_pad|>") else {
            throw GGUFLoadError.missingMetadata("tokenizer vocabulary: <|image_pad|>")
        }
        guard let videoTokenId = tokenizer.tokenID(for: "<|video_pad|>") else {
            throw GGUFLoadError.missingMetadata("tokenizer vocabulary: <|video_pad|>")
        }
        guard let visionStartTokenId = tokenizer.tokenID(for: "<|vision_start|>") else {
            throw GGUFLoadError.missingMetadata("tokenizer vocabulary: <|vision_start|>")
        }
        guard let visionEndTokenId = tokenizer.tokenID(for: "<|vision_end|>") else {
            throw GGUFLoadError.missingMetadata("tokenizer vocabulary: <|vision_end|>")
        }

        let vlmConfig = Qwen25VLConfiguration(
            text: textConfig,
            vision: visionConfig,
            imageTokenId: imageTokenId,
            videoTokenId: videoTokenId,
            visionStartTokenId: visionStartTokenId,
            visionEndTokenId: visionEndTokenId
        )
        let vlmModel = Qwen25VLModel(vlmConfig)

        let imageProcessor = Qwen25VLImageProcessor(config: visionConfig)

        // Capture loadedVisionEncoder for deferred weight loading
        let capturedVisionEncoder = loadedVisionEncoder
        let capturedVLMModel = vlmModel

        return GGUFLoadResult(
            model: vlmModel,
            mapper: LlamaTensorNameMapper(),
            visionLoader: { _ in
                let visionParams = capturedVisionEncoder.parameters()
                capturedVLMModel.visionEncoder.update(parameters: visionParams)
                eval(capturedVLMModel.visionEncoder)
            },
            makeProcessor: { tokenizer, chatTemplate, bosToken, eosToken, addBosToken in
                VLMUserInputProcessor(
                    tokenizer: tokenizer,
                    chatTemplate: chatTemplate,
                    bosToken: bosToken,
                    eosToken: eosToken,
                    addBosToken: addBosToken,
                    vlmInputConfig: vlmConfig,
                    preprocessImage: { try imageProcessor.preprocess(image: $0) }
                )
            }
        )
    }
}

// MARK: - LoRA

extension Qwen25VLModel: LoRAModel {
    var loraLayers: [Module] {
        textModel.layers
    }
}
