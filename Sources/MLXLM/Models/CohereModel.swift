import Foundation
import GGUFParser
import MLX
import MLXFast
import MLXNN

// MARK: - Configuration

struct CohereConfiguration: Sendable {
    var hiddenSize: Int
    var hiddenLayers: Int
    var intermediateSize: Int
    var attentionHeads: Int
    var headDimensions: Int?
    var layerNormEps: Float
    var vocabularySize: Int
    var kvHeads: Int
    var maxPositionEmbeddings: Int?
    var ropeTheta: Float
    var ropeTraditional: Bool
    var ropeScaling: [String: StringOrNumber]?
    var tieWordEmbeddings: Bool
    var useQKNorm: Bool
    var logitScale: Float?

    init(
        hiddenSize: Int,
        hiddenLayers: Int,
        intermediateSize: Int,
        attentionHeads: Int,
        headDimensions: Int? = nil,
        layerNormEps: Float = 1e-5,
        vocabularySize: Int,
        kvHeads: Int? = nil,
        maxPositionEmbeddings: Int? = nil,
        ropeTheta: Float = 10_000,
        ropeTraditional: Bool = false,
        ropeScaling: [String: StringOrNumber]? = nil,
        tieWordEmbeddings: Bool = true,
        useQKNorm: Bool = false,
        logitScale: Float? = nil
    ) {
        self.hiddenSize = hiddenSize
        self.hiddenLayers = hiddenLayers
        self.intermediateSize = intermediateSize
        self.attentionHeads = attentionHeads
        self.headDimensions = headDimensions
        self.layerNormEps = layerNormEps
        self.vocabularySize = vocabularySize
        self.kvHeads = kvHeads ?? attentionHeads
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.ropeTheta = ropeTheta
        self.ropeTraditional = ropeTraditional
        self.ropeScaling = ropeScaling
        self.tieWordEmbeddings = tieWordEmbeddings
        self.useQKNorm = useQKNorm
        self.logitScale = logitScale
    }

    var resolvedHeadDimensions: Int {
        headDimensions ?? (hiddenSize / attentionHeads)
    }

    // MARK: - GGUF Init

    package init(from file: GGUFFile) throws {
        let embed = try file.require(.embeddingLength)
        let blocks = try file.require(.blockCount)
        let heads = try file.require(.headCount)

        let ffn = file[.feedForwardLength] ?? (embed * 4)
        let kv = file[.headCountKV] ?? heads
        let normEps = file[.attentionLayerNormEpsilon]
            ?? file[.attentionLayerNormRMSEpsilon] ?? 1e-5
        let ropeTheta = file[.ropeFreqBase] ?? 10_000.0
        let vocabSize = file.vocabularySize ?? 0
        let tieWordEmbeddings = detectTieWordEmbeddings(from: file)
        let ropeScaling = extractRopeScaling(from: file)
        let hasQKNorm = file.tensors.contains { $0.name == "blk.0.attn_q_norm.weight" }

        let headDim: Int? = file[.attentionKeyLength]

        self.init(
            hiddenSize: embed,
            hiddenLayers: blocks,
            intermediateSize: ffn,
            attentionHeads: heads,
            headDimensions: headDim,
            layerNormEps: normEps,
            vocabularySize: vocabSize,
            kvHeads: kv,
            maxPositionEmbeddings: file[.contextLength],
            ropeTheta: ropeTheta,
            ropeTraditional: false,
            ropeScaling: ropeScaling,
            tieWordEmbeddings: tieWordEmbeddings,
            useQKNorm: hasQKNorm,
            logitScale: file[.logitScale]
        )
    }
}

// MARK: - Attention

class CohereAttention: Module {

    let args: CohereConfiguration
    let scale: Float

    @ModuleInfo(key: "q_proj") var wq: Linear
    @ModuleInfo(key: "k_proj") var wk: Linear
    @ModuleInfo(key: "v_proj") var wv: Linear
    @ModuleInfo(key: "o_proj") var wo: Linear

    @ModuleInfo(key: "q_norm") var qNorm: LayerNorm?
    @ModuleInfo(key: "k_norm") var kNorm: LayerNorm?

    let rope: RoPELayer

    init(_ args: CohereConfiguration) {
        self.args = args
        let headDim = args.resolvedHeadDimensions
        self.scale = pow(Float(headDim), -0.5)

        let dim = args.hiddenSize
        let heads = args.attentionHeads
        let kvHeads = args.kvHeads

        self._wq.wrappedValue = Linear(dim, heads * headDim, bias: false)
        self._wk.wrappedValue = Linear(dim, kvHeads * headDim, bias: false)
        self._wv.wrappedValue = Linear(dim, kvHeads * headDim, bias: false)
        self._wo.wrappedValue = Linear(heads * headDim, dim, bias: false)

        if args.useQKNorm {
            self._qNorm.wrappedValue = LayerNorm(dimensions: headDim, eps: args.layerNormEps)
            self._kNorm.wrappedValue = LayerNorm(dimensions: headDim, eps: args.layerNormEps)
        }

        self.rope = initializeRope(
            dims: headDim, base: args.ropeTheta,
            traditional: args.ropeTraditional,
            scalingConfig: args.ropeScaling,
            maxPositionEmbeddings: args.maxPositionEmbeddings)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))

        var queries = wq(x)
        var keys = wk(x)
        var values = wv(x)

        queries = queries.reshaped(B, L, args.attentionHeads, -1).transposed(0, 2, 1, 3)
        keys = keys.reshaped(B, L, args.kvHeads, -1).transposed(0, 2, 1, 3)
        values = values.reshaped(B, L, args.kvHeads, -1).transposed(0, 2, 1, 3)

        // Apply QK normalization before RoPE
        if let qNorm, let kNorm {
            queries = qNorm(queries)
            keys = kNorm(keys)
        }

        let offset = cache?.offset ?? 0
        queries = rope(queries, offset: offset)
        keys = rope(keys, offset: offset)

        let output = attentionWithCacheUpdate(
            queries: queries, keys: keys, values: values,
            cache: cache, scale: scale, mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, -1)

        return wo(output)
    }
}

// MARK: - MLP

class CohereMLP: Module {

    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "down_proj") var down: Linear
    @ModuleInfo(key: "up_proj") var up: Linear

    init(_ args: CohereConfiguration) {
        self._gate.wrappedValue = Linear(args.hiddenSize, args.intermediateSize, bias: false)
        self._down.wrappedValue = Linear(args.intermediateSize, args.hiddenSize, bias: false)
        self._up.wrappedValue = Linear(args.hiddenSize, args.intermediateSize, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        down(silu(gate(x)) * up(x))
    }
}

// MARK: - Transformer Block

class CohereTransformerBlock: Module {

    @ModuleInfo(key: "self_attn") var attention: CohereAttention
    @ModuleInfo(key: "mlp") var mlp: CohereMLP

    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: LayerNorm

    init(_ args: CohereConfiguration) {
        self._attention.wrappedValue = CohereAttention(args)
        self._mlp.wrappedValue = CohereMLP(args)
        self._inputLayerNorm.wrappedValue = LayerNorm(
            dimensions: args.hiddenSize, eps: args.layerNormEps)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        // Cohere uses parallel attention + FFN with shared norm
        let normed = inputLayerNorm(x)
        let attnOut = attention(normed, mask: mask, cache: cache)
        let mlpOut = mlp(normed)
        return x + attnOut + mlpOut
    }
}

// MARK: - Inner Model

class CohereModelInner: Module {

    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding

    let layers: [CohereTransformerBlock]
    let norm: LayerNorm

    init(_ args: CohereConfiguration) {
        precondition(args.vocabularySize > 0)
        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: args.vocabularySize, dimensions: args.hiddenSize)
        self.layers = (0..<args.hiddenLayers).map { _ in CohereTransformerBlock(args) }
        self.norm = LayerNorm(dimensions: args.hiddenSize, eps: args.layerNormEps)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var h = embedTokens(inputs)
        let mask = createAttentionMask(h: h, cache: cache?.first)
        for (i, layer) in layers.enumerated() {
            h = layer(h, mask: mask, cache: cache?[i])
        }
        return norm(h)
    }
}

// MARK: - Top-Level Model

/// Cohere Command-R architecture language model.
///
/// Features QK normalization, logit scaling, LayerNorm (not RMSNorm),
/// and parallel attention + FFN computation.
class CohereModel: Module, LanguageModel, KVCacheDimensionProvider {

    let vocabularySize: Int
    let kvHeads: [Int]
    var layerCount: Int { model.layers.count }

    let model: CohereModelInner
    let configuration: CohereConfiguration

    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    init(_ args: CohereConfiguration) {
        self.configuration = args
        self.vocabularySize = args.vocabularySize
        self.kvHeads = (0..<args.hiddenLayers).map { _ in args.kvHeads }
        self.model = CohereModelInner(args)
        if !args.tieWordEmbeddings {
            self._lmHead.wrappedValue = Linear(args.hiddenSize, args.vocabularySize, bias: false)
        }
    }

    func callAsFunction(_ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?) -> LMOutput {
        let out = model(input.tokens, cache: cache)
        var logits: MLXArray
        if let lmHead {
            logits = lmHead(out)
        } else {
            logits = model.embedTokens.asLinear(out)
        }

        // Apply logit scaling
        if let scale = configuration.logitScale {
            logits = logits * scale
        }

        return LMOutput(logits: logits)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let out = model(inputs, cache: cache)
        var logits: MLXArray
        if let lmHead {
            logits = lmHead(out)
        } else {
            logits = model.embedTokens.asLinear(out)
        }

        if let scale = configuration.logitScale {
            logits = logits * scale
        }

        return logits
    }

    func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        weights.filter { !$0.key.contains("self_attn.rotary_emb.inv_freq") }
    }
}

// MARK: - GGUF Loading

extension CohereModel: GGUFLoadableModel {

    /// Detects Cohere by its unique parallel attention + FFN structure.
    ///
    /// Cohere uses a single shared norm (`input_layernorm`) without a separate
    /// `post_attention_layernorm` (`ffn_norm`). Combined with QK normalization,
    /// this combination uniquely identifies the architecture — other models
    /// with QK norm (e.g. Qwen 3.5) still have `ffn_norm`.
    package static func canLoad(from file: GGUFFile, context: GGUFLoadContext) -> Bool {
        let hasQKNorm = file.tensors.contains { $0.name == "blk.0.attn_q_norm.weight" }
        let hasFFNNorm = file.tensors.contains { $0.name == "blk.0.ffn_norm.weight" }
        return hasQKNorm && !hasFFNNorm
    }

    package static func load(
        from file: GGUFFile, context: GGUFLoadContext
    ) throws -> GGUFLoadResult {
        let config = try CohereConfiguration(from: file)
        let model = CohereModel(config)

        return GGUFLoadResult(
            model: model,
            mapper: CohereTensorNameMapper(),
            makeProcessor: { tokenizer, chatTemplate, bosToken, eosToken, addBosToken in
                GGUFUserInputProcessor(
                    tokenizer: tokenizer, chatTemplate: chatTemplate,
                    bosToken: bosToken, eosToken: eosToken, addBosToken: addBosToken)
            }
        )
    }
}

// MARK: - LoRA

extension CohereModel: LoRAModel {
    var loraLayers: [Module] {
        model.layers
    }
}
