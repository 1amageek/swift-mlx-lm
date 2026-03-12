import Foundation
import GGUFParser
import MLX
import MLXFast
import MLXNN

// MARK: - Unified Configuration

/// Configuration for all standard transformer architectures.
///
/// Covers Llama, Qwen2, Mistral, Phi-3, StarCoder2, Gemma 2, and Mixtral
/// through configuration flags rather than separate model classes.
struct TransformerConfiguration: Sendable {

    // MARK: Core Dimensions

    var hiddenSize: Int
    var hiddenLayers: Int
    var intermediateSize: Int
    var attentionHeads: Int
    var headDimensions: Int?
    var vocabularySize: Int
    var kvHeads: Int

    // MARK: Normalization

    var normEps: Float

    /// Gemma 2: apply RMSNorm after attention and FFN sublayers in addition to before.
    var hasPostNorm: Bool

    // MARK: Activation

    enum ActivationType: Sendable { case silu, gelu }
    var activation: ActivationType

    // MARK: Attention

    var attentionBias: Bool
    var mlpBias: Bool

    /// Gemma 2: cap attention logits with `cap * tanh(scores / cap)`.
    var attnLogitSoftcap: Float?

    /// Gemma 2: scale queries by `1 / sqrt(scalar)` instead of `1 / sqrt(headDim)`.
    var queryPreAttnScalar: Float?

    // MARK: Position Embedding

    var ropeTheta: Float
    var ropeTraditional: Bool
    var ropeScaling: [String: StringOrNumber]?
    var maxPositionEmbeddings: Int?

    /// Phi-3: number of RoPE dimensions (partial rotary when < headDim).
    var ropeDimensionCount: Int?

    // MARK: Sliding Window

    var slidingWindow: Int?

    /// How sliding window is applied across layers.
    enum SlidingWindowPattern: Sendable {
        /// No sliding window.
        case none
        /// All layers use sliding window.
        case allLayers
        /// Even-indexed layers use sliding window, odd layers use full attention (Gemma 2).
        case evenLayers
    }
    var slidingWindowPattern: SlidingWindowPattern

    // MARK: Mixture of Experts

    var expertCount: Int?
    var expertUsedCount: Int?

    var isMoE: Bool { (expertCount ?? 0) > 0 }

    // MARK: Output

    var tieWordEmbeddings: Bool

    /// Gemma 2: cap final logits with `cap * tanh(logits / cap)`.
    var finalLogitSoftcap: Float?

    /// Gemma: scale embeddings by `sqrt(hiddenSize)`.
    var embedScale: Float?

    // MARK: Init

    init(
        hiddenSize: Int,
        hiddenLayers: Int,
        intermediateSize: Int,
        attentionHeads: Int,
        headDimensions: Int? = nil,
        vocabularySize: Int,
        kvHeads: Int? = nil,
        normEps: Float = 1e-5,
        hasPostNorm: Bool = false,
        activation: ActivationType = .silu,
        attentionBias: Bool = false,
        mlpBias: Bool = false,
        attnLogitSoftcap: Float? = nil,
        queryPreAttnScalar: Float? = nil,
        ropeTheta: Float = 10_000,
        ropeTraditional: Bool = false,
        ropeScaling: [String: StringOrNumber]? = nil,
        maxPositionEmbeddings: Int? = nil,
        ropeDimensionCount: Int? = nil,
        slidingWindow: Int? = nil,
        slidingWindowPattern: SlidingWindowPattern = .none,
        expertCount: Int? = nil,
        expertUsedCount: Int? = nil,
        tieWordEmbeddings: Bool = true,
        finalLogitSoftcap: Float? = nil,
        embedScale: Float? = nil
    ) {
        self.hiddenSize = hiddenSize
        self.hiddenLayers = hiddenLayers
        self.intermediateSize = intermediateSize
        self.attentionHeads = attentionHeads
        self.headDimensions = headDimensions
        self.vocabularySize = vocabularySize
        self.kvHeads = kvHeads ?? attentionHeads
        self.normEps = normEps
        self.hasPostNorm = hasPostNorm
        self.activation = activation
        self.attentionBias = attentionBias
        self.mlpBias = mlpBias
        self.attnLogitSoftcap = attnLogitSoftcap
        self.queryPreAttnScalar = queryPreAttnScalar
        self.ropeTheta = ropeTheta
        self.ropeTraditional = ropeTraditional
        self.ropeScaling = ropeScaling
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.ropeDimensionCount = ropeDimensionCount
        self.slidingWindow = slidingWindow
        self.slidingWindowPattern = slidingWindowPattern
        self.expertCount = expertCount
        self.expertUsedCount = expertUsedCount
        self.tieWordEmbeddings = tieWordEmbeddings
        self.finalLogitSoftcap = finalLogitSoftcap
        self.embedScale = embedScale
    }

    var resolvedHeadDimensions: Int {
        headDimensions ?? (hiddenSize / attentionHeads)
    }

    var resolvedRopeDimensions: Int {
        ropeDimensionCount ?? resolvedHeadDimensions
    }

    // MARK: - GGUF Init

    /// Construct configuration from GGUF metadata using structural feature detection.
    package init(from file: GGUFFile) throws {
        let embed = try file.require(.embeddingLength)
        let blocks = try file.require(.blockCount)
        let heads = try file.require(.headCount)

        let ffn = file[.feedForwardLength] ?? (embed * 4)
        let kv = file[.headCountKV] ?? heads
        let normEps = file[.attentionLayerNormRMSEpsilon] ?? 1e-5
        let ropeTheta = file[.ropeFreqBase] ?? 10_000.0
        let vocabSize = file.vocabularySize ?? 0

        let tieWordEmbeddings = detectTieWordEmbeddings(from: file)
        let hasAttentionBias = file.tensors.contains { $0.name == "blk.0.attn_q.bias" }
        let hasMlpBias = file.tensors.contains { $0.name == "blk.0.ffn_gate.bias" }
        let ropeScaling = extractRopeScaling(from: file)

        // Feature detection from structure — no architecture strings
        let hasPostNorm = file.tensors.contains { $0.name == "blk.0.attn_post_norm.weight" }
        let isMoE = (file[.expertCount] ?? 0) > 0

        var activation: ActivationType = .silu
        var softcap: Float? = nil
        var queryScalar: Float? = nil
        var finalSoftcap: Float? = nil
        var swPattern: SlidingWindowPattern = .none
        var scale: Float? = nil
        var headDims: Int? = file[.attentionKeyLength]
            ?? (embed / heads != (file[.attentionValueLength] ?? embed / heads)
                ? file[.attentionKeyLength] : nil)
        var experts: Int? = nil
        var expertsUsed: Int? = nil

        // Post-normalization signals Gemma 2-style architecture
        if hasPostNorm {
            let headDim = file[.attentionKeyLength] ?? embed / heads
            headDims = headDim
            activation = .gelu
            softcap = file[.attnLogitSoftcapping]
            finalSoftcap = file[.finalLogitSoftcapping]
            queryScalar = Float(headDim)
            scale = Float(embed).squareRoot()
            if file[.slidingWindow] != nil {
                swPattern = .evenLayers
            }
        } else {
            // Detect Phi-3 / StarCoder2 sliding window pattern
            if file[.slidingWindow] != nil {
                // StarCoder2 uses sliding window on all layers
                // (distinguished from Gemma 2 by absence of post-norm)
                swPattern = .allLayers
            }
        }

        if isMoE {
            experts = file[.expertCount]
            expertsUsed = file[.expertUsedCount] ?? 2
        }

        // Phi-3 has explicit ropeDimensionCount
        let ropeDimCount = file[.ropeDimensionCount]

        self.init(
            hiddenSize: embed,
            hiddenLayers: blocks,
            intermediateSize: ffn,
            attentionHeads: heads,
            headDimensions: headDims,
            vocabularySize: vocabSize,
            kvHeads: kv,
            normEps: normEps,
            hasPostNorm: hasPostNorm,
            activation: activation,
            attentionBias: hasAttentionBias,
            mlpBias: hasMlpBias,
            attnLogitSoftcap: softcap,
            queryPreAttnScalar: queryScalar,
            ropeTheta: ropeTheta,
            ropeTraditional: false,
            ropeScaling: ropeScaling,
            maxPositionEmbeddings: file[.contextLength],
            ropeDimensionCount: ropeDimCount,
            slidingWindow: file[.slidingWindow],
            slidingWindowPattern: swPattern,
            expertCount: experts,
            expertUsedCount: expertsUsed,
            tieWordEmbeddings: tieWordEmbeddings,
            finalLogitSoftcap: finalSoftcap,
            embedScale: scale
        )
    }
}

// MARK: - Attention

class TransformerAttention: Module {

    let config: TransformerConfiguration
    let scale: Float
    let logitSoftcap: Float?

    @ModuleInfo(key: "q_proj") var wq: Linear
    @ModuleInfo(key: "k_proj") var wk: Linear
    @ModuleInfo(key: "v_proj") var wv: Linear
    @ModuleInfo(key: "o_proj") var wo: Linear

    let rope: RoPELayer

    init(_ config: TransformerConfiguration) {
        self.config = config
        let headDim = config.resolvedHeadDimensions

        if let scalar = config.queryPreAttnScalar {
            self.scale = pow(scalar, -0.5)
        } else {
            self.scale = pow(Float(headDim), -0.5)
        }

        if let cap = config.attnLogitSoftcap, cap > 0 {
            self.logitSoftcap = cap
        } else {
            self.logitSoftcap = nil
        }

        let dim = config.hiddenSize
        let heads = config.attentionHeads
        let kvHeads = config.kvHeads

        self._wq.wrappedValue = Linear(dim, heads * headDim, bias: config.attentionBias)
        self._wk.wrappedValue = Linear(dim, kvHeads * headDim, bias: config.attentionBias)
        self._wv.wrappedValue = Linear(dim, kvHeads * headDim, bias: config.attentionBias)
        self._wo.wrappedValue = Linear(heads * headDim, dim, bias: config.attentionBias)

        self.rope = initializeRope(
            dims: config.resolvedRopeDimensions,
            base: config.ropeTheta,
            traditional: config.ropeTraditional,
            scalingConfig: config.ropeScaling,
            maxPositionEmbeddings: config.maxPositionEmbeddings)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXArray?, cache: KVCache?
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))

        var queries = wq(x)
        var keys = wk(x)
        var values = wv(x)

        queries = queries.reshaped(B, L, config.attentionHeads, -1).transposed(0, 2, 1, 3)
        keys = keys.reshaped(B, L, config.kvHeads, -1).transposed(0, 2, 1, 3)
        values = values.reshaped(B, L, config.kvHeads, -1).transposed(0, 2, 1, 3)

        let offset = cache?.offset ?? 0
        queries = rope(queries, offset: offset)
        keys = rope(keys, offset: offset)

        // Update KV cache
        if let cache {
            let (cachedKeys, cachedValues) = cache.update(keys: keys, values: values)
            keys = cachedKeys
            values = cachedValues
        }

        let output: MLXArray

        if let cap = logitSoftcap {
            // Expand KV heads for GQA (manual attention path)
            let repeats = config.attentionHeads / config.kvHeads
            if repeats > 1 {
                keys = expandedGQA(keys, repeats: repeats)
                values = expandedGQA(values, repeats: repeats)
            }

            // Manual attention with logit softcapping (Gemma 2)
            var scores = MLX.matmul(queries, keys.transposed(0, 1, 3, 2)) * scale
            scores = cap * MLX.tanh(scores / cap)
            if let mask { scores = scores + mask }
            let weights = softmax(scores, axis: -1)
            output = MLX.matmul(weights, values)
        } else {
            // Fused attention (fast path)
            let maskMode: MLXFast.ScaledDotProductAttentionMaskMode
            if let mask {
                maskMode = .array(mask)
            } else if L > 1 {
                maskMode = .causal
            } else {
                maskMode = .none
            }
            output = MLXFast.scaledDotProductAttention(
                queries: queries, keys: keys, values: values,
                scale: scale, mask: maskMode
            )
        }

        return wo(output.transposed(0, 2, 1, 3).reshaped(B, L, -1))
    }

    private func expandedGQA(_ x: MLXArray, repeats: Int) -> MLXArray {
        let expanded = MLX.expandedDimensions(x, axis: 2)
        let repeated = MLX.repeated(expanded, count: repeats, axis: 2)
        let (B, _, _, L, D) = (repeated.dim(0), repeated.dim(1), repeated.dim(2), repeated.dim(3), repeated.dim(4))
        return repeated.reshaped(B, -1, L, D)
    }
}

// MARK: - MLP

class TransformerMLP: Module {

    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "down_proj") var down: Linear
    @ModuleInfo(key: "up_proj") var up: Linear

    let useGelu: Bool

    init(_ config: TransformerConfiguration) {
        self._gate.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: config.mlpBias)
        self._down.wrappedValue = Linear(config.intermediateSize, config.hiddenSize, bias: config.mlpBias)
        self._up.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: config.mlpBias)
        self.useGelu = (config.activation == .gelu)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let activated = useGelu ? gelu(gate(x)) : silu(gate(x))
        return down(activated * up(x))
    }
}

// MARK: - Mixture of Experts

class TransformerMoE: Module {

    @ModuleInfo(key: "gate") var routerGate: Linear

    let experts: [TransformerMLP]
    let expertUsedCount: Int

    init(_ config: TransformerConfiguration) {
        let numExperts = config.expertCount ?? 8
        self._routerGate.wrappedValue = Linear(config.hiddenSize, numExperts, bias: false)
        self.experts = (0..<numExperts).map { _ in TransformerMLP(config) }
        self.expertUsedCount = config.expertUsedCount ?? 2
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (B, L, D) = (x.dim(0), x.dim(1), x.dim(2))
        let flat = x.reshaped(-1, D)

        let gateLogits = routerGate(flat)
        let topKIndices = MLX.argSort(gateLogits, axis: -1)[0..., (gateLogits.dim(-1) - expertUsedCount)...]
        let topKGateLogits = MLX.takeAlong(gateLogits, topKIndices, axis: -1)
        let gateWeights = softmax(topKGateLogits, axis: -1)

        var output = MLXArray.zeros(like: flat)

        for (expertIdx, expert) in experts.enumerated() {
            for k in 0..<expertUsedCount {
                let kMask = topKIndices[0..., k..<(k+1)] .== MLXArray(Int32(expertIdx))
                let kMaskFloat = kMask.asType(.float32).squeezed(axis: -1)

                let tokenCount = MLX.sum(kMaskFloat).item(Int.self)
                guard tokenCount > 0 else { continue }

                let kWeight = gateWeights[0..., k..<(k+1)]
                let expertOut = expert(flat)
                output = output + expertOut * kWeight * kMaskFloat.expandedDimensions(axis: -1)
            }
        }

        return output.reshaped(B, L, D)
    }
}

// MARK: - Transformer Block

class TransformerBlock: Module {

    let config: TransformerConfiguration
    let layerIndex: Int

    @ModuleInfo(key: "self_attn") var attention: TransformerAttention

    // Dense FFN (nil for MoE models)
    @ModuleInfo(key: "mlp") var mlp: TransformerMLP?

    // MoE FFN (nil for dense models)
    @ModuleInfo(key: "block_sparse_moe") var moe: TransformerMoE?

    // Primary norms (always present)
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    // Gemma 2 extra norms (nil for standard models)
    @ModuleInfo(key: "pre_feedforward_layernorm") var preFeedforwardLayerNorm: RMSNorm?
    @ModuleInfo(key: "post_feedforward_layernorm") var postFeedforwardLayerNorm: RMSNorm?

    init(_ config: TransformerConfiguration, layerIndex: Int) {
        self.config = config
        self.layerIndex = layerIndex

        self._attention.wrappedValue = TransformerAttention(config)

        if config.isMoE {
            self._moe.wrappedValue = TransformerMoE(config)
        } else {
            self._mlp.wrappedValue = TransformerMLP(config)
        }

        self._inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.normEps)
        self._postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.normEps)

        if config.hasPostNorm {
            self._preFeedforwardLayerNorm.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize, eps: config.normEps)
            self._postFeedforwardLayerNorm.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize, eps: config.normEps)
        }
    }

    /// Whether this layer uses sliding window attention.
    var usesSlidingWindow: Bool {
        switch config.slidingWindowPattern {
        case .none: return false
        case .allLayers: return true
        case .evenLayers: return layerIndex % 2 == 0
        }
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXArray?, cache: KVCache?
    ) -> MLXArray {
        if config.hasPostNorm {
            // Gemma 2 mode: pre-norm + post-norm on both sublayers
            var r = attention(inputLayerNorm(x), mask: mask, cache: cache)
            r = postAttentionLayerNorm(r)
            let h = x + r
            r = ffn(preFeedforwardLayerNorm!(h))
            r = postFeedforwardLayerNorm!(r)
            return h + r
        } else {
            // Standard mode: pre-norm only
            var r = attention(inputLayerNorm(x), mask: mask, cache: cache)
            let h = x + r
            r = ffn(postAttentionLayerNorm(h))
            return h + r
        }
    }

    private func ffn(_ x: MLXArray) -> MLXArray {
        if let moe { return moe(x) }
        if let mlp { return mlp(x) }
        fatalError("TransformerBlock has neither MLP nor MoE")
    }
}

// MARK: - Inner Model

class TransformerModelInner: Module {

    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding

    let layers: [TransformerBlock]
    let norm: RMSNorm
    let config: TransformerConfiguration

    init(_ config: TransformerConfiguration) {
        self.config = config
        precondition(config.vocabularySize > 0)
        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize, dimensions: config.hiddenSize)
        self.layers = (0..<config.hiddenLayers).map { TransformerBlock(config, layerIndex: $0) }
        self.norm = RMSNorm(dimensions: config.hiddenSize, eps: config.normEps)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var h = embedTokens(inputs)

        // Gemma: scale embeddings
        if let scale = config.embedScale {
            h = h * MLXArray(scale)
        }

        for (i, layer) in layers.enumerated() {
            let mask = createLayerMask(
                queryLength: h.dim(1),
                cache: cache?[i],
                layer: layer
            )
            h = layer(h, mask: mask, cache: cache?[i])
        }
        return norm(h)
    }

    private func createLayerMask(
        queryLength: Int, cache: KVCache?, layer: TransformerBlock
    ) -> MLXArray? {
        let needsSlidingWindow = layer.usesSlidingWindow && config.slidingWindow != nil

        if !needsSlidingWindow && config.attnLogitSoftcap == nil {
            // Standard path: use built-in causal masking
            // Return nil to signal the attention module to use the fast path
            if queryLength > 1 {
                // Prefill: create causal mask as MLXArray for the manual attention path
                // For the fast path (no softcap), return nil and let SDPA handle it
                return nil
            }
            return nil
        }

        let offset = cache?.offset ?? 0
        let keyLength = queryLength + offset

        if queryLength == 1 && !needsSlidingWindow {
            return nil
        }

        // Create causal mask
        let rowIndices = MLXArray(0..<queryLength).reshaped(queryLength, 1) + MLXArray(offset)
        let colIndices = MLXArray(0..<keyLength).reshaped(1, keyLength)
        var mask = MLX.where(rowIndices .>= colIndices, MLXArray(Float(0)), MLXArray(Float(-1e9)))

        // Apply sliding window constraint
        if needsSlidingWindow, let sw = config.slidingWindow {
            let distanceMask = MLX.where(
                (rowIndices - colIndices) .< MLXArray(sw),
                MLXArray(Float(0)),
                MLXArray(Float(-1e9))
            )
            mask = mask + distanceMask
        }

        return mask.reshaped(1, 1, queryLength, keyLength)
    }
}

// MARK: - Top-Level Model

/// Unified transformer language model.
///
/// Handles Llama, Qwen2, Mistral, Phi-3, StarCoder2, Gemma 2, and Mixtral
/// through configuration flags. Architecture differences like activation function,
/// post-normalization, attention softcapping, and MoE routing are all
/// parameterized in `TransformerConfiguration`.
class TransformerModel: Module, LanguageModel, KVCacheDimensionProvider {

    let vocabularySize: Int
    let kvHeads: [Int]
    var layerCount: Int { model.layers.count }

    let model: TransformerModelInner
    let configuration: TransformerConfiguration

    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    /// Cached compiled decode function for fixed [1,1] input shape.
    /// Created lazily on first decode call. Reduces graph construction overhead
    /// by reusing the pre-traced computation graph.
    private var compiledDecode: (@Sendable ([MLXArray]) -> [MLXArray])?

    /// The cache array used during compilation. Must match the cache used in generation.
    private var compiledDecodeCache: [KVCache]?

    init(_ config: TransformerConfiguration) {
        self.configuration = config
        self.vocabularySize = config.vocabularySize
        self.kvHeads = (0..<config.hiddenLayers).map { _ in config.kvHeads }
        self.model = TransformerModelInner(config)
        if !config.tieWordEmbeddings {
            self._lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabularySize, bias: false)
        }
    }

    func callAsFunction(_ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?) -> LMOutput {
        let tokens = input.tokens
        let tokenCount = tokens.dim(tokens.ndim - 1)

        // Use compiled decode for single-token inputs (autoregressive decode phase)
        if tokenCount == 1, let cache {
            let logits = compiledForwardLogits(tokens, cache: cache)
            return LMOutput(logits: logits)
        }

        var logits = forwardLogits(tokens, cache: cache)

        if let cap = configuration.finalLogitSoftcap, cap > 0 {
            logits = cap * MLX.tanh(logits / cap)
        }

        return LMOutput(logits: logits)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        var logits = forwardLogits(inputs, cache: cache)

        if let cap = configuration.finalLogitSoftcap, cap > 0 {
            logits = cap * MLX.tanh(logits / cap)
        }

        return logits
    }

    private func forwardLogits(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let out = model(inputs, cache: cache)
        if let lmHead {
            return lmHead(out)
        } else {
            return model.embedTokens.asLinear(out)
        }
    }

    /// Forward pass using compiled (graph-cached) decode for single-token inputs.
    ///
    /// On first call, traces the decode computation and caches the graph.
    /// Subsequent calls reuse the cached graph, saving graph construction time.
    private func compiledForwardLogits(_ inputs: MLXArray, cache: [KVCache]) -> MLXArray {
        // Rebuild compiled function if cache changed (new generation session)
        if compiledDecodeCache == nil || !cacheIdentical(compiledDecodeCache!, cache) {
            let updatableCaches = cache.compactMap { $0 as? (any Updatable) }

            // Only compile if all caches support Updatable
            guard updatableCaches.count == cache.count else {
                return forwardLogits(inputs, cache: cache)
            }

            compiledDecodeCache = cache
            compiledDecode = compile(
                inputs: [self] + updatableCaches,
                outputs: [self] + updatableCaches
            ) { [self] args in
                let tokenInput = args[0]
                var logits = forwardLogits(tokenInput, cache: cache)
                if let cap = configuration.finalLogitSoftcap, cap > 0 {
                    logits = cap * MLX.tanh(logits / cap)
                }
                return [logits]
            }
        }

        guard let compiled = compiledDecode else {
            return forwardLogits(inputs, cache: cache)
        }

        return compiled([inputs])[0]
    }

    /// Check if two cache arrays refer to the same objects.
    private func cacheIdentical(_ a: [KVCache], _ b: [KVCache]) -> Bool {
        guard a.count == b.count else { return false }
        return zip(a, b).allSatisfy { $0 === $1 }
    }

    func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        weights.filter { !$0.key.contains("self_attn.rotary_emb.inv_freq") }
    }
}

// MARK: - GGUF Loading

extension TransformerModel: GGUFLoadableModel {

    /// Universal fallback: can handle any standard Transformer GGUF.
    package static func canLoad(from file: GGUFFile, context: GGUFLoadContext) -> Bool { true }

    package static func load(
        from file: GGUFFile, context: GGUFLoadContext
    ) throws -> GGUFLoadResult {
        let config = try TransformerConfiguration(from: file)
        let model = TransformerModel(config)

        // Mapper chosen by structure, not by name
        let hasPostNorm = file.tensors.contains { $0.name == "blk.0.attn_post_norm.weight" }
        let hasExperts = (file[.expertCount] ?? 0) > 0

        let mapper: any GGUFTensorNameMapper
        if hasExperts {
            mapper = MoETensorNameMapper()
        } else if hasPostNorm {
            mapper = PostNormTransformerTensorNameMapper()
        } else {
            mapper = TransformerTensorNameMapper()
        }

        return GGUFLoadResult(
            model: model,
            mapper: mapper,
            makeProcessor: { tokenizer, chatTemplate, bosToken, eosToken, addBosToken in
                GGUFUserInputProcessor(
                    tokenizer: tokenizer, chatTemplate: chatTemplate,
                    bosToken: bosToken, eosToken: eosToken, addBosToken: addBosToken)
            }
        )
    }
}

// MARK: - LoRA

extension TransformerModel: LoRAModel {
    var loraLayers: [Module] {
        model.layers
    }
}
