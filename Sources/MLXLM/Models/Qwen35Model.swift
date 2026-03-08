import Foundation
import MLX
import MLXFast
import MLXNN

// MARK: - Configuration

/// Configuration for the Qwen 3.5 hybrid DeltaNet + Full Attention architecture.
///
/// Alternates between Gated DeltaNet (linear attention, O(1) per token)
/// and standard Full Attention layers in a configurable ratio
/// (default 3:1 via `fullAttentionInterval=4`).
struct Qwen35Configuration: Sendable {

    // MARK: Core Dimensions

    var hiddenSize: Int
    var hiddenLayers: Int
    var intermediateSize: Int
    var vocabularySize: Int
    var normEps: Float
    var tieWordEmbeddings: Bool

    // MARK: Full Attention Parameters

    var attentionHeads: Int
    var kvHeads: Int
    var headDim: Int
    var ropeTheta: Float
    var ropeTraditional: Bool
    var ropeScaling: [String: StringOrNumber]?
    var maxPositionEmbeddings: Int?
    var partialRotaryFactor: Float

    // MARK: DeltaNet Parameters

    var linearKeyHeads: Int
    var linearValueHeads: Int
    var linearKeyHeadDim: Int
    var linearValueHeadDim: Int
    var convKernelSize: Int

    // MARK: Layer Routing

    var fullAttentionInterval: Int

    // MARK: Computed

    var keyDim: Int { linearKeyHeads * linearKeyHeadDim }
    var valueDim: Int { linearValueHeads * linearValueHeadDim }
    var convDim: Int { 2 * keyDim + valueDim }
    var ropePartialDim: Int { Int(Float(headDim) * partialRotaryFactor) }

    func isFullAttentionLayer(_ index: Int) -> Bool {
        (index + 1) % fullAttentionInterval == 0
    }
}

// MARK: - DeltaNet Cache

/// Cache for DeltaNet linear attention layers.
///
/// Stores a Conv1D sliding window buffer and a fixed-size recurrent state matrix,
/// unlike the growing KV cache used by standard attention.
final class DeltaNetCache: KVCache {

    var convState: MLXArray?
    var recurrentState: MLXArray?
    private(set) var offset: Int = 0

    var maxSize: Int? { nil }
    var isTrimmable: Bool { false }

    func incrementOffset(by n: Int) { offset += n }

    func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        fatalError("DeltaNetCache uses recurrent state, not KV pairs")
    }

    @discardableResult
    func trim(_ n: Int) -> Int { 0 }

    func innerState() -> [MLXArray] {
        [convState, recurrentState].compactMap { $0 }
    }

    var state: [MLXArray] {
        get { [convState, recurrentState].compactMap { $0 } }
        set {
            guard newValue.count >= 2 else { return }
            convState = newValue[0]
            recurrentState = newValue[1]
        }
    }

    var metaState: [String] {
        get { [String(offset)] }
        set { if let v = newValue.first.flatMap(Int.init) { offset = v } }
    }

    func makeMask(queryLength: Int) -> MLXFast.ScaledDotProductAttentionMaskMode { .none }
}

// MARK: - Gated RMSNorm

/// RMSNorm with SiLU gating for DeltaNet output.
///
/// Uses `(1 + weight)` scaling with zero-initialized weight (HuggingFace convention).
class Qwen35GatedRMSNorm: Module {

    let weight: MLXArray
    let eps: Float

    init(dimensions: Int, eps: Float = 1e-5) {
        self.weight = MLXArray.zeros([dimensions])
        self.eps = eps
    }

    func callAsFunction(_ x: MLXArray, gate: MLXArray) -> MLXArray {
        MLXFast.rmsNorm(x, weight: 1 + weight, eps: eps) * silu(gate)
    }
}

// MARK: - Gated DeltaNet

/// Gated DeltaNet linear attention layer.
///
/// Uses a fixed-size state matrix S instead of a growing KV cache.
/// State update: `S_t = exp(g) * S_{t-1} + k ⊗ [β(v − exp(g)·S^T·k)]`
/// Output: `o = S^T · (q / √d_k)`
class Qwen35GatedDeltaNet: Module {

    let config: Qwen35Configuration
    let scale: Float

    @ModuleInfo(key: "in_proj_qkv") var inProjQKV: Linear
    @ModuleInfo(key: "in_proj_z") var inProjZ: Linear
    @ModuleInfo(key: "in_proj_b") var inProjB: Linear
    @ModuleInfo(key: "in_proj_a") var inProjA: Linear
    @ModuleInfo(key: "conv1d") var conv: Conv1d
    @ModuleInfo(key: "norm") var gatedNorm: Qwen35GatedRMSNorm
    @ModuleInfo(key: "out_proj") var outProj: Linear

    var dt_bias: MLXArray
    var A_log: MLXArray

    init(_ config: Qwen35Configuration) {
        self.config = config
        self.scale = 1.0 / Float(config.linearKeyHeadDim).squareRoot()

        let h = config.hiddenSize
        self._inProjQKV.wrappedValue = Linear(h, config.convDim, bias: false)
        self._inProjZ.wrappedValue = Linear(h, config.valueDim, bias: false)
        self._inProjB.wrappedValue = Linear(h, config.linearValueHeads, bias: false)
        self._inProjA.wrappedValue = Linear(h, config.linearValueHeads, bias: false)

        self._conv.wrappedValue = Conv1d(
            inputChannels: config.convDim, outputChannels: config.convDim,
            kernelSize: config.convKernelSize, padding: 0,
            groups: config.convDim, bias: false)

        self._gatedNorm.wrappedValue = Qwen35GatedRMSNorm(
            dimensions: config.linearValueHeadDim, eps: config.normEps)
        self._outProj.wrappedValue = Linear(config.valueDim, h, bias: false)

        self.dt_bias = MLXArray.ones([config.linearValueHeads])
        self.A_log = MLXArray.zeros([config.linearValueHeads])
    }

    func callAsFunction(_ x: MLXArray, cache: DeltaNetCache?) -> MLXArray {
        let B = x.dim(0)
        let T = x.dim(1)
        let K = config.convKernelSize

        // Projections
        let mixedQKV = inProjQKV(x)
        let z = inProjZ(x)
        let b = inProjB(x)
        let a = inProjA(x)

        // Causal Conv1D: prepend state (or zeros), apply conv, take causal output
        let prefix: MLXArray
        if let existing = cache?.convState {
            prefix = existing
        } else {
            prefix = MLXArray.zeros([B, K, config.convDim])
        }
        let convInput = concatenated([prefix, mixedQKV], axis: 1)  // [B, K+T, C]
        cache?.convState = convInput[0..., (convInput.dim(1) - K)..., 0...]

        let rawConv = conv(convInput)                  // [B, T+1, C]
        let activated = silu(rawConv[0..., 1..., 0...])  // [B, T, C] skip warmup

        // Split Q, K, V
        let kd = config.keyDim
        let parts = activated.split(indices: [kd, 2 * kd], axis: -1)
        let query = parts[0].reshaped(B, T, config.linearKeyHeads, config.linearKeyHeadDim)
        let key = parts[1].reshaped(B, T, config.linearKeyHeads, config.linearKeyHeadDim)
        let value = parts[2].reshaped(B, T, config.linearValueHeads, config.linearValueHeadDim)

        // Gates
        let beta = sigmoid(b)
        let g = -MLX.exp(A_log) * softplus(a + dt_bias)
        let decay = MLX.exp(g)

        // Delta rule recurrence
        let (attnOut, newState) = recurrence(
            query: query, key: key, value: value,
            decay: decay, beta: beta, state: cache?.recurrentState)
        cache?.recurrentState = newState
        cache?.incrementOffset(by: T)

        // Gated output norm: applied per-head on the last dimension (dv)
        // attnOut: [B, T, H, dv], z: [B, T, H*dv]
        let numHeads = config.linearValueHeads
        let dv = config.linearValueHeadDim
        let flat = attnOut.reshaped(B * T * numHeads, dv)
        let zFlat = z.reshaped(B, T, numHeads, dv).reshaped(B * T * numHeads, dv)
        let gated = gatedNorm(flat, gate: zFlat).reshaped(B, T, config.valueDim)

        return outProj(gated)
    }

    private func recurrence(
        query: MLXArray, key: MLXArray, value: MLXArray,
        decay: MLXArray, beta: MLXArray, state: MLXArray?
    ) -> (MLXArray, MLXArray) {
        let B = query.dim(0), T = query.dim(1), H = query.dim(2)
        let dk = query.dim(3), dv = value.dim(3)

        let qN = l2Norm(query) * MLXArray(scale)
        let kN = l2Norm(key)

        var S = state ?? MLXArray.zeros([B, H, dk, dv])
        var outputs = [MLXArray]()

        for t in 0..<T {
            let qt = qN[0..., t..<(t + 1), 0..., 0...].squeezed(axis: 1)
            let kt = kN[0..., t..<(t + 1), 0..., 0...].squeezed(axis: 1)
            let vt = value[0..., t..<(t + 1), 0..., 0...].squeezed(axis: 1)
            let gt = decay[0..., t..<(t + 1), 0...].squeezed(axis: 1)
            let bt = beta[0..., t..<(t + 1), 0...].squeezed(axis: 1)

            let gE = gt.expandedDimensions(axes: [-1, -2])
            S = S * gE

            let kE = kt.expandedDimensions(axis: -1)
            let kvMem = (S * kE).sum(axis: -2)
            let delta = bt.expandedDimensions(axis: -1) * (vt - kvMem)
            S = S + kE * delta.expandedDimensions(axis: -2)

            let qE = qt.expandedDimensions(axis: -1)
            let ot = (S * qE).sum(axis: -2)
            outputs.append(ot.expandedDimensions(axis: 1))
        }

        return (concatenated(outputs, axis: 1), S)
    }

    private func l2Norm(_ x: MLXArray, eps: Float = 1e-6) -> MLXArray {
        x / MLX.sqrt((x * x).sum(axis: -1, keepDims: true) + MLXArray(eps))
    }
}

// MARK: - Full Attention

/// Full attention layer with sigmoid gate and QK normalization.
///
/// q_proj outputs 2x dim: first half = queries, second half = sigmoid gate.
/// Partial RoPE is applied only to the first `partialRotaryFactor * headDim` dimensions.
class Qwen35FullAttention: Module {

    let config: Qwen35Configuration
    let scale: Float

    @ModuleInfo(key: "q_proj") var wq: Linear
    @ModuleInfo(key: "k_proj") var wk: Linear
    @ModuleInfo(key: "v_proj") var wv: Linear
    @ModuleInfo(key: "o_proj") var wo: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    let rope: RoPELayer

    init(_ config: Qwen35Configuration) {
        self.config = config
        self.scale = pow(Float(config.headDim), -0.5)

        let h = config.hiddenSize
        self._wq.wrappedValue = Linear(h, config.attentionHeads * config.headDim * 2, bias: false)
        self._wk.wrappedValue = Linear(h, config.kvHeads * config.headDim, bias: false)
        self._wv.wrappedValue = Linear(h, config.kvHeads * config.headDim, bias: false)
        self._wo.wrappedValue = Linear(config.attentionHeads * config.headDim, h, bias: false)

        self._qNorm.wrappedValue = RMSNorm(dimensions: config.headDim, eps: config.normEps)
        self._kNorm.wrappedValue = RMSNorm(dimensions: config.headDim, eps: config.normEps)

        self.rope = initializeRope(
            dims: config.ropePartialDim, base: config.ropeTheta,
            traditional: config.ropeTraditional, scalingConfig: config.ropeScaling,
            maxPositionEmbeddings: config.maxPositionEmbeddings)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))

        let qFull = wq(x).reshaped(B, L, config.attentionHeads, config.headDim * 2)
        let queries = qFull[0..., 0..., 0..., 0..<config.headDim]
        let gate = qFull[0..., 0..., 0..., config.headDim...].reshaped(B, L, -1)

        var keys = wk(x).reshaped(B, L, config.kvHeads, config.headDim)
        var values = wv(x).reshaped(B, L, config.kvHeads, config.headDim)

        var q = qNorm(queries).transposed(0, 2, 1, 3)
        keys = kNorm(keys).transposed(0, 2, 1, 3)
        values = values.transposed(0, 2, 1, 3)

        let offset = cache?.offset ?? 0
        let rpd = config.ropePartialDim
        if rpd < config.headDim {
            let qRot = rope(q[0..., 0..., 0..., 0..<rpd], offset: offset)
            let kRot = rope(keys[0..., 0..., 0..., 0..<rpd], offset: offset)
            q = concatenated([qRot, q[0..., 0..., 0..., rpd...]], axis: -1)
            keys = concatenated([kRot, keys[0..., 0..., 0..., rpd...]], axis: -1)
        } else {
            q = rope(q, offset: offset)
            keys = rope(keys, offset: offset)
        }

        if let cache {
            let (ck, cv) = cache.update(keys: keys, values: values)
            keys = ck
            values = cv
        }

        let attnOut = MLXFast.scaledDotProductAttention(
            queries: q, keys: keys, values: values, scale: scale, mask: mask)

        var output = attnOut.transposed(0, 2, 1, 3).reshaped(B, L, -1)
        output = output * sigmoid(gate)
        return wo(output)
    }
}

// MARK: - MLP

class Qwen35MLP: Module {

    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "down_proj") var down: Linear
    @ModuleInfo(key: "up_proj") var up: Linear

    init(_ config: Qwen35Configuration) {
        self._gate.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        self._down.wrappedValue = Linear(config.intermediateSize, config.hiddenSize, bias: false)
        self._up.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        down(silu(gate(x)) * up(x))
    }
}

// MARK: - Decoder Layer

/// Routes to DeltaNet or Full Attention based on layer index.
class Qwen35DecoderLayer: Module {

    let isFullAttention: Bool

    @ModuleInfo(key: "self_attn") var fullAttn: Qwen35FullAttention?
    @ModuleInfo(key: "linear_attn") var deltaNet: Qwen35GatedDeltaNet?
    @ModuleInfo(key: "mlp") var mlp: Qwen35MLP
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    init(_ config: Qwen35Configuration, layerIndex: Int) {
        self.isFullAttention = config.isFullAttentionLayer(layerIndex)

        if isFullAttention {
            self._fullAttn.wrappedValue = Qwen35FullAttention(config)
        } else {
            self._deltaNet.wrappedValue = Qwen35GatedDeltaNet(config)
        }

        self._mlp.wrappedValue = Qwen35MLP(config)
        self._inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.normEps)
        self._postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.normEps)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let r: MLXArray
        if isFullAttention, let attn = fullAttn {
            r = attn(inputLayerNorm(x), mask: mask, cache: cache)
        } else if let dn = deltaNet {
            r = dn(inputLayerNorm(x), cache: cache as? DeltaNetCache)
        } else {
            fatalError("Layer has neither full attention nor DeltaNet")
        }
        let h = x + r
        return h + mlp(postAttentionLayerNorm(h))
    }
}

// MARK: - Inner Model

class Qwen35ModelInner: Module {

    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    let layers: [Qwen35DecoderLayer]
    let norm: RMSNorm
    let config: Qwen35Configuration

    init(_ config: Qwen35Configuration) {
        self.config = config
        precondition(config.vocabularySize > 0)
        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize, dimensions: config.hiddenSize)
        self.layers = (0..<config.hiddenLayers).map {
            Qwen35DecoderLayer(config, layerIndex: $0)
        }
        self.norm = RMSNorm(dimensions: config.hiddenSize, eps: config.normEps)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var h = embedTokens(inputs)
        for (i, layer) in layers.enumerated() {
            let mask: MLXFast.ScaledDotProductAttentionMaskMode
            if layer.isFullAttention && h.dim(1) > 1 {
                mask = .causal
            } else {
                mask = .none
            }
            h = layer(h, mask: mask, cache: cache?[i])
        }
        return norm(h)
    }
}

// MARK: - Top-Level Model

class Qwen35Model: Module, LanguageModel, KVCacheDimensionProvider {

    let vocabularySize: Int
    let kvHeads: [Int]
    var layerCount: Int { model.layers.count }

    let model: Qwen35ModelInner
    let configuration: Qwen35Configuration

    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    init(_ config: Qwen35Configuration) {
        self.configuration = config
        self.vocabularySize = config.vocabularySize
        self.kvHeads = (0..<config.hiddenLayers).map { i in
            config.isFullAttentionLayer(i) ? config.kvHeads : 0
        }
        self.model = Qwen35ModelInner(config)
        if !config.tieWordEmbeddings {
            self._lmHead.wrappedValue = Linear(
                config.hiddenSize, config.vocabularySize, bias: false)
        }
    }

    func callAsFunction(
        _ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?
    ) -> LMOutput {
        LMOutput(logits: forwardLogits(input.tokens, cache: cache))
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        forwardLogits(inputs, cache: cache)
    }

    private func forwardLogits(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let out = model(inputs, cache: cache)
        if let lmHead { return lmHead(out) }
        return model.embedTokens.asLinear(out)
    }

    /// Hybrid cache: DeltaNetCache for DeltaNet layers, KVCacheSimple for full attention.
    func newCache(parameters: GenerateParameters?) -> [KVCache] {
        let params = parameters ?? GenerateParameters()
        return (0..<configuration.hiddenLayers).map { i in
            if configuration.isFullAttentionLayer(i) {
                if let maxSize = params.maxKVSize {
                    return RotatingKVCache(maxSize: maxSize)
                }
                return KVCacheSimple()
            }
            return DeltaNetCache()
        }
    }

    func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var result = weights.filter { !$0.key.contains("rotary_emb.inv_freq") }
        for key in Array(result.keys) where key.contains("conv1d.weight") {
            if let w = result[key] {
                if w.ndim == 2 {
                    // GGUF dimensions are reversed by GGUFTensorBridge,
                    // so [4, 6144] in GGUF becomes [6144, 4] = [C, K].
                    // MLX Conv1d expects [C_out, K, C_in/groups] = [C, K, 1].
                    result[key] = w.expandedDimensions(axis: -1)
                } else if w.ndim == 3 {
                    // PyTorch [O, I/G, K] → MLX [O, K, I/G]
                    result[key] = w.transposed(0, 2, 1)
                }
            }
        }
        return result
    }
}

// MARK: - LoRA

extension Qwen35Model: LoRAModel {
    var loraLayers: [Module] { model.layers }
}
