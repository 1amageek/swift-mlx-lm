import GGUFParser
@preconcurrency import MLX
import MLXCompiler
import SwiftLM

// MARK: - DetectedArchitecture

/// Architecture variant detected from GGUF tensor name patterns.
///
/// Detection is based exclusively on tensor name patterns, never on
/// `general.architecture` metadata, to ensure robustness against
/// incorrect or missing metadata.
public enum DetectedArchitecture: Sendable, Equatable {

    /// Standard transformer (Llama, Qwen2, Mistral, Gemma, Phi, StarCoder2).
    /// Sequential residual blocks: [norm+attn] → [norm+mlp].
    case transformer

    /// Shared-norm parallel attention + MLP transformer.
    /// No separate FFN norm. QK norm weights present.
    case parallelAttentionMLP

    /// MoE transformer (Mixtral): standard transformer with expert FFN.
    /// Has `ffn_gate_inp` (router) and per-expert `ffn_gate.{e}` tensors.
    case moe

    /// Hybrid DeltaNet / full-attention decoder.
    /// Has DeltaNet tensors (ssm_beta, ssm_alpha, ssm_conv1d) on some layers.
    case hybridDeltaNetAttention
}

// MARK: - GGUFArchitectureDetector

/// Detects architecture from GGUF tensor name patterns.
///
/// Each pattern check is a sufficient condition. Detection order
/// (most specific first):
/// 1. hybridDeltaNetAttention — DeltaNet tensors
/// 2. parallelAttentionMLP — QK norm + no FFN norm
/// 3. moe — expert tensors
/// 4. transformer — universal fallback
public struct GGUFArchitectureDetector: Sendable {

    public init() {}

    /// Detect architecture from tensor name patterns in the GGUF file.
    public func detect(file: GGUFFile) -> DetectedArchitecture {
        let names = Set(file.tensors.map(\.name))
        return detect(tensorNames: names)
    }

    /// Detect architecture from a set of tensor names.
    ///
    /// Exposed for testing with synthetic tensor name sets.
    public func detect(tensorNames names: Set<String>) -> DetectedArchitecture {
        // Priority 1: hybrid DeltaNet / full-attention decoder
        if names.contains("blk.0.ssm_beta.weight") {
            return .hybridDeltaNetAttention
        }

        // Priority 2: shared-norm parallel attention + MLP
        if names.contains("blk.0.attn_q_norm.weight")
            && !names.contains("blk.0.ffn_norm.weight")
        {
            return .parallelAttentionMLP
        }

        // Priority 3: MoE (expert routing gate)
        if names.contains("blk.0.ffn_gate_inp.weight") {
            return .moe
        }

        // Fallback: standard transformer
        return .transformer
    }
}

// MARK: - GGUFModelConfig

/// All configuration values extracted from GGUF metadata for graph construction.
///
/// Architecture-agnostic container. The builder uses only the fields
/// relevant to the detected architecture.
public struct GGUFModelConfig: Sendable {

    // MARK: Core Dimensions

    public let hiddenSize: Int
    public let layerCount: Int
    public let intermediateSize: Int
    public let vocabSize: Int

    // MARK: Attention

    public let attentionHeads: Int
    public let kvHeads: Int
    public let headDim: Int
    public let attentionBias: Bool
    public let mlpBias: Bool

    // MARK: Normalization

    public let normEps: Float
    public let normKind: NormKind

    // MARK: RoPE

    public let ropeTheta: Float
    public let ropeDimension: Int
    public let ropeScaling: RoPEScaling?

    // MARK: Output

    public let tiedEmbeddings: Bool

    // MARK: MoE (optional)

    public let expertCount: Int?
    public let expertsPerToken: Int?

    // MARK: Shared-Norm Parallel Attention/MLP (optional)

    public let qkNorm: Bool

    // MARK: Hybrid State-Space / Attention (optional)

    public let fullAttentionInterval: Int?
    public let linearKeyHeads: Int?
    public let linearValueHeads: Int?
    public let linearKeyHeadDim: Int?
    public let linearValueHeadDim: Int?
    public let convKernelSize: Int?
    public let partialRotaryFactor: Float?

    // MARK: Sliding Window

    public let slidingWindow: Int?

    // MARK: M-RoPE (VLM only)

    /// Multi-axis RoPE configuration for VLM. Nil for text-only models.
    public let mropeAxes: MRoPEAxes?

    /// Normalization layer kind.
    public enum NormKind: Sendable, Equatable {
        case rmsNorm
        case layerNorm
    }

    /// Return a copy with M-RoPE axes set.
    func withMRoPEAxes(_ axes: MRoPEAxes?) -> GGUFModelConfig {
        GGUFModelConfig(
            hiddenSize: hiddenSize, layerCount: layerCount,
            intermediateSize: intermediateSize, vocabSize: vocabSize,
            attentionHeads: attentionHeads, kvHeads: kvHeads, headDim: headDim,
            attentionBias: attentionBias, mlpBias: mlpBias,
            normEps: normEps, normKind: normKind,
            ropeTheta: ropeTheta, ropeDimension: ropeDimension,
            ropeScaling: ropeScaling, tiedEmbeddings: tiedEmbeddings,
            expertCount: expertCount, expertsPerToken: expertsPerToken,
            qkNorm: qkNorm,
            fullAttentionInterval: fullAttentionInterval,
            linearKeyHeads: linearKeyHeads, linearValueHeads: linearValueHeads,
            linearKeyHeadDim: linearKeyHeadDim, linearValueHeadDim: linearValueHeadDim,
            convKernelSize: convKernelSize, partialRotaryFactor: partialRotaryFactor,
            slidingWindow: slidingWindow, mropeAxes: axes
        )
    }
}

// MARK: - GGUFConfigExtractor

/// Extracts model configuration from GGUF metadata and tensor patterns.
public struct GGUFConfigExtractor: Sendable {

    public init() {}

    /// Extract configuration from GGUF file.
    public func extract(
        from file: GGUFFile,
        architecture: DetectedArchitecture
    ) throws -> GGUFModelConfig {
        let hiddenSize = try require(file.embeddingLength, "embedding_length")
        let layerCount = try require(file.blockCount, "block_count")
        let headCount = try require(file.headCount, "head_count")
        let kvHeads = file.headCountKV ?? headCount
        let headDim = file.headDimension ?? (hiddenSize / headCount)
        let intermediateSize = try require(file.feedForwardLength, "feed_forward_length")
        let vocabSize = file.vocabularyLength ?? file.vocabularySize ?? 0

        let normEps = file.attentionLayerNormRMSEpsilon
            ?? file.attentionLayerNormEpsilon
            ?? 1e-5

        let ropeTheta = file.ropeFreqBase ?? 10000.0
        let ropeDim = file.ropeDimensionCount ?? headDim

        // Bias detection from tensor names
        let tensorNames = Set(file.tensors.map(\.name))
        let attentionBias = tensorNames.contains("blk.0.attn_q.bias")
        let mlpBias = tensorNames.contains("blk.0.ffn_gate.bias")
            || tensorNames.contains("blk.0.ffn_up.bias")

        // Tied embeddings: output.weight absent means tied to token_embd.weight
        let tiedEmbeddings = !tensorNames.contains("output.weight")

        // Norm kind
        let normKind: GGUFModelConfig.NormKind
        switch architecture {
        case .parallelAttentionMLP:
            normKind = .layerNorm
        default:
            normKind = .rmsNorm
        }

        // QK norm
        let qkNorm = tensorNames.contains("blk.0.attn_q_norm.weight")

        // RoPE scaling
        let ropeScaling = extractRoPEScaling(from: file)

        // MoE
        let expertCount = file.expertCount
        let expertsPerToken = file.expertUsedCount

        // Hybrid state-space / attention
        let fullAttentionInterval = file.fullAttentionInterval
        let linearKeyHeads = file.linearKeyHeadCount
        let linearValueHeads = file.linearValueHeadCount
        let linearKeyHeadDim = file.linearKeyHeadDim
        let linearValueHeadDim = file.linearValueHeadDim
        let convKernelSize = file.linearConvKernelSize
        let partialRotaryFactor: Float?
        switch architecture {
        case .hybridDeltaNetAttention:
            partialRotaryFactor = try require(
                file.partialRotaryFactor, "rope.partial_rotary_factor", in: file)
        default:
            partialRotaryFactor = file.partialRotaryFactor
        }

        // Sliding window
        let slidingWindow = file.slidingWindow

        return GGUFModelConfig(
            hiddenSize: hiddenSize,
            layerCount: layerCount,
            intermediateSize: intermediateSize,
            vocabSize: vocabSize,
            attentionHeads: headCount,
            kvHeads: kvHeads,
            headDim: headDim,
            attentionBias: attentionBias,
            mlpBias: mlpBias,
            normEps: normEps,
            normKind: normKind,
            ropeTheta: ropeTheta,
            ropeDimension: ropeDim,
            ropeScaling: ropeScaling,
            tiedEmbeddings: tiedEmbeddings,
            expertCount: expertCount,
            expertsPerToken: expertsPerToken,
            qkNorm: qkNorm,
            fullAttentionInterval: fullAttentionInterval,
            linearKeyHeads: linearKeyHeads,
            linearValueHeads: linearValueHeads,
            linearKeyHeadDim: linearKeyHeadDim,
            linearValueHeadDim: linearValueHeadDim,
            convKernelSize: convKernelSize,
            partialRotaryFactor: partialRotaryFactor,
            slidingWindow: slidingWindow,
            mropeAxes: nil
        )
    }

    // MARK: - Private

    private func require<T>(_ value: T?, _ name: String, in file: GGUFFile? = nil) throws -> T {
        guard let v = value else {
            let message = file.map { GGUFMetadataDiagnostics.missingMetadataMessage(name, in: $0) } ?? name
            throw GGUFGraphBuildError.missingMetadata(message)
        }
        return v
    }

    private func extractRoPEScaling(from file: GGUFFile) -> RoPEScaling? {
        guard let typeStr = file.ropeScalingType else { return nil }
        let factor = file.ropeScalingFactor ?? 1.0

        let kind: RoPEScalingKind
        switch typeStr.lowercased() {
        case "linear":
            kind = .linear
        case "dynamic":
            kind = .dynamic
        case "yarn":
            kind = .yarn
        case "su", "longrope":
            kind = .custom("su")
        default:
            return nil
        }

        let origMaxPos = file.ropeScalingOriginalMaxPositionEmbeddings
        return RoPEScaling(kind: kind, factor: factor, originalMaxPositions: origMaxPos)
    }
}

// MARK: - IRGraphAssembler

/// Assembles a ModelGraph from config and architecture.
///
/// Pure function: (GGUFModelConfig, DetectedArchitecture) → ModelGraph.
/// No GGUF or weight knowledge. Directly constructs SSA IR.
public struct IRGraphAssembler: Sendable {

    public init() {}

    /// Assemble a ModelGraph for the given architecture and config.
    public func assemble(
        config: GGUFModelConfig,
        architecture: DetectedArchitecture
    ) throws -> ModelGraph {
        var ctx = IRContext()

        let rootRegion: Region
        switch architecture {
        case .transformer:
            rootRegion = assembleTransformer(config: config, ctx: &ctx)
        case .parallelAttentionMLP:
            rootRegion = assembleParallelAttentionMLP(config: config, ctx: &ctx)
        case .moe:
            rootRegion = try assembleMoE(config: config, ctx: &ctx)
        case .hybridDeltaNetAttention:
            rootRegion = assembleHybridDeltaNetAttention(config: config, ctx: &ctx)
        }

        return ModelGraph(rootRegion: rootRegion)
    }

}

// MARK: - IRContext (SSA Counter)

/// Internal SSA counter for ModelGraph construction.
struct IRContext {
    private var nextVal = 0
    private var nextKey = 0

    mutating func freshValue() -> ValueID {
        defer { nextVal += 1 }
        return ValueID(rawValue: nextVal)
    }

    mutating func freshKey() -> OperationKey {
        defer { nextKey += 1 }
        return OperationKey(rawValue: nextKey)
    }

    /// Create a primitive operation that consumes operands and produces one result.
    mutating func makePrimitive(
        kind: OperationKind,
        operands: [ValueID]
    ) -> (Operation, ValueID) {
        let resultID = freshValue()
        let op = Operation(
            key: freshKey(),
            kind: kind,
            operands: operands.map { Operand(value: $0) },
            results: [OperationResult(id: resultID)]
        )
        return (op, resultID)
    }

    /// Create a source operation (no operands, one result).
    mutating func makeSource(kind: OperationKind) -> (Operation, ValueID) {
        return makePrimitive(kind: kind, operands: [])
    }

    /// Create a region parameter and return the ValueID it introduces.
    mutating func makeParam() -> (RegionParameter, ValueID) {
        let id = freshValue()
        return (RegionParameter(id: id), id)
    }
}

// MARK: - IRGraphAssembler: Architecture Builders

private extension IRGraphAssembler {

    // MARK: Shared Post-Embedding Assembly

    /// Build the decoder + final norm + output head.
    ///
    /// Used by both text-only and VLM graph assembly to avoid duplication.
    func assemblePostEmbedding(
        config: GGUFModelConfig,
        architecture: DetectedArchitecture,
        inputVal: ValueID,
        ctx: inout IRContext
    ) throws -> (operations: [Operation], resultVal: ValueID) {
        var ops: [Operation] = []

        // Decoder layers
        let decoderVal: ValueID
        switch architecture {
        case .transformer:
            let body = makeTransformerDecoderBlock(config: config, ctx: &ctx)
            let (op, val) = ctx.makePrimitive(
                kind: .repeating(count: config.layerCount, body: body),
                operands: [inputVal])
            ops.append(op)
            decoderVal = val

        case .parallelAttentionMLP:
            let body = makeParallelAttentionMLPDecoderBlock(config: config, ctx: &ctx)
            let (op, val) = ctx.makePrimitive(
                kind: .repeating(count: config.layerCount, body: body),
                operands: [inputVal])
            ops.append(op)
            decoderVal = val

        case .moe:
            let body = try makeMoEDecoderBlock(config: config, ctx: &ctx)
            let (op, val) = ctx.makePrimitive(
                kind: .repeating(count: config.layerCount, body: body),
                operands: [inputVal])
            ops.append(op)
            decoderVal = val

        case .hybridDeltaNetAttention:
            let interval = config.fullAttentionInterval ?? 4
            var layers: [Region] = []
            for i in 0..<config.layerCount {
                let isFullAttention = (i + 1) % interval == 0
                if isFullAttention {
                    layers.append(makeHybridDeltaNetAttentionFullAttnBlock(config: config, ctx: &ctx))
                } else {
                    layers.append(makeHybridDeltaNetAttentionStateSpaceBlock(config: config, ctx: &ctx))
                }
            }
            let (op, val) = ctx.makePrimitive(
                kind: .layerStack(layers: layers),
                operands: [inputVal])
            ops.append(op)
            decoderVal = val
        }

        // Final norm
        let normKind: OperationKind
        switch architecture {
        case .hybridDeltaNetAttention:
            normKind = .rmsNorm(RMSNormAttributes(dimension: config.hiddenSize, epsilon: config.normEps))
        default:
            normKind = makeNorm(config: config)
        }
        let (normOp, normVal) = ctx.makePrimitive(kind: normKind, operands: [decoderVal])
        ops.append(normOp)

        // Output head
        let (headOp, headVal) = ctx.makePrimitive(
            kind: .outputHead(OutputHeadAttributes(
                inputSize: config.hiddenSize,
                vocabSize: config.vocabSize,
                tiedToEmbedding: config.tiedEmbeddings,
                bias: false
            )),
            operands: [normVal]
        )
        ops.append(headOp)

        return (operations: ops, resultVal: headVal)
    }

    // MARK: Standard Transformer

    func assembleTransformer(config: GGUFModelConfig, ctx: inout IRContext) -> Region {
        // Root region: embed → repeating(N, decoderBlock) → norm → outputHead
        var ops: [Operation] = []

        let (embedOp, embedVal) = ctx.makeSource(kind: .tokenEmbedding(
            TokenEmbeddingAttributes(vocabSize: config.vocabSize, embeddingSize: config.hiddenSize)
        ))
        ops.append(embedOp)

        // Safe to force-try: transformer assembly does not throw
        let (postOps, resultVal) = try! assemblePostEmbedding(
            config: config, architecture: .transformer, inputVal: embedVal, ctx: &ctx)
        ops.append(contentsOf: postOps)

        return Region(
            parameters: [],
            operations: ops,
            results: [ValueUse(value: resultVal)]
        )
    }

    // MARK: Shared-Norm Parallel Attention/MLP

    func assembleParallelAttentionMLP(config: GGUFModelConfig, ctx: inout IRContext) -> Region {
        var ops: [Operation] = []

        let (embedOp, embedVal) = ctx.makeSource(kind: .tokenEmbedding(
            TokenEmbeddingAttributes(vocabSize: config.vocabSize, embeddingSize: config.hiddenSize)
        ))
        ops.append(embedOp)

        // Safe to force-try: shared-norm parallel assembly does not throw
        let (postOps, resultVal) = try! assemblePostEmbedding(
            config: config, architecture: .parallelAttentionMLP, inputVal: embedVal, ctx: &ctx)
        ops.append(contentsOf: postOps)

        return Region(
            parameters: [],
            operations: ops,
            results: [ValueUse(value: resultVal)]
        )
    }

    // MARK: MoE

    func assembleMoE(config: GGUFModelConfig, ctx: inout IRContext) throws -> Region {
        var ops: [Operation] = []

        let (embedOp, embedVal) = ctx.makeSource(kind: .tokenEmbedding(
            TokenEmbeddingAttributes(vocabSize: config.vocabSize, embeddingSize: config.hiddenSize)
        ))
        ops.append(embedOp)

        let (postOps, resultVal) = try assemblePostEmbedding(
            config: config, architecture: .moe, inputVal: embedVal, ctx: &ctx)
        ops.append(contentsOf: postOps)

        return Region(
            parameters: [],
            operations: ops,
            results: [ValueUse(value: resultVal)]
        )
    }

    // MARK: Hybrid State-Space / Attention

    func assembleHybridDeltaNetAttention(config: GGUFModelConfig, ctx: inout IRContext) -> Region {
        var ops: [Operation] = []

        let (embedOp, embedVal) = ctx.makeSource(kind: .tokenEmbedding(
            TokenEmbeddingAttributes(vocabSize: config.vocabSize, embeddingSize: config.hiddenSize)
        ))
        ops.append(embedOp)

        // Safe to force-try: hybridDeltaNetAttention assembly does not throw
        let (postOps, resultVal) = try! assemblePostEmbedding(
            config: config, architecture: .hybridDeltaNetAttention, inputVal: embedVal, ctx: &ctx)
        ops.append(contentsOf: postOps)

        return Region(
            parameters: [],
            operations: ops,
            results: [ValueUse(value: resultVal)]
        )
    }
}

// MARK: - IRGraphAssembler: Block Builders

private extension IRGraphAssembler {

    // MARK: Shared Helpers

    func makeNorm(config: GGUFModelConfig) -> OperationKind {
        switch config.normKind {
        case .rmsNorm:
            return .rmsNorm(RMSNormAttributes(dimension: config.hiddenSize, epsilon: config.normEps))
        case .layerNorm:
            return .layerNorm(LayerNormAttributes(dimension: config.hiddenSize, epsilon: config.normEps, affine: true))
        }
    }

    func makeAttentionAttrs(config: GGUFModelConfig) -> AttentionAttributes {
        AttentionAttributes(
            hiddenSize: config.hiddenSize,
            headCount: config.attentionHeads,
            kvHeadCount: config.kvHeads,
            headDimension: config.headDim,
            bias: config.attentionBias,
            causal: true,
            rope: RoPEAttributes(
                dimension: config.ropeDimension,
                base: config.ropeTheta,
                scaling: config.ropeScaling,
                mropeAxes: config.mropeAxes
            ),
            qkNorm: config.qkNorm ? .rmsNorm : nil
        )
    }

    func makeMLPAttrs(config: GGUFModelConfig) -> MLPAttributes {
        MLPAttributes(
            inputSize: config.hiddenSize,
            outputSize: config.hiddenSize,
            intermediateSize: config.intermediateSize,
            activation: .silu,
            gating: .swiglu,
            bias: config.mlpBias
        )
    }

    func makeMoEAttrs(config: GGUFModelConfig) throws -> MoEAttributes {
        guard let expertCount = config.expertCount else {
            throw GGUFGraphBuildError.missingMetadata("expert_count")
        }
        guard let expertsPerToken = config.expertsPerToken else {
            throw GGUFGraphBuildError.missingMetadata("expert_used_count")
        }
        return MoEAttributes(
            expertCount: expertCount,
            expertsPerToken: expertsPerToken,
            gateKind: .topK,
            expertMLP: makeMLPAttrs(config: config)
        )
    }

    /// Build a residual block: norm → body operation, wrapped in residual(.add).
    func makeResidualBlock(
        normKind: OperationKind,
        bodyKind: OperationKind,
        ctx: inout IRContext
    ) -> Region {
        let (param, paramVal) = ctx.makeParam()
        var bodyOps: [Operation] = []

        let (normOp, normVal) = ctx.makePrimitive(kind: normKind, operands: [paramVal])
        bodyOps.append(normOp)

        let (bodyOp, bodyVal) = ctx.makePrimitive(kind: bodyKind, operands: [normVal])
        bodyOps.append(bodyOp)

        return Region(
            parameters: [param],
            operations: bodyOps,
            results: [ValueUse(value: bodyVal)]
        )
    }

    // MARK: Standard Transformer Decoder Block

    func makeTransformerDecoderBlock(config: GGUFModelConfig, ctx: inout IRContext) -> Region {
        let (param, paramVal) = ctx.makeParam()
        var ops: [Operation] = []

        // Residual 1: norm + attention
        let attnBody = makeResidualBlock(
            normKind: makeNorm(config: config),
            bodyKind: .attention(makeAttentionAttrs(config: config)),
            ctx: &ctx
        )
        let (attnResidual, attnVal) = ctx.makePrimitive(
            kind: .residual(strategy: .add, body: attnBody),
            operands: [paramVal]
        )
        ops.append(attnResidual)

        // Residual 2: norm + mlp
        let mlpBody = makeResidualBlock(
            normKind: makeNorm(config: config),
            bodyKind: .mlp(makeMLPAttrs(config: config)),
            ctx: &ctx
        )
        let (mlpResidual, mlpVal) = ctx.makePrimitive(
            kind: .residual(strategy: .add, body: mlpBody),
            operands: [attnVal]
        )
        ops.append(mlpResidual)

        return Region(
            parameters: [param],
            operations: ops,
            results: [ValueUse(value: mlpVal)]
        )
    }

    // MARK: Shared-Norm Parallel Decoder Block

    func makeParallelAttentionMLPDecoderBlock(config: GGUFModelConfig, ctx: inout IRContext) -> Region {
        // Shared LayerNorm → Parallel(Attention, MLP) → residual add.
        let (param, paramVal) = ctx.makeParam()
        var ops: [Operation] = []

        let residualBody = makeParallelAttentionMLPResidualBody(config: config, ctx: &ctx)
        let (residualOp, residualVal) = ctx.makePrimitive(
            kind: .residual(strategy: .add, body: residualBody),
            operands: [paramVal]
        )
        ops.append(residualOp)

        return Region(
            parameters: [param],
            operations: ops,
            results: [ValueUse(value: residualVal)]
        )
    }

    /// Shared-norm residual body: LayerNorm → Parallel(.add, [Attention, MLP]).
    func makeParallelAttentionMLPResidualBody(config: GGUFModelConfig, ctx: inout IRContext) -> Region {
        let (param, paramVal) = ctx.makeParam()
        var ops: [Operation] = []

        // Shared LayerNorm
        let (normOp, normVal) = ctx.makePrimitive(
            kind: makeNorm(config: config),
            operands: [paramVal]
        )
        ops.append(normOp)

        // Attention branch
        let attnBranch: Region = {
            let (bp, bpVal) = ctx.makeParam()
            let (attnOp, attnVal) = ctx.makePrimitive(
                kind: .attention(makeParallelAttentionMLPAttentionAttrs(config: config)),
                operands: [bpVal]
            )
            return Region(
                parameters: [bp],
                operations: [attnOp],
                results: [ValueUse(value: attnVal)]
            )
        }()

        // MLP branch
        let mlpBranch: Region = {
            let (bp, bpVal) = ctx.makeParam()
            let (mlpOp, mlpVal) = ctx.makePrimitive(
                kind: .mlp(makeMLPAttrs(config: config)),
                operands: [bpVal]
            )
            return Region(
                parameters: [bp],
                operations: [mlpOp],
                results: [ValueUse(value: mlpVal)]
            )
        }()

        // Parallel merge: attention + MLP
        let (parallelOp, parallelVal) = ctx.makePrimitive(
            kind: .parallel(merge: .add, branches: [attnBranch, mlpBranch]),
            operands: [normVal]
        )
        ops.append(parallelOp)

        return Region(
            parameters: [param],
            operations: ops,
            results: [ValueUse(value: parallelVal)]
        )
    }

    /// Shared-norm parallel attention attributes: QK norm uses layerNorm.
    func makeParallelAttentionMLPAttentionAttrs(config: GGUFModelConfig) -> AttentionAttributes {
        AttentionAttributes(
            hiddenSize: config.hiddenSize,
            headCount: config.attentionHeads,
            kvHeadCount: config.kvHeads,
            headDimension: config.headDim,
            bias: config.attentionBias,
            causal: true,
            rope: RoPEAttributes(
                dimension: config.ropeDimension,
                base: config.ropeTheta,
                scaling: config.ropeScaling,
                mropeAxes: config.mropeAxes
            ),
            qkNorm: config.qkNorm ? .layerNorm : nil
        )
    }

    // MARK: MoE Decoder Block

    func makeMoEDecoderBlock(config: GGUFModelConfig, ctx: inout IRContext) throws -> Region {
        let (param, paramVal) = ctx.makeParam()
        var ops: [Operation] = []

        // Residual 1: norm + attention
        let attnBody = makeResidualBlock(
            normKind: makeNorm(config: config),
            bodyKind: .attention(makeAttentionAttrs(config: config)),
            ctx: &ctx
        )
        let (attnResidual, attnVal) = ctx.makePrimitive(
            kind: .residual(strategy: .add, body: attnBody),
            operands: [paramVal]
        )
        ops.append(attnResidual)

        // Residual 2: norm + moe
        let moeBody = makeResidualBlock(
            normKind: makeNorm(config: config),
            bodyKind: .moe(try makeMoEAttrs(config: config)),
            ctx: &ctx
        )
        let (moeResidual, moeVal) = ctx.makePrimitive(
            kind: .residual(strategy: .add, body: moeBody),
            operands: [attnVal]
        )
        ops.append(moeResidual)

        return Region(
            parameters: [param],
            operations: ops,
            results: [ValueUse(value: moeVal)]
        )
    }

    // MARK: Hybrid State-Space / Attention Blocks

    func makeHybridDeltaNetAttentionFullAttnBlock(config: GGUFModelConfig, ctx: inout IRContext) -> Region {
        let ropePartialDim: Int
        if let factor = config.partialRotaryFactor {
            ropePartialDim = Int(Float(config.headDim) * factor)
        } else {
            ropePartialDim = config.headDim
        }

        let attnAttrs = AttentionAttributes(
            hiddenSize: config.hiddenSize,
            headCount: config.attentionHeads,
            kvHeadCount: config.kvHeads,
            headDimension: config.headDim,
            bias: config.attentionBias,
            causal: true,
            rope: RoPEAttributes(
                dimension: ropePartialDim,
                base: config.ropeTheta,
                scaling: config.ropeScaling,
                mropeAxes: config.mropeAxes
            ),
            qkNorm: .rmsNorm,
            outputGate: .sigmoidPackedInQProj
        )

        let (param, paramVal) = ctx.makeParam()
        var ops: [Operation] = []

        let attnBody = makeResidualBlock(
            normKind: .rmsNorm(RMSNormAttributes(dimension: config.hiddenSize, epsilon: config.normEps)),
            bodyKind: .attention(attnAttrs),
            ctx: &ctx
        )
        let (attnResidual, attnVal) = ctx.makePrimitive(
            kind: .residual(strategy: .add, body: attnBody),
            operands: [paramVal]
        )
        ops.append(attnResidual)

        let mlpBody = makeResidualBlock(
            normKind: .rmsNorm(RMSNormAttributes(dimension: config.hiddenSize, epsilon: config.normEps)),
            bodyKind: .mlp(makeMLPAttrs(config: config)),
            ctx: &ctx
        )
        let (mlpResidual, mlpVal) = ctx.makePrimitive(
            kind: .residual(strategy: .add, body: mlpBody),
            operands: [attnVal]
        )
        ops.append(mlpResidual)

        return Region(
            parameters: [param],
            operations: ops,
            results: [ValueUse(value: mlpVal)]
        )
    }

    func makeHybridDeltaNetAttentionStateSpaceBlock(config: GGUFModelConfig, ctx: inout IRContext) -> Region {
        let ssAttrs = StateSpaceAttributes(
            hiddenSize: config.hiddenSize,
            stateSize: config.linearKeyHeadDim ?? config.headDim,
            variant: DeltaNet.Variant.gated.rawValue
        )

        let (param, paramVal) = ctx.makeParam()
        var ops: [Operation] = []

        let ssBody = makeResidualBlock(
            normKind: .rmsNorm(RMSNormAttributes(dimension: config.hiddenSize, epsilon: config.normEps)),
            bodyKind: .stateSpace(ssAttrs),
            ctx: &ctx
        )
        let (ssResidual, ssVal) = ctx.makePrimitive(
            kind: .residual(strategy: .add, body: ssBody),
            operands: [paramVal]
        )
        ops.append(ssResidual)

        let mlpBody = makeResidualBlock(
            normKind: .rmsNorm(RMSNormAttributes(dimension: config.hiddenSize, epsilon: config.normEps)),
            bodyKind: .mlp(makeMLPAttrs(config: config)),
            ctx: &ctx
        )
        let (mlpResidual, mlpVal) = ctx.makePrimitive(
            kind: .residual(strategy: .add, body: mlpBody),
            operands: [ssVal]
        )
        ops.append(mlpResidual)

        return Region(
            parameters: [param],
            operations: ops,
            results: [ValueUse(value: mlpVal)]
        )
    }
}

// MARK: - GGUFGraphBuilder

/// Builds ModelGraph IR directly from a GGUF file.
///
/// Replaces the DSL-based `CompiledModelEntry` approach:
/// - Detects architecture from tensor name patterns (not metadata)
/// - Reads configuration from GGUF metadata
/// - Constructs ModelGraph IR directly (no DSL intermediate)
///
/// Usage:
/// ```swift
/// let builder = GGUFGraphBuilder()
/// let result = try builder.build(file: ggufFile)
/// // result.graph → ModelGraph ready for MLXInferenceCompiler
/// // result.architecture → for selecting GGUFTensorNameMapper
/// ```
public struct GGUFGraphBuilder: Sendable {

    public init() {}

    /// Result of building from a GGUF file.
    public struct BuildResult: Sendable {

        /// The constructed ModelGraph IR.
        public let graph: ModelGraph

        /// Detected architecture.
        public let architecture: DetectedArchitecture

        /// Extracted configuration.
        public let config: GGUFModelConfig
    }

    /// Build a text decoder ModelGraph from a GGUF file.
    ///
    /// For VLM models, pass `mmprojFile` to resolve M-RoPE sections.
    /// The resulting graph is the same text decoder IR as text-only models —
    /// the vision encoder is loaded separately by `GGUFVisionLoader`.
    public func build(
        file: GGUFFile,
        mmprojFile: GGUFFile? = nil
    ) throws -> BuildResult {
        let detector = GGUFArchitectureDetector()
        let architecture = detector.detect(file: file)

        let extractor = GGUFConfigExtractor()
        var config = try extractor.extract(from: file, architecture: architecture)

        // Resolve M-RoPE for VLM (from text GGUF metadata)
        if mmprojFile != nil {
            let mropeAxes = resolveMRoPEAxes(
                file: file, config: config, architecture: architecture)
            config = config.withMRoPEAxes(mropeAxes)
        }

        let assembler = IRGraphAssembler()
        let graph = try assembler.assemble(config: config, architecture: architecture)

        return BuildResult(
            graph: graph,
            architecture: architecture,
            config: config
        )
    }

    // MARK: - Private Helpers

    /// Resolve M-RoPE axes for VLM from GGUF metadata.
    ///
    /// Returns `nil` if `rope.scaling.sections` is not present in the GGUF file,
    /// meaning the model does not use M-RoPE. No architecture-specific heuristics
    /// are applied — the sections must come from metadata.
    private func resolveMRoPEAxes(
        file: GGUFFile,
        config: GGUFModelConfig,
        architecture: DetectedArchitecture
    ) -> MRoPEAxes? {
        guard let sections: [Int] = file[.ropeScalingSections], !sections.isEmpty else {
            return nil
        }
        let interleaved = architecture == .hybridDeltaNetAttention
        return MRoPEAxes(sections: sections, interleaved: interleaved)
    }

    /// Return the appropriate GGUFTensorNameMapper for a detected architecture.
    public func mapper(for architecture: DetectedArchitecture) -> any GGUFTensorNameMapper {
        switch architecture {
        case .transformer:
            return TransformerTensorNameMapper()
        case .parallelAttentionMLP:
            return ParallelAttentionMLPTensorNameMapper()
        case .moe:
            return MoETensorNameMapper()
        case .hybridDeltaNetAttention:
            return HybridDeltaNetAttentionTensorNameMapper()
        }
    }

    /// Sanitize raw weights after GGUF → TensorData conversion.
    ///
    /// Applies tensor-pattern-driven transforms (no architecture switch):
    /// - Removes `rotary_emb.inv_freq` (unused at inference time)
    /// - Reshapes `conv1d.weight` for MLX layout (if present)
    ///
    /// Safe to apply unconditionally — operations are no-ops when
    /// the relevant tensor patterns are absent.
    public static let sanitizeWeights: @Sendable ([String: TensorData]) -> [String: TensorData] = { weights in
        var result = weights.filter { !$0.key.contains("rotary_emb.inv_freq") }
        for key in Array(result.keys) where key.contains("conv1d.weight") {
            guard let td = result[key],
                  let storage = td.storage as? MLXTensorStorage
            else { continue }

            let array: MLXArray
            switch storage {
            case .dense(let a):
                array = a
            case .affineQuantized(let qt):
                array = dequantized(
                    qt.packedWeight, scales: qt.scales, biases: qt.zeroBiases,
                    groupSize: qt.groupSize, bits: qt.bits
                )
            }

            let reshaped: MLXArray
            if array.ndim == 2 {
                reshaped = array.expandedDimensions(axis: -1)
            } else if array.ndim == 3 {
                reshaped = array.transposed(0, 2, 1)
            } else {
                continue
            }

            let newShape = (0..<reshaped.ndim).map { reshaped.dim($0) }
            result[key] = TensorData(
                shape: newShape,
                dtype: td.dtype,
                storage: MLXTensorStorage.dense(reshaped)
            )
        }
        return result
    }
}

// MARK: - GGUFGraphBuildError

/// Errors encountered during GGUF → ModelGraph construction.
public enum GGUFGraphBuildError: Error, Sendable, CustomStringConvertible {

    /// A required GGUF metadata key is missing.
    case missingMetadata(String)

    /// Configuration values are invalid or inconsistent.
    case invalidConfig(String)

    public var description: String {
        switch self {
        case .missingMetadata(let key):
            return "Missing required GGUF metadata: \(key)"
        case .invalidConfig(let msg):
            return "Invalid model configuration: \(msg)"
        }
    }
}
