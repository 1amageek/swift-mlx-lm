import SwiftLM

// MARK: - IRGraphAssembler

/// Assembles a ModelGraph from config and architecture.
///
/// Pure function: (ModelConfig, DetectedArchitecture) → ModelGraph.
/// No GGUF or weight knowledge. Directly constructs SSA IR.
public struct IRGraphAssembler: Sendable {

    public init() {}

    /// Assemble a ModelGraph for the given architecture and config.
    public func assemble(
        config: ModelConfig,
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
            rootRegion = try assembleHybridDeltaNetAttention(config: config, ctx: &ctx)
        case .hybridConvAttention:
            rootRegion = try assembleHybridConvAttention(config: config, ctx: &ctx)
        }

        let graph = ModelGraph(rootRegion: rootRegion)
        try GraphValidator.validate(graph)
        try DimensionValidator.validate(graph)
        return graph
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

// MARK: - Layer Schedule Resolution

extension IRGraphAssembler {

    /// Resolve per-layer full-attention schedule from config.
    ///
    /// Uses `layerTypes` if available (explicit per-layer schedule from config.json),
    /// otherwise falls back to `fullAttentionInterval` (GGUF metadata).
    /// Throws if neither is available.
    func resolveLayerSchedule(config: ModelConfig) throws -> [Bool] {
        if let layerTypes = config.layerTypes {
            guard layerTypes.count == config.layerCount else {
                throw ModelGraphBuildError.invalidConfig(
                    "layerTypes count (\(layerTypes.count)) != layerCount (\(config.layerCount))")
            }
            return layerTypes.map { $0 == "full_attention" }
        }

        guard let interval = config.fullAttentionInterval else {
            throw ModelGraphBuildError.missingMetadata(
                "Neither layerTypes nor fullAttentionInterval is available for hybrid model")
        }

        return (0..<config.layerCount).map { i in
            (i + 1) % interval == 0
        }
    }

    /// Resolve per-layer conv schedule from config.
    ///
    /// Uses `layerTypes` with "conv" indicating conv layers and "full_attention" indicating attention.
    /// Returns array of booleans where `true` = conv layer, `false` = attention layer.
    func resolveConvSchedule(config: ModelConfig) throws -> [Bool] {
        guard let layerTypes = config.layerTypes else {
            throw ModelGraphBuildError.missingMetadata(
                "layerTypes is required for hybrid conv-attention model")
        }
        guard layerTypes.count == config.layerCount else {
            throw ModelGraphBuildError.invalidConfig(
                "layerTypes count (\(layerTypes.count)) != layerCount (\(config.layerCount))")
        }
        return layerTypes.map { $0 == "conv" }
    }
}

// MARK: - IRGraphAssembler: Architecture Builders

private extension IRGraphAssembler {

    // MARK: Shared Post-Embedding Assembly

    /// Build the decoder + final norm + output head.
    ///
    /// Used by both text-only and VLM graph assembly to avoid duplication.
    func assemblePostEmbedding(
        config: ModelConfig,
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
            let layerSchedule = try resolveLayerSchedule(config: config)
            var layers: [Region] = []
            for i in 0..<config.layerCount {
                let isFullAttention = layerSchedule[i]
                if isFullAttention {
                    layers.append(makeHybridDeltaNetAttentionFullAttnBlock(config: config, ctx: &ctx))
                } else {
                    layers.append(try makeHybridDeltaNetAttentionStateSpaceBlock(config: config, ctx: &ctx))
                }
            }
            let (op, val) = ctx.makePrimitive(
                kind: .layerStack(layers: layers),
                operands: [inputVal])
            ops.append(op)
            decoderVal = val

        case .hybridConvAttention:
            let convSchedule = try resolveConvSchedule(config: config)
            var layers: [Region] = []
            for i in 0..<config.layerCount {
                if convSchedule[i] {
                    layers.append(try makeConvDecoderBlock(config: config, ctx: &ctx))
                } else {
                    layers.append(makeConvAttentionDecoderBlock(config: config, ctx: &ctx))
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
        case .hybridDeltaNetAttention, .hybridConvAttention:
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

    func assembleTransformer(config: ModelConfig, ctx: inout IRContext) -> Region {
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

    func assembleParallelAttentionMLP(config: ModelConfig, ctx: inout IRContext) -> Region {
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

    func assembleMoE(config: ModelConfig, ctx: inout IRContext) throws -> Region {
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

    func assembleHybridDeltaNetAttention(config: ModelConfig, ctx: inout IRContext) throws -> Region {
        var ops: [Operation] = []

        let (embedOp, embedVal) = ctx.makeSource(kind: .tokenEmbedding(
            TokenEmbeddingAttributes(vocabSize: config.vocabSize, embeddingSize: config.hiddenSize)
        ))
        ops.append(embedOp)

        let (postOps, resultVal) = try assemblePostEmbedding(
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

    func makeNorm(config: ModelConfig) -> OperationKind {
        switch config.normKind {
        case .rmsNorm:
            return .rmsNorm(RMSNormAttributes(dimension: config.hiddenSize, epsilon: config.normEps))
        case .layerNorm:
            return .layerNorm(LayerNormAttributes(dimension: config.hiddenSize, epsilon: config.normEps, affine: true))
        }
    }

    func makeAttentionAttrs(config: ModelConfig) -> AttentionAttributes {
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

    func makeMLPAttrs(config: ModelConfig) -> MLPAttributes {
        MLPAttributes(
            inputSize: config.hiddenSize,
            outputSize: config.hiddenSize,
            intermediateSize: config.intermediateSize,
            activation: .silu,
            gating: .swiglu,
            bias: config.mlpBias
        )
    }

    func makeMoEAttrs(config: ModelConfig) throws -> MoEAttributes {
        guard let expertCount = config.expertCount else {
            throw ModelGraphBuildError.missingMetadata("expert_count")
        }
        guard let expertsPerToken = config.expertsPerToken else {
            throw ModelGraphBuildError.missingMetadata("expert_used_count")
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

    func makeTransformerDecoderBlock(config: ModelConfig, ctx: inout IRContext) -> Region {
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

    func makeParallelAttentionMLPDecoderBlock(config: ModelConfig, ctx: inout IRContext) -> Region {
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
    func makeParallelAttentionMLPResidualBody(config: ModelConfig, ctx: inout IRContext) -> Region {
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
    func makeParallelAttentionMLPAttentionAttrs(config: ModelConfig) -> AttentionAttributes {
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

    func makeMoEDecoderBlock(config: ModelConfig, ctx: inout IRContext) throws -> Region {
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

    func makeHybridDeltaNetAttentionFullAttnBlock(config: ModelConfig, ctx: inout IRContext) -> Region {
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

    // MARK: Hybrid Short-Convolution / Attention

    func assembleHybridConvAttention(config: ModelConfig, ctx: inout IRContext) throws -> Region {
        var ops: [Operation] = []

        let (embedOp, embedVal) = ctx.makeSource(kind: .tokenEmbedding(
            TokenEmbeddingAttributes(vocabSize: config.vocabSize, embeddingSize: config.hiddenSize)
        ))
        ops.append(embedOp)

        // LFM2 applies RMSNorm after embedding (model.embedding_norm)
        let (embedNormOp, embedNormVal) = ctx.makePrimitive(
            kind: .rmsNorm(RMSNormAttributes(dimension: config.hiddenSize, epsilon: config.normEps)),
            operands: [embedVal])
        ops.append(embedNormOp)

        let (postOps, resultVal) = try assemblePostEmbedding(
            config: config, architecture: .hybridConvAttention, inputVal: embedNormVal, ctx: &ctx)
        ops.append(contentsOf: postOps)

        return Region(
            parameters: [],
            operations: ops,
            results: [ValueUse(value: resultVal)]
        )
    }

    /// Conv decoder block: RMSNorm + ShortConv residual → RMSNorm + MLP residual.
    func makeConvDecoderBlock(config: ModelConfig, ctx: inout IRContext) throws -> Region {
        guard let convLCache = config.convLCache else {
            throw ModelGraphBuildError.missingMetadata("conv_L_cache is required for hybrid conv-attention model")
        }

        let scAttrs = ShortConvAttributes(
            hiddenSize: config.hiddenSize,
            kernelSize: convLCache
        )

        let (param, paramVal) = ctx.makeParam()
        var ops: [Operation] = []

        let scBody = makeResidualBlock(
            normKind: .rmsNorm(RMSNormAttributes(dimension: config.hiddenSize, epsilon: config.normEps)),
            bodyKind: .shortConv(scAttrs),
            ctx: &ctx
        )
        let (scResidual, scVal) = ctx.makePrimitive(
            kind: .residual(strategy: .add, body: scBody),
            operands: [paramVal]
        )
        ops.append(scResidual)

        let mlpBody = makeResidualBlock(
            normKind: .rmsNorm(RMSNormAttributes(dimension: config.hiddenSize, epsilon: config.normEps)),
            bodyKind: .mlp(makeMLPAttrs(config: config)),
            ctx: &ctx
        )
        let (mlpResidual, mlpVal) = ctx.makePrimitive(
            kind: .residual(strategy: .add, body: mlpBody),
            operands: [scVal]
        )
        ops.append(mlpResidual)

        return Region(
            parameters: [param],
            operations: ops,
            results: [ValueUse(value: mlpVal)]
        )
    }

    /// Attention decoder block: RMSNorm + GQA residual → RMSNorm + MLP residual.
    func makeConvAttentionDecoderBlock(config: ModelConfig, ctx: inout IRContext) -> Region {
        let attnAttrs = AttentionAttributes(
            hiddenSize: config.hiddenSize,
            headCount: config.attentionHeads,
            kvHeadCount: config.kvHeads,
            headDimension: config.headDim,
            bias: config.attentionBias,
            causal: true,
            rope: RoPEAttributes(
                dimension: config.ropeDimension,
                base: config.ropeTheta,
                scaling: config.ropeScaling
            ),
            qkNorm: .rmsNorm
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

    func makeHybridDeltaNetAttentionStateSpaceBlock(config: ModelConfig, ctx: inout IRContext) throws -> Region {
        guard let numHeads = config.ssmNumHeads else {
            throw ModelGraphBuildError.missingMetadata("ssm.time_step_rank (ssmNumHeads) is required for DeltaNet")
        }
        guard let keyHeadDim = config.ssmKeyHeadDim else {
            throw ModelGraphBuildError.missingMetadata("ssm.state_size (ssmKeyHeadDim) is required for DeltaNet")
        }
        guard let valueHeadDim = config.ssmValueHeadDim else {
            throw ModelGraphBuildError.missingMetadata("ssm.state_size (ssmValueHeadDim) is required for DeltaNet")
        }
        let groupCount = config.ssmGroupCount ?? numHeads

        let ssAttrs = StateSpaceAttributes(
            hiddenSize: config.hiddenSize,
            numHeads: numHeads,
            groupCount: groupCount,
            keyHeadDim: keyHeadDim,
            valueHeadDim: valueHeadDim,
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
