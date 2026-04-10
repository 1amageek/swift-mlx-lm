import LMArchitecture

/// Standard pre-norm decoder-only transformer.
///
/// Covers Llama, Qwen2, Mistral, Gemma, Phi, StarCoder2, and MoE variants (Mixtral).
/// Each layer: Residual(Norm + Attention) -> Residual(Norm + MLP/MoE).
///
/// Accepts `ModelConfig` directly. All fields are read from config without adaptation.
public struct Transformer: ModelComponent {

    public let config: ModelConfig

    public init(config: ModelConfig) {
        self.config = config
    }

    @ModelComponentBuilder
    public var body: some ModelComponent {
        TokenEmbedding(vocabSize: config.vocabSize, embeddingSize: config.hiddenSize)

        Repeat(count: config.layerCount, label: "layers") {
            TransformerDecoderLayer(config: config)
        }

        makeNorm()
        OutputHead(inputSize: config.hiddenSize, vocabSize: config.vocabSize,
                   tiedToEmbedding: config.tiedEmbeddings)
    }

    @ModelComponentBuilder
    private func makeNorm() -> some ModelComponent {
        switch config.normKind {
        case .rmsNorm:
            RMSNorm(dimension: config.hiddenSize, epsilon: config.normEps)
        case .layerNorm:
            LayerNorm(dimension: config.hiddenSize, epsilon: config.normEps)
        }
    }
}

/// Decoder layer for the standard pre-norm transformer.
///
/// Residual(Norm + Attention) -> Residual(Norm + MLP/MoE).
struct TransformerDecoderLayer: ModelComponent {
    let config: ModelConfig

    @ModelComponentBuilder
    var body: some ModelComponent {
        Residual {
            makeNorm()
            Attention(
                hiddenSize: config.hiddenSize,
                headCount: config.attentionHeads,
                kvHeadCount: config.kvHeads,
                headDimension: config.headDim,
                bias: config.attentionBias,
                rope: RoPEAttributes(
                    dimension: config.ropeDimension,
                    base: config.ropeTheta,
                    scaling: config.ropeScaling
                ),
                qkNorm: config.qkNorm ? .rmsNorm : nil,
                window: config.slidingWindow.map { AttentionWindow(left: $0, right: 0) }
            )
        }
        Residual {
            makeNorm()
            makeFeedForward()
        }
    }

    @ModelComponentBuilder
    private func makeNorm() -> some ModelComponent {
        switch config.normKind {
        case .rmsNorm:
            RMSNorm(dimension: config.hiddenSize, epsilon: config.normEps)
        case .layerNorm:
            LayerNorm(dimension: config.hiddenSize, epsilon: config.normEps)
        }
    }

    @ModelComponentBuilder
    private func makeFeedForward() -> some ModelComponent {
        if let expertCount = config.expertCount, let expertsPerToken = config.expertsPerToken {
            MoE(
                expertCount: expertCount,
                expertsPerToken: expertsPerToken,
                expertInputSize: config.hiddenSize,
                expertOutputSize: config.hiddenSize,
                expertIntermediateSize: config.intermediateSize,
                expertActivation: .silu,
                expertGating: .swiglu,
                expertBias: config.mlpBias
            )
        } else {
            MLP(inputSize: config.hiddenSize, intermediateSize: config.intermediateSize,
                activation: .silu, gating: .swiglu, bias: config.mlpBias)
        }
    }
}
