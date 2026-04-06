import LMArchitecture

struct Gemma4DecoderLayer: ModelComponent {

    let config: ModelConfig
    let layerIndex: Int

    private var layerType: String {
        config.layerTypes![layerIndex]
    }

    private var isFullAttention: Bool {
        layerType == "full_attention"
    }

    private var usesDoubleWideMLP: Bool {
        guard config.useDoubleWideMLP,
              let numKVSharedLayers = config.numKVSharedLayers,
              numKVSharedLayers > 0 else {
            return false
        }
        let firstSharedLayerIndex = config.layerCount - numKVSharedLayers
        return layerIndex >= firstSharedLayerIndex
    }

    private var attentionHeadDimension: Int {
        isFullAttention ? (config.globalHeadDim ?? config.headDim) : config.headDim
    }

    private var attentionKVHeads: Int {
        if isFullAttention, config.attentionKEqualsV {
            return config.globalKVHeads ?? config.kvHeads
        }
        return config.kvHeads
    }

    private var ropeDimension: Int {
        guard isFullAttention,
              let factor = config.fullAttentionPartialRotaryFactor else {
            return config.ropeDimension
        }
        return Int(Float(attentionHeadDimension) * factor)
    }

    private var ropeBase: Float {
        isFullAttention ? (config.fullAttentionRopeTheta ?? config.ropeTheta) : config.ropeTheta
    }

    private var ropeScaling: RoPEScaling? {
        isFullAttention ? config.fullAttentionRoPEScaling : config.ropeScaling
    }

    private var mlpIntermediateSize: Int {
        usesDoubleWideMLP ? config.intermediateSize * 2 : config.intermediateSize
    }

    @ModelComponentBuilder
    var body: some ModelComponent {
        Residual {
            RMSNorm(dimension: config.hiddenSize, epsilon: config.normEps)
            Attention(
                hiddenSize: config.hiddenSize,
                headCount: config.attentionHeads,
                kvHeadCount: attentionKVHeads,
                headDimension: attentionHeadDimension,
                bias: config.attentionBias,
                rope: RoPEAttributes(
                    dimension: ropeDimension,
                    base: ropeBase,
                    scaling: ropeScaling
                ),
                qkNorm: .rmsNorm,
                window: isFullAttention ? nil : config.slidingWindow.map {
                    AttentionWindow(left: $0, right: $0)
                }
            )
            RMSNorm(dimension: config.hiddenSize, epsilon: config.normEps)
        }

        Residual {
            RMSNorm(dimension: config.hiddenSize, epsilon: config.normEps)
            MLP(
                inputSize: config.hiddenSize,
                intermediateSize: mlpIntermediateSize,
                activation: .custom("gelu_pytorch_tanh"),
                gating: .none,
                bias: config.mlpBias
            )
            RMSNorm(dimension: config.hiddenSize, epsilon: config.normEps)
        }

        if let perLayerInputSize = config.hiddenSizePerLayerInput,
           let vocabSizePerLayerInput = config.vocabSizePerLayerInput {
            Residual {
                PerLayerInput(
                    hiddenSize: config.hiddenSize,
                    perLayerInputSize: perLayerInputSize,
                    vocabSize: vocabSizePerLayerInput,
                    activation: .custom("gelu_pytorch_tanh")
                )
            }
        }
    }
}
