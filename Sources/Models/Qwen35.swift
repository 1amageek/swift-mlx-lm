import SwiftLM

/// Qwen 3.5 hybrid DeltaNet + Full Attention model.
///
/// Supports both text-only and vision-language modes. When `vision` config
/// is provided, includes a Vision Transformer encoder with M-RoPE for
/// multi-axis position encoding. When `nil`, operates as a text-only decoder.
///
/// Every `fullAttentionInterval`-th layer uses full multi-head attention;
/// all other layers use Gated DeltaNet (linear attention with O(1) per-token
/// state update).
///
/// ```swift
/// // Text-only
/// let textModel = Qwen35(config: .init(
///     hiddenSize: 1024, hiddenLayers: 24, intermediateSize: 3584,
///     vocabularySize: 248320, attentionHeads: 8, kvHeads: 2,
///     linearKeyHeads: 16, linearValueHeads: 16
/// ))
///
/// // VLM
/// let vlm = Qwen35(config: .qwen35_0_8B)
/// ```
public struct Qwen35: ModelComponent {

    /// Vision encoder and M-RoPE configuration for VLM mode.
    public struct VisionConfig: Sendable {

        // MARK: - Vision Encoder

        public let hiddenSize: Int
        public let depth: Int
        public let headCount: Int
        public let patchSize: Int
        public let intermediateSize: Int
        public let outputSize: Int
        public let spatialMergeSize: Int
        public let temporalPatchSize: Int

        // MARK: - Vision-Text Merge

        public let imageTokenId: Int
        public let videoTokenId: Int

        // MARK: - M-RoPE

        /// Dimension allocation per axis [temporal, height, width].
        /// Sum must equal ropePartialDim / 2 (half-dimensions).
        public let mropeSections: [Int]

        /// Whether M-RoPE dimensions cycle across axes (interleaved)
        /// or are allocated in contiguous blocks.
        public let mropeInterleaved: Bool

        public init(
            hiddenSize: Int = 768,
            depth: Int = 12,
            headCount: Int = 12,
            patchSize: Int = 16,
            intermediateSize: Int = 3072,
            outputSize: Int,
            spatialMergeSize: Int = 2,
            temporalPatchSize: Int = 2,
            imageTokenId: Int = 248056,
            videoTokenId: Int = 248057,
            mropeSections: [Int] = [11, 11, 10],
            mropeInterleaved: Bool = true
        ) {
            self.hiddenSize = hiddenSize
            self.depth = depth
            self.headCount = headCount
            self.patchSize = patchSize
            self.intermediateSize = intermediateSize
            self.outputSize = outputSize
            self.spatialMergeSize = spatialMergeSize
            self.temporalPatchSize = temporalPatchSize
            self.imageTokenId = imageTokenId
            self.videoTokenId = videoTokenId
            self.mropeSections = mropeSections
            self.mropeInterleaved = mropeInterleaved
        }
    }

    /// Architecture configuration for Qwen 3.5 models.
    public struct Config: Sendable {

        // MARK: - Core Dimensions

        public let hiddenSize: Int
        public let hiddenLayers: Int
        public let intermediateSize: Int
        public let vocabularySize: Int
        public let normEps: Float

        // MARK: - Full Attention Parameters

        public let attentionHeads: Int
        public let kvHeads: Int
        public let headDim: Int

        // MARK: - RoPE

        public let ropeTheta: Float
        public let ropeScaling: RoPEScaling?
        public let partialRotaryFactor: Float

        // MARK: - DeltaNet Parameters

        public let linearKeyHeads: Int
        public let linearValueHeads: Int
        public let linearKeyHeadDim: Int
        public let linearValueHeadDim: Int
        public let convKernelSize: Int

        // MARK: - Hybrid Routing

        public let fullAttentionInterval: Int

        // MARK: - Vision (optional, nil = text-only)

        public let vision: VisionConfig?

        // MARK: - Output

        public let tieWordEmbeddings: Bool

        // MARK: - Computed

        /// RoPE dimension for partial rotation (25% of head_dim by default).
        public var ropePartialDim: Int {
            Int(Float(headDim) * partialRotaryFactor)
        }

        /// Number of hybrid groups (each = interval-1 DeltaNet + 1 FullAttention).
        public var hybridGroupCount: Int {
            hiddenLayers / fullAttentionInterval
        }

        /// Number of DeltaNet layers per hybrid group.
        public var deltaNetLayersPerGroup: Int {
            fullAttentionInterval - 1
        }

        /// Whether this config includes vision encoder.
        public var isVLM: Bool { vision != nil }

        public init(
            hiddenSize: Int,
            hiddenLayers: Int,
            intermediateSize: Int,
            vocabularySize: Int,
            normEps: Float = 1e-6,
            attentionHeads: Int,
            kvHeads: Int,
            headDim: Int = 256,
            ropeTheta: Float = 10_000_000.0,
            ropeScaling: RoPEScaling? = nil,
            partialRotaryFactor: Float = 0.25,
            linearKeyHeads: Int,
            linearValueHeads: Int,
            linearKeyHeadDim: Int = 128,
            linearValueHeadDim: Int = 128,
            convKernelSize: Int = 4,
            fullAttentionInterval: Int = 4,
            vision: VisionConfig? = nil,
            tieWordEmbeddings: Bool = true
        ) {
            precondition(
                fullAttentionInterval > 0,
                "fullAttentionInterval must be positive"
            )
            precondition(
                hiddenLayers % fullAttentionInterval == 0,
                "hiddenLayers must be divisible by fullAttentionInterval"
            )
            self.hiddenSize = hiddenSize
            self.hiddenLayers = hiddenLayers
            self.intermediateSize = intermediateSize
            self.vocabularySize = vocabularySize
            self.normEps = normEps
            self.attentionHeads = attentionHeads
            self.kvHeads = kvHeads
            self.headDim = headDim
            self.ropeTheta = ropeTheta
            self.ropeScaling = ropeScaling
            self.partialRotaryFactor = partialRotaryFactor
            self.linearKeyHeads = linearKeyHeads
            self.linearValueHeads = linearValueHeads
            self.linearKeyHeadDim = linearKeyHeadDim
            self.linearValueHeadDim = linearValueHeadDim
            self.convKernelSize = convKernelSize
            self.fullAttentionInterval = fullAttentionInterval
            self.vision = vision
            self.tieWordEmbeddings = tieWordEmbeddings
        }
    }

    public let config: Config

    public init(config: Config) {
        self.config = config
    }

    @ModelComponentBuilder
    public var body: some ModelComponent {
        // Token embedding — same IR for text-only and VLM.
        // Vision encoder is loaded separately (not part of the IR graph).
        TokenEmbedding(vocabSize: config.vocabularySize, embeddingSize: config.hiddenSize)

        // Hybrid decoder: [DeltaNet x (interval-1), FullAttention x 1] groups
        Repeat(count: config.hybridGroupCount, label: "hybrid_groups") {
            Repeat(count: config.deltaNetLayersPerGroup, label: "deltanet_layers") {
                Qwen35DeltaNetDecoderLayer(config: config)
            }
            Qwen35AttnDecoderLayer(config: config)
        }

        RMSNorm(dimension: config.hiddenSize, epsilon: config.normEps)
        OutputHead(
            inputSize: config.hiddenSize,
            vocabSize: config.vocabularySize,
            tiedToEmbedding: config.tieWordEmbeddings
        )
    }
}

// MARK: - Decoder Layers

/// DeltaNet decoder layer: RMSNorm + Gated DeltaNet residual, then RMSNorm + MLP residual.
///
/// Linear attention with O(1) per-token state update.
struct Qwen35DeltaNetDecoderLayer: ModelComponent {

    let config: Qwen35.Config

    @ModelComponentBuilder
    var body: some ModelComponent {
        Residual {
            RMSNorm(dimension: config.hiddenSize, epsilon: config.normEps)
            DeltaNet(
                hiddenSize: config.hiddenSize,
                stateSize: config.linearKeyHeadDim,
                variant: .gated
            )
        }
        Residual {
            RMSNorm(dimension: config.hiddenSize, epsilon: config.normEps)
            MLP(inputSize: config.hiddenSize, intermediateSize: config.intermediateSize)
        }
    }
}

/// Full attention decoder layer: RMSNorm + GQA residual, then RMSNorm + MLP residual.
///
/// Standard GQA with partial RoPE (M-RoPE for VLM, standard for text-only),
/// QK RMSNorm, and sigmoid output gate packed in Q projection.
struct Qwen35AttnDecoderLayer: ModelComponent {

    let config: Qwen35.Config

    @ModelComponentBuilder
    var body: some ModelComponent {
        Residual {
            RMSNorm(dimension: config.hiddenSize, epsilon: config.normEps)
            Attention(
                hiddenSize: config.hiddenSize,
                headCount: config.attentionHeads,
                kvHeadCount: config.kvHeads,
                headDimension: config.headDim,
                rope: RoPEAttributes(
                    dimension: config.ropePartialDim,
                    base: config.ropeTheta,
                    scaling: config.ropeScaling,
                    mropeAxes: config.vision.map {
                        MRoPEAxes(
                            sections: $0.mropeSections,
                            interleaved: $0.mropeInterleaved
                        )
                    }
                ),
                qkNorm: .rmsNorm,
                outputGate: .sigmoidPackedInQProj
            )
        }
        Residual {
            RMSNorm(dimension: config.hiddenSize, epsilon: config.normEps)
            MLP(inputSize: config.hiddenSize, intermediateSize: config.intermediateSize)
        }
    }
}

// MARK: - Preset Configurations

extension Qwen35.Config {

    /// Qwen 3.5 0.8B VLM preset.
    public static let qwen35_0_8B = Qwen35.Config(
        hiddenSize: 1024,
        hiddenLayers: 24,
        intermediateSize: 3584,
        vocabularySize: 248320,
        attentionHeads: 8,
        kvHeads: 2,
        headDim: 256,
        linearKeyHeads: 16,
        linearValueHeads: 16,
        linearKeyHeadDim: 128,
        linearValueHeadDim: 128,
        vision: .init(outputSize: 1024)
    )
}
