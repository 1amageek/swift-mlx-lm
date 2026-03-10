import SwiftLM

/// Qwen 3.5 hybrid vision-language model.
///
/// Architecture: Vision Transformer encoder + hybrid Gated DeltaNet / Full Attention
/// text decoder with M-RoPE for multi-axis position encoding.
///
/// Every `fullAttentionInterval`-th layer uses full multi-head attention;
/// all other layers use Gated DeltaNet (linear attention with O(1) per-token
/// state update). Vision features are merged into the text sequence via
/// placeholder token replacement.
///
/// ```swift
/// let qwen = Qwen35(config: .qwen35_0_8B)
/// let graph = try qwen.makeModelGraph()
/// ```
public struct Qwen35: ModelComponent {

    /// Architecture configuration for Qwen 3.5 VLM models.
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

        // MARK: - M-RoPE (multi-axis for vision-language)

        /// Dimension allocation per axis [temporal, height, width].
        /// Sum must equal ropePartialDim / 2 (half-dimensions).
        public let mropeSections: [Int]

        /// Whether M-RoPE dimensions cycle across axes (interleaved)
        /// or are allocated in contiguous blocks.
        public let mropeInterleaved: Bool

        // MARK: - DeltaNet Parameters

        public let linearKeyHeads: Int
        public let linearValueHeads: Int
        public let linearKeyHeadDim: Int
        public let linearValueHeadDim: Int
        public let convKernelSize: Int

        // MARK: - Hybrid Routing

        public let fullAttentionInterval: Int

        // MARK: - Vision Encoder

        public let visionHiddenSize: Int
        public let visionDepth: Int
        public let visionHeadCount: Int
        public let visionPatchSize: Int
        public let visionIntermediateSize: Int
        public let visionOutputSize: Int
        public let spatialMergeSize: Int
        public let temporalPatchSize: Int

        // MARK: - Vision-Text Merge

        public let imageTokenId: Int
        public let videoTokenId: Int

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
            mropeSections: [Int] = [11, 11, 10],
            mropeInterleaved: Bool = true,
            linearKeyHeads: Int,
            linearValueHeads: Int,
            linearKeyHeadDim: Int = 128,
            linearValueHeadDim: Int = 128,
            convKernelSize: Int = 4,
            fullAttentionInterval: Int = 4,
            visionHiddenSize: Int = 768,
            visionDepth: Int = 12,
            visionHeadCount: Int = 12,
            visionPatchSize: Int = 16,
            visionIntermediateSize: Int = 3072,
            visionOutputSize: Int? = nil,
            spatialMergeSize: Int = 2,
            temporalPatchSize: Int = 2,
            imageTokenId: Int = 248056,
            videoTokenId: Int = 248057,
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
            self.mropeSections = mropeSections
            self.mropeInterleaved = mropeInterleaved
            self.linearKeyHeads = linearKeyHeads
            self.linearValueHeads = linearValueHeads
            self.linearKeyHeadDim = linearKeyHeadDim
            self.linearValueHeadDim = linearValueHeadDim
            self.convKernelSize = convKernelSize
            self.fullAttentionInterval = fullAttentionInterval
            self.visionHiddenSize = visionHiddenSize
            self.visionDepth = visionDepth
            self.visionHeadCount = visionHeadCount
            self.visionPatchSize = visionPatchSize
            self.visionIntermediateSize = visionIntermediateSize
            self.visionOutputSize = visionOutputSize ?? hiddenSize
            self.spatialMergeSize = spatialMergeSize
            self.temporalPatchSize = temporalPatchSize
            self.imageTokenId = imageTokenId
            self.videoTokenId = videoTokenId
            self.tieWordEmbeddings = tieWordEmbeddings
        }
    }

    public let config: Config

    public init(config: Config) {
        self.config = config
    }

    @ModelComponentBuilder
    public var body: some ModelComponent {
        // Vision-text parallel input: vision encoder + token embedding,
        // merged by replacing placeholder tokens with vision features.
        Parallel(merge: .visionMerge(VisionMergeConfig(
            imageTokenId: config.imageTokenId,
            videoTokenId: config.videoTokenId
        ))) {
            TokenEmbedding(vocabSize: config.vocabularySize, embeddingSize: config.hiddenSize)
            VisionEncoder(
                hiddenSize: config.visionHiddenSize,
                outputSize: config.visionOutputSize,
                depth: config.visionDepth,
                headCount: config.visionHeadCount,
                patchSize: config.visionPatchSize,
                spatialMergeSize: config.spatialMergeSize,
                intermediateSize: config.visionIntermediateSize,
                temporalPatchSize: config.temporalPatchSize
            )
        }

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
            StateSpace(
                hiddenSize: config.hiddenSize,
                stateSize: config.linearKeyHeadDim,
                variant: "gated_deltanet"
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
/// Standard GQA with partial M-RoPE and QK RMSNorm.
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
                    mropeAxes: MRoPEAxes(
                        sections: config.mropeSections,
                        interleaved: config.mropeInterleaved
                    )
                ),
                qkNorm: .rmsNorm
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

    /// Qwen 3.5 0.8B
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
        linearValueHeadDim: 128
    )
}
