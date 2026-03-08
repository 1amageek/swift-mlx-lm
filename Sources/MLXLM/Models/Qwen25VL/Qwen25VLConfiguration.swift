/// Configuration for Qwen2.5-VL vision-language model.
struct Qwen25VLConfiguration: Sendable {

    let text: TextConfiguration
    let vision: VisionConfiguration
    let mrope: MRoPEConfiguration

    /// Token IDs for vision placeholders in the text sequence.
    let imageTokenId: Int
    let videoTokenId: Int

    init(
        text: TextConfiguration,
        vision: VisionConfiguration,
        mrope: MRoPEConfiguration = MRoPEConfiguration(),
        imageTokenId: Int = 151655,
        videoTokenId: Int = 151656
    ) {
        self.text = text
        self.vision = vision
        self.mrope = mrope
        self.imageTokenId = imageTokenId
        self.videoTokenId = videoTokenId
    }

    // MARK: - Text Configuration

    /// LLM decoder configuration (standard Qwen2 architecture).
    struct TextConfiguration: Sendable {
        var hiddenSize: Int
        var hiddenLayers: Int
        var intermediateSize: Int
        var attentionHeads: Int
        var kvHeads: Int
        var vocabularySize: Int
        var normEps: Float
        var ropeTheta: Float
        var maxPositionEmbeddings: Int?
        var tieWordEmbeddings: Bool

        init(
            hiddenSize: Int,
            hiddenLayers: Int,
            intermediateSize: Int,
            attentionHeads: Int,
            kvHeads: Int? = nil,
            vocabularySize: Int,
            normEps: Float = 1e-6,
            ropeTheta: Float = 1_000_000,
            maxPositionEmbeddings: Int? = nil,
            tieWordEmbeddings: Bool = true
        ) {
            self.hiddenSize = hiddenSize
            self.hiddenLayers = hiddenLayers
            self.intermediateSize = intermediateSize
            self.attentionHeads = attentionHeads
            self.kvHeads = kvHeads ?? attentionHeads
            self.vocabularySize = vocabularySize
            self.normEps = normEps
            self.ropeTheta = ropeTheta
            self.maxPositionEmbeddings = maxPositionEmbeddings
            self.tieWordEmbeddings = tieWordEmbeddings
        }

        var headDim: Int { hiddenSize / attentionHeads }
    }

    // MARK: - Vision Configuration

    /// Vision encoder configuration (shared across 3B/7B/72B, only outHiddenSize differs).
    struct VisionConfiguration: Sendable {
        var hiddenSize: Int
        var intermediateSize: Int
        var depth: Int
        var numHeads: Int
        var outHiddenSize: Int
        var patchSize: Int
        var spatialMergeSize: Int
        var temporalPatchSize: Int
        var inChannels: Int
        var normEps: Float
        var windowSize: Int
        var fullAttBlockIndexes: [Int]

        init(
            hiddenSize: Int = 1280,
            intermediateSize: Int = 3420,
            depth: Int = 32,
            numHeads: Int = 16,
            outHiddenSize: Int = 2048,
            patchSize: Int = 14,
            spatialMergeSize: Int = 2,
            temporalPatchSize: Int = 2,
            inChannels: Int = 3,
            normEps: Float = 1e-6,
            windowSize: Int = 112,
            fullAttBlockIndexes: [Int] = [7, 15, 23, 31]
        ) {
            self.hiddenSize = hiddenSize
            self.intermediateSize = intermediateSize
            self.depth = depth
            self.numHeads = numHeads
            self.outHiddenSize = outHiddenSize
            self.patchSize = patchSize
            self.spatialMergeSize = spatialMergeSize
            self.temporalPatchSize = temporalPatchSize
            self.inChannels = inChannels
            self.normEps = normEps
            self.windowSize = windowSize
            self.fullAttBlockIndexes = fullAttBlockIndexes
        }

        var headDim: Int { hiddenSize / numHeads }

        /// Factor for image dimensions (must be divisible by this).
        var imageFactor: Int { patchSize * spatialMergeSize }
    }

    // MARK: - M-RoPE Configuration

    /// Multimodal Rotary Position Embedding configuration.
    ///
    /// Splits head dimensions into 3 sections for temporal/height/width positioning.
    struct MRoPEConfiguration: Sendable {
        /// Head dimension split: [temporal, height, width].
        var sections: [Int]

        init(sections: [Int] = [16, 24, 24]) {
            self.sections = sections
        }

        var totalDimensions: Int { sections.reduce(0, +) }
    }
}
