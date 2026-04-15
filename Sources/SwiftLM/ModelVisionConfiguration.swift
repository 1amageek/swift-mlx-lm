/// Vision-related metadata declared by a model bundle.
///
/// This mirrors the multimodal markers published in Hugging Face model
/// configurations such as Qwen3.5.
public struct ModelVisionConfiguration: Sendable, Equatable {
    public var hiddenSize: Int?
    public var depth: Int?
    public var intermediateSize: Int?
    public var outHiddenSize: Int?
    public var headCount: Int?
    public var numPositionEmbeddings: Int?
    public var inChannels: Int?
    public var hiddenAct: String?
    public var deepstackVisualIndexes: [Int]
    public var processorClass: String?
    public var imageTokenID: Int?
    public var videoTokenID: Int?
    public var visionStartTokenID: Int?
    public var visionEndTokenID: Int?
    public var imageProcessorType: String?
    public var videoProcessorType: String?
    public var patchSize: Int?
    public var poolingKernelSize: Int?
    public var temporalPatchSize: Int?
    public var mergeSize: Int?
    public var spatialMergeSize: Int?
    public var positionEmbeddingSize: Int?
    public var defaultOutputLength: Int?
    public var ropeTheta: Float?
    public var standardize: Bool?
    public var minimumPixelCount: Int?
    public var maximumPixelCount: Int?
    public var videoFramesPerSecond: Double?
    public var minimumFrameCount: Int?
    public var maximumFrameCount: Int?
    public var imageMean: [Double]
    public var imageStd: [Double]

    public init(
        hiddenSize: Int? = nil,
        depth: Int? = nil,
        intermediateSize: Int? = nil,
        outHiddenSize: Int? = nil,
        headCount: Int? = nil,
        numPositionEmbeddings: Int? = nil,
        inChannels: Int? = nil,
        hiddenAct: String? = nil,
        deepstackVisualIndexes: [Int] = [],
        processorClass: String? = nil,
        imageTokenID: Int? = nil,
        videoTokenID: Int? = nil,
        visionStartTokenID: Int? = nil,
        visionEndTokenID: Int? = nil,
        imageProcessorType: String? = nil,
        videoProcessorType: String? = nil,
        patchSize: Int? = nil,
        poolingKernelSize: Int? = nil,
        temporalPatchSize: Int? = nil,
        mergeSize: Int? = nil,
        spatialMergeSize: Int? = nil,
        positionEmbeddingSize: Int? = nil,
        defaultOutputLength: Int? = nil,
        ropeTheta: Float? = nil,
        standardize: Bool? = nil,
        minimumPixelCount: Int? = nil,
        maximumPixelCount: Int? = nil,
        videoFramesPerSecond: Double? = nil,
        minimumFrameCount: Int? = nil,
        maximumFrameCount: Int? = nil,
        imageMean: [Double] = [],
        imageStd: [Double] = []
    ) {
        self.hiddenSize = hiddenSize
        self.depth = depth
        self.intermediateSize = intermediateSize
        self.outHiddenSize = outHiddenSize
        self.headCount = headCount
        self.numPositionEmbeddings = numPositionEmbeddings
        self.inChannels = inChannels
        self.hiddenAct = hiddenAct
        self.deepstackVisualIndexes = deepstackVisualIndexes
        self.processorClass = processorClass
        self.imageTokenID = imageTokenID
        self.videoTokenID = videoTokenID
        self.visionStartTokenID = visionStartTokenID
        self.visionEndTokenID = visionEndTokenID
        self.imageProcessorType = imageProcessorType
        self.videoProcessorType = videoProcessorType
        self.patchSize = patchSize
        self.poolingKernelSize = poolingKernelSize
        self.temporalPatchSize = temporalPatchSize
        self.mergeSize = mergeSize
        self.spatialMergeSize = spatialMergeSize
        self.positionEmbeddingSize = positionEmbeddingSize
        self.defaultOutputLength = defaultOutputLength
        self.ropeTheta = ropeTheta
        self.standardize = standardize
        self.minimumPixelCount = minimumPixelCount
        self.maximumPixelCount = maximumPixelCount
        self.videoFramesPerSecond = videoFramesPerSecond
        self.minimumFrameCount = minimumFrameCount
        self.maximumFrameCount = maximumFrameCount
        self.imageMean = imageMean
        self.imageStd = imageStd
    }
}
