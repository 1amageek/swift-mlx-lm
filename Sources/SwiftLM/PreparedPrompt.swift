/// Prompt preparation output produced from ``ModelInput``.
///
/// `PreparedPrompt` captures the rendered prompt text, token IDs, and optional
/// high-level multimodal metadata emitted by model-specific prompt processors.
/// Raw vision execution payloads remain internal to `SwiftLM`.
public struct PreparedPrompt: Sendable {
    /// Rendered prompt text after chat templating and placeholder expansion.
    public var renderedText: String
    /// Token IDs produced by the tokenizer for `renderedText`.
    public var tokenIDs: [Int]
    /// Optional attention mask.
    public var attentionMask: [Int]?
    /// Optional multimodal prompt metadata derived during preparation.
    public var multimodalMetadata: Multimodal?

    public init(
        renderedText: String,
        tokenIDs: [Int],
        attentionMask: [Int]? = nil,
        multimodalMetadata: Multimodal? = nil
    ) {
        self.renderedText = renderedText
        self.tokenIDs = tokenIDs
        self.attentionMask = attentionMask
        self.multimodalMetadata = multimodalMetadata
    }

    /// Prompt-time multimodal metadata derived from the model processor.
    public struct Multimodal: Sendable {
        public var images: [Image]
        public var videos: [Video]
        var mmTokenTypeIDs: [Int]

        public init(
            images: [Image] = [],
            videos: [Video] = []
        ) {
            self.images = images
            self.videos = videos
            self.mmTokenTypeIDs = []
        }

        init(
            mmTokenTypeIDs: [Int] = [],
            images: [Image] = [],
            videos: [Video] = []
        ) {
            self.images = images
            self.videos = videos
            self.mmTokenTypeIDs = mmTokenTypeIDs
        }

        public struct Image: Sendable {
            public var gridTHW: [Int]
            public var placeholderTokenCount: Int
            public var resizedSize: [Int]
            var pixelValuesShape: [Int]
            var pixelValues: [Float]

            public init(
                gridTHW: [Int],
                placeholderTokenCount: Int,
                resizedSize: [Int] = []
            ) {
                self.gridTHW = gridTHW
                self.placeholderTokenCount = placeholderTokenCount
                self.resizedSize = resizedSize
                self.pixelValuesShape = []
                self.pixelValues = []
            }

            init(
                gridTHW: [Int],
                placeholderTokenCount: Int,
                pixelValuesShape: [Int] = [],
                pixelValues: [Float] = [],
                resizedSize: [Int] = []
            ) {
                self.gridTHW = gridTHW
                self.placeholderTokenCount = placeholderTokenCount
                self.resizedSize = resizedSize
                self.pixelValuesShape = pixelValuesShape
                self.pixelValues = pixelValues
            }
        }

        public struct Video: Sendable {
            public var gridTHW: [Int]
            public var placeholderTokenCount: Int
            public var frameTimestamps: [Double]
            public var sampledFrameCount: Int
            public var resizedSize: [Int]
            var pixelValuesShape: [Int]
            var pixelValues: [Float]

            public init(
                gridTHW: [Int] = [],
                placeholderTokenCount: Int,
                frameTimestamps: [Double] = [],
                sampledFrameCount: Int = 0,
                resizedSize: [Int] = []
            ) {
                self.gridTHW = gridTHW
                self.placeholderTokenCount = placeholderTokenCount
                self.frameTimestamps = frameTimestamps
                self.sampledFrameCount = sampledFrameCount
                self.resizedSize = resizedSize
                self.pixelValuesShape = []
                self.pixelValues = []
            }

            init(
                gridTHW: [Int] = [],
                placeholderTokenCount: Int,
                pixelValuesShape: [Int] = [],
                pixelValues: [Float] = [],
                frameTimestamps: [Double] = [],
                sampledFrameCount: Int = 0,
                resizedSize: [Int] = []
            ) {
                self.gridTHW = gridTHW
                self.placeholderTokenCount = placeholderTokenCount
                self.frameTimestamps = frameTimestamps
                self.sampledFrameCount = sampledFrameCount
                self.resizedSize = resizedSize
                self.pixelValuesShape = pixelValuesShape
                self.pixelValues = pixelValues
            }
        }
    }
}
