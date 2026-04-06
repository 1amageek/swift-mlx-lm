/// Prompt preparation output produced from ``ModelInput``.
///
/// `PreparedInput` captures the rendered prompt text, token IDs, and optional
/// multimodal metadata emitted by model-specific prompt processors such as the
/// Qwen3-VL image placeholder expander.
public struct PreparedInput: Sendable {
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
        public var mmTokenTypeIDs: [Int]
        public var images: [Image]
        public var videos: [Video]

        public init(
            mmTokenTypeIDs: [Int] = [],
            images: [Image] = [],
            videos: [Video] = []
        ) {
            self.mmTokenTypeIDs = mmTokenTypeIDs
            self.images = images
            self.videos = videos
        }

        public struct Image: Sendable {
            public var gridTHW: [Int]
            public var placeholderTokenCount: Int
            public var pixelValuesShape: [Int]
            public var pixelValues: [Float]
            public var resizedSize: [Int]

            public init(
                gridTHW: [Int],
                placeholderTokenCount: Int,
                pixelValuesShape: [Int] = [],
                pixelValues: [Float] = [],
                resizedSize: [Int] = []
            ) {
                self.gridTHW = gridTHW
                self.placeholderTokenCount = placeholderTokenCount
                self.pixelValuesShape = pixelValuesShape
                self.pixelValues = pixelValues
                self.resizedSize = resizedSize
            }
        }

        public struct Video: Sendable {
            public var gridTHW: [Int]
            public var placeholderTokenCount: Int
            public var pixelValuesShape: [Int]
            public var pixelValues: [Float]
            public var frameTimestamps: [Double]
            public var sampledFrameCount: Int
            public var resizedSize: [Int]

            public init(
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
                self.pixelValuesShape = pixelValuesShape
                self.pixelValues = pixelValues
                self.frameTimestamps = frameTimestamps
                self.sampledFrameCount = sampledFrameCount
                self.resizedSize = resizedSize
            }
        }
    }
}
