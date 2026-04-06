public struct VisualContext: Sendable {
    public var layout: PromptLayout
    public var imageTokenEmbeddings: [[Float]]
    public var imageDeepstackFeaturesByLayer: [Int: [[Float]]]
    public var videoTokenEmbeddings: [[Float]]
    public var videoDeepstackFeaturesByLayer: [Int: [[Float]]]

    public init(
        layout: PromptLayout,
        imageTokenEmbeddings: [[Float]],
        imageDeepstackFeaturesByLayer: [Int: [[Float]]],
        videoTokenEmbeddings: [[Float]],
        videoDeepstackFeaturesByLayer: [Int: [[Float]]]
    ) {
        self.layout = layout
        self.imageTokenEmbeddings = imageTokenEmbeddings
        self.imageDeepstackFeaturesByLayer = imageDeepstackFeaturesByLayer
        self.videoTokenEmbeddings = videoTokenEmbeddings
        self.videoDeepstackFeaturesByLayer = videoDeepstackFeaturesByLayer
    }
}
