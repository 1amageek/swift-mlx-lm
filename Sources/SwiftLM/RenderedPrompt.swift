struct RenderedPrompt: Sendable {
    let text: String
    let tokenIDs: [Int]?
    let multimodal: PreparedPrompt.Multimodal?

    init(
        text: String,
        tokenIDs: [Int]? = nil,
        multimodal: PreparedPrompt.Multimodal?
    ) {
        self.text = text
        self.tokenIDs = tokenIDs
        self.multimodal = multimodal
    }
}
