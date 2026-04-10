struct PreparedPrompt: Sendable {
    let text: String
    let tokenIDs: [Int]?
    let multimodal: PreparedInput.Multimodal?

    init(
        text: String,
        tokenIDs: [Int]? = nil,
        multimodal: PreparedInput.Multimodal?
    ) {
        self.text = text
        self.tokenIDs = tokenIDs
        self.multimodal = multimodal
    }
}
