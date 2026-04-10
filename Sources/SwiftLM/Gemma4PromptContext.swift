struct Gemma4PromptContext: Sendable {
    let promptEmbeddings: [[Float]]
    let perLayerInputs: [[[Float]]]
    let usesEmbeddingOverrides: Bool
}
