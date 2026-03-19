import MetalCompiler

public struct PromptState: @unchecked Sendable {
    let metalState: MetalPromptState
    public let promptTokenCount: Int

    init(metalState: MetalPromptState, promptTokenCount: Int) {
        self.metalState = metalState
        self.promptTokenCount = promptTokenCount
    }
}

public enum ModelContainerError: Error {
    case invalidPrefillResult
}
