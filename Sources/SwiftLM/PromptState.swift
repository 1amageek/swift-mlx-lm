import MetalCompiler

/// A reusable snapshot of decode state for a prepared prompt prefix.
///
/// Build a prompt state with ``ModelContainer/makePromptState(prompt:)`` or
/// ``ModelContainer/makePromptState(input:)-(ModelInput)`` and reuse it with
/// ``ModelContainer/generate(from:parameters:)``.
public struct PromptState: @unchecked Sendable {
    let metalState: MetalPromptState
    let ropePositionOffset: Int
    let samplingSeed: UInt64
    let promptTokenTail: [Int]
    /// Number of tokens in the prompt prefix used to create this state.
    public let promptTokenCount: Int

    init(
        metalState: MetalPromptState,
        promptTokenCount: Int,
        ropePositionOffset: Int = 0,
        samplingSeed: UInt64,
        promptTokenTail: [Int]
    ) {
        self.metalState = metalState
        self.ropePositionOffset = ropePositionOffset
        self.samplingSeed = samplingSeed
        self.promptTokenTail = promptTokenTail
        self.promptTokenCount = promptTokenCount
    }
}

/// Errors produced by ``ModelContainer``.
public enum ModelContainerError: Error {
    /// Prefill did not produce a valid first token.
    case invalidPrefillResult
    /// The input asks for a modality the loaded model does not declare.
    case unsupportedInputForModel(String)
    /// The input includes multimodal content that the active runtime or model
    /// family does not currently support.
    case multimodalInputNotSupported(String)
    /// Restoring a prompt state failed before generation could start.
    case promptStateRestoreFailed(String)
}
