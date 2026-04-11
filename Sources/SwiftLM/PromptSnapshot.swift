import MetalCompiler

/// A reusable snapshot of decode state for a prepared prompt prefix.
///
/// Build a prompt snapshot with ``LanguageModelContext/makePromptSnapshot(from:)`` or
/// ``LanguageModelContext/makePromptSnapshot(from:)-(ModelInput)`` and reuse it with
/// ``LanguageModelContext/generate(from:parameters:)``.
///
/// `PromptSnapshot` is context-affine runtime state. Reuse it only with the same
/// `LanguageModelContext` instance and the same loaded model bundle that produced it.
public struct PromptSnapshot: @unchecked Sendable {
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

/// Errors produced by ``LanguageModelContext``.
public enum LanguageModelContextError: Error {
    /// Prefill did not produce a valid first token.
    case invalidPrefillResult
    /// The input asks for a modality the loaded model does not declare.
    case unsupportedInputForModel(String)
    /// The input includes multimodal content that the active runtime or model
    /// family does not currently support.
    case multimodalInputNotSupported(String)
    /// Restoring a prompt snapshot failed before generation could start.
    case promptSnapshotRestoreFailed(String)
    /// Prompt preparation options contained incompatible thinking controls.
    case conflictingPromptThinkingConfiguration(String)
    /// A known prompt-template variable was provided with an invalid value type.
    case invalidPromptTemplateVariable(String)
}
