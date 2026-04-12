import Foundation

/// Immutable, shareable container for a compiled language model bundle.
///
/// A container owns the loaded model assets, tokenizer, templates, and compile
/// products. This is the primary public entry point for most application code.
/// Initialize ``LanguageModelContext`` with it only when you need isolated
/// mutable inference state, explicit prompt staging, or prompt snapshot reuse.
public final class LanguageModelContainer: @unchecked Sendable {
    let prototypeContext: LanguageModelContext

    init(prototypeContext: LanguageModelContext) {
        self.prototypeContext = prototypeContext
    }

    /// Model configuration (name, EOS tokens, capabilities).
    public var configuration: ModelConfiguration {
        prototypeContext.configuration
    }

    /// Prepare user-facing input into rendered text, tokens, and prompt metadata.
    ///
    /// Most callers can skip this and use ``generate(_:parameters:)`` directly.
    public func prepare(_ input: ModelInput) async throws -> PreparedPrompt {
        try await prototypeContext.prepare(input)
    }

    /// Decode token IDs to text.
    public func decode(_ tokenIDs: [Int], skipSpecialTokens: Bool = true) -> String {
        prototypeContext.decode(tokenIDs, skipSpecialTokens: skipSpecialTokens)
    }

    /// Encode text to token IDs.
    public func encode(_ text: String, addSpecialTokens: Bool = true) -> [Int] {
        prototypeContext.encode(text, addSpecialTokens: addSpecialTokens)
    }

    /// Convenience one-shot generation from an executable prompt.
    ///
    /// Internally creates a fresh ``LanguageModelContext`` so repeated requests
    /// do not share decode-time mutable state. Prefer
    /// ``generate(_:parameters:)`` unless you need explicit prompt staging.
    public func generate(
        from prompt: ExecutablePrompt,
        parameters: GenerationParameters = GenerationParameters()
    ) throws -> AsyncStream<GenerationEvent> {
        let context = try LanguageModelContext(self)
        return try context.generate(from: prompt, parameters: parameters)
    }

    /// Convenience one-shot generation from user input.
    ///
    /// Internally creates a fresh ``LanguageModelContext`` so repeated requests
    /// do not share decode-time mutable state. This is the recommended
    /// high-level entry point for most applications.
    public func generate(
        _ input: ModelInput,
        parameters: GenerationParameters = GenerationParameters()
    ) async throws -> AsyncStream<GenerationEvent> {
        let context = try LanguageModelContext(self)
        return try await context.generate(input, parameters: parameters)
    }
}
