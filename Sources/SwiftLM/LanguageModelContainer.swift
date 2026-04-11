import Foundation

/// Immutable, shareable container for a compiled language model bundle.
///
/// A container owns the loaded model assets, tokenizer, templates, and compile
/// products. Create ``LanguageModelContext`` instances from it when you need
/// isolated mutable inference state.
public final class LanguageModelContainer: @unchecked Sendable {
    private let prototypeContext: LanguageModelContext

    init(prototypeContext: LanguageModelContext) {
        self.prototypeContext = prototypeContext
    }

    /// Model configuration (name, EOS tokens, capabilities).
    public var configuration: ModelConfiguration {
        prototypeContext.configuration
    }

    /// Create a fresh mutable context backed by this container's compiled model.
    public func makeContext() throws -> LanguageModelContext {
        try prototypeContext.cloneContext(compiledModel: prototypeContext.debugCompiledModel)
    }

    /// Prepare user-facing input into rendered text, tokens, and prompt metadata.
    public func prepare(_ input: ModelInput) async throws -> PreparedPrompt {
        try await prototypeContext.prepare(input)
    }

    /// Convert a prepared prompt into the executable runtime form.
    public func makeExecutablePrompt(from preparedPrompt: PreparedPrompt) throws -> ExecutablePrompt {
        try prototypeContext.makeExecutablePrompt(from: preparedPrompt)
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
    /// do not share decode-time mutable state.
    public func generate(
        from prompt: ExecutablePrompt,
        parameters: GenerationParameters = GenerationParameters()
    ) throws -> AsyncStream<GenerationEvent> {
        let context = try makeContext()
        return try context.generate(from: prompt, parameters: parameters)
    }

    /// Convenience one-shot generation from user input.
    ///
    /// Internally creates a fresh ``LanguageModelContext`` so repeated requests
    /// do not share decode-time mutable state.
    public func generate(
        _ input: ModelInput,
        parameters: GenerationParameters = GenerationParameters()
    ) async throws -> AsyncStream<GenerationEvent> {
        let context = try makeContext()
        return try await context.generate(input, parameters: parameters)
    }
}
