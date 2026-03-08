import GGUFTokenizer

/// Bundles a loaded model with its tokenizer and input processor.
///
/// Safety: Always used within `ModelContainer` actor which serializes all access.
/// Contains non-Sendable `Module` subclass references that cannot conform to `Sendable`.
public struct ModelContext: @unchecked Sendable {

    public var configuration: ModelConfiguration
    public var model: any LanguageModel
    public var processor: any UserInputProcessor
    public var tokenizer: any Tokenizer

    public init(
        configuration: ModelConfiguration,
        model: any LanguageModel,
        processor: any UserInputProcessor,
        tokenizer: any Tokenizer
    ) {
        self.configuration = configuration
        self.model = model
        self.processor = processor
        self.tokenizer = tokenizer
    }
}
