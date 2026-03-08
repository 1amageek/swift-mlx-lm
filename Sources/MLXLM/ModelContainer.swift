import MLX
import GGUFTokenizer

/// Thread-safe container providing serial access to a loaded model.
///
/// All model operations (prepare, generate, decode) are serialized
/// through this actor to prevent concurrent GPU access.
public actor ModelContainer {

    private var context: ModelContext

    public init(context: ModelContext) {
        self.context = context
    }

    /// Model configuration (accessible without awaiting).
    public var configuration: ModelConfiguration {
        context.configuration
    }

    /// Execute an arbitrary action with serial access to the model context.
    public func perform<R: Sendable>(
        _ action: @Sendable (ModelContext) throws -> R
    ) throws -> R {
        try action(context)
    }

    /// Execute an action with additional values passed in.
    public func perform<V: Sendable, R: Sendable>(
        values: V,
        _ action: @Sendable (ModelContext, V) throws -> R
    ) throws -> R {
        try action(context, values)
    }

    /// Mutate the model context.
    public func update(_ action: @Sendable (inout ModelContext) -> Void) {
        action(&context)
    }

    // MARK: - Convenience Methods

    /// Convert user input into tokenized model input.
    public func prepare(input: UserInput) async throws -> LMInput {
        try await context.processor.prepare(input: input)
    }

    /// Convert user input into prefix-only tokenized input (for cache warming).
    public func preparePrefix(input: UserInput) async throws -> LMInput {
        try await context.processor.preparePrefix(input: input)
    }

    /// Decode token IDs back to text.
    public func decode(tokens: [Int]) -> String {
        context.tokenizer.decode(tokens: tokens)
    }

    /// Encode text to token IDs.
    public func encode(_ text: String) -> [Int] {
        context.tokenizer.encode(text: text)
    }

    /// Generate text from tokenized input.
    ///
    /// Returns an async stream of `Generation` elements:
    /// - `.chunk(String)` for decoded text segments
    /// - `.toolCall(ToolCall)` for detected tool invocations
    /// - `.info(GenerateCompletionInfo)` as the final element
    public func generate(
        input: LMInput,
        cache: [KVCache]? = nil,
        parameters: GenerateParameters
    ) throws -> AsyncStream<Generation> {
        try MLXLM.generate(
            input: input,
            cache: cache,
            parameters: parameters,
            context: context
        )
    }
}
