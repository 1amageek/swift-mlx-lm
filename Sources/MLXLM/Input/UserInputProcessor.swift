/// Converts user-facing input into tokenized model input.
public protocol UserInputProcessor: Sendable {
    /// Prepare full input for generation.
    func prepare(input: UserInput) async throws -> LMInput

    /// Prepare prefix-only input for cache warming.
    func preparePrefix(input: UserInput) async throws -> LMInput
}

extension UserInputProcessor {
    public func preparePrefix(input: UserInput) async throws -> LMInput {
        try await prepare(input: input)
    }
}
