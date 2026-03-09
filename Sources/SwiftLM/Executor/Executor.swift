/// Runtime layer responsible for forward execution of compiled models.
///
/// The executor owns all runtime concerns:
/// - Forward pass execution
/// - KV cache management
/// - Mask construction
/// - Device/backend/dtype selection
/// - Generation loop integration
///
/// The executor is intentionally separated from model declaration.
/// Forward-pass logic belongs here, not in `LanguageModel` or `ModelComponent`.
public protocol Executor: Sendable {

    /// Execute a compiled model on the given inputs.
    func run(
        _ model: CompiledModel,
        inputs: ModelInputs
    ) async throws -> ModelOutputs
}
