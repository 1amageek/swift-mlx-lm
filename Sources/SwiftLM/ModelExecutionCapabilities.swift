/// Execution capabilities of the current runtime for a loaded bundle.
///
/// This is distinct from ``ModelInputCapabilities``, which describes only what
/// the bundle declares in its metadata.
public struct ModelExecutionCapabilities: Sendable, Equatable {
    public var supportsTextGeneration: Bool
    public var supportsPromptStateReuse: Bool
    public var supportsImagePromptPreparation: Bool
    public var supportsImageExecution: Bool
    public var supportsVideoPromptPreparation: Bool
    public var supportsVideoExecution: Bool

    public init(
        supportsTextGeneration: Bool = true,
        supportsPromptStateReuse: Bool = true,
        supportsImagePromptPreparation: Bool = false,
        supportsImageExecution: Bool = false,
        supportsVideoPromptPreparation: Bool = false,
        supportsVideoExecution: Bool = false
    ) {
        self.supportsTextGeneration = supportsTextGeneration
        self.supportsPromptStateReuse = supportsPromptStateReuse
        self.supportsImagePromptPreparation = supportsImagePromptPreparation
        self.supportsImageExecution = supportsImageExecution
        self.supportsVideoPromptPreparation = supportsVideoPromptPreparation
        self.supportsVideoExecution = supportsVideoExecution
    }

}
