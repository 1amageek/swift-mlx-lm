import Foundation

/// Identity and configuration of a loaded model.
public struct ModelConfiguration: Sendable {
    /// Model name (from config.json or directory name).
    public var name: String
    /// EOS token IDs for stopping generation.
    public var eosTokenIds: Set<Int>
    /// Declared input modalities from the model bundle metadata.
    public var inputCapabilities: ModelInputCapabilities
    /// Execution capabilities of the active runtime for this bundle.
    public var executionCapabilities: ModelExecutionCapabilities
    /// Vision-related metadata declared by the model bundle.
    public var vision: ModelVisionConfiguration?

    public init(
        name: String = "model",
        eosTokenIds: Set<Int> = [],
        inputCapabilities: ModelInputCapabilities = .textOnly,
        executionCapabilities: ModelExecutionCapabilities = .init(),
        vision: ModelVisionConfiguration? = nil
    ) {
        self.name = name
        self.eosTokenIds = eosTokenIds
        self.inputCapabilities = inputCapabilities
        self.executionCapabilities = executionCapabilities
        self.vision = vision
    }
}
