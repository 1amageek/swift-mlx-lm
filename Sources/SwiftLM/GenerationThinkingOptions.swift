/// Per-request controls for whether reasoning tags are included in generated user-visible output.
public struct GenerationThinkingOptions: Sendable, Equatable {
    /// When true, streamed and final visible output includes `<think>...</think>` content.
    public var includeInOutput: Bool

    public init(includeInOutput: Bool = false) {
        self.includeInOutput = includeInOutput
    }

    public static let hidden = GenerationThinkingOptions(includeInOutput: false)
    public static let visible = GenerationThinkingOptions(includeInOutput: true)
}
