/// Per-request controls for whether template-defined reasoning content is
/// included in user-visible output.
public struct GenerationThinkingOptions: Sendable, Equatable {
    /// When true, streamed and final visible output keeps reasoning inline.
    public var includeInOutput: Bool

    /// When true, reasoning extracted from template-defined reasoning tags is emitted via
    /// `Generation.reasoningChunk` instead of remaining inline in visible output.
    ///
    /// This takes precedence over inline visibility so callers can keep answer and
    /// reasoning on separate channels.
    public var emitSeparately: Bool

    public init(includeInOutput: Bool = false, emitSeparately: Bool = false) {
        self.includeInOutput = includeInOutput
        self.emitSeparately = emitSeparately
    }

    public static let hidden = GenerationThinkingOptions(includeInOutput: false, emitSeparately: false)
    public static let visible = GenerationThinkingOptions(includeInOutput: true, emitSeparately: false)
    public static let separate = GenerationThinkingOptions(includeInOutput: false, emitSeparately: true)
}
