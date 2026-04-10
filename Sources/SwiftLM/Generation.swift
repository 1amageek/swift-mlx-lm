/// Streaming output from inference.
public enum Generation: Sendable {
    /// A text chunk (one or more decoded tokens).
    case chunk(String)
    /// A reasoning chunk extracted using the active chat template's reasoning tags.
    case reasoningChunk(String)
    /// Completion information (token count, timing, etc.).
    case info(CompletionInfo)

    /// The text chunk, if this is a `.chunk` case.
    public var chunk: String? {
        if case .chunk(let text) = self { return text }
        return nil
    }

    /// The reasoning chunk, if this is a `.reasoningChunk` case.
    public var reasoningChunk: String? {
        if case .reasoningChunk(let text) = self { return text }
        return nil
    }

    /// The completion info, if this is an `.info` case.
    public var info: CompletionInfo? {
        if case .info(let info) = self { return info }
        return nil
    }
}

/// Information about a completed generation.
public struct CompletionInfo: Sendable {
    /// Total tokens generated.
    public let tokenCount: Int
    /// Tokens per second.
    public let tokensPerSecond: Double
    /// Total generation time in seconds.
    public let totalTime: Double

    public init(tokenCount: Int, tokensPerSecond: Double, totalTime: Double) {
        self.tokenCount = tokenCount
        self.tokensPerSecond = tokensPerSecond
        self.totalTime = totalTime
    }
}
