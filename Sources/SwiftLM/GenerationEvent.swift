/// Streaming output from inference.
public enum GenerationEvent: Sendable {
    /// A text event containing one or more decoded tokens.
    case text(String)
    /// A reasoning event extracted using the active chat template's reasoning tags.
    case reasoning(String)
    /// Completion information (token count, timing, etc.).
    case completed(CompletionInfo)

    /// The visible text payload, if this is a `.text` case.
    public var text: String? {
        if case .text(let text) = self { return text }
        return nil
    }

    /// The reasoning payload, if this is a `.reasoning` case.
    public var reasoning: String? {
        if case .reasoning(let text) = self { return text }
        return nil
    }

    /// The completion info, if this is a `.completed` case.
    public var completion: CompletionInfo? {
        if case .completed(let info) = self { return info }
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
