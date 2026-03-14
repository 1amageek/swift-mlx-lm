import Foundation

/// Stop reason for text generation.
public enum GenerateStopReason: Sendable {
    case stop
    case length
    case cancelled
}

/// Completion statistics for a generation run.
public struct GenerateCompletionInfo: Sendable {

    public let promptTokenCount: Int
    public let generationTokenCount: Int
    public let promptTime: TimeInterval
    public let generateTime: TimeInterval
    public let stopReason: GenerateStopReason

    public var promptTokensPerSecond: Double {
        promptTokenCount > 0 && promptTime > 0
            ? Double(promptTokenCount) / promptTime : 0
    }

    public var tokensPerSecond: Double {
        generationTokenCount > 0 && generateTime > 0
            ? Double(generationTokenCount) / generateTime : 0
    }

    public init(
        promptTokenCount: Int,
        generationTokenCount: Int,
        promptTime: TimeInterval,
        generateTime: TimeInterval,
        stopReason: GenerateStopReason = .stop
    ) {
        self.promptTokenCount = promptTokenCount
        self.generationTokenCount = generationTokenCount
        self.promptTime = promptTime
        self.generateTime = generateTime
        self.stopReason = stopReason
    }

    public func summary() -> String {
        let pp = String(format: "%.1f", promptTokensPerSecond)
        let tp = String(format: "%.1f", tokensPerSecond)
        return "Prompt: \(promptTokenCount) tokens (\(pp) tok/s), "
            + "Generation: \(generationTokenCount) tokens (\(tp) tok/s)"
    }
}

/// Streamed generation output element.
public enum Generation: Sendable {
    /// A chunk of decoded text.
    case chunk(String)
    /// A parsed tool call.
    case toolCall(ToolCall)
    /// Final completion info (always the last element).
    case info(GenerateCompletionInfo)

    public var chunk: String? {
        if case .chunk(let text) = self { return text }
        return nil
    }

    public var info: GenerateCompletionInfo? {
        if case .info(let info) = self { return info }
        return nil
    }

    public var toolCall: ToolCall? {
        if case .toolCall(let call) = self { return call }
        return nil
    }
}
