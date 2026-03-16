/// Parameters controlling text generation.
public struct GenerateParameters: Sendable {
    /// Maximum tokens to generate. nil = unlimited.
    public var maxTokens: Int?
    /// Sampling temperature. 0 = greedy.
    public var temperature: Float
    /// Top-p (nucleus) sampling threshold.
    public var topP: Float
    /// Repetition penalty factor. nil = disabled.
    public var repetitionPenalty: Float?
    /// Number of recent tokens to consider for repetition penalty.
    public var repetitionContextSize: Int

    public init(
        maxTokens: Int? = nil,
        temperature: Float = 0.6,
        topP: Float = 1.0,
        repetitionPenalty: Float? = nil,
        repetitionContextSize: Int = 20
    ) {
        self.maxTokens = maxTokens
        self.temperature = temperature
        self.topP = topP
        self.repetitionPenalty = repetitionPenalty
        self.repetitionContextSize = repetitionContextSize
    }
}
