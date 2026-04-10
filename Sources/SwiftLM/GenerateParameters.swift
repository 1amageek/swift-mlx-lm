/// Parameters controlling text generation.
public struct GenerateParameters: Sendable {
    /// Maximum tokens to generate. nil uses the runtime default cap.
    public var maxTokens: Int?
    /// Maximum number of tokens to coalesce into one streamed text chunk.
    public var streamChunkTokenCount: Int
    /// Sampling temperature. 0 = greedy.
    public var temperature: Float
    /// Top-p (nucleus) sampling threshold.
    public var topP: Float
    /// Limit sampling to the highest-probability K tokens. nil = disabled.
    public var topK: Int?
    /// Minimum probability threshold relative to the best candidate. 0 = disabled.
    public var minP: Float
    /// Repetition penalty factor. nil = disabled.
    public var repetitionPenalty: Float?
    /// Penalize tokens that have already appeared in the recent context.
    public var presencePenalty: Float?
    /// Number of recent tokens to consider for repetition penalty.
    public var repetitionContextSize: Int
    /// Request-level controls for whether reasoning tags are included in visible output.
    public var thinking: GenerationThinkingOptions

    public init(
        maxTokens: Int? = nil,
        streamChunkTokenCount: Int = 8,
        temperature: Float = 0.6,
        topP: Float = 1.0,
        topK: Int? = nil,
        minP: Float = 0,
        repetitionPenalty: Float? = nil,
        presencePenalty: Float? = nil,
        repetitionContextSize: Int = 20,
        thinking: GenerationThinkingOptions = .hidden
    ) {
        self.maxTokens = maxTokens
        self.streamChunkTokenCount = streamChunkTokenCount
        self.temperature = temperature
        self.topP = topP
        self.topK = topK
        self.minP = minP
        self.repetitionPenalty = repetitionPenalty
        self.presencePenalty = presencePenalty
        self.repetitionContextSize = repetitionContextSize
        self.thinking = thinking
    }
}
