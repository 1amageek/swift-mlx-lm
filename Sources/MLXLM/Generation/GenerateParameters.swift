/// Parameters controlling text generation.
public struct GenerateParameters: Sendable {

    /// Maximum tokens per prefill step.
    public var prefillStepSize: Int

    /// Maximum tokens to generate (nil = unlimited).
    public var maxTokens: Int?

    /// Maximum KV cache size (nil = unlimited).
    public var maxKVSize: Int?

    /// Quantization bits for KV cache (nil = no quantization).
    public var kvBits: Int?

    /// Group size for KV cache quantization.
    public var kvGroupSize: Int

    /// Layer index at which to start KV quantization.
    public var quantizedKVStart: Int

    /// Sampling temperature (0 = greedy).
    public var temperature: Float

    /// Top-p (nucleus) sampling threshold.
    public var topP: Float

    /// Repetition penalty factor (nil = disabled).
    public var repetitionPenalty: Float?

    /// Token window size for repetition penalty.
    public var repetitionContextSize: Int

    /// Number of prefix tokens reused from a previous generation.
    public var reusedPrefixTokenCount: Int

    public init(
        maxTokens: Int? = nil,
        maxKVSize: Int? = nil,
        kvBits: Int? = nil,
        kvGroupSize: Int = 64,
        quantizedKVStart: Int = 0,
        temperature: Float = 0.6,
        topP: Float = 1.0,
        repetitionPenalty: Float? = nil,
        repetitionContextSize: Int = 20,
        reusedPrefixTokenCount: Int = 0,
        prefillStepSize: Int = 512
    ) {
        self.maxTokens = maxTokens
        self.maxKVSize = maxKVSize
        self.kvBits = kvBits
        self.kvGroupSize = kvGroupSize
        self.quantizedKVStart = quantizedKVStart
        self.temperature = temperature
        self.topP = topP
        self.repetitionPenalty = repetitionPenalty
        self.repetitionContextSize = repetitionContextSize
        self.reusedPrefixTokenCount = reusedPrefixTokenCount
        self.prefillStepSize = prefillStepSize
    }
}
