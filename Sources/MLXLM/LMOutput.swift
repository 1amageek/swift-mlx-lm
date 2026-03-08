import MLX

/// Model forward-pass output.
public struct LMOutput {

    public let logits: MLXArray
    public let state: State?

    public struct State {
        public let crossAttentionStates: MLXArray?

        public init(crossAttentionStates: MLXArray? = nil) {
            self.crossAttentionStates = crossAttentionStates
        }
    }

    public init(logits: MLXArray, state: State? = nil) {
        self.logits = logits
        self.state = state
    }
}

/// Result of `LanguageModel.prepare(_:cache:windowSize:)`.
public enum PrepareResult {
    /// More tokens to process (prefill chunking).
    case tokens(LMInput.Text)
    /// Final logits ready for sampling.
    case logits(LMOutput)
}
