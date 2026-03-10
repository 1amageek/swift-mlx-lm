import MLX
import MLXFast

/// Selects the next token from logits.
protocol LogitSampler {
    func sample(logits: MLXArray) -> MLXArray
}

/// Transforms logits before sampling (e.g., repetition penalty).
protocol LogitProcessor: Sendable {
    mutating func prompt(_ prompt: MLXArray)
    func process(logits: MLXArray) -> MLXArray
    mutating func didSample(token: MLXArray)
}

// MARK: - Sampler Implementations

/// Greedy sampler: always picks the highest-probability token.
struct ArgMaxSampler: LogitSampler {
    init() {}

    func sample(logits: MLXArray) -> MLXArray {
        MLX.argMax(logits, axis: -1)
    }
}

/// Top-p (nucleus) sampler with temperature.
struct TopPSampler: LogitSampler {
    let temperature: Float
    let topP: Float

    init(temperature: Float, topP: Float) {
        self.temperature = temperature
        self.topP = topP
    }

    func sample(logits: MLXArray) -> MLXArray {
        var logits = logits
        if temperature != 1.0 {
            logits = logits * (1.0 / temperature)
        }

        let probs = softmax(logits, axis: -1)

        let sortedIndices = argSort(probs, axis: -1)
        let sortedProbs = probs.take(sortedIndices, axis: -1)

        let cumProbs = cumsum(sortedProbs, axis: -1)
        let mask = cumProbs .< (1.0 - topP)
        let maskedProbs = MLX.where(mask, sortedProbs, MLXArray(0.0))

        let totalProb = maskedProbs.sum(axis: -1, keepDims: true)
        let safeTotal = MLX.where(totalProb .> 0, totalProb, MLXArray(1.0))
        let normalizedProbs = maskedProbs / safeTotal

        let sampled = categorical(normalizedProbs)
        return sortedIndices.take(sampled, axis: -1)
    }
}

/// Categorical sampler with temperature (no top-p filtering).
struct CategoricalSampler: LogitSampler {
    let temperature: Float

    init(temperature: Float) {
        self.temperature = temperature
    }

    func sample(logits: MLXArray) -> MLXArray {
        var logits = logits
        if temperature != 1.0 {
            logits = logits * (1.0 / temperature)
        }
        return categorical(softmax(logits, axis: -1))
    }
}

// MARK: - Processor Implementations

/// Applies a repetition penalty to recently-generated tokens.
struct RepetitionProcessor: LogitProcessor, Sendable {
    let penalty: Float
    let contextSize: Int
    var recentTokens: [Int]

    init(repetitionPenalty: Float, repetitionContextSize: Int) {
        self.penalty = repetitionPenalty
        self.contextSize = repetitionContextSize
        self.recentTokens = []
    }

    mutating func prompt(_ prompt: MLXArray) {
        let flat = prompt.flattened()
        eval(flat)
        let tokens: [Int32] = flat.asArray(Int32.self)
        recentTokens = tokens.map { Int($0) }
        if recentTokens.count > contextSize {
            recentTokens = Array(recentTokens.suffix(contextSize))
        }
    }

    func process(logits: MLXArray) -> MLXArray {
        guard !recentTokens.isEmpty, penalty != 1.0 else {
            return logits
        }

        let originalShape = logits.shape
        let flat = logits.reshaped(-1)
        let vocabSize = flat.dim(0)

        let validTokens = recentTokens.filter { $0 >= 0 && $0 < vocabSize }
        guard !validTokens.isEmpty else { return logits }

        let indices = MLXArray(validTokens.map { Int32($0) })
        let selected = flat.take(indices, axis: 0)

        // Divide positive logits by penalty, multiply negative logits by penalty
        let penalized = MLX.where(
            selected .> 0,
            selected / MLXArray(penalty),
            selected * MLXArray(penalty)
        )

        // Scatter penalized values back in one operation
        let result = flat
        result[indices] = penalized
        return result.reshaped(originalShape)
    }

    mutating func didSample(token: MLXArray) {
        // token is already materialized by the caller's .item() call
        let tokenID: Int32 = token.item()
        recentTokens.append(Int(tokenID))
        if recentTokens.count > contextSize {
            recentTokens.removeFirst()
        }
    }
}

// MARK: - Factory

extension GenerateParameters {

    /// Create a sampler matching these parameters.
    func sampler() -> any LogitSampler {
        if temperature == 0 {
            return ArgMaxSampler()
        } else if topP < 1.0 {
            return TopPSampler(temperature: temperature, topP: topP)
        } else {
            return CategoricalSampler(temperature: temperature)
        }
    }

    /// Create a logit processor if repetition penalty is enabled.
    func processor() -> (any LogitProcessor)? {
        guard let penalty = repetitionPenalty, penalty != 1.0 else { return nil }
        return RepetitionProcessor(
            repetitionPenalty: penalty,
            repetitionContextSize: repetitionContextSize
        )
    }
}
