import MLX

/// Iterates over generated tokens from a language model.
///
/// Handles prefill (prompt processing) and autoregressive decoding.
/// Uses asyncEval pipelining: each `next()` returns the previous token
/// while GPU computes the next one in parallel.
struct TokenIterator: Sequence, IteratorProtocol {

    private let model: any LanguageModel
    private let cache: [KVCache]
    private var sampler: any LogitSampler
    private var processor: (any LogitProcessor)?
    private let prefillStepSize: Int
    private let maxTokens: Int?
    private let eosTokenIds: Set<Int>

    private var fullInput: LMInput
    private var inputText: LMInput.Text
    private var prefillComplete: Bool = false
    private var tokenCount: Int = 0

    /// Pending token from the previous step, already submitted via asyncEval.
    private var pendingToken: MLXArray?

    /// Create a token iterator.
    ///
    /// - Parameters:
    ///   - input: Tokenized input to generate from.
    ///   - model: Language model for forward passes.
    ///   - cache: KV caches (one per layer).
    ///   - parameters: Generation parameters.
    ///   - eosTokenIds: Token IDs that signal end of generation.
    init(
        input: LMInput,
        model: any LanguageModel,
        cache: [KVCache]? = nil,
        parameters: GenerateParameters,
        eosTokenIds: Set<Int> = []
    ) throws {
        self.model = model
        self.cache = cache ?? model.newCache(parameters: parameters)
        self.sampler = parameters.sampler()
        self.processor = parameters.processor()
        self.prefillStepSize = parameters.prefillStepSize
        self.maxTokens = parameters.maxTokens
        self.eosTokenIds = eosTokenIds
        self.fullInput = input
        self.inputText = input.text

        // Initialize repetition processor with prompt tokens
        self.processor?.prompt(input.text.tokens)
    }

    /// Sample a token from model output logits.
    private mutating func sampleToken(from output: LMOutput) -> MLXArray {
        var logits = output.logits[0..., (-1)..., 0...]
        logits = logits.squeezed(axis: 0)
        if let proc = processor {
            logits = proc.process(logits: logits)
        }
        return sampler.sample(logits: logits)
    }

    /// Run one forward pass on the current inputText and sample.
    private mutating func step() -> MLXArray {
        let output = model.callAsFunction(inputText, cache: cache, state: nil)
        return sampleToken(from: output)
    }

    mutating func next() -> Int? {
        if let max = maxTokens, tokenCount >= max {
            return nil
        }

        do {
            // Prefill: process prompt chunks, then sample first token
            if !prefillComplete {
                let output: LMOutput
                while true {
                    let result = try model.prepare(
                        fullInput,
                        cache: cache,
                        windowSize: prefillStepSize
                    )
                    switch result {
                    case .tokens(let remaining):
                        inputText = remaining
                        fullInput = LMInput(text: inputText, image: fullInput.image, video: fullInput.video)
                        continue
                    case .logits(let logits):
                        output = logits
                        break
                    }
                    break
                }

                let token = sampleToken(from: output)
                pendingToken = token
                asyncEval(token)
                prefillComplete = true
            }

            guard let previousToken = pendingToken else { return nil }

            // Extract the previous token value (blocks until asyncEval completes)
            let tokenID: Int32 = previousToken.item()
            let token = Int(tokenID)

            // Update processor state with the materialized token
            processor?.didSample(token: previousToken)

            // Check EOS
            if eosTokenIds.contains(token) {
                pendingToken = nil
                return nil
            }

            // Build computation graph for the next token (lazy, no GPU wait)
            inputText = LMInput.Text(tokens: MLXArray([tokenID]).reshaped([1, 1]))
            let nextToken = step()
            pendingToken = nextToken
            // Submit next token to GPU — computation runs in parallel with caller's CPU work
            asyncEval(nextToken)

            tokenCount += 1
            return token
        } catch {
            print("[TokenIterator] error during generation: \(error)")
            return nil
        }
    }
}
