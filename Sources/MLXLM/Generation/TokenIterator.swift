import MLX

/// Iterates over generated tokens from a language model.
///
/// Handles prefill (prompt processing) and autoregressive decoding.
/// Each call to `next()` performs one forward pass and samples a token.
struct TokenIterator: Sequence, IteratorProtocol {

    private let model: any LanguageModel
    private let cache: [KVCache]
    private var sampler: any LogitSampler
    private var processor: (any LogitProcessor)?
    private let prefillStepSize: Int
    private let maxTokens: Int?
    private let eosTokenIds: Set<Int>

    private var inputText: LMInput.Text
    private var prefillComplete: Bool = false
    private var tokenCount: Int = 0

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
        self.inputText = input.text

        // Initialize repetition processor with prompt tokens
        self.processor?.prompt(input.text.tokens)
    }

    mutating func next() -> Int? {
        if let max = maxTokens, tokenCount >= max {
            return nil
        }

        do {
            let output: LMOutput

            if !prefillComplete {
                let result = try model.prepare(
                    LMInput(text: inputText),
                    cache: cache,
                    windowSize: prefillStepSize
                )
                switch result {
                case .tokens(let remaining):
                    inputText = remaining
                    // Prefill chunk processed, recurse for next chunk
                    return next()
                case .logits(let logits):
                    output = logits
                    prefillComplete = true
                }
            } else {
                output = model.callAsFunction(inputText, cache: cache, state: nil)
            }

            // Sample from logits
            var logits = output.logits[0..., (-1)..., 0...]
            logits = logits.squeezed(axis: 0)

            if let proc = processor {
                logits = proc.process(logits: logits)
            }

            let tokenArray = sampler.sample(logits: logits)
            eval(tokenArray)

            let tokenID: Int32 = tokenArray.item()
            let token = Int(tokenID)

            processor?.didSample(token: tokenArray)

            // Check EOS
            if eosTokenIds.contains(token) {
                return nil
            }

            // Prepare next input
            inputText = LMInput.Text(tokens: MLXArray([tokenID]).reshaped([1, 1]))
            tokenCount += 1

            return token
        } catch {
            return nil
        }
    }
}
