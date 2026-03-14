import MLX

/// Speculative decoding engine.
///
/// Uses a smaller draft model to propose candidate tokens, then verifies them
/// against the target model in a single batched forward pass. When the draft
/// model's predictions align with the target, multiple tokens are accepted per
/// step, yielding significant throughput improvement.
///
/// Algorithm (greedy verification):
/// 1. Draft model generates K candidate tokens autoregressively.
/// 2. Target model verifies all K+1 positions (current + K candidates) in one forward pass.
/// 3. For each position i, compare target's greedy choice with draft's token:
///    - Match → accept, continue to next position
///    - Mismatch → accept target's choice at position i, reject remaining
/// 4. At least 1 token is always accepted (the target's first choice).
///
/// Both draft and target models must use trimmable caches (pure attention).
/// DeltaNet/recurrent draft models are not supported (rollback requires trimming).
struct SpeculativeGenerator {

    let draft: any LanguageModel
    let draftCache: [KVCache]
    let target: any LanguageModel
    let targetCache: [KVCache]
    let candidateCount: Int
    var sampler: any LogitSampler

    /// Run one speculative decoding step.
    ///
    /// - Parameter lastToken: The last accepted token from the previous step.
    /// - Returns: Array of accepted token IDs (at least 1).
    mutating func step(lastToken: Int32) -> [Int] {
        // 1. Draft K candidate tokens
        var candidates: [Int32] = []
        var currentToken = lastToken

        for _ in 0..<candidateCount {
            let input = LMInput.Text(tokens: MLXArray([currentToken]).reshaped([1, 1]))
            let output = draft.callAsFunction(input, cache: draftCache, state: nil)
            let logits = output.logits[0..., (-1)..., 0...].squeezed(axis: 0)
            let sampled = sampler.sample(logits: logits)
            let tokenID: Int32 = sampled.item()
            candidates.append(tokenID)
            currentToken = tokenID
        }

        // 2. Build verification input: [lastToken, d1, d2, ..., dK]
        let verifyTokens = [lastToken] + candidates
        let verifyInput = LMInput.Text(
            tokens: MLXArray(verifyTokens).reshaped([1, verifyTokens.count])
        )

        // 3. Target forward pass on all K+1 positions
        let targetOutput = target.callAsFunction(verifyInput, cache: targetCache, state: nil)
        let targetLogits = targetOutput.logits.squeezed(axis: 0)  // [K+1, vocab]

        // 4. Verify: compare target's greedy choice at each position with draft's token
        var accepted: [Int] = []

        for i in 0..<candidateCount {
            let posLogits = targetLogits[i]
            let targetSampled = sampler.sample(logits: posLogits)
            let targetToken: Int32 = targetSampled.item()

            if targetToken == candidates[i] {
                accepted.append(Int(targetToken))
            } else {
                // Mismatch: accept target's choice, reject rest
                accepted.append(Int(targetToken))

                // Trim draft cache: remove tokens after the divergence point.
                // We fed K tokens to draft. Accepted i+1 total (0..i from draft match, plus
                // the target's choice at i). Draft cache has advanced by K tokens beyond
                // lastToken. We need to roll back K - (i+1) = K - accepted.count tokens.
                // But actually we need to roll back to just after the last accepted token.
                // Draft saw: lastToken → d0 → d1 → ... → d(K-1). That's K decode steps.
                // We accepted i+1 tokens. Draft should have cache for i+1 tokens beyond lastToken.
                // Roll back: K - (i + 1) tokens.
                let draftRollback = candidateCount - (i + 1)
                if draftRollback > 0 {
                    for cache in draftCache { cache.trim(draftRollback) }
                }

                // Target cache: we fed K+1 tokens. We accepted i+1 tokens.
                // Target should have cache up to position of accepted[i].
                // Roll back: K+1 - (i+1) = K - i tokens from target.
                let targetRollback = candidateCount - i
                if targetRollback > 0 {
                    for cache in targetCache { cache.trim(targetRollback) }
                }

                return accepted
            }
        }

        // All K candidates accepted. The target's output at position K gives the next token.
        let lastPosLogits = targetLogits[candidateCount]
        let bonusToken = sampler.sample(logits: lastPosLogits)
        let bonusTokenID: Int32 = bonusToken.item()
        accepted.append(Int(bonusTokenID))

        // Draft cache needs rollback: draft produced K tokens, but the bonus token
        // at position K+1 was produced by the target, not the draft. The draft's
        // cache is K positions ahead. We don't need to roll back because we'll
        // feed the bonus token to the draft on the next step.
        // However, we DO need to trim the draft's last token because the target
        // produced a potentially different token at position K (the bonus).
        // Actually, if all K matched, the draft's cache is exactly right for
        // K candidates. The bonus is an extra token from the target that the
        // draft hasn't seen. We'll feed it to the draft in the next step.
        // No rollback needed for draft or target in this all-match case.

        return accepted
    }

    /// Prefill both draft and target models with the prompt.
    ///
    /// After prefill, both models' caches contain the prompt state and are ready
    /// for speculative decode steps.
    ///
    /// - Parameters:
    ///   - input: The full prompt input.
    ///   - prefillStepSize: Chunk size for prefill (nil = single pass).
    /// - Returns: The first sampled token from the target model.
    mutating func prefill(input: LMInput, prefillStepSize: Int?) throws -> Int32 {
        // Prefill target model
        let targetOutput = try prefillModel(
            target, input: input, cache: targetCache,
            prefillStepSize: prefillStepSize
        )

        // Prefill draft model
        _ = try prefillModel(
            draft, input: input, cache: draftCache,
            prefillStepSize: draft.recommendedPrefillStepSize
        )

        // Sample first token from target
        let logits = targetOutput.logits[0..., (-1)..., 0...].squeezed(axis: 0)
        let token = sampler.sample(logits: logits)
        let tokenID: Int32 = token.item()

        // Feed first token to draft cache (so draft is in sync with target)
        let firstTokenInput = LMInput.Text(tokens: MLXArray([tokenID]).reshaped([1, 1]))
        _ = draft.callAsFunction(firstTokenInput, cache: draftCache, state: nil)

        return tokenID
    }

    private func prefillModel(
        _ model: any LanguageModel,
        input: LMInput,
        cache: [KVCache],
        prefillStepSize: Int?
    ) throws -> LMOutput {
        while true {
            let result = try model.prepare(input, cache: cache, windowSize: prefillStepSize)
            switch result {
            case .tokens:
                continue
            case .logits(let output):
                return output
            }
        }
    }
}
