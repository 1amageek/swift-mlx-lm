import Foundation
import MLX
import MLXNN
import MLXFast

/// Protocol for autoregressive language models.
public protocol LanguageModel: Module {

    /// Prepare input for generation, potentially chunking large prompts.
    ///
    /// - Parameters:
    ///   - input: Full tokenized input.
    ///   - cache: KV caches (one per layer).
    ///   - windowSize: Maximum chunk size for prefill.
    /// - Returns: Either more tokens to process or final logits.
    func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws -> PrepareResult

    /// Forward pass: compute logits for the given token sequence.
    func callAsFunction(_ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?) -> LMOutput

    /// Forward pass with raw token array (convenience).
    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray

    /// Create a fresh set of KV caches for this model.
    func newCache(parameters: GenerateParameters?) -> [KVCache]

    /// Filter/transform loaded weights before applying to the model.
    func sanitize(weights: [String: MLXArray]) -> [String: MLXArray]

    /// Number of transformer layers (for cache creation).
    var layerCount: Int { get }

    /// Number of KV heads per layer (for quantized cache sizing).
    var kvHeads: [Int] { get }
}

// MARK: - Default Implementations

extension LanguageModel {

    public func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws -> PrepareResult {
        let tokens = input.text.tokens
        let tokenCount = tokens.dim(tokens.ndim - 1)
        let windowSize = windowSize ?? tokenCount
        let prefillOffset = cache.first?.offset ?? 0

        if prefillOffset + windowSize < tokenCount {
            let chunkStart = Date()
            let chunk = tokens[0..., prefillOffset..<(prefillOffset + windowSize)]
            let chunkInput = LMInput.Text(tokens: chunk)
            let output = callAsFunction(chunkInput, cache: cache, state: nil)
            // Force evaluation per chunk to prevent computation graph accumulation.
            // Critical for recurrent layers (DeltaNet) where per-token loops create
            // O(T * layers) graph nodes per chunk.
            eval(output.logits)
            let chunkTime = Date().timeIntervalSince(chunkStart)
            print("[prepare] prefill chunk offset=\(prefillOffset)/\(tokenCount) window=\(windowSize) time=\(String(format: "%.2f", chunkTime))s")
            return .tokens(LMInput.Text(tokens: tokens))
        }

        let remaining = tokens[0..., prefillOffset...]
        let output = callAsFunction(
            LMInput.Text(tokens: remaining),
            cache: cache,
            state: nil
        )
        return .logits(output)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let input = LMInput.Text(tokens: inputs)
        return callAsFunction(input, cache: cache, state: nil).logits
    }

    public func newCache(parameters: GenerateParameters?) -> [KVCache] {
        let params = parameters ?? GenerateParameters()
        return createKVCaches(layerCount: layerCount, parameters: params)
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        weights
    }
}

// MARK: - KVCacheDimensionProvider

/// Models that can report their KV head counts per layer.
protocol KVCacheDimensionProvider {
    var kvHeads: [Int] { get }
}

// MARK: - Attention Utilities

/// Create a causal attention mask suitable for the given hidden state and cache.
func createAttentionMask(h: MLXArray, cache: KVCache?) -> MLXFast.ScaledDotProductAttentionMaskMode {
    let queryLength = h.dim(1)
    if queryLength > 1 {
        return .causal
    }
    return .none
}
