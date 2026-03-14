import Foundation
import MLX
import MLXNN
import MLXFast
import MLXCompiler

/// LanguageModel backed by a compiled MPSGraph.
///
/// The MPSGraph compiler fuses all layers into a single execution plan.
/// Each forward pass replays the full token history (no incremental KV cache).
public final class MPSGraphLanguageModel: Module, LanguageModel, @unchecked Sendable {

    private let compiled: MPSGraphCompiledModel

    public init(compiled: MPSGraphCompiledModel) {
        self.compiled = compiled
        super.init()
    }

    // MARK: - LanguageModel

    public var layerCount: Int { compiled.metadata.cacheSlotCount }

    public var kvHeads: [Int] {
        compiled.metadata.cacheDescriptors.map { _ in 1 }
    }

    public func newCache(parameters: GenerateParameters?) -> [KVCache] {
        [TokenHistoryCache()]
    }

    public func callAsFunction(
        _ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?
    ) -> LMOutput {
        let flat = input.tokens.flattened()
        eval(flat)
        let newTokens: [Int32] = flat.asArray(Int32.self)

        let allTokens: [Int32]
        if let history = cache?.first as? TokenHistoryCache {
            history.append(newTokens)
            allTokens = history.tokens
        } else {
            allTokens = newTokens
        }

        return LMOutput(logits: compiled.forward(allTokens))
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        callAsFunction(LMInput.Text(tokens: inputs), cache: cache, state: nil).logits
    }

    public func prepare(
        _ input: LMInput, cache: [KVCache], windowSize: Int?
    ) throws -> PrepareResult {
        .logits(callAsFunction(input.text, cache: cache, state: nil))
    }
}

// MARK: - Token History Cache

/// Tracks token history for MPSGraph full-sequence replay.
private final class TokenHistoryCache: KVCache, @unchecked Sendable {

    private(set) var tokens: [Int32] = []

    var offset: Int { tokens.count }
    var maxSize: Int? { nil }
    var isTrimmable: Bool { true }

    func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) { (keys, values) }

    @discardableResult
    func trim(_ n: Int) -> Int {
        let trimmed = min(n, tokens.count)
        tokens.removeLast(trimmed)
        return trimmed
    }

    var state: [MLXArray] {
        get { [] }
        set { }
    }

    var metaState: [String] {
        get { [String(tokens.count)] }
        set {
            if let first = newValue.first, let count = Int(first) {
                tokens = Array(tokens.prefix(count))
            }
        }
    }

    func innerState() -> [MLXArray] { [] }

    func makeMask(queryLength: Int) -> MLXFast.ScaledDotProductAttentionMaskMode {
        queryLength > 1 ? .causal : .none
    }

    func append(_ newTokens: [Int32]) {
        tokens.append(contentsOf: newTokens)
    }
}
