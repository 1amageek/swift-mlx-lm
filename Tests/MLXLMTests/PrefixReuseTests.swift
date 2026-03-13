import MLX
import MLXNN
import Testing
@testable import MLXLM

@Suite("Prefix Reuse", .tags(.unit))
struct PrefixReuseTests {

    @Test("Pool reuses only compatible cache layouts")
    func poolSeparatesLayouts() {
        let pool = PrefixCachePool(maxSlots: 2)
        let simpleParameters = GenerateParameters()
        let simpleLayout = CacheLayout(parameters: simpleParameters)
        let simpleCache = createKVCaches(layerCount: 1, parameters: simpleParameters)

        pool.release(cache: simpleCache, layout: simpleLayout, tokens: [1, 2, 3])

        var quantizedParameters = GenerateParameters()
        quantizedParameters.kvBits = 4
        let quantizedLayout = CacheLayout(parameters: quantizedParameters)

        let (cache, reusedPrefixLength) = pool.acquire(
            for: [1, 2, 3],
            layout: quantizedLayout,
            newCacheFactory: { createKVCaches(layerCount: 1, parameters: quantizedParameters) }
        )

        #expect(reusedPrefixLength == 0)
        #expect(cache[0] is QuantizedKVCache)
        #expect(cache[0] !== simpleCache[0])
    }

    @Test("Pool reuses compatible layouts by longest prefix")
    func poolReusesMatchingLayout() {
        let pool = PrefixCachePool(maxSlots: 2)
        let parameters = GenerateParameters()
        let layout = CacheLayout(parameters: parameters)
        let cache = createKVCaches(layerCount: 1, parameters: parameters)

        pool.release(cache: cache, layout: layout, tokens: [10, 20, 30, 40])

        let (reusedCache, reusedPrefixLength) = pool.acquire(
            for: [10, 20, 99],
            layout: layout,
            newCacheFactory: { createKVCaches(layerCount: 1, parameters: parameters) }
        )

        #expect(reusedPrefixLength == 2)
        #expect(reusedCache[0] === cache[0])
    }

    @Test("Text-only inputs use automatic prefix reuse")
    func textOnlyInputsUseAutomaticReuse() async {
        let container = makeContainer(model: TextModel())
        let input = LMInput(tokens: MLXArray([Int32(1), Int32(2)]).reshaped([1, 2]))

        let usesAutomaticReuse = await container.usesAutomaticPrefixReuse(for: input)

        #expect(usesAutomaticReuse)
    }

    @Test("Explicit position IDs disable automatic prefix reuse")
    func positionIdsDisableAutomaticReuse() async {
        let container = makeContainer(model: TextModel())
        let input = LMInput(
            text: LMInput.Text(
                tokens: MLXArray([Int32(1), Int32(2)]).reshaped([1, 2]),
                positionIds: MLXArray([Int32(0), Int32(1)]).reshaped([1, 2])
            )
        )

        let usesAutomaticReuse = await container.usesAutomaticPrefixReuse(for: input)

        #expect(!usesAutomaticReuse)
    }

private func makeContainer(model: any LanguageModel) -> ModelContainer {
        ModelContainer(
            context: ModelContext(
                configuration: ModelConfiguration(name: "test"),
                model: model,
                processor: TestProcessor(),
                tokenizer: TestTokenizer()
            )
        )
    }

    // MARK: - Non-Trimmable Cache (DeltaNet-like)

    @Test("Non-trimmable cache with exact prefix match is reused")
    func nonTrimmableExactPrefixReuse() {
        let pool = PrefixCachePool(maxSlots: 2)
        let layout = CacheLayout.simple
        let cache: [KVCache] = [NonTrimmableCache(), KVCacheSimple()]

        pool.release(cache: cache, layout: layout, tokens: [1, 2, 3])

        // Request extends stored tokens — trimCount == 0
        let (reused, prefixLen) = pool.acquire(
            for: [1, 2, 3, 4, 5],
            layout: layout,
            newCacheFactory: { [NonTrimmableCache(), KVCacheSimple()] }
        )

        #expect(prefixLen == 3)
        #expect(reused[0] === cache[0])
    }

    @Test("Non-trimmable cache with trimCount > 0 falls back without slot leak")
    func nonTrimmableNoSlotLeak() {
        let pool = PrefixCachePool(maxSlots: 4)
        let layout = CacheLayout.simple
        let cache: [KVCache] = [NonTrimmableCache()]

        // Store [1,2,3,4,5] — longer than the new request's common prefix
        pool.release(cache: cache, layout: layout, tokens: [1, 2, 3, 4, 5])
        #expect(pool.count == 1)

        // Request [1,2,9] — prefix=2, trimCount=3 — can't trim
        let (_, prefixLen) = pool.acquire(
            for: [1, 2, 9],
            layout: layout,
            newCacheFactory: { [NonTrimmableCache()] }
        )

        // Should fall back to fresh cache
        #expect(prefixLen == 0)
        // Slot should NOT be leaked — it stays in the pool for future use
        #expect(pool.count == 1)
    }

    @Test("Pool tries shorter candidate when best match is non-trimmable")
    func fallbackToShorterTrimmableCandidate() {
        let pool = PrefixCachePool(maxSlots: 4)
        let layout = CacheLayout.simple

        // Slot A: non-trimmable, stored [1,2,3,4,5] — will be best prefix match (5)
        // but needs trimCount=3 for request [1,2,X] which it can't handle
        let slotA: [KVCache] = [NonTrimmableCache()]
        pool.release(cache: slotA, layout: layout, tokens: [1, 2, 3, 4, 5])

        // Slot B: trimmable, stored [1,2,3] — shorter match (2) but CAN trim
        let slotB: [KVCache] = [KVCacheSimple()]
        pool.release(cache: slotB, layout: layout, tokens: [1, 2, 3])

        // Request [1,2,9] — prefix match with A=2 (trimCount=3), B=2 (trimCount=1)
        let (reused, prefixLen) = pool.acquire(
            for: [1, 2, 9],
            layout: layout,
            newCacheFactory: { [KVCacheSimple()] }
        )

        // Should fall back to slot B (trimmable) with prefix=2
        #expect(prefixLen == 2)
        #expect(reused[0] === slotB[0])
        // Slot A should still be in pool (not leaked)
        #expect(pool.count == 1)
    }

    @Test("Pool prefers exact-match non-trimmable over trimmable shorter match")
    func exactMatchNonTrimmablePreferred() {
        let pool = PrefixCachePool(maxSlots: 4)
        let layout = CacheLayout.simple

        // Slot A: non-trimmable, stored [1,2,3]
        let slotA: [KVCache] = [NonTrimmableCache()]
        pool.release(cache: slotA, layout: layout, tokens: [1, 2, 3])

        // Slot B: trimmable, stored [1,2]
        let slotB: [KVCache] = [KVCacheSimple()]
        pool.release(cache: slotB, layout: layout, tokens: [1, 2])

        // Request [1,2,3,4] — A matches 3 (trimCount=0), B matches 2 (trimCount=0)
        let (reused, prefixLen) = pool.acquire(
            for: [1, 2, 3, 4],
            layout: layout,
            newCacheFactory: { [KVCacheSimple()] }
        )

        // Prefer longer prefix match (slot A with 3 tokens)
        #expect(prefixLen == 3)
        #expect(reused[0] === slotA[0])
    }

    /// Mock non-trimmable cache simulating DeltaNet recurrent state.
    private final class NonTrimmableCache: KVCache, @unchecked Sendable {
        var offset: Int = 0
        var maxSize: Int? { nil }
        var isTrimmable: Bool { false }

        func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
            fatalError("Not used in tests")
        }

        @discardableResult
        func trim(_ n: Int) -> Int { 0 }

        var state: [MLXArray] {
            get { [] }
            set {}
        }

        var metaState: [String] {
            get { ["0"] }
            set {}
        }

        func innerState() -> [MLXArray] { [] }

        func makeMask(queryLength: Int) -> MLXFast.ScaledDotProductAttentionMaskMode { .none }
    }

    private struct TestProcessor: UserInputProcessor {
        func prepare(input: UserInput) async throws -> LMInput {
            LMInput(tokens: MLXArray([Int32(1)]).reshaped([1, 1]))
        }
    }

    private struct TestTokenizer: Tokenizer {
        var bosTokenID: Int? { 1 }
        var eosTokenID: Int? { 2 }
        var vocabularySize: Int { 16 }

        func encode(text: String) -> [Int] {
            [1]
        }

        func decode(tokens: [Int]) -> String {
            ""
        }

        func tokenToString(_ id: Int) -> String? {
            String(id)
        }

        func tokenID(for string: String) -> Int? {
            nil
        }
    }

    private final class TextModel: Module, LanguageModel {
        var layerCount: Int { 1 }
        var kvHeads: [Int] { [1] }

        func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws -> PrepareResult {
            .logits(callAsFunction(input.text, cache: cache, state: nil))
        }

        func callAsFunction(_ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?) -> LMOutput {
            let batchSize = input.tokens.dim(0)
            let sequenceLength = input.tokens.dim(1)
            let logits = MLXArray.zeros([batchSize, sequenceLength, 4])
            return LMOutput(logits: logits)
        }

        func newCache(parameters: GenerateParameters?) -> [KVCache] {
            [KVCacheSimple()]
        }

        func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
            weights
        }
    }

}
