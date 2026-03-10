import GGUFTokenizer
import MLX
import MLXNN
import Testing
@testable import MLXLM

@Suite("Prefix Reuse")
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

    @Test("Vision-language models bypass automatic prefix reuse")
    func visionModelsBypassAutomaticReuse() async {
        let container = makeContainer(model: VisionModel())
        let input = LMInput(tokens: MLXArray([Int32(1), Int32(2)]).reshaped([1, 2]))

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

    private final class VisionModel: Module, VisionLanguageModel {
        var imageTokenId: Int { 11 }
        var videoTokenId: Int { 12 }
        var spatialMergeSize: Int { 2 }
        var nextPosition: Int = 0
        var layerCount: Int { 1 }
        var kvHeads: [Int] { [1] }

        func encodeVision(
            image: LMInput.ProcessedImage?,
            video: LMInput.ProcessedVideo?
        ) throws -> MLXArray? {
            nil
        }

        func embedTokens(_ tokens: MLXArray) -> MLXArray {
            let batchSize = tokens.dim(0)
            let sequenceLength = tokens.dim(1)
            return MLXArray.zeros([batchSize, sequenceLength, 1])
        }

        func forwardTextModel(
            _ inputs: MLXArray,
            cache: [KVCache]?,
            inputEmbeddings: MLXArray?,
            positionIds: MLXArray?
        ) -> MLXArray {
            let batchSize = inputs.dim(0)
            let sequenceLength = inputs.dim(1)
            return MLXArray.zeros([batchSize, sequenceLength, 1])
        }

        func forwardLogits(_ h: MLXArray) -> MLXArray {
            MLXArray.zeros([h.dim(0), h.dim(1), 4])
        }

        func newCache(parameters: GenerateParameters?) -> [KVCache] {
            nextPosition = 0
            return [KVCacheSimple()]
        }

        func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
            weights
        }
    }
}
