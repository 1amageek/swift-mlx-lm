import Foundation
import MLX
import GGUFTokenizer

/// Thread-safe container providing serial access to a loaded model.
///
/// All model operations (prepare, generate, decode) are serialized
/// through this actor to prevent concurrent GPU access.
///
/// Integrates a `PrefixCachePool` for automatic KV cache reuse across
/// generations. When no external cache is provided, the container finds
/// the longest common prefix between the new request and cached state,
/// trims the divergent tail, and only prefills new tokens. This makes
/// multi-turn chat efficient: only the new user message is processed,
/// not the entire conversation history.
///
/// Analogous to Ollama's scheduler + InputCache architecture
/// (`server/sched.go` + `runner/llamarunner/cache.go`).
public actor ModelContainer {

    private var context: ModelContext
    private let cachePool: PrefixCachePool

    public init(context: ModelContext, maxCacheSlots: Int = 4) {
        self.context = context
        self.cachePool = PrefixCachePool(maxSlots: maxCacheSlots)
    }

    /// Model configuration (accessible without awaiting).
    public var configuration: ModelConfiguration {
        context.configuration
    }

    /// Execute an arbitrary action with serial access to the model context.
    public func perform<R: Sendable>(
        _ action: @Sendable (ModelContext) throws -> R
    ) throws -> R {
        try action(context)
    }

    /// Execute an action with additional values passed in.
    public func perform<V: Sendable, R: Sendable>(
        values: V,
        _ action: @Sendable (ModelContext, V) throws -> R
    ) throws -> R {
        try action(context, values)
    }

    /// Mutate the model context.
    public func update(_ action: @Sendable (inout ModelContext) -> Void) {
        action(&context)
    }

    // MARK: - Convenience Methods

    /// Convert user input into tokenized model input.
    public func prepare(input: UserInput) async throws -> LMInput {
        try await context.processor.prepare(input: input)
    }

    /// Convert user input into prefix-only tokenized input (for cache warming).
    public func preparePrefix(input: UserInput) async throws -> LMInput {
        try await context.processor.preparePrefix(input: input)
    }

    /// Decode token IDs back to text.
    public func decode(tokens: [Int]) -> String {
        context.tokenizer.decode(tokens: tokens)
    }

    /// Encode text to token IDs.
    public func encode(_ text: String) -> [Int] {
        context.tokenizer.encode(text: text)
    }

    /// Generate text from tokenized input.
    ///
    /// When no external cache is provided, automatically reuses KV caches from
    /// previous generations via token-level prefix matching. Only tokens after
    /// the longest common prefix need to be prefilled.
    ///
    /// Automatic prefix reuse is limited to text-only models with token-only
    /// conditioning. Vision-language models and inputs with explicit position IDs
    /// bypass the pool because their runtime state cannot currently be restored
    /// from token prefixes alone.
    ///
    /// When an external cache is provided, uses it directly (caller manages lifecycle).
    ///
    /// Returns an async stream of `Generation` elements:
    /// - `.chunk(String)` for decoded text segments
    /// - `.toolCall(ToolCall)` for detected tool invocations
    /// - `.info(GenerateCompletionInfo)` as the final element
    public func generate(
        input: LMInput,
        cache: [KVCache]? = nil,
        parameters: GenerateParameters
    ) throws -> AsyncStream<Generation> {
        if let cache {
            // External cache management (caller controls lifecycle)
            return try MLXLM.generate(
                input: input,
                cache: cache,
                parameters: parameters,
                context: context
            )
        }

        if !usesAutomaticPrefixReuse(for: input) {
            return try MLXLM.generate(
                input: input,
                parameters: parameters,
                context: context
            )
        }

        // Auto-managed cache with prefix reuse
        return try generateWithPrefixReuse(input: input, parameters: parameters)
    }

    /// Clear all cached prefix state.
    ///
    /// Call when switching conversations, unloading the model, or when
    /// memory pressure requires freeing GPU resources held by cached KV state.
    public func clearCachePool() {
        cachePool.clear()
    }

    /// Number of cache slots currently occupied.
    public var cacheSlotCount: Int {
        cachePool.count
    }

    // MARK: - Private: Prefix-Aware Generation

    /// Generate with automatic prefix cache reuse.
    ///
    /// Flow:
    /// 1. Extract token IDs from input for prefix matching
    /// 2. Acquire best cache slot (longest common prefix, trim divergent tail)
    /// 3. Run generation loop (collecting generated token IDs)
    /// 4. Return cache + full tokens to pool for future reuse
    private func generateWithPrefixReuse(
        input: LMInput,
        parameters: GenerateParameters
    ) throws -> AsyncStream<Generation> {
        let model = context.model
        let tokenizer = context.tokenizer
        let configuration = context.configuration

        // Extract input token IDs for prefix matching (GPU→CPU sync, before generation)
        let inputTokens = extractTokenIDs(input.text.tokens)
        let layout = CacheLayout(parameters: parameters)

        // Acquire cache from pool (finds best prefix match, trims divergent tail)
        let (cache, reusedLen) = cachePool.acquire(
            for: inputTokens,
            layout: layout,
            newCacheFactory: { model.newCache(parameters: parameters) }
        )

        print("[generate] prefix reuse: \(reusedLen)/\(inputTokens.count) tokens cached")

        // Build EOS set
        var allEosIds = configuration.eosTokenIds
        if let eos = tokenizer.eosTokenID {
            allEosIds.insert(eos)
        }

        let promptTokenCount = input.text.tokens.dim(input.text.tokens.ndim - 1)
        var effectiveParameters = parameters
        effectiveParameters.reusedPrefixTokenCount = reusedLen

        let iterator = try TokenIterator(
            input: input,
            model: model,
            cache: cache,
            parameters: effectiveParameters,
            eosTokenIds: allEosIds
        )

        // Wrap non-Sendable state for safe transfer into Task.
        // Safety: ModelContainer actor serializes all model access.
        let state = CacheAwareGenerationState(
            iterator: iterator,
            detokenizer: StreamingDetokenizer(tokenizer: tokenizer),
            rawTokenLogger: RawTokenTraceLogger(tokenizer: tokenizer)
        )

        let maxTokens = parameters.maxTokens

        return AsyncStream { continuation in
            let task = Task { [weak self] in
                var s = state
                var generationTokenCount = 0
                var generatedTokens: [Int] = []
                let promptStart = Date()
                var generateStart = promptStart
                var stopReason: GenerateStopReason = .stop

                print("[generate] prefill starting promptTokens=\(promptTokenCount) reused=\(reusedLen)")
                while let token = s.iterator.next() {
                    if generationTokenCount == 0 {
                        generateStart = Date()
                        let prefillTime = generateStart.timeIntervalSince(promptStart)
                        print("[generate] first token latency=\(String(format: "%.2f", prefillTime))s")
                    }

                    generationTokenCount += 1
                    generatedTokens.append(token)
                    print("[generate-debug] step=\(generationTokenCount) token=\(token)")

                    let text = s.detokenizer.append(token: token)
                    s.rawTokenLogger.logOutputToken(
                        step: generationTokenCount,
                        token: token,
                        chunk: text
                    )

                    if let text {
                        continuation.yield(.chunk(text))
                    }

                    if Task.isCancelled {
                        stopReason = .cancelled
                        break
                    }

                    if let max = maxTokens, generationTokenCount >= max {
                        stopReason = .length
                        break
                    }
                }

                // Synchronize pending asyncEval operations before measuring final time
                Stream().synchronize()

                let now = Date()
                let promptTime = generateStart.timeIntervalSince(promptStart)
                let generateTime = now.timeIntervalSince(generateStart)

                // Return cache to pool with full token sequence (input + generated)
                let allTokens = inputTokens + generatedTokens
                await self?.returnCacheToPool(cache: cache, layout: layout, tokens: allTokens)

                let info = GenerateCompletionInfo(
                    promptTokenCount: promptTokenCount,
                    generationTokenCount: generationTokenCount,
                    promptTime: promptTime,
                    generateTime: generateTime,
                    stopReason: stopReason
                )
                print("[generate] done tokens=\(generationTokenCount) prefill=\(String(format: "%.2f", promptTime))s generate=\(String(format: "%.2f", generateTime))s stop=\(stopReason)")
                continuation.yield(.info(info))
                continuation.finish()
            }

            continuation.onTermination = { _ in
                task.cancel()
            }
        }
    }

    /// Return a used cache to the pool for future prefix reuse.
    private func returnCacheToPool(cache: [KVCache], layout: CacheLayout, tokens: [Int]) {
        cachePool.release(cache: cache, layout: layout, tokens: tokens)
    }

    /// Extract token IDs from an MLXArray for prefix matching.
    private func extractTokenIDs(_ tokens: MLXArray) -> [Int] {
        let flat = tokens.flattened()
        eval(flat)
        let int32Tokens: [Int32] = flat.asArray(Int32.self)
        return int32Tokens.map { Int($0) }
    }

    func usesAutomaticPrefixReuse(for input: LMInput) -> Bool {
        if context.model is any VisionLanguageModel {
            return false
        }

        if input.image != nil || input.video != nil {
            return false
        }

        return input.text.positionIds == nil
    }
}

/// Wraps non-Sendable generation state for safe transfer into a Task.
///
/// Safety: Used exclusively within `ModelContainer.generateWithPrefixReuse()`,
/// which is actor-isolated. The model and cache are never accessed concurrently.
private struct CacheAwareGenerationState: @unchecked Sendable {
    var iterator: TokenIterator
    var detokenizer: StreamingDetokenizer
    var rawTokenLogger: RawTokenTraceLogger
}
