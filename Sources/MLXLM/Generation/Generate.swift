import Foundation
import MLX
import GGUFTokenizer

/// Generate text from tokenized input using a model context.
///
/// Returns an async stream that yields `.chunk(String)` for each decoded
/// text segment and `.info(GenerateCompletionInfo)` as the final element.
public func generate(
    input: LMInput,
    cache existingCache: [KVCache]? = nil,
    parameters: GenerateParameters,
    context: ModelContext
) throws -> AsyncStream<Generation> {
    let model = context.model
    let tokenizer = context.tokenizer
    let configuration = context.configuration

    var allEosIds = configuration.eosTokenIds
    if let eos = tokenizer.eosTokenID {
        allEosIds.insert(eos)
    }

    let cache = existingCache ?? model.newCache(parameters: parameters)

    let promptTokenCount = input.text.tokens.dim(input.text.tokens.ndim - 1)

    let iterator = try TokenIterator(
        input: input,
        model: model,
        cache: cache,
        parameters: parameters,
        eosTokenIds: allEosIds
    )

    // Wrap non-Sendable state for safe transfer into Task.
    // This is safe because ModelContainer actor serializes all model access.
    let state = GenerationTaskState(
        iterator: iterator,
        detokenizer: StreamingDetokenizer(tokenizer: tokenizer)
    )

    let maxTokens = parameters.maxTokens

    return AsyncStream { continuation in
        let task = Task {
            var s = state
            var generationTokenCount = 0
            let promptStart = Date()
            var generateStart = promptStart
            var stopReason: GenerateStopReason = .stop

            while let token = s.iterator.next() {
                if generationTokenCount == 0 {
                    generateStart = Date()
                }

                generationTokenCount += 1

                if let text = s.detokenizer.append(token: token) {
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

            let now = Date()
            let promptTime = generateStart.timeIntervalSince(promptStart)
            let generateTime = now.timeIntervalSince(generateStart)

            let info = GenerateCompletionInfo(
                promptTokenCount: promptTokenCount,
                generationTokenCount: generationTokenCount,
                promptTime: promptTime,
                generateTime: generateTime,
                stopReason: stopReason
            )
            continuation.yield(.info(info))
            continuation.finish()
        }

        continuation.onTermination = { _ in
            task.cancel()
        }
    }
}

/// Wraps non-Sendable generation state for safe transfer into a Task.
///
/// Safety: This type is used exclusively within `ModelContainer.generate()`,
/// which is actor-isolated. The model and cache are never accessed concurrently.
private struct GenerationTaskState: @unchecked Sendable {
    var iterator: TokenIterator
    var detokenizer: StreamingDetokenizer
}
