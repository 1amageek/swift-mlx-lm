import Foundation
import Metal
import OrderedCollections
import MetalCompiler
import Tokenizers
import Jinja

/// Thread-safe container for a compiled inference model.
///
/// Wraps MetalInferenceModel + Tokenizer. Provides the public API
/// consumed by AnyFoundationModels and Jardis.
///
/// ```swift
/// let container = try await ModelBundleLoader().load(repo: "mlx-community/Qwen2.5-0.5B-Instruct")
/// let stream = try await container.generate(
///     input: LMInput(tokens: tokenizer.encode(text: "Hello")),
///     parameters: GenerateParameters(maxTokens: 100)
/// )
/// for await generation in stream {
///     if let text = generation.chunk { print(text, terminator: "") }
/// }
/// ```
public final class ModelContainer: @unchecked Sendable {

    private var inferenceModel: MetalInferenceModel
    private let modelTokenizer: any Tokenizer
    private let modelConfiguration: ModelConfiguration
    /// Jinja chat template loaded from model bundle. nil if not available.
    private let chatTemplate: Template?

    public init(
        inferenceModel: MetalInferenceModel,
        tokenizer: any Tokenizer,
        configuration: ModelConfiguration,
        chatTemplate: Template? = nil
    ) {
        self.inferenceModel = inferenceModel
        self.modelTokenizer = tokenizer
        self.modelConfiguration = configuration
        self.chatTemplate = chatTemplate
    }

    /// Model configuration (name, EOS tokens).
    public var configuration: ModelConfiguration {
        modelConfiguration
    }

    /// The tokenizer used by this model.
    public var tokenizer: any Tokenizer {
        modelTokenizer
    }

    /// Tokenize text into LMInput.
    ///
    /// For chat messages, applies the Jinja chat template from the model bundle
    /// (chat_template.jinja or tokenizer_config.json). Falls back to simple
    /// role-prefixed formatting if no template is available.
    public func prepare(input: UserInput) throws -> LMInput {
        let text: String
        switch input.prompt {
        case .text(let prompt):
            text = prompt
        case .chat(let messages):
            text = try applyChatTemplate(messages: messages)
        }
        let tokens = modelTokenizer.encode(text: text)
        return LMInput(tokens: tokens)
    }

    /// Apply the Jinja chat template to format chat messages into a prompt string.
    private func applyChatTemplate(messages: [ChatMessage]) throws -> String {
        // Convert to Jinja-compatible format: [{"role": "user", "content": "..."}]
        let jinjaMessages: [[String: String]] = messages.map {
            ["role": $0.role.rawValue, "content": $0.content]
        }

        if let template = chatTemplate {
            // Convert messages to Jinja Value format
            let messageValues: [Value] = jinjaMessages.map { dict in
                var ordered = OrderedDictionary<String, Value>()
                for (key, value) in dict {
                    ordered[key] = .string(value)
                }
                return .object(ordered)
            }

            let context: [String: Value] = [
                "messages": .array(messageValues),
                "add_generation_prompt": .boolean(true),
                "bos_token": .string(modelTokenizer.bosToken ?? ""),
                "eos_token": .string(modelTokenizer.eosToken ?? ""),
            ]
            let rendered = try template.render(context)
            return rendered
        }

        // Fallback: simple role-prefixed format
        return messages.map { "\($0.role.rawValue): \($0.content)" }.joined(separator: "\n")
    }

    private func streamGeneration(
        firstToken: Int32,
        promptTokenCount: Int,
        preparationTime: Double,
        requestStartTime: Double,
        parameters: GenerateParameters,
        continuation: AsyncStream<Generation>.Continuation
    ) {
        var tokenCount = 0
        let maxTokens = parameters.maxTokens ?? 1024
        let chunkTokenCount = max(1, parameters.streamChunkTokenCount)
        var bufferedTokenIDs: [Int] = []

        func emitBufferedChunkIfNeeded(force: Bool = false) {
            guard !bufferedTokenIDs.isEmpty else { return }
            guard force || bufferedTokenIDs.count >= chunkTokenCount else { return }
            let text = self.modelTokenizer.decode(tokens: bufferedTokenIDs)
            bufferedTokenIDs.removeAll(keepingCapacity: true)
            continuation.yield(.chunk(text))
        }

        guard firstToken >= 0 else {
            continuation.finish()
            return
        }
        if self.modelConfiguration.eosTokenIds.contains(Int(firstToken)) {
            continuation.finish()
            return
        }

        bufferedTokenIDs.append(Int(firstToken))
        tokenCount += 1

        if tokenCount < maxTokens {
            _ = self.inferenceModel.decode(tokenID: firstToken)
        }

        emitBufferedChunkIfNeeded()

        while tokenCount < maxTokens {
            let outputToken = self.inferenceModel.flush()

            if outputToken < 0 { break }

            let isEOS = self.modelConfiguration.eosTokenIds.contains(Int(outputToken))
            if !isEOS, tokenCount + 1 < maxTokens {
                _ = self.inferenceModel.decode(tokenID: outputToken)
            }
            if isEOS {
                emitBufferedChunkIfNeeded(force: true)
                break
            }

            bufferedTokenIDs.append(Int(outputToken))
            tokenCount += 1
            emitBufferedChunkIfNeeded()
        }

        emitBufferedChunkIfNeeded(force: true)

        let totalTime = CFAbsoluteTimeGetCurrent() - requestStartTime
        let tokensPerSecond = totalTime > 0 ? Double(tokenCount) / totalTime : 0
        let preparationTokPerSec = preparationTime > 0 ? Double(promptTokenCount) / preparationTime : 0
        print("[ModelContainer] \(tokenCount) tokens (\(String(format: "%.0f", preparationTokPerSec)) prefill, \(String(format: "%.1f", tokensPerSecond)) decode tok/s) [\(String(format: "%.1f", totalTime))s]")
        continuation.yield(.info(CompletionInfo(
            tokenCount: tokenCount,
            tokensPerSecond: tokensPerSecond,
            totalTime: totalTime
        )))
        continuation.finish()
    }

    /// Build a reusable prompt state from tokenized input.
    ///
    /// This runs prefill once, snapshots the decode state, and stores the
    /// first predicted token so the same prompt prefix can be reused later.
    public func makePromptState(input: LMInput) throws -> PromptState {
        inferenceModel.resetCaches()
        let promptTokens = input.text.tokens.map { Int32($0) }
        let firstToken = inferenceModel.prefill(tokens: promptTokens)
        guard firstToken >= 0 else {
            throw ModelContainerError.invalidPrefillResult
        }
        let metalState = try inferenceModel.makePromptState(firstToken: firstToken)
        return PromptState(metalState: metalState, promptTokenCount: input.text.tokens.count)
    }

    /// Build a reusable prompt state from user input.
    public func makePromptState(input: UserInput) throws -> PromptState {
        let prepared = try prepare(input: input)
        return try makePromptState(input: prepared)
    }

    /// Generate text from tokenized input.
    ///
    /// Returns an AsyncStream of Generation values (text chunks + completion info).
    /// Each `.chunk` may contain one or more decoded tokens.
    public func generate(
        input: LMInput,
        parameters: GenerateParameters = GenerateParameters()
    ) -> AsyncStream<Generation> {
        AsyncStream { continuation in
            Task {
                let startTime = CFAbsoluteTimeGetCurrent()
                let prefillStart = CFAbsoluteTimeGetCurrent()
                let promptTokens = input.text.tokens.map { Int32($0) }
                let firstToken = self.inferenceModel.prefill(tokens: promptTokens)
                let prefillTime = CFAbsoluteTimeGetCurrent() - prefillStart
                self.streamGeneration(
                    firstToken: firstToken,
                    promptTokenCount: input.text.tokens.count,
                    preparationTime: prefillTime,
                    requestStartTime: startTime,
                    parameters: parameters,
                    continuation: continuation
                )
            }
        }
    }

    /// Generate text by restoring a reusable prompt state instead of re-running prefill.
    public func generate(
        from promptState: PromptState,
        parameters: GenerateParameters = GenerateParameters()
    ) -> AsyncStream<Generation> {
        AsyncStream { continuation in
            Task {
                let startTime = CFAbsoluteTimeGetCurrent()
                let restoreStart = CFAbsoluteTimeGetCurrent()
                do {
                    try self.inferenceModel.restore(promptState: promptState.metalState)
                } catch {
                    print("[ModelContainer] Failed to restore prompt state: \(error)")
                    continuation.finish()
                    return
                }
                let restoreTime = CFAbsoluteTimeGetCurrent() - restoreStart
                self.streamGeneration(
                    firstToken: promptState.metalState.firstToken,
                    promptTokenCount: promptState.promptTokenCount,
                    preparationTime: restoreTime,
                    requestStartTime: startTime,
                    parameters: parameters,
                    continuation: continuation
                )
            }
        }
    }

    /// Decode token IDs to text.
    public func decode(tokens: [Int]) -> String {
        modelTokenizer.decode(tokens: tokens)
    }

    /// Encode text to token IDs.
    public func encode(_ text: String) -> [Int] {
        modelTokenizer.encode(text: text)
    }

    /// Reset KV cache (call between independent conversations).
    public func resetCaches() {
        inferenceModel.resetCaches()
    }
}
