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

    /// Generate text from tokenized input.
    ///
    /// Returns an AsyncStream of Generation values (text chunks + completion info).
    /// Each `.chunk` contains one decoded token's text.
    public func generate(
        input: LMInput,
        parameters: GenerateParameters = GenerateParameters()
    ) -> AsyncStream<Generation> {
        AsyncStream { continuation in
            Task {
                let startTime = CFAbsoluteTimeGetCurrent()
                var tokenCount = 0
                let maxTokens = parameters.maxTokens ?? 1024

                // Prefill: batch all prompt tokens with minimal GPU sync
                let prefillStart = CFAbsoluteTimeGetCurrent()
                let promptTokens = input.text.tokens.map { Int32($0) }
                let firstToken = self.inferenceModel.prefill(tokens: promptTokens)
                let prefillTime = CFAbsoluteTimeGetCurrent() - prefillStart

                // First generated token comes from prefill argmax
                guard firstToken >= 0 else {
                    continuation.finish()
                    return
                }
                if self.modelConfiguration.eosTokenIds.contains(Int(firstToken)) {
                    continuation.finish()
                    return
                }

                // Yield the first token (from prefill)
                let firstText = self.modelTokenizer.decode(tokens: [Int(firstToken)])
                continuation.yield(.chunk(firstText))
                tokenCount += 1

                // Decode subsequent tokens
                var previousToken = firstToken
                while tokenCount < maxTokens {
                    let outputToken = self.inferenceModel.decodeSync(tokenID: previousToken)

                    if outputToken < 0 { break }
                    if self.modelConfiguration.eosTokenIds.contains(Int(outputToken)) { break }

                    let text = self.modelTokenizer.decode(tokens: [Int(outputToken)])
                    continuation.yield(.chunk(text))

                    previousToken = outputToken
                    tokenCount += 1
                }

                let totalTime = CFAbsoluteTimeGetCurrent() - startTime
                let tokensPerSecond = totalTime > 0 ? Double(tokenCount) / totalTime : 0
                let prefillTokPerSec = prefillTime > 0 ? Double(input.text.tokens.count) / prefillTime : 0
                print("[ModelContainer] \(tokenCount) tokens (\(String(format: "%.0f", prefillTokPerSec)) prefill, \(String(format: "%.1f", tokensPerSecond)) decode tok/s) [\(String(format: "%.1f", totalTime))s]")
                continuation.yield(.info(CompletionInfo(
                    tokenCount: tokenCount,
                    tokensPerSecond: tokensPerSecond,
                    totalTime: totalTime
                )))
                continuation.finish()
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
