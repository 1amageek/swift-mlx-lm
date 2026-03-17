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
            print("[ModelContainer] prepare: text prompt, \(prompt.count) chars")
        case .chat(let messages):
            text = try applyChatTemplate(messages: messages)
            print("[ModelContainer] prepare: chat template applied, \(messages.count) messages → \(text.count) chars")
        }
        let tokens = modelTokenizer.encode(text: text)
        print("[ModelContainer] prepare: \(tokens.count) tokens, first=\(tokens.first ?? -1) last=\(tokens.last ?? -1)")
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

                print("[ModelContainer] generate: \(input.text.tokens.count) prompt tokens, maxTokens=\(maxTokens)")
                print("[ModelContainer] dispatch plan: \(self.inferenceModel.plan.steps.count) steps")
                // Dump ALL tokens for diagnostic reproduction
                let allTokens = input.text.tokens.map(String.init).joined(separator: ",")
                print("[ModelContainer] ALL_TOKENS=[\(allTokens)]")

                // Prefill: batch all prompt tokens with minimal GPU sync
                let prefillStart = CFAbsoluteTimeGetCurrent()
                let promptTokens = input.text.tokens.map { Int32($0) }
                let firstToken = self.inferenceModel.prefill(tokens: promptTokens)
                let prefillTime = CFAbsoluteTimeGetCurrent() - prefillStart
                print("[ModelContainer] prefill done: \(input.text.tokens.count) tokens [\(String(format: "%.3f", prefillTime))s] \(String(format: "%.0f", Double(input.text.tokens.count) / prefillTime)) tok/s")
                print("[ModelContainer] first predicted token: \(firstToken)")

                // First generated token comes from prefill argmax
                guard firstToken >= 0 else {
                    print("[ModelContainer] prefill returned negative token, aborting")
                    continuation.finish()
                    return
                }
                if self.modelConfiguration.eosTokenIds.contains(Int(firstToken)) {
                    print("[ModelContainer] prefill predicted EOS token \(firstToken)")
                    continuation.finish()
                    return
                }

                // Yield the first token (from prefill)
                let firstText = self.modelTokenizer.decode(tokens: [Int(firstToken)])
                print("[ModelContainer] token 0: id=\(firstToken) text='\(firstText)'")
                continuation.yield(.chunk(firstText))
                tokenCount += 1

                // Decode subsequent tokens
                var previousToken = firstToken
                while tokenCount < maxTokens {
                    let decodeStart = CFAbsoluteTimeGetCurrent()
                    let outputToken = self.inferenceModel.decodeSync(tokenID: previousToken)
                    let decodeTime = CFAbsoluteTimeGetCurrent() - decodeStart

                    if tokenCount < 10 || tokenCount % 50 == 0 {
                        print("[ModelContainer] decode step \(tokenCount): input=\(previousToken) → output=\(outputToken) [\(String(format: "%.3f", decodeTime))s]")
                    }

                    if outputToken < 0 {
                        print("[ModelContainer] decode returned negative token, stopping")
                        break
                    }
                    if self.modelConfiguration.eosTokenIds.contains(Int(outputToken)) {
                        print("[ModelContainer] EOS token \(outputToken), stopping after \(tokenCount) tokens")
                        break
                    }

                    let text = self.modelTokenizer.decode(tokens: [Int(outputToken)])
                    if tokenCount < 10 {
                        print("[ModelContainer] token \(tokenCount): id=\(outputToken) text='\(text)'")
                    }
                    continuation.yield(.chunk(text))

                    previousToken = outputToken
                    tokenCount += 1
                }

                let totalTime = CFAbsoluteTimeGetCurrent() - startTime
                let tokensPerSecond = totalTime > 0 ? Double(tokenCount) / totalTime : 0
                print("[ModelContainer] generation complete: \(tokenCount) tokens, \(String(format: "%.1f", tokensPerSecond)) tok/s, \(String(format: "%.3f", totalTime))s")
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
