import Foundation
import Jinja
import OrderedCollections

/// Renders chat messages into a prompt string using a Jinja chat template.
///
/// Uses the `tokenizer.chat_template` from GGUF metadata to format
/// messages with the correct special tokens and formatting.
struct ChatTemplateRenderer: Sendable {

    private let template: Template
    private let bosToken: String?
    private let eosToken: String?

    /// Create a renderer with a Jinja template string.
    ///
    /// - Parameters:
    ///   - templateString: Raw Jinja2 template from GGUF `tokenizer.chat_template`.
    ///   - bosToken: Beginning-of-sequence token string (e.g., "<|begin_of_text|>").
    ///   - eosToken: End-of-sequence token string (e.g., "<|eot_id|>").
    init(templateString: String, bosToken: String?, eosToken: String?) throws {
        self.template = try Template(templateString)
        self.bosToken = bosToken
        self.eosToken = eosToken
    }

    /// Render a list of chat messages into a prompt string.
    ///
    /// - Parameters:
    ///   - messages: Chat messages to render.
    ///   - tools: Optional tool specifications for function-calling models.
    ///   - addGenerationPrompt: Whether to append the assistant turn prefix.
    /// - Returns: The fully formatted prompt string.
    func render(
        messages: [Chat.Message],
        tools: [ToolSpec]? = nil,
        additionalContext: [String: any Sendable]? = nil,
        addGenerationPrompt: Bool = true
    ) throws -> String {
        var context: [String: Value] = [:]

        // messages: array of {role, content} dicts
        let messageValues: [Value] = messages.map { msg in
            var dict = OrderedDictionary<String, Value>()
            dict["role"] = .string(msg.role.rawValue)
            dict["content"] = .string(msg.content)
            return .object(dict)
        }
        context["messages"] = .array(messageValues)

        // Special tokens
        if let bos = bosToken {
            context["bos_token"] = .string(bos)
        } else {
            context["bos_token"] = .string("")
        }

        if let eos = eosToken {
            context["eos_token"] = .string(eos)
        } else {
            context["eos_token"] = .string("")
        }

        context["add_generation_prompt"] = .boolean(addGenerationPrompt)

        // Tools
        if let tools {
            let toolValues: [Value] = tools.compactMap { spec in
                do {
                    return try Value(any: spec)
                } catch {
                    return nil
                }
            }
            context["tools"] = .array(toolValues)
        }

        // Additional context (e.g. enable_thinking)
        if let additional = additionalContext {
            for (key, value) in additional {
                do {
                    context[key] = try Value(any: value)
                } catch {
                    // Skip values that cannot be converted to Jinja Value
                }
            }
        }

        return try template.render(context)
    }
}
