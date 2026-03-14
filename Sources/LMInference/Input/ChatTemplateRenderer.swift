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
        context["bos_token"] = .string(bosToken ?? "")
        context["eos_token"] = .string(eosToken ?? "")
        context["add_generation_prompt"] = .boolean(addGenerationPrompt)

        // Tools — use sorted-key conversion for deterministic serialization.
        // Swift Dictionary iteration order is non-deterministic, which causes
        // different JSON field ordering across calls and breaks prefix cache matching.
        if let tools {
            let toolValues: [Value] = tools.compactMap { spec in
                do {
                    return try Self.sortedValue(from: spec)
                } catch {
                    return nil
                }
            }
            context["tools"] = .array(toolValues)
        }

        // Additional context (e.g. enable_thinking)
        if let additional = additionalContext {
            for key in additional.keys.sorted() {
                do {
                    context[key] = try Self.sortedValue(from: additional[key] as Any)
                } catch {
                    // Skip values that cannot be converted to Jinja Value
                }
            }
        }

        // Create a fresh Environment per call — Environment is a mutable class
        // that template.render() writes context into, so sharing is unsafe.
        let env = Environment()
        env["tojson"] = Self.deterministicTojson
        return try template.render(context, environment: env)
    }

    // MARK: - Deterministic Serialization

    /// A `tojson` Jinja filter that uses `JSONEncoder` with `.sortedKeys`.
    ///
    /// swift-jinja's built-in `tojson` uses `JSONEncoder` without `.sortedKeys`,
    /// and `Value.encode(to:)` converts `OrderedDictionary` to an unordered
    /// `[String: Value]`. This produces non-deterministic JSON field ordering,
    /// changing the token sequence across calls and breaking `PrefixCachePool`
    /// prefix matching.
    private static let deterministicTojson: Value = .function {
        @Sendable (args: [Value], kwargs: [String: Value], _: Environment) throws -> Value in
        guard let value = args.first else { return .string("null") }

        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]

        if let indentArg = kwargs["indent"] ?? args.dropFirst().first,
           case .int(let count) = indentArg, count > 0 {
            encoder.outputFormatting.insert(.prettyPrinted)
        }

        guard let jsonData = try? encoder.encode(value),
              let jsonString = String(data: jsonData, encoding: .utf8)
        else {
            return .string("null")
        }

        let ensureASCII: Bool
        if let flag = kwargs["ensure_ascii"] {
            ensureASCII = flag.isTruthy
        } else {
            ensureASCII = true
        }

        if ensureASCII {
            return .string(escapeNonASCII(jsonString))
        }
        return .string(jsonString)
    }

    /// Escapes non-ASCII characters as `\uXXXX` sequences.
    private static func escapeNonASCII(_ string: String) -> String {
        var result = ""
        result.reserveCapacity(string.utf16.count)
        for codeUnit in string.utf16 {
            if codeUnit > 127 {
                result += String(format: "\\u%04x", codeUnit)
            } else if let scalar = UnicodeScalar(codeUnit) {
                result.append(Character(scalar))
            }
        }
        return result
    }

    /// Convert an arbitrary value to a Jinja `Value` with dictionary keys sorted.
    ///
    /// Swift `Dictionary` has non-deterministic iteration order. This method
    /// recursively sorts all dictionary keys before building `Value`, ensuring
    /// identical `Value` structure for identical logical input.
    private static func sortedValue(from value: Any?) throws -> Value {
        switch value {
        case let v as Value:
            return v
        case nil:
            return .null
        case let str as String:
            return .string(str)
        case let int as Int:
            return .int(int)
        case let double as Double:
            return .double(double)
        case let float as Float:
            return .double(Double(float))
        case let bool as Bool:
            return .boolean(bool)
        case let array as [Any?]:
            return .array(try array.map { try sortedValue(from: $0) })
        case let dict as [String: Any?]:
            var ordered = OrderedDictionary<String, Value>()
            for key in dict.keys.sorted() {
                ordered[key] = try sortedValue(from: dict[key] ?? nil)
            }
            return .object(ordered)
        default:
            throw JinjaError.runtime(
                "Cannot convert value of type \(type(of: value)) to Jinja Value"
            )
        }
    }
}
