import MLX

/// Converts user input to tokenized model input using chat template.
///
/// Pipeline: UserInput → ChatTemplateRenderer → tokenize → LMInput
struct ChatTemplateInputProcessor: UserInputProcessor {

    private let tokenizer: any Tokenizer
    private let renderer: ChatTemplateRenderer?
    private let addBosToken: Bool

    /// Create a processor for a model.
    ///
    /// - Parameters:
    ///   - tokenizer: The tokenizer for encoding text to tokens.
    ///   - chatTemplate: Optional Jinja chat template string.
    ///   - bosToken: BOS token string for template rendering.
    ///   - eosToken: EOS token string for template rendering.
    ///   - addBosToken: Whether to prepend BOS token when no chat template is used.
    init(
        tokenizer: any Tokenizer,
        chatTemplate: String?,
        bosToken: String?,
        eosToken: String?,
        addBosToken: Bool
    ) {
        self.tokenizer = tokenizer
        self.addBosToken = addBosToken

        if let template = chatTemplate {
            do {
                self.renderer = try ChatTemplateRenderer(
                    templateString: template,
                    bosToken: bosToken,
                    eosToken: eosToken
                )
            } catch {
                let preview = String(template.prefix(200)).replacingOccurrences(of: "\n", with: "\\n")
                print(
                    "[ChatTemplateInputProcessor] chat_template parse failed: \(error). Falling back to plain prompt formatting. templatePreview=\"\(preview)\""
                )
                self.renderer = nil
            }
        } else {
            self.renderer = nil
        }
    }

    func prepare(input: UserInput) async throws -> LMInput {
        let prompt = try renderPrompt(
            messages: input.chat,
            tools: input.tools,
            additionalContext: input.additionalContext,
            addGenerationPrompt: true
        )
        return tokenize(prompt)
    }

    func preparePrefix(input: UserInput) async throws -> LMInput {
        let prompt = try renderPrompt(
            messages: input.chat,
            tools: input.tools,
            additionalContext: input.additionalContext,
            addGenerationPrompt: false
        )
        return tokenize(prompt)
    }

    // MARK: - Private

    private func renderPrompt(
        messages: [Chat.Message],
        tools: [ToolSpec]?,
        additionalContext: [String: any Sendable]?,
        addGenerationPrompt: Bool
    ) throws -> String {
        if let renderer {
            return try renderer.render(
                messages: messages,
                tools: tools,
                additionalContext: additionalContext,
                addGenerationPrompt: addGenerationPrompt
            )
        }

        // Fallback: simple concatenation without template
        return formatWithoutTemplate(messages: messages)
    }

    /// Simple fallback formatting when no chat template is available.
    private func formatWithoutTemplate(messages: [Chat.Message]) -> String {
        var parts: [String] = []
        for message in messages {
            switch message.role {
            case .system:
                parts.append("System: \(message.content)")
            case .user:
                parts.append("User: \(message.content)")
            case .assistant:
                parts.append("Assistant: \(message.content)")
            case .tool:
                parts.append("Tool: \(message.content)")
            }
        }
        parts.append("Assistant:")
        return parts.joined(separator: "\n")
    }

    private func tokenize(_ text: String) -> LMInput {
        let tokenIDs = tokenizer.encode(text: text)
        RawTokenTraceLogger(tokenizer: tokenizer).logPrompt(prompt: text, tokens: tokenIDs)
        let tokenArray = MLXArray(tokenIDs.map { Int32($0) }).reshaped([1, tokenIDs.count])
        let textInput = LMInput.Text(tokens: tokenArray, cpuTokenIDs: tokenIDs)
        return LMInput(text: textInput)
    }
}
