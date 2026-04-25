import Testing
@testable import SwiftLM

@Suite("Release Smoke Output", .serialized)
struct ReleaseSmokeOutputTests {
    @Test("Local LFM chat greeting emits reasoning and final answer", .timeLimit(.minutes(10)))
    func localLFMChatGreetingEmitsReasoningAndFinalAnswer() async throws {
        guard let localModelDirectory = ReleaseSmokeTestSupport.readableLocalModelDirectoryOrSkip() else { return }

        let loader = ModelBundleLoader()
        let container = try await loader.load(directory: localModelDirectory)
        let context = try LanguageModelContext(container)
        let stream = try await context.generate(
            ModelInput(
                prompt: "hi",
                promptOptions: PromptPreparationOptions(isThinkingEnabled: true)
            ),
            parameters: GenerationParameters(
                maxTokens: 32,
                streamChunkTokenCount: 1,
                reasoning: .separate
            )
        )

        var answer = ""
        var reasoning = ""
        for await generation in stream {
            switch generation {
            case .text(let chunk):
                answer += chunk
            case .reasoning(let chunk):
                reasoning += chunk
            case .completed:
                break
            }
        }

        print("[LFM chat greeting reasoning prefix] \(String(reasoning.prefix(400)))")
        print("[LFM chat greeting answer] \(answer)")

        #expect(!reasoning.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
        #expect(!answer.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
        #expect(!answer.contains("<think>"))
        #expect(!answer.contains("</think>"))
        #expect(answer.localizedCaseInsensitiveContains("hello"))
    }

    @Test("Local LFM thinking chat prefix matches HuggingFace reference", .timeLimit(.minutes(2)))
    func localLFMThinkingChatPrefixMatchesReference() async throws {
        guard let localModelDirectory = ReleaseSmokeTestSupport.readableLocalModelDirectoryOrSkip() else { return }

        let loader = ModelBundleLoader()
        let container = try await loader.load(directory: localModelDirectory)
        let context = try LanguageModelContext(container)
        let prepared = try await context.prepare(ModelInput(
            prompt: RealOutputAssertionSupport.strictCapitalPrompt,
            promptOptions: PromptPreparationOptions(isThinkingEnabled: true)
        ))
        let executable = try ExecutablePrompt(preparedPrompt: prepared, using: context)
        let tokenIDs = try context.debugRawGeneratedTokenIDs(
            prompt: executable,
            parameters: RealOutputAssertionSupport.greedyParameters(maxTokens: 8)
        )
        let rawText = context.tokenizer.decode(tokens: tokenIDs, skipSpecialTokens: false)

        print("[LFM thinking chat reference token ids] \(tokenIDs)")
        print("[LFM thinking chat reference raw text] \(rawText)")

        #expect(tokenIDs.count >= 3, "LFM thinking chat should emit the reference prefix")
        #expect(Array(tokenIDs.prefix(3)) == [64400, 9095, 892])
        #expect(rawText.hasPrefix("<think> Okay"))
    }

    @Test("Local LFM hidden thinking chat emits visible final answer", .timeLimit(.minutes(10)))
    func localLFMHiddenThinkingChatEmitsVisibleFinalAnswer() async throws {
        guard let localModelDirectory = ReleaseSmokeTestSupport.readableLocalModelDirectoryOrSkip() else { return }

        let loader = ModelBundleLoader()
        let container = try await loader.load(directory: localModelDirectory)
        let context = try LanguageModelContext(container)
        let prepared = try await context.prepare(ModelInput(
            prompt: "hi",
            promptOptions: PromptPreparationOptions(isThinkingEnabled: false)
        ))
        let executable = try ExecutablePrompt(preparedPrompt: prepared, using: context)
        let tokenIDs = try context.debugGeneratedTokenIDs(
            prompt: executable,
            parameters: GenerationParameters(
                maxTokens: 32,
                streamChunkTokenCount: 1
            )
        )
        let text = RealOutputAssertionSupport.normalized(context.decode(tokenIDs))

        print("[LFM hidden thinking visible token ids] \(tokenIDs)")
        print("[LFM hidden thinking visible text] \(text)")

        #expect(!text.isEmpty, "LFM hidden thinking should produce a visible final answer")
        #expect(!text.contains("<think>"))
        #expect(!text.contains("</think>"))
        #expect(text.localizedCaseInsensitiveContains("hello"))
    }

}
