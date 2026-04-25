import Foundation
import Testing
@testable import SwiftLM

@Suite("Qwen Vision Real Bundle Text", .serialized)
struct QwenVisionRealBundleTextTests {
    @Test("Real Qwen3.5 bundle generates text from a simple prompt", .timeLimit(.minutes(10)))
    func realBundleTextPrompt() async throws {
        guard let directory = try QwenVisionTestSupport.optionalRealQwen3VLDirectory() else {
            print("[Skip] No local Qwen3.5 snapshot found")
            return
        }
        let loaded = try await ModelBundleLoader().load(directory: directory)
        let container = try LanguageModelContext(loaded)

        container.resetState()
        let prepared = try await container.prepare(ModelInput(
            chat: [.user([.text(RealOutputAssertionSupport.strictCapitalPrompt)])],
            promptOptions: PromptPreparationOptions(isThinkingEnabled: false)
        ))
        let prompt = try ExecutablePrompt(preparedPrompt: prepared, using: container)
        container.resetState()
        let tokenIDs = try container.debugGeneratedTokenIDs(
            prompt: prompt,
            parameters: RealOutputAssertionSupport.greedyParameters(maxTokens: 6)
        )
        let text = RealOutputAssertionSupport.normalized(container.decode(tokenIDs))
        print("[Qwen3.5 real greedy token ids] \(tokenIDs)")
        print("[Qwen3.5 real greedy text] \(text)")
        RealOutputAssertionSupport.assertHasPrefix(
            text,
            prefix: "Japan's capital is Tokyo",
            label: "Qwen3.5 real greedy"
        )
        #expect(tokenIDs.count >= 2)
    }

    @Test("Real Qwen3.5 bundle can expose thinking content for chat prompts", .timeLimit(.minutes(10)))
    func realBundleChatThinkingVisible() async throws {
        guard let directory = try QwenVisionTestSupport.optionalRealQwen3VLDirectory() else {
            print("[Skip] No local Qwen3.5 snapshot found")
            return
        }
        let loaded = try await ModelBundleLoader().load(directory: directory)
        let container = try LanguageModelContext(loaded)

        let parameters = GenerationParameters(
            maxTokens: 64,
            streamChunkTokenCount: 1,
            temperature: 0,
            reasoning: .inline
        )
        let prepared = try await container.prepare( ModelInput(
            chat: [
                .user([.text("Hello")])
            ],
            promptOptions: PromptPreparationOptions(isThinkingEnabled: true)
        ))
        print("[Qwen3.5 thinking rendered text prefix]")
        print(String(prepared.renderedText.prefix(400)))
        let prompt = try ExecutablePrompt(preparedPrompt: prepared, using: container)

        container.resetState()
        let visibleTokenIDs = try container.debugGeneratedTokenIDs(
            prompt: prompt,
            parameters: parameters
        )

        container.resetState()
        let rawTokenIDs = try container.debugRawGeneratedTokenIDs(
            prompt: prompt,
            parameters: parameters
        )

        container.resetState()
        let stream = try container.generate(from: prompt, parameters: parameters)
        let streamed = await QwenVisionTestSupport.collectGeneration(from: stream)

        let visibleText = container.tokenizer.decode(tokens: visibleTokenIDs, skipSpecialTokens: false)
        let rawText = container.tokenizer.decode(tokens: rawTokenIDs, skipSpecialTokens: false)
        let streamedText = streamed.chunks.joined()

        print("[Qwen3.5 thinking visible text prefix]")
        print(String(visibleText.prefix(400)))
        print("[Qwen3.5 thinking streamed text prefix]")
        print(String(streamedText.prefix(400)))

        #expect(!rawTokenIDs.isEmpty)
        #expect(prepared.renderedText.contains("<think>"))
        #expect(visibleTokenIDs == rawTokenIDs)
        #expect(visibleText == rawText)
        #expect(streamedText == visibleText)
        #expect(!visibleText.isEmpty)
    }
}
