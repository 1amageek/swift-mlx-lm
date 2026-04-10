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
        let container = try await ModelBundleLoader().load(directory: directory)

        container.resetCaches()
        let prepared = try await container.prepare(
            input: ModelInput(prompt: RealOutputAssertionSupport.strictCapitalPrompt)
        )
        let prompt = try container.makeExecutablePrompt(from: prepared)
        let comparison = try RealOutputAssertionSupport.assertGreedyDirectMatchesPromptState(
            container: container,
            prompt: prompt,
            label: "Qwen3.5 real greedy"
        )
        let tokenIDs = comparison.directTokenIDs
        RealOutputAssertionSupport.assertStartsWithTokyo(
            comparison.directText,
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
        let container = try await ModelBundleLoader().load(directory: directory)

        let parameters = GenerateParameters(
            maxTokens: 64,
            streamChunkTokenCount: 1,
            temperature: 0,
            thinking: .visible
        )
        let prepared = try await container.prepare(input: ModelInput(
            chat: [
                .user([.text("Hello")])
            ],
            promptOptions: PromptPreparationOptions(thinkingEnabled: true)
        ))
        print("[Qwen3.5 thinking rendered text prefix]")
        print(String(prepared.renderedText.prefix(400)))
        let prompt = try container.makeExecutablePrompt(from: prepared)

        container.resetCaches()
        let visibleTokenIDs = try container.debugGeneratedTokenIDs(
            prompt: prompt,
            parameters: parameters
        )

        container.resetCaches()
        let rawTokenIDs = try container.debugRawGeneratedTokenIDs(
            prompt: prompt,
            parameters: parameters
        )

        container.resetCaches()
        let stream = try container.generate(prompt: prompt, parameters: parameters)
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
