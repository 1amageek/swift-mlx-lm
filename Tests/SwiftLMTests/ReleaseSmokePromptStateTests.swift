import Testing
@testable import SwiftLM

@Suite("Release Smoke PromptSnapshot", .serialized)
struct ReleaseSmokePromptStateTests {
    @Test("Local model bundle loads and generates", .timeLimit(.minutes(2)))
    func localBundleLoadPrefillDecodeSmoke() async throws {
        guard let localModelDirectory = ReleaseSmokeTestSupport.readableLocalModelDirectoryOrSkip() else { return }

        let loader = ModelBundleLoader()
        let container = try await loader.load(directory: localModelDirectory)
        let context = try container.makeContext()
        let input = try await context.prepare(ModelInput(prompt: "Hello"))
        let executable = try context.makeExecutablePrompt(from: input)
        let promptState = try context.makePromptSnapshot(from: executable)

        var chunks: [String] = []
        var completion: CompletionInfo?
        for await generation in try context.generate(
            from: promptState,
            parameters: GenerationParameters(maxTokens: 4, streamChunkTokenCount: 1)
        ) {
            if let chunk = generation.text {
                chunks.append(chunk)
            }
            if let info = generation.completion {
                completion = info
            }
        }

        let info = try #require(completion)
        #expect(info.tokenCount > 0)
        #expect(!chunks.joined().isEmpty)
    }

    @Test("Local LFM prompt-state sampling matches direct sampling", .timeLimit(.minutes(2)))
    func localBundlePromptStateSamplingMatchesDirect() async throws {
        guard let localModelDirectory = ReleaseSmokeTestSupport.readableLocalModelDirectoryOrSkip() else { return }

        let loader = ModelBundleLoader()
        let container = try await loader.load(directory: localModelDirectory)
        let context = try container.makeContext()
        let prepared = try await context.prepare(ModelInput(prompt: RealOutputAssertionSupport.strictCapitalPrompt)
        )
        let executable = try context.makeExecutablePrompt(from: prepared)

        try RealOutputAssertionSupport.assertPromptStateSamplingMatchesDirect(
            container: context,
            prompt: executable,
            label: "LFM sampling"
        )
    }
}
