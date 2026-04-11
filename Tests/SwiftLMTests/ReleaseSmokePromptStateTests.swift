import Testing
@testable import SwiftLM

@Suite("Release Smoke PromptSnapshot", .serialized)
struct ReleaseSmokePromptStateTests {
    @Test("Local model bundle loads and generates", .timeLimit(.minutes(2)))
    func localBundleLoadPrefillDecodeSmoke() async throws {
        guard let localModelDirectory = ReleaseSmokeTestSupport.readableLocalModelDirectoryOrSkip() else { return }

        let loader = ModelBundleLoader()
        let container = try await loader.load(directory: localModelDirectory)
        let input = try await container.prepare( ModelInput(prompt: "Hello"))
        let executable = try container.makeExecutablePrompt(from: input)
        let promptState = try container.makePromptSnapshot(from: executable)

        var chunks: [String] = []
        var completion: CompletionInfo?
        for await generation in try container.generate(
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
        let prepared = try await container.prepare( ModelInput(prompt: RealOutputAssertionSupport.strictCapitalPrompt)
        )
        let executable = try container.makeExecutablePrompt(from: prepared)

        try RealOutputAssertionSupport.assertPromptStateSamplingMatchesDirect(
            container: container,
            prompt: executable,
            label: "LFM sampling"
        )
    }
}
