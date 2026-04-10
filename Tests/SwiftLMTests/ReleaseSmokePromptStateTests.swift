import Testing
@testable import SwiftLM

@Suite("Release Smoke PromptState", .serialized)
struct ReleaseSmokePromptStateTests {
    @Test("Local model bundle loads and generates", .timeLimit(.minutes(2)))
    func localBundleLoadPrefillDecodeSmoke() async throws {
        guard let localModelDirectory = ReleaseSmokeTestSupport.readableLocalModelDirectoryOrSkip() else { return }

        let loader = ModelBundleLoader()
        let container = try await loader.load(directory: localModelDirectory)
        let input = try await container.prepare(input: ModelInput(prompt: "Hello"))
        let executable = try container.makeExecutablePrompt(from: input)
        let promptState = try container.makePromptState(prompt: executable)

        var chunks: [String] = []
        var completion: CompletionInfo?
        for await generation in try container.generate(
            from: promptState,
            parameters: GenerateParameters(maxTokens: 4, streamChunkTokenCount: 1)
        ) {
            if let chunk = generation.chunk {
                chunks.append(chunk)
            }
            if let info = generation.info {
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
        let prepared = try await container.prepare(
            input: ModelInput(prompt: RealOutputAssertionSupport.strictCapitalPrompt)
        )
        let executable = try container.makeExecutablePrompt(from: prepared)

        try RealOutputAssertionSupport.assertPromptStateSamplingMatchesDirect(
            container: container,
            prompt: executable,
            label: "LFM sampling"
        )
    }
}
