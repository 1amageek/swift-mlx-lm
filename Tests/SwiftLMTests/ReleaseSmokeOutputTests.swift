import Testing
@testable import SwiftLM

@Suite("Release Smoke Output", .serialized)
struct ReleaseSmokeOutputTests {
    @Test("Local LFM bundle answers the capital-of-Japan prompt", .timeLimit(.minutes(2)))
    func localBundleCapitalOfJapanOutput() async throws {
        guard let localModelDirectory = ReleaseSmokeTestSupport.readableLocalModelDirectoryOrSkip() else { return }

        let loader = ModelBundleLoader()
        let container = try await loader.load(directory: localModelDirectory)
        let prepared = try await container.prepare(
            input: ModelInput(prompt: RealOutputAssertionSupport.strictCapitalPrompt)
        )
        let executable = try container.makeExecutablePrompt(from: prepared)
        let comparison = try RealOutputAssertionSupport.assertGreedyDirectMatchesPromptState(
            container: container,
            prompt: executable,
            label: "LFM text greedy"
        )
        RealOutputAssertionSupport.assertStartsWithTokyo(
            comparison.directText,
            label: "LFM text greedy"
        )
    }

    @Test("Local LFM chat prompt starts a strict factual answer with Tokyo", .timeLimit(.minutes(2)))
    func localBundleCapitalOfJapanChatOutput() async throws {
        guard let localModelDirectory = ReleaseSmokeTestSupport.readableLocalModelDirectoryOrSkip() else { return }

        let loader = ModelBundleLoader()
        let container = try await loader.load(directory: localModelDirectory)
        let prepared = try await container.prepare(
            input: ModelInput(chat: [
                .user([.text(RealOutputAssertionSupport.strictCapitalPrompt)])
            ])
        )
        let executable = try container.makeExecutablePrompt(from: prepared)
        let comparison = try RealOutputAssertionSupport.assertGreedyDirectMatchesPromptState(
            container: container,
            prompt: executable,
            label: "LFM chat greedy"
        )
        RealOutputAssertionSupport.assertStartsWithTokyo(
            comparison.directText,
            label: "LFM chat greedy"
        )
    }
}
