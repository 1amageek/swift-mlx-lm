import Testing
@testable import SwiftLM

@Suite("Release Smoke Output", .serialized)
struct ReleaseSmokeOutputTests {
    @Test("Local LFM bundle answers the capital-of-Japan prompt", .timeLimit(.minutes(2)))
    func localBundleCapitalOfJapanOutput() async throws {
        guard let localModelDirectory = ReleaseSmokeTestSupport.readableLocalModelDirectoryOrSkip() else { return }

        let loader = ModelBundleLoader()
        let container = try await loader.load(directory: localModelDirectory)
        let context = try container.makeContext()
        let prepared = try await context.prepare(ModelInput(prompt: RealOutputAssertionSupport.strictCapitalPrompt)
        )
        let executable = try context.makeExecutablePrompt(from: prepared)
        let comparison = try RealOutputAssertionSupport.assertGreedyDirectMatchesPromptState(
            container: context,
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
        let context = try container.makeContext()
        let prepared = try await context.prepare(ModelInput(chat: [
                .user([.text(RealOutputAssertionSupport.strictCapitalPrompt)])
            ])
        )
        let executable = try context.makeExecutablePrompt(from: prepared)
        let comparison = try RealOutputAssertionSupport.assertGreedyDirectMatchesPromptState(
            container: context,
            prompt: executable,
            label: "LFM chat greedy"
        )
        RealOutputAssertionSupport.assertStartsWithTokyo(
            comparison.directText,
            label: "LFM chat greedy"
        )
    }
}
