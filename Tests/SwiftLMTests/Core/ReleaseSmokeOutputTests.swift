import Testing
@testable import SwiftLM

@Suite("Release Smoke Output", .serialized)
struct ReleaseSmokeOutputTests {
    @Test("Local LFM bundle answers the capital-of-Japan prompt", .timeLimit(.minutes(2)))
    func localBundleCapitalOfJapanOutput() async throws {
        guard let localModelDirectory = ReleaseSmokeTestSupport.readableLocalModelDirectoryOrSkip() else { return }

        let loader = ModelBundleLoader()
        let container = try await loader.load(directory: localModelDirectory)
        let context = try LanguageModelContext(container)
        let prepared = RealOutputAssertionSupport.directTextPrompt(
            RealOutputAssertionSupport.capitalCompletionPrompt,
            using: context
        )
        let executable = try ExecutablePrompt(preparedPrompt: prepared, using: context)
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

}
