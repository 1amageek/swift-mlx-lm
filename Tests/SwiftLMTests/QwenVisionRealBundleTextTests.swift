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
}
