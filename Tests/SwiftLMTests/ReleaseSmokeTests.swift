import Foundation
import Testing
@testable import SwiftLM

@Suite("Release Smoke")
struct ReleaseSmokeTests {
    private static let localModelDirectory = URL(
        fileURLWithPath: "/Users/1amageek/Desktop/swift-lm/TestData/LFM2.5-1.2B-Thinking"
    )

    @Test("Local model bundle loads and generates", .timeLimit(.minutes(2)))
    func localBundleLoadPrefillDecodeSmoke() async throws {
        let configURL = Self.localModelDirectory.appendingPathComponent("config.json")
        guard FileManager.default.fileExists(atPath: configURL.path) else {
            print("[Skip] Local release smoke bundle not found at \(Self.localModelDirectory.path)")
            return
        }

        let loader = ModelBundleLoader()
        let container = try await loader.load(directory: Self.localModelDirectory)
        let input = try container.prepare(input: UserInput(prompt: "Hello"))
        let promptState = try container.makePromptState(input: input)

        var chunks: [String] = []
        var completion: CompletionInfo?
        for await generation in container.generate(
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
}
