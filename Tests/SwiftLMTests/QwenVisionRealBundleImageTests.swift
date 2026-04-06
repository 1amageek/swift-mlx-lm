import Foundation
import Testing
@testable import SwiftLM

@Suite("Qwen Vision Real Bundle Image", .serialized)
struct QwenVisionRealBundleImageTests {
    @Test("Real Qwen3.5/VL bundle can prepare and generate from an image prompt", .timeLimit(.minutes(10)))
    func realBundleImagePrompt() async throws {
        guard let container = try await QwenVisionTestSupport.realQwen3VLContainer() else {
            print("[Skip] No local Qwen3.5/VL snapshot found")
            return
        }
        guard container.configuration.executionCapabilities.supportsImageExecution else {
            print("[Skip] Loaded local Qwen vision bundle does not execute image prompts")
            return
        }
        container.resetCaches()

        let imageData = try TestImageFixtures.makeOnePixelPNGData()
        let prepared = try await container.prepare(
            input: ModelInput(chat: [
                .user([
                    .text("Describe"),
                    .image(InputImage(data: imageData, mimeType: "image/png")),
                ])
            ])
        )
        let executable = try container.makeExecutablePrompt(from: prepared)
        let stream = try container.generate(
            prompt: executable,
            parameters: GenerateParameters(maxTokens: 1, streamChunkTokenCount: 1)
        )
        let result = await QwenVisionTestSupport.collectGeneration(from: stream)

        #expect(executable.visualContext != nil)
        #expect(!result.chunks.isEmpty || result.completion != nil)
    }
}
