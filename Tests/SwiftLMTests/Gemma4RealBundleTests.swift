import Foundation
import Testing
@testable import SwiftLM

@Suite("Gemma4 Real Bundle", .serialized)
struct Gemma4RealBundleTests {
    @Test("Real Gemma4 bundle can prepare and generate from an image prompt", .timeLimit(.minutes(10)))
    func realBundleImagePrompt() async throws {
        guard let container = try await Gemma4TestSupport.realGemma4Container() else {
            print("[Skip] No local official Gemma4 E2B snapshot found")
            return
        }
        guard container.configuration.executionCapabilities.supportsImageExecution else {
            print("[Skip] Loaded Gemma4 bundle does not execute image prompts")
            return
        }

        container.resetCaches()
        let imageData = try TestImageFixtures.makeOnePixelPNGData()
        let prepared = try await container.prepare(
            input: ModelInput(chat: [
                .user([
                    .text("Describe the image"),
                    .image(InputImage(data: imageData, mimeType: "image/png")),
                ])
            ])
        )
        let prompt = try container.makeExecutablePrompt(from: prepared)
        let stream = try container.generate(
            prompt: prompt,
            parameters: GenerateParameters(maxTokens: 1, streamChunkTokenCount: 1)
        )
        var sawOutput = false
        for await generation in stream {
            if generation.chunk != nil || generation.info != nil {
                sawOutput = true
                break
            }
        }

        #expect(prompt.gemma4PromptContext != nil)
        #expect(sawOutput)
    }
}
