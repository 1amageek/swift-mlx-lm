import Foundation
import Testing
@testable import SwiftLM

@Suite("Qwen Vision Real Bundle PromptSnapshot", .serialized)
struct QwenVisionRealBundlePromptStateTests {
    @Test("Real Qwen3.5/VL bundle can reuse a multimodal prompt state", .timeLimit(.minutes(10)))
    func realBundlePromptStateReuse() async throws {
        guard let container = try await QwenVisionTestSupport.realQwen3VLContainer() else {
            print("[Skip] No local Qwen3.5/VL snapshot found")
            return
        }
        guard container.configuration.executionCapabilities.supportsImageExecution else {
            print("[Skip] Loaded local Qwen vision bundle does not execute image prompts")
            return
        }
        container.resetState()

        let imageData = try TestImageFixtures.makeOnePixelPNGData()
        let promptStateStart = CFAbsoluteTimeGetCurrent()
        let promptState = try await container.makePromptSnapshot(from: ModelInput(chat: [
                .user([
                    .text("Reuse"),
                    .image(InputImage(data: imageData, mimeType: "image/png")),
                ])
            ])
        )
        let promptStateTime = CFAbsoluteTimeGetCurrent() - promptStateStart
        print("[RealQwenVision] promptState build=\(String(format: "%.3f", promptStateTime))s tokens=\(promptState.promptTokenCount)")
        let generationStart = CFAbsoluteTimeGetCurrent()
        let stream = try container.generate(
            from: promptState,
            parameters: GenerationParameters(maxTokens: 1, streamChunkTokenCount: 1)
        )
        let result = await QwenVisionTestSupport.collectGeneration(from: stream)
        let generationTime = CFAbsoluteTimeGetCurrent() - generationStart
        print("[RealQwenVision] promptState generation=\(String(format: "%.3f", generationTime))s")

        #expect(promptState.promptTokenCount > 0)
        #expect(!result.chunks.isEmpty || result.completion != nil)
    }
}
