import Foundation
import Testing
@testable import SwiftLM

@Suite("Qwen Vision Real Bundle Mixed", .serialized)
struct QwenVisionRealBundleMixedTests {
    @Test("Real Qwen3.5/VL bundle can prepare mixed image and video chats", .timeLimit(.minutes(10)))
    func realBundleMixedPrompt() async throws {
        guard let container = try await QwenVisionTestSupport.realQwen3VLContainer() else {
            print("[Skip] No local Qwen3.5/VL snapshot found")
            return
        }
        guard container.configuration.executionCapabilities.supportsImageExecution,
              container.configuration.executionCapabilities.supportsVideoExecution else {
            print("[Skip] Loaded local Qwen vision bundle does not execute both image and video prompts")
            return
        }
        container.resetCaches()

        let imageData = try TestImageFixtures.makeOnePixelPNGData()
        let videoData = try await TestVideoFixtures.makeMP4Data()
        let stream = try await container.generate(
            input: ModelInput(chat: [
                .user([
                    .text("Compare"),
                    .image(InputImage(data: imageData, mimeType: "image/png")),
                    .text("and"),
                    .video(InputVideo(data: videoData, mimeType: "video/mp4")),
                ])
            ]),
            parameters: GenerateParameters(maxTokens: 1, streamChunkTokenCount: 1)
        )
        let result = await QwenVisionTestSupport.collectGeneration(from: stream)

        #expect(!result.chunks.isEmpty || result.completion != nil)
    }
}
