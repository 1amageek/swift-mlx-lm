import Foundation
import Testing
@testable import SwiftLM

@Suite("Qwen Vision Integration", .serialized)
struct QwenVisionIntegrationTests {
    @Test("Prepare image then make executable prompt then generate", .timeLimit(.minutes(2)))
    func prepareImageMakeExecutableAndGenerate() async throws {
        guard let container = try await QwenVisionTestSupport.syntheticMultimodalContainer() else {
            print("[Skip] No local text bundle available for synthetic multimodal integration tests")
            return
        }
        container.resetState()

        let imageData = try TestImageFixtures.makeOnePixelPNGData()
        let prepared = try await container.prepare( ModelInput(chat: [
                .user([
                    .text("Inspect"),
                    .image(InputImage(data: imageData, mimeType: "image/png")),
                    .text("now"),
                ])
            ])
        )
        let executable = try container.makeExecutablePrompt(from: prepared)
        let stream = try container.generate(from: executable,
            parameters: GenerationParameters(maxTokens: 2, streamChunkTokenCount: 1)
        )
        let result = await QwenVisionTestSupport.collectGeneration(from: stream)

        #expect(prepared.multimodalMetadata?.images.count == 1)
        #expect(executable.visualContext != nil)
        #expect(!result.chunks.isEmpty)
    }

    @Test("Prepare video then make executable prompt then generate", .timeLimit(.minutes(2)))
    func prepareVideoMakeExecutableAndGenerate() async throws {
        guard let container = try await QwenVisionTestSupport.syntheticMultimodalContainer() else {
            print("[Skip] No local text bundle available for synthetic multimodal integration tests")
            return
        }
        container.resetState()

        let videoData = try await TestVideoFixtures.makeMP4Data()
        let prepared = try await container.prepare( ModelInput(chat: [
                .user([
                    .text("Inspect"),
                    .video(InputVideo(data: videoData, mimeType: "video/mp4")),
                    .text("now"),
                ])
            ])
        )
        let executable = try container.makeExecutablePrompt(from: prepared)
        let stream = try container.generate(from: executable,
            parameters: GenerationParameters(maxTokens: 2, streamChunkTokenCount: 1)
        )
        let result = await QwenVisionTestSupport.collectGeneration(from: stream)

        #expect(prepared.multimodalMetadata?.videos.count == 1)
        #expect(executable.visualContext != nil)
        #expect(!result.chunks.isEmpty)
    }

    @Test("Prepare mixed image and video chat then generate", .timeLimit(.minutes(2)))
    func prepareMixedImageAndVideoChatThenGenerate() async throws {
        guard let container = try await QwenVisionTestSupport.syntheticMultimodalContainer() else {
            print("[Skip] No local text bundle available for synthetic multimodal integration tests")
            return
        }
        container.resetState()

        let imageData = try TestImageFixtures.makeOnePixelPNGData()
        let videoData = try await TestVideoFixtures.makeMP4Data()
        let stream = try await container.generate( ModelInput(chat: [
                .user([
                    .text("Compare"),
                    .image(InputImage(data: imageData, mimeType: "image/png")),
                    .text("and"),
                    .video(InputVideo(data: videoData, mimeType: "video/mp4")),
                    .text("summarize"),
                ])
            ]),
            parameters: GenerationParameters(maxTokens: 2, streamChunkTokenCount: 1)
        )
        let result = await QwenVisionTestSupport.collectGeneration(from: stream)

        #expect(!result.chunks.isEmpty)
        #expect(try #require(result.completion).tokenCount > 0)
    }

    @Test("Multimodal prompt state can be reused for generation", .timeLimit(.minutes(2)))
    func multimodalPromptStateReuse() async throws {
        guard let container = try await QwenVisionTestSupport.syntheticMultimodalContainer() else {
            print("[Skip] No local text bundle available for synthetic multimodal integration tests")
            return
        }
        container.resetState()

        let imageData = try TestImageFixtures.makeOnePixelPNGData()
        let promptState = try await container.makePromptSnapshot(from: ModelInput(chat: [
                .user([
                    .text("Reuse"),
                    .image(InputImage(data: imageData, mimeType: "image/png")),
                    .text("this"),
                ])
            ])
        )
        let stream = try container.generate(
            from: promptState,
            parameters: GenerationParameters(maxTokens: 2, streamChunkTokenCount: 1)
        )
        let result = await QwenVisionTestSupport.collectGeneration(from: stream)

        #expect(promptState.promptTokenCount > 0)
        #expect(!result.chunks.isEmpty)
    }
}
