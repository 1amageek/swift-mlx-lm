import Foundation
import Testing
@testable import SwiftLM

@Suite("Qwen Vision Runtime", .serialized)
struct QwenVisionExecutionTests {
    @Test("Prepared text-only input becomes executable text-only input", .timeLimit(.minutes(2)))
    func preparedTextOnlyBecomesExecutable() async throws {
        guard let container = try await QwenVisionTestSupport.syntheticMultimodalContainer() else {
            print("[Skip] No local text bundle available for synthetic multimodal runtime tests")
            return
        }
        container.resetCaches()

        let prepared = try await container.prepare(input: ModelInput(prompt: "Hello multimodal runtime"))
        let executable = try container.makeExecutablePrompt(from: prepared)

        #expect(executable.visualContext == nil)
        #expect(executable.tokenIDs == prepared.tokenIDs)
    }

    @Test("Prepared image input becomes executable multimodal input", .timeLimit(.minutes(2)))
    func preparedImageBecomesExecutableMultimodal() async throws {
        guard let container = try await QwenVisionTestSupport.syntheticMultimodalContainer() else {
            print("[Skip] No local text bundle available for synthetic multimodal runtime tests")
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

        let multimodal = try #require(executable.visualContext)
        #expect(multimodal.layout.mmTokenTypeIDs.contains(1))
        #expect(!multimodal.imageTokenEmbeddings.isEmpty)
        #expect(multimodal.imageTokenEmbeddings.count == multimodal.layout.mmTokenTypeIDs.filter { $0 == 1 }.count)
    }

    @Test("Prepared video input becomes executable multimodal input", .timeLimit(.minutes(2)))
    func preparedVideoBecomesExecutableMultimodal() async throws {
        guard let container = try await QwenVisionTestSupport.syntheticMultimodalContainer() else {
            print("[Skip] No local text bundle available for synthetic multimodal runtime tests")
            return
        }
        container.resetCaches()

        let videoData = try await TestVideoFixtures.makeMP4Data()
        let prepared = try await container.prepare(
            input: ModelInput(chat: [
                .user([
                    .text("Describe"),
                    .video(InputVideo(data: videoData, mimeType: "video/mp4")),
                ])
            ])
        )
        let executable = try container.makeExecutablePrompt(from: prepared)

        let multimodal = try #require(executable.visualContext)
        #expect(multimodal.layout.mmTokenTypeIDs.contains(2))
        #expect(!multimodal.videoTokenEmbeddings.isEmpty)
        #expect(multimodal.videoTokenEmbeddings.count == multimodal.layout.mmTokenTypeIDs.filter { $0 == 2 }.count)
    }

    @Test("Synthetic multimodal model input runs through generation", .timeLimit(.minutes(2)))
    func syntheticModelInputGenerates() async throws {
        guard let container = try await QwenVisionTestSupport.syntheticMultimodalContainer() else {
            print("[Skip] No local text bundle available for synthetic multimodal runtime tests")
            return
        }
        container.resetCaches()

        let imageData = try TestImageFixtures.makeOnePixelPNGData()
        let stream = try await container.generate(
            input: ModelInput(chat: [
                .user([
                    .text("Hello"),
                    .image(InputImage(data: imageData, mimeType: "image/png")),
                    .text("world"),
                ])
            ]),
            parameters: GenerateParameters(maxTokens: 2, streamChunkTokenCount: 1)
        )
        let result = await QwenVisionTestSupport.collectGeneration(from: stream)

        #expect(!result.chunks.isEmpty)
        #expect(try #require(result.completion).tokenCount > 0)
    }

    @Test("Synthetic video model input can build a prompt state", .timeLimit(.minutes(2)))
    func syntheticVideoModelInputBuildsPromptState() async throws {
        guard let container = try await QwenVisionTestSupport.syntheticMultimodalContainer() else {
            print("[Skip] No local text bundle available for synthetic multimodal runtime tests")
            return
        }
        container.resetCaches()

        let videoData = try await TestVideoFixtures.makeMP4Data()
        let promptState = try await container.makePromptState(
            input: ModelInput(chat: [
                .user([
                    .text("Watch"),
                    .video(InputVideo(data: videoData, mimeType: "video/mp4")),
                    .text("carefully"),
                ])
            ])
        )

        #expect(promptState.promptTokenCount > 0)
    }

    @Test("Prompt state reuse preserves multimodal execution flow", .timeLimit(.minutes(2)))
    func promptStateReuseForMultimodalInput() async throws {
        guard let container = try await QwenVisionTestSupport.syntheticMultimodalContainer() else {
            print("[Skip] No local text bundle available for synthetic multimodal runtime tests")
            return
        }
        container.resetCaches()

        let imageData = try TestImageFixtures.makeOnePixelPNGData()
        let prepared = try await container.prepare(
            input: ModelInput(chat: [
                .user([
                    .text("Reuse"),
                    .image(InputImage(data: imageData, mimeType: "image/png")),
                    .text("prompt"),
                ])
            ])
        )
        let executable = try container.makeExecutablePrompt(from: prepared)
        let directStream = try container.generate(
            prompt: executable,
            parameters: GenerateParameters(maxTokens: 2, streamChunkTokenCount: 1)
        )
        let direct = await QwenVisionTestSupport.collectGeneration(from: directStream)

        container.resetCaches()
        let promptState = try container.makePromptState(prompt: executable)
        let restoredStream = try container.generate(
            from: promptState,
            parameters: GenerateParameters(maxTokens: 2, streamChunkTokenCount: 1)
        )
        let restored = await QwenVisionTestSupport.collectGeneration(from: restoredStream)

        #expect(direct.chunks == restored.chunks)
        #expect(direct.completion?.tokenCount == restored.completion?.tokenCount)
    }
}
