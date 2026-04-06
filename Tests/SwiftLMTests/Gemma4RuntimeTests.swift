import Foundation
import Testing
@testable import SwiftLM

@Suite("Gemma4 Runtime", .serialized)
struct Gemma4RuntimeTests {
    @Test("Text-only prepared prompt becomes executable Gemma4 prompt", .timeLimit(.minutes(2)))
    func textOnlyPreparedPromptBecomesExecutable() async throws {
        guard let container = try await Gemma4TestSupport.syntheticGemma4Container() else {
            print("[Skip] No Metal device available for Gemma4 runtime tests")
            return
        }
        let prepared = try await container.prepare(input: ModelInput(prompt: "hello gemma4"))
        let executable = try container.makeExecutablePrompt(from: prepared)

        #expect(executable.visualContext == nil)
        #expect(executable.tokenIDs == prepared.tokenIDs)
    }

    @Test("Image prepared prompt becomes executable Gemma4 prompt", .timeLimit(.minutes(2)))
    func imagePreparedPromptBecomesExecutable() async throws {
        guard let container = try await Gemma4TestSupport.syntheticGemma4Container() else {
            print("[Skip] No Metal device available for Gemma4 runtime tests")
            return
        }

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

        #expect(executable.visualContext == nil)
        #expect(executable.tokenIDs == prepared.tokenIDs)
    }

    @Test("Gemma4 text generation runs end-to-end", .timeLimit(.minutes(2)))
    func textGeneration() async throws {
        guard let container = try await Gemma4TestSupport.syntheticGemma4Container() else {
            print("[Skip] No Metal device available for Gemma4 runtime tests")
            return
        }
        let stream = try await container.generate(
            input: ModelInput(prompt: "hello gemma4"),
            parameters: GenerateParameters(maxTokens: 2, streamChunkTokenCount: 1)
        )
        let result = await QwenVisionTestSupport.collectGeneration(from: stream)

        #expect(!result.chunks.isEmpty)
        #expect(try #require(result.completion).tokenCount > 0)
    }

    @Test("Gemma4 image generation runs end-to-end", .timeLimit(.minutes(2)))
    func imageGeneration() async throws {
        guard let container = try await Gemma4TestSupport.syntheticGemma4Container() else {
            print("[Skip] No Metal device available for Gemma4 runtime tests")
            return
        }

        let imageData = try TestImageFixtures.makeOnePixelPNGData()
        let stream = try await container.generate(
            input: ModelInput(chat: [
                .user([
                    .text("Describe"),
                    .image(InputImage(data: imageData, mimeType: "image/png")),
                    .text("briefly"),
                ])
            ]),
            parameters: GenerateParameters(maxTokens: 2, streamChunkTokenCount: 1)
        )
        let result = await QwenVisionTestSupport.collectGeneration(from: stream)

        #expect(!result.chunks.isEmpty)
        #expect(try #require(result.completion).tokenCount > 0)
    }

    @Test("Gemma4 multimodal prompt state can be reused", .timeLimit(.minutes(2)))
    func promptStateReuse() async throws {
        guard let container = try await Gemma4TestSupport.syntheticGemma4Container() else {
            print("[Skip] No Metal device available for Gemma4 runtime tests")
            return
        }

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
        let prompt = try container.makeExecutablePrompt(from: prepared)

        let direct = await QwenVisionTestSupport.collectGeneration(
            from: try container.generate(
                prompt: prompt,
                parameters: GenerateParameters(maxTokens: 2, streamChunkTokenCount: 1)
            )
        )

        container.resetCaches()
        let promptState = try container.makePromptState(prompt: prompt)
        let restored = await QwenVisionTestSupport.collectGeneration(
            from: try container.generate(
                from: promptState,
                parameters: GenerateParameters(maxTokens: 2, streamChunkTokenCount: 1)
            )
        )

        #expect(direct.chunks == restored.chunks)
        #expect(direct.completion?.tokenCount == restored.completion?.tokenCount)
    }
}

