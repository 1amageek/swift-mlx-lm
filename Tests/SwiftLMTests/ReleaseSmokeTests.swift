import Foundation
import Testing
@testable import SwiftLM

@Suite("Release Smoke", .serialized)
struct ReleaseSmokeTests {
    private static let localModelDirectory = URL(
        fileURLWithPath: "/Users/1amageek/Desktop/swift-lm/TestData/LFM2.5-1.2B-Thinking"
    )

    private func readableLocalModelDirectoryOrSkip() -> URL? {
        let configURL = Self.localModelDirectory.appendingPathComponent("config.json")
        do {
            _ = try Data(contentsOf: configURL)
            return Self.localModelDirectory
        } catch {
            print("[Skip] Local release smoke bundle is not readable at \(Self.localModelDirectory.path): \(error)")
            return nil
        }
    }

    @Test("Local model bundle loads and generates", .timeLimit(.minutes(2)))
    func localBundleLoadPrefillDecodeSmoke() async throws {
        guard let localModelDirectory = readableLocalModelDirectoryOrSkip() else { return }

        let loader = ModelBundleLoader()
        let container = try await loader.load(directory: localModelDirectory)
        let input = try await container.prepare(input: ModelInput(prompt: "Hello"))
        let executable = try container.makeExecutablePrompt(from: input)
        let promptState = try container.makePromptState(prompt: executable)

        var chunks: [String] = []
        var completion: CompletionInfo?
        for await generation in try container.generate(
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

    @Test("Local LFM bundle answers the capital-of-Japan prompt", .timeLimit(.minutes(2)))
    func localBundleCapitalOfJapanOutput() async throws {
        guard let localModelDirectory = readableLocalModelDirectoryOrSkip() else { return }

        let loader = ModelBundleLoader()
        let container = try await loader.load(directory: localModelDirectory)
        let prepared = try await container.prepare(
            input: ModelInput(prompt: RealOutputAssertionSupport.strictCapitalPrompt)
        )
        let executable = try container.makeExecutablePrompt(from: prepared)
        let comparison = try RealOutputAssertionSupport.assertGreedyDirectMatchesPromptState(
            container: container,
            prompt: executable,
            label: "LFM text greedy"
        )
        RealOutputAssertionSupport.assertStartsWithTokyo(
            comparison.directText,
            label: "LFM text greedy"
        )
    }

    @Test("Local LFM chat prompt starts a strict factual answer with Tokyo", .timeLimit(.minutes(2)))
    func localBundleCapitalOfJapanChatOutput() async throws {
        guard let localModelDirectory = readableLocalModelDirectoryOrSkip() else { return }

        let loader = ModelBundleLoader()
        let container = try await loader.load(directory: localModelDirectory)
        let prepared = try await container.prepare(
            input: ModelInput(chat: [
                .user([.text(RealOutputAssertionSupport.strictCapitalPrompt)])
            ])
        )
        let executable = try container.makeExecutablePrompt(from: prepared)
        let comparison = try RealOutputAssertionSupport.assertGreedyDirectMatchesPromptState(
            container: container,
            prompt: executable,
            label: "LFM chat greedy"
        )
        RealOutputAssertionSupport.assertStartsWithTokyo(
            comparison.directText,
            label: "LFM chat greedy"
        )
    }

    @Test("Local LFM prompt-state sampling matches direct sampling", .timeLimit(.minutes(2)))
    func localBundlePromptStateSamplingMatchesDirect() async throws {
        guard let localModelDirectory = readableLocalModelDirectoryOrSkip() else { return }

        let loader = ModelBundleLoader()
        let container = try await loader.load(directory: localModelDirectory)
        let prepared = try await container.prepare(
            input: ModelInput(prompt: RealOutputAssertionSupport.strictCapitalPrompt)
        )
        let executable = try container.makeExecutablePrompt(from: prepared)

        try RealOutputAssertionSupport.assertPromptStateSamplingMatchesDirect(
            container: container,
            prompt: executable,
            label: "LFM sampling"
        )
    }

    @Test("Text-only bundle rejects multimodal input", .timeLimit(.minutes(2)))
    func textOnlyBundleRejectsMultimodalInput() async throws {
        guard let localModelDirectory = readableLocalModelDirectoryOrSkip() else { return }

        let loader = ModelBundleLoader()
        let container = try await loader.load(directory: localModelDirectory)
        let supportsImages = container.configuration.inputCapabilities.supportsImages

        if !supportsImages {
            do {
                _ = try await container.prepare(input: ModelInput(chat: [
                    .user([
                        .text("Describe this image."),
                        .image(InputImage(data: try TestImageFixtures.makeOnePixelPNGData(), mimeType: "image/png")),
                    ])
                ]))
                Issue.record("Expected text-only bundle to reject image-bearing input")
            } catch ModelContainerError.unsupportedInputForModel {
                return
            }
            return
        }

        let prepared = try await container.prepare(input: ModelInput(chat: [
            .user([
                .text("Describe this image."),
                .image(InputImage(data: try TestImageFixtures.makeOnePixelPNGData(), mimeType: "image/png")),
            ])
        ]))
        #expect(prepared.multimodalMetadata != nil)
        #expect(container.configuration.executionCapabilities.supportsImagePromptPreparation)
        #expect(!container.configuration.executionCapabilities.supportsImageExecution)

        do {
            _ = try await container.generate(input: ModelInput(chat: [
                .user([
                    .text("Describe this image."),
                    .image(InputImage(data: try TestImageFixtures.makeOnePixelPNGData(), mimeType: "image/png")),
                ])
            ]))
            Issue.record("Expected multimodal generate(input: ModelInput) to throw")
        } catch ModelContainerError.multimodalInputNotSupported {
            // Text-only bundles must still reject multimodal execution.
        }

        do {
            _ = try container.makeExecutablePrompt(from: prepared)
            Issue.record("Expected multimodal prepared input to remain non-executable")
        } catch ModelContainerError.multimodalInputNotSupported {
            // Text-only bundles must still reject multimodal execution.
        }

    }
}
