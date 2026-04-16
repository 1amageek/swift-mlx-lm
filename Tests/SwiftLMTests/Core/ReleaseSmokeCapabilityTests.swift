import Testing
@testable import SwiftLM

@Suite("Release Smoke Capability", .serialized)
struct ReleaseSmokeCapabilityTests {
    @Test("Text-only bundle rejects multimodal input", .timeLimit(.minutes(2)))
    func textOnlyBundleRejectsMultimodalInput() async throws {
        guard let localModelDirectory = ReleaseSmokeTestSupport.readableLocalModelDirectoryOrSkip() else { return }

        let loader = ModelBundleLoader()
        let container = try await loader.load(directory: localModelDirectory)
        let supportsImages = container.configuration.inputCapabilities.supportsImages

        if !supportsImages {
            do {
                _ = try await container.prepare( ModelInput(chat: [
                    .user([
                        .text("Describe this image."),
                        .image(InputImage(data: try TestImageFixtures.makeOnePixelPNGData(), mimeType: "image/png")),
                    ])
                ]))
                Issue.record("Expected text-only bundle to reject image-bearing input")
            } catch LanguageModelContextError.unsupportedInputForModel {
                return
            }
            return
        }

        let prepared = try await container.prepare( ModelInput(chat: [
            .user([
                .text("Describe this image."),
                .image(InputImage(data: try TestImageFixtures.makeOnePixelPNGData(), mimeType: "image/png")),
            ])
        ]))
        #expect(prepared.multimodalMetadata != nil)
        #expect(container.configuration.executionCapabilities.supportsImagePromptPreparation)
        #expect(!container.configuration.executionCapabilities.supportsImageExecution)

        do {
            _ = try await container.generate( ModelInput(chat: [
                .user([
                    .text("Describe this image."),
                    .image(InputImage(data: try TestImageFixtures.makeOnePixelPNGData(), mimeType: "image/png")),
                ])
            ]))
            Issue.record("Expected multimodal generate( ModelInput) to throw")
        } catch LanguageModelContextError.multimodalInputNotSupported {
        }

        do {
            _ = try ExecutablePrompt(preparedPrompt: prepared, using: container)
            Issue.record("Expected multimodal prepared input to remain non-executable")
        } catch LanguageModelContextError.multimodalInputNotSupported {
        }
    }
}
