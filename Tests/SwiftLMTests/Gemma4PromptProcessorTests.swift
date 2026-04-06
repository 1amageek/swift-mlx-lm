import Foundation
import Testing
@testable import SwiftLM

@Suite("Gemma4 Prompt Processor", .serialized)
struct Gemma4PromptProcessorTests {
    @Test("Gemma4 image prompt expands soft-token placeholders", .timeLimit(.minutes(2)))
    func expandsImagePrompt() async throws {
        guard let container = try await Gemma4TestSupport.syntheticGemma4Container() else {
            print("[Skip] No Metal device available for Gemma4 prompt processor tests")
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

        let multimodal = try #require(prepared.multimodalMetadata)
        #expect(multimodal.images.count == 1)
        #expect(prepared.renderedText.contains("<|image>"))
        #expect(prepared.renderedText.contains("<image|>"))
        #expect(multimodal.mmTokenTypeIDs.filter { $0 == 1 }.count == multimodal.images[0].placeholderTokenCount)
    }
}

