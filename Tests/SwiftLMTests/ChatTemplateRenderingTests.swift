import Foundation
import Testing
@testable import SwiftLM

@Suite("Chat Template Rendering", .serialized)
struct ChatTemplateRenderingTests {
    private static let lfmDirectory = URL(
        fileURLWithPath: "/Users/1amageek/Desktop/swift-lm/TestData/LFM2.5-1.2B-Thinking"
    )

    @Test("LFM Jinja chat template renders plain text content, not JSON payloads", .timeLimit(.minutes(2)))
    func lfmJinjaTemplateRendersPlainTextContent() async throws {
        let configURL = Self.lfmDirectory.appendingPathComponent("config.json")
        guard FileManager.default.fileExists(atPath: configURL.path) else {
            print("[Skip] No local LFM bundle found at \(Self.lfmDirectory.path)")
            return
        }

        let loader = ModelBundleLoader()
        let container = try await loader.load(directory: Self.lfmDirectory)
        let prepared = try await container.prepare(
            input: ModelInput(chat: [
                .user([.text("What is the capital of Japan? Answer with exactly one word.")])
            ])
        )

        print("[LFM rendered chat prompt]")
        print(prepared.renderedText)

        #expect(prepared.renderedText.contains("What is the capital of Japan?"))
        #expect(!prepared.renderedText.contains("\"type\":\"text\""))
        #expect(!prepared.renderedText.contains(".\"}]"))
    }
}
