import Foundation
import Testing
@testable import SwiftLM

@Suite("Qwen Vision Real Bundle Image", .serialized)
struct QwenVisionRealBundleImageTests {
    @Test("Real Qwen3.5/VL bundle can prepare and generate from an image prompt", .timeLimit(.minutes(10)))
    func realBundleImagePrompt() async throws {
        setenv("SWIFTLM_PROFILE_MULTIMODAL", "1", 1)
        guard let container = try await QwenVisionTestSupport.realQwen3VLContainer() else {
            print("[Skip] No local Qwen3.5/VL snapshot found")
            return
        }
        guard container.configuration.executionCapabilities.supportsImageExecution else {
            print("[Skip] Loaded local Qwen vision bundle does not execute image prompts")
            return
        }
        container.resetCaches()

        let imageData = try TestImageFixtures.makeOnePixelPNGData()
        let prepareStart = CFAbsoluteTimeGetCurrent()
        let prepared = try await container.prepare(
            input: ModelInput(chat: [
                .user([
                    .text("Describe"),
                    .image(InputImage(data: imageData, mimeType: "image/png")),
                ])
            ])
        )
        let prepareTime = CFAbsoluteTimeGetCurrent() - prepareStart
        let executableStart = CFAbsoluteTimeGetCurrent()
        let executable = try container.makeExecutablePrompt(from: prepared)
        let executableTime = CFAbsoluteTimeGetCurrent() - executableStart
        let imageTokenCount = executable.visualContext?.layout.mmTokenTypeIDs.filter { $0 == 1 }.count ?? 0
        print("[RealQwenVision] image prompt prepare=\(String(format: "%.3f", prepareTime))s executable=\(String(format: "%.3f", executableTime))s tokens=\(prepared.tokenIDs.count) imageTokens=\(imageTokenCount)")
        let streamStart = CFAbsoluteTimeGetCurrent()
        let stream = try container.generate(
            prompt: executable,
            parameters: GenerateParameters(maxTokens: 1, streamChunkTokenCount: 1)
        )
        let streamTime = CFAbsoluteTimeGetCurrent() - streamStart
        print("[RealQwenVision] image prompt stream-create=\(String(format: "%.3f", streamTime))s")
        let collectionStart = CFAbsoluteTimeGetCurrent()
        let result = await QwenVisionTestSupport.collectGeneration(from: stream)
        let collectionTime = CFAbsoluteTimeGetCurrent() - collectionStart
        print("[RealQwenVision] image prompt stream-collect=\(String(format: "%.3f", collectionTime))s")

        #expect(executable.visualContext != nil)
        #expect(!result.chunks.isEmpty || result.completion != nil)
    }
}
