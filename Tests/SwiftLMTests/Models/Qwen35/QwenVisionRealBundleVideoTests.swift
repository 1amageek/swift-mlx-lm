import Foundation
import Testing
@testable import SwiftLM

@Suite("Qwen Vision Real Bundle Video", .serialized)
struct QwenVisionRealBundleVideoTests {
    @Test("Real Qwen3.5/VL bundle can prepare and generate from a video prompt", .timeLimit(.minutes(10)))
    func realBundleVideoPrompt() async throws {
        guard let container = try await QwenVisionTestSupport.realQwen3VLContainer() else {
            print("[Skip] No local Qwen3.5/VL snapshot found")
            return
        }
        guard container.configuration.executionCapabilities.supportsVideoExecution else {
            print("[Skip] Loaded local Qwen vision bundle does not execute video prompts")
            return
        }
        container.resetState()

        let videoData = try await TestVideoFixtures.makeMP4Data()
        let prepareStart = CFAbsoluteTimeGetCurrent()
        let prepared = try await container.prepare( ModelInput(chat: [
                .user([
                    .text("Describe"),
                    .video(InputVideo(data: videoData, mimeType: "video/mp4")),
                ])
            ])
        )
        let prepareTime = CFAbsoluteTimeGetCurrent() - prepareStart
        let executableStart = CFAbsoluteTimeGetCurrent()
        let executable = try ExecutablePrompt(preparedPrompt: prepared, using: container)
        let executableTime = CFAbsoluteTimeGetCurrent() - executableStart
        let videoTokenCount = executable.visualContext?.layout.mmTokenTypeIDs.filter { $0 == 2 }.count ?? 0
        print("[RealQwenVision] video prompt prepare=\(String(format: "%.3f", prepareTime))s executable=\(String(format: "%.3f", executableTime))s tokens=\(prepared.tokenIDs.count) videoTokens=\(videoTokenCount)")
        let generationStart = CFAbsoluteTimeGetCurrent()
        let stream = try container.generate(from: executable,
            parameters: GenerationParameters(maxTokens: 1, streamChunkTokenCount: 1)
        )
        let result = await QwenVisionTestSupport.collectGeneration(from: stream)
        let generationTime = CFAbsoluteTimeGetCurrent() - generationStart
        print("[RealQwenVision] video prompt generation=\(String(format: "%.3f", generationTime))s")

        #expect(executable.visualContext != nil)
        #expect(!result.chunks.isEmpty || result.completion != nil)
    }
}
