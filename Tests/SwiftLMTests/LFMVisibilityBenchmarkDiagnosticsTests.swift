import Foundation
import Testing
@testable import SwiftLM

@Suite("LFM Visibility Benchmark Diagnostics", .serialized)
struct LFMVisibilityBenchmarkDiagnosticsTests {
    @Test("benchmark prompt reports raw-to-visible expansion", .timeLimit(.minutes(10)))
    func benchmarkPromptReportsRawToVisibleExpansion() async throws {
        let localModelDirectory = URL(
            fileURLWithPath: "/Users/1amageek/Desktop/swift-lm/TestData/LFM2.5-1.2B-Thinking"
        )
        let configURL = localModelDirectory.appendingPathComponent("config.json")
        guard FileManager.default.fileExists(atPath: configURL.path) else {
            print("[Skip] No local LFM bundle found at \(localModelDirectory.path)")
            return
        }

        let container = try await ModelBundleLoader().load(directory: localModelDirectory)
        let prompt = ExecutablePrompt(tokenIDs: [1, 1, 6, 6423, 708])
        let parameters = GenerateParameters(maxTokens: 50, streamChunkTokenCount: 8, temperature: 0)

        container.resetCaches()
        let visibleTokenIDs = try container.debugGeneratedTokenIDs(
            prompt: prompt,
            parameters: parameters
        )

        container.resetCaches()
        let rawTokenIDs = try container.debugRawGeneratedTokenIDs(
            prompt: prompt,
            parameters: parameters
        )

        let visibleText = container.decode(tokens: visibleTokenIDs)
        let rawText = container.tokenizer.decode(tokens: rawTokenIDs, skipSpecialTokens: false)
        let expansion = Double(rawTokenIDs.count) / Double(max(visibleTokenIDs.count, 1))

        print("[LFM benchmark visible token count] \(visibleTokenIDs.count)")
        print("[LFM benchmark raw token count] \(rawTokenIDs.count)")
        print(String(format: "[LFM benchmark raw/visible ratio] %.2f", expansion))
        print("[LFM benchmark visible text prefix]")
        print(String(visibleText.prefix(200)))
        print("[LFM benchmark raw text prefix]")
        print(String(rawText.prefix(400)))

        #expect(!visibleTokenIDs.isEmpty)
        #expect(!rawTokenIDs.isEmpty)
        #expect(rawTokenIDs.count >= visibleTokenIDs.count)
        #expect(!visibleText.contains("<think>"))
    }

    @Test("hello prompt reports raw-to-visible expansion", .timeLimit(.minutes(10)))
    func helloPromptReportsRawToVisibleExpansion() async throws {
        let localModelDirectory = URL(
            fileURLWithPath: "/Users/1amageek/Desktop/swift-lm/TestData/LFM2.5-1.2B-Thinking"
        )
        let configURL = localModelDirectory.appendingPathComponent("config.json")
        guard FileManager.default.fileExists(atPath: configURL.path) else {
            print("[Skip] No local LFM bundle found at \(localModelDirectory.path)")
            return
        }

        let container = try await ModelBundleLoader().load(directory: localModelDirectory)
        let prepared = try await container.prepare(input: ModelInput(prompt: "Hello"))
        let prompt = try container.makeExecutablePrompt(from: prepared)
        let parameters = GenerateParameters(maxTokens: 4, streamChunkTokenCount: 1, temperature: 0)

        container.resetCaches()
        let visibleTokenIDs = try container.debugGeneratedTokenIDs(
            prompt: prompt,
            parameters: parameters
        )

        container.resetCaches()
        let rawTokenIDs = try container.debugRawGeneratedTokenIDs(
            prompt: prompt,
            parameters: GenerateParameters(maxTokens: 1024, streamChunkTokenCount: 1, temperature: 0)
        )

        let visibleText = container.decode(tokens: visibleTokenIDs)
        let rawText = container.tokenizer.decode(tokens: rawTokenIDs, skipSpecialTokens: false)
        let expansion = Double(rawTokenIDs.count) / Double(max(visibleTokenIDs.count, 1))

        print("[LFM hello visible token count] \(visibleTokenIDs.count)")
        print("[LFM hello raw token count] \(rawTokenIDs.count)")
        print(String(format: "[LFM hello raw/visible ratio] %.2f", expansion))
        print("[LFM hello visible text]")
        print(visibleText)
        print("[LFM hello raw text prefix]")
        print(String(rawText.prefix(400)))

        #expect(!rawTokenIDs.isEmpty)
        #expect(rawTokenIDs.count >= visibleTokenIDs.count)
        #expect(!visibleText.contains("<think>"))
    }
}
