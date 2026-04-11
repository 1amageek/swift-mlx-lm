import Foundation
import Testing
@testable import SwiftLM

@Suite("LFM Output Diagnostics", .serialized)
struct LFMOutputDiagnosticsTests {
    @Test("Inspect LFM real output", .timeLimit(.minutes(10)))
    func inspectLFMRealOutput() async throws {
        let localModelDirectory = URL(
            fileURLWithPath: "/Users/1amageek/Desktop/swift-lm/TestData/LFM2.5-1.2B-Thinking"
        )
        let configURL = localModelDirectory.appendingPathComponent("config.json")
        guard FileManager.default.fileExists(atPath: configURL.path) else {
            print("[Skip] No local LFM bundle found at \(localModelDirectory.path)")
            return
        }

        let container = try await ModelBundleLoader().load(directory: localModelDirectory)
        container.resetState()
        let prepared = try await container.prepare( ModelInput(prompt: RealOutputAssertionSupport.strictCapitalPrompt)
        )
        print("[LFM prepared]")
        print(prepared.renderedText)
        print("[LFM prepared token count] \(prepared.tokenIDs.count)")

        let prompt = try container.makeExecutablePrompt(from: prepared)
        let topLogits = try container.debugPrefillTopLogits(prompt: prompt, topK: 20)
        print("[LFM prefill top logits]")
        for entry in topLogits {
            let formattedLogit = String(format: "%.4f", entry.logit)
            print("  id=\(entry.tokenID) logit=\(formattedLogit) token=\(String(reflecting: entry.decoded))")
        }

        let tokenIDs = try container.debugRawGeneratedTokenIDs(
            prompt: prompt,
            parameters: GenerationParameters(
                maxTokens: 64,
                streamChunkTokenCount: 8,
                temperature: 0
            )
        )
        let rawText = container.tokenizer.decode(tokens: tokenIDs, skipSpecialTokens: false)
        let visibleText = visibleTextDroppingLeadingThinkingBlock(from: rawText)

        print("[LFM generated token ids]")
        print(tokenIDs)
        print("[LFM generated raw text]")
        print(rawText)
        print("[LFM generated visible text]")
        print(visibleText)
    }

    private func visibleTextDroppingLeadingThinkingBlock(from text: String) -> String {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard trimmed.hasPrefix("<think>") else {
            return trimmed
        }
        guard let closingRange = trimmed.range(of: "</think>") else {
            return trimmed
        }
        return trimmed[closingRange.upperBound...]
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }
}
