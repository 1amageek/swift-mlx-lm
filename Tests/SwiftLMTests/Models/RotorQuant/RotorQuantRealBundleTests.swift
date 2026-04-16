import Foundation
import Testing
@testable import MetalCompiler
@testable import SwiftLM

private enum RotorQuantRealBundleTestSupport {
    static let promptText = "What is the capital of Japan? Answer briefly."

    static func assertGemma4StartsWithTokyo(
        inferencePolicy: InferencePolicy,
        label: String
    ) async throws {
        guard let directory = try Gemma4TestSupport.optionalRealGemma4Directory() else {
            print("[Skip] No local official Gemma4 E2B snapshot found")
            return
        }

        let loaded = try await ModelBundleLoader().load(
            directory: directory,
            inferencePolicy: inferencePolicy
        )
        let container = try autoreleasepool {
            try LanguageModelContext(loaded)
        }

        container.resetState()
        let prepared = try await container.prepare(ModelInput(prompt: promptText))
        let prompt = try autoreleasepool {
            try ExecutablePrompt(preparedPrompt: prepared, using: container)
        }
        let promptState = try autoreleasepool {
            try PromptSnapshot(from: prompt, using: container)
        }
        let firstToken = Int(promptState.metalState.firstToken)
        let rawDecoded = container.tokenizer.decode(tokens: [firstToken], skipSpecialTokens: false)
        print("[Gemma4 \(label) prompt-state first token] \(firstToken) -> \(String(reflecting: rawDecoded))")

        let stream = try container.generate(from: prompt,
            parameters: GenerationParameters(
                maxTokens: 1,
                streamChunkTokenCount: 1,
                temperature: 0
            )
        )

        var firstChunk = ""
        for await generation in stream {
            if let text = generation.text, !text.isEmpty {
                firstChunk = text
                break
            }
        }

        let normalized = firstChunk.trimmingCharacters(in: .whitespacesAndNewlines)
        print("[Gemma4 \(label) real first chunk] \(firstChunk)")
        #expect(!normalized.isEmpty)
        #expect(normalized.hasPrefix("Tokyo"))
    }
}

@Suite("RotorQuant Real Bundle Baseline", .serialized)
struct RotorQuantRealBundleBaselineTests {
    @Test("Gemma4 FP16 starts a factual answer with Tokyo", .timeLimit(.minutes(10)))
    func gemma4FP16RealOutput() async throws {
        try await RotorQuantRealBundleTestSupport.assertGemma4StartsWithTokyo(
            inferencePolicy: InferencePolicy(maximumSequenceLength: 256, kvCache: .automatic),
            label: "FP16"
        )
    }
}

@Suite("RotorQuant Real Bundle Key Path", .serialized)
struct RotorQuantRealBundleKeyPathTests {
    @Test("Gemma4 RotorQ8-K + FP16-V starts a factual answer with Tokyo", .timeLimit(.minutes(10)))
    func gemma4RotorQ8KeyOnlyRealOutput() async throws {
        let policy = InferencePolicy(
            maximumSequenceLength: 256,
            kvCache: KVCachePolicy(
                keyScheme: .fixed(.rotorQ8Group32ScaleF16),
                valueScheme: .automatic
            )
        )
        try await RotorQuantRealBundleTestSupport.assertGemma4StartsWithTokyo(
            inferencePolicy: policy,
            label: "RotorQ8-K/FP16-V"
        )
    }
}

@Suite("RotorQuant Real Bundle Value Path", .serialized)
struct RotorQuantRealBundleValuePathTests {
    @Test("Gemma4 FP16-K + RotorQ8-V starts a factual answer with Tokyo", .timeLimit(.minutes(10)))
    func gemma4RotorQ8ValueOnlyRealOutput() async throws {
        let policy = InferencePolicy(
            maximumSequenceLength: 256,
            kvCache: KVCachePolicy(
                keyScheme: .automatic,
                valueScheme: .fixed(.rotorQ8Group32ScaleF16)
            )
        )
        try await RotorQuantRealBundleTestSupport.assertGemma4StartsWithTokyo(
            inferencePolicy: policy,
            label: "FP16-K/RotorQ8-V"
        )
    }
}

@Suite("RotorQuant Real Bundle Full", .serialized)
struct RotorQuantRealBundleFullTests {
    @Test("Gemma4 RotorQ8 starts a factual answer with Tokyo", .timeLimit(.minutes(10)))
    func gemma4RotorQ8RealOutput() async throws {
        let policy = InferencePolicy(
            maximumSequenceLength: 256,
            kvCache: KVCachePolicy(
                keyScheme: .fixed(.rotorQ8Group32ScaleF16),
                valueScheme: .fixed(.rotorQ8Group32ScaleF16)
            )
        )
        try await RotorQuantRealBundleTestSupport.assertGemma4StartsWithTokyo(
            inferencePolicy: policy,
            label: "RotorQ8"
        )
    }

    @Test("Gemma4 RotorQ4 starts a factual answer with Tokyo", .timeLimit(.minutes(10)))
    func gemma4RotorQ4RealOutput() async throws {
        let policy = InferencePolicy(
            maximumSequenceLength: 256,
            kvCache: KVCachePolicy(
                keyScheme: .fixed(.rotorQ4Group64ScaleF16),
                valueScheme: .fixed(.rotorQ4Group64ScaleF16)
            )
        )
        try await RotorQuantRealBundleTestSupport.assertGemma4StartsWithTokyo(
            inferencePolicy: policy,
            label: "RotorQ4"
        )
    }
}
