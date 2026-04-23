import Foundation
import Testing
@testable import MetalCompiler
@testable import SwiftLM

private enum RotorQuantRealBundleTestSupport {
    static let promptText = "What is the capital of Japan? Answer briefly."
    static let expectedAnswer = "Tokyo"
    static let answerWindowTokens = 32

    /// Assert that Gemma4 mentions "Tokyo" in its short factual answer.
    ///
    /// The earlier version of this helper required the model's very first
    /// token to equal "Tokyo", which is unrealistic for an instruct-tuned
    /// model whose natural response is "The capital of Japan is Tokyo." —
    /// the prefix is "The", not "Tokyo". Instead, generate a small window
    /// and assert the expected fact appears anywhere in the decoded answer.
    static func assertGemma4AnswersWithTokyo(
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
                maxTokens: answerWindowTokens,
                streamChunkTokenCount: 1,
                temperature: 0
            )
        )

        var fullAnswer = ""
        for await generation in stream {
            if let text = generation.text {
                fullAnswer += text
            }
        }

        let normalized = fullAnswer.trimmingCharacters(in: .whitespacesAndNewlines)
        print("[Gemma4 \(label) real answer] \(String(reflecting: normalized))")
        #expect(!normalized.isEmpty, "Model produced no output for \(label)")
        #expect(
            normalized.contains(expectedAnswer),
            "Expected \"\(expectedAnswer)\" in \(label) answer; got: \(String(reflecting: normalized))"
        )
    }
}

@Suite("RotorQuant Real Bundle Baseline", .serialized)
struct RotorQuantRealBundleBaselineTests {
    @Test("Gemma4 FP16 answers Tokyo to capital-of-Japan", .timeLimit(.minutes(10)))
    func gemma4FP16RealOutput() async throws {
        try await RotorQuantRealBundleTestSupport.assertGemma4AnswersWithTokyo(
            inferencePolicy: InferencePolicy(maximumSequenceLength: 256, kvCache: .automatic),
            label: "FP16"
        )
    }
}

@Suite("RotorQuant Real Bundle Key Path", .serialized)
struct RotorQuantRealBundleKeyPathTests {
    @Test("Gemma4 RotorQ8-K + FP16-V answers Tokyo to capital-of-Japan", .timeLimit(.minutes(10)))
    func gemma4RotorQ8KeyOnlyRealOutput() async throws {
        let policy = InferencePolicy(
            maximumSequenceLength: 256,
            kvCache: KVCachePolicy(
                keyScheme: .fixed(.rotorQ8Group32ScaleF16),
                valueScheme: .automatic
            )
        )
        try await RotorQuantRealBundleTestSupport.assertGemma4AnswersWithTokyo(
            inferencePolicy: policy,
            label: "RotorQ8-K/FP16-V"
        )
    }
}

@Suite("RotorQuant Real Bundle Value Path", .serialized)
struct RotorQuantRealBundleValuePathTests {
    @Test("Gemma4 FP16-K + RotorQ8-V answers Tokyo to capital-of-Japan", .timeLimit(.minutes(10)))
    func gemma4RotorQ8ValueOnlyRealOutput() async throws {
        let policy = InferencePolicy(
            maximumSequenceLength: 256,
            kvCache: KVCachePolicy(
                keyScheme: .automatic,
                valueScheme: .fixed(.rotorQ8Group32ScaleF16)
            )
        )
        try await RotorQuantRealBundleTestSupport.assertGemma4AnswersWithTokyo(
            inferencePolicy: policy,
            label: "FP16-K/RotorQ8-V"
        )
    }
}

@Suite("RotorQuant Real Bundle Full", .serialized)
struct RotorQuantRealBundleFullTests {
    @Test("Gemma4 RotorQ8 answers Tokyo to capital-of-Japan", .timeLimit(.minutes(10)))
    func gemma4RotorQ8RealOutput() async throws {
        let policy = InferencePolicy(
            maximumSequenceLength: 256,
            kvCache: KVCachePolicy(
                keyScheme: .fixed(.rotorQ8Group32ScaleF16),
                valueScheme: .fixed(.rotorQ8Group32ScaleF16)
            )
        )
        try await RotorQuantRealBundleTestSupport.assertGemma4AnswersWithTokyo(
            inferencePolicy: policy,
            label: "RotorQ8"
        )
    }

    @Test("Gemma4 RotorQ4 answers Tokyo to capital-of-Japan", .timeLimit(.minutes(10)))
    func gemma4RotorQ4RealOutput() async throws {
        let policy = InferencePolicy(
            maximumSequenceLength: 256,
            kvCache: KVCachePolicy(
                keyScheme: .fixed(.rotorQ4Group64ScaleF16),
                valueScheme: .fixed(.rotorQ4Group64ScaleF16)
            )
        )
        try await RotorQuantRealBundleTestSupport.assertGemma4AnswersWithTokyo(
            inferencePolicy: policy,
            label: "RotorQ4"
        )
    }
}
