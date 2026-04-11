import Foundation
import Testing
@testable import SwiftLM

enum RealOutputAssertionSupport {
    static let strictCapitalPrompt = "What is the capital of Japan? Answer with exactly one word."

    static func greedyParameters(maxTokens: Int = 8) -> GenerationParameters {
        GenerationParameters(
            maxTokens: maxTokens,
            streamChunkTokenCount: 1,
            temperature: 0
        )
    }

    static func samplingParameters(maxTokens: Int = 1) -> GenerationParameters {
        GenerationParameters(
            maxTokens: maxTokens,
            streamChunkTokenCount: 1,
            temperature: 0.6
        )
    }

    static func normalized(_ text: String) -> String {
        text.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    @discardableResult
    static func assertGreedyDirectMatchesPromptState(
        container: LanguageModelContext,
        prompt: ExecutablePrompt,
        label: String,
        parameters: GenerationParameters = greedyParameters()
    ) throws -> (directTokenIDs: [Int], restoredTokenIDs: [Int], directText: String, restoredText: String) {
        container.resetState()
        let directTokenIDs = try container.debugGeneratedTokenIDs(
            prompt: prompt,
            parameters: parameters
        )

        container.resetState()
        let promptState = try container.makePromptSnapshot(from: prompt)
        let restoredTokenIDs = try container.debugPromptStateGeneratedTokenIDs(
            promptState: promptState,
            parameters: parameters
        )

        let directText = normalized(container.decode( directTokenIDs))
        let restoredText = normalized(container.decode( restoredTokenIDs))

        print("[\(label) direct token ids] \(directTokenIDs)")
        print("[\(label) restored token ids] \(restoredTokenIDs)")
        print("[\(label) direct text] \(directText)")
        print("[\(label) restored text] \(restoredText)")

        #expect(!directTokenIDs.isEmpty, "\(label) direct generation should emit at least one token")
        #expect(
            directTokenIDs == restoredTokenIDs,
            "\(label) prompt-state generation must match direct greedy token IDs"
        )
        let promptStateFirstToken = Int(promptState.metalState.firstToken)
        let promptStateFirstTokenText = container.tokenizer.decode(
            tokens: [promptStateFirstToken],
            skipSpecialTokens: false
        )
        if promptStateFirstTokenText != "<think>" {
            #expect(
                promptStateFirstToken == directTokenIDs.first,
                "\(label) prompt-state first token should match direct greedy first token"
            )
        }
        #expect(
            directText == restoredText,
            "\(label) prompt-state decoded text must match direct greedy decoded text"
        )

        return (directTokenIDs, restoredTokenIDs, directText, restoredText)
    }

    static func assertPromptStateSamplingMatchesDirect(
        container: LanguageModelContext,
        prompt: ExecutablePrompt,
        label: String,
        parameters: GenerationParameters = samplingParameters()
    ) throws {
        container.resetState()
        let sampled = try container.debugPromptStateSampledFirstTokens(
            prompt: prompt,
            parameters: parameters
        )

        print("[\(label) sampled direct token] \(sampled.direct)")
        print("[\(label) sampled restored token] \(sampled.restored)")
        print("[\(label) sampled direct recent ids] \(sampled.directRecentTokenIDs)")
        print("[\(label) sampled restored recent ids] \(sampled.restoredRecentTokenIDs)")
        print("[\(label) sampled direct top logits] \(sampled.directTopLogits.map(\.tokenID))")
        print("[\(label) sampled restored top logits] \(sampled.restoredTopLogits.map(\.tokenID))")

        #expect(
            sampled.directRecentTokenIDs == sampled.restoredRecentTokenIDs,
            "\(label) prompt-state recent-token context must match direct sampling context"
        )
        #expect(
            sampled.directTopLogits.map(\.tokenID) == sampled.restoredTopLogits.map(\.tokenID),
            "\(label) prompt-state top-logit token ordering must match direct sampling"
        )
        #expect(
            sampled.directTopLogits.map(\.logit) == sampled.restoredTopLogits.map(\.logit),
            "\(label) prompt-state top-logit values must match direct sampling"
        )
        #expect(
            sampled.direct == sampled.restored,
            "\(label) prompt-state sampled first token must match direct sampling"
        )
    }

    static func assertStartsWithTokyo(_ text: String, label: String) {
        let normalizedText = normalized(text)
        print("[\(label) normalized output] \(normalizedText)")
        #expect(!normalizedText.isEmpty, "\(label) output must not be empty")
        #expect(
            normalizedText.hasPrefix("Tokyo"),
            "\(label) should start with Tokyo for the strict capital prompt"
        )
    }
}
