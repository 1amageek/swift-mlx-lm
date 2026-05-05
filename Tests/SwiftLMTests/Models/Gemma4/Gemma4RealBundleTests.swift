import Testing
@testable import MetalCompiler
@testable import SwiftLM

@Suite("Gemma4 Real Bundle", .serialized)
struct Gemma4RealBundleTests {
    @Test("Gemma4 answers the strict capital prompt with Tokyo", .timeLimit(.minutes(2)))
    func strictCapitalPromptOutput() async throws {
        guard let directory = try Gemma4TestSupport.optionalRealGemma4Directory() else {
            print("[Skip] No local official Gemma4 E2B snapshot found")
            return
        }

        print("[Gemma4 real-bundle model directory] \(directory.path)")
        let loaded = try await ModelBundleLoader().load(
            directory: directory,
            inferencePolicy: InferencePolicy(maximumSequenceLength: 256, kvCache: .automatic)
        )
        let context = try LanguageModelContext(loaded)
        let tokenIDs = [2] + context.encode(
            RealOutputAssertionSupport.capitalCompletionPrompt,
            addSpecialTokens: false
        )
        let prepared = PreparedPrompt(
            renderedText: "<bos>" + RealOutputAssertionSupport.capitalCompletionPrompt,
            tokenIDs: tokenIDs
        )
        let executable = try ExecutablePrompt(preparedPrompt: prepared, using: context)
        let generatedTokenIDs = try context.debugRawGeneratedTokenIDs(
            prompt: executable,
            parameters: RealOutputAssertionSupport.greedyParameters(maxTokens: 1)
        )
        let text = RealOutputAssertionSupport.normalized(context.decode(generatedTokenIDs))

        print("[Gemma4 real-bundle visible token ids] \(generatedTokenIDs)")
        print("[Gemma4 real-bundle greedy text] \(text)")
        RealOutputAssertionSupport.assertStartsWithTokyo(
            text,
            label: "Gemma4 real-bundle greedy"
        )
    }
}
