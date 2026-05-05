import Foundation
import Testing
@testable import MetalCompiler
@testable import SwiftLM

@Suite("Real Output Evidence", .serialized)
struct RealOutputEvidenceTests {
    @Test("Qwen3.5 answers the capital prompt with Tokyo", .timeLimit(.minutes(2)))
    func qwen35StrictCapitalPromptOutput() async throws {
        guard let directory = try QwenVisionTestSupport.optionalRealQwen3VLDirectory() else {
            print("[Skip] No local Qwen3.5 snapshot found")
            return
        }

        print("[Qwen3.5 evidence model directory] \(directory.path)")
        let loaded = try await ModelBundleLoader().load(directory: directory)
        let context = try LanguageModelContext(loaded)
        let output = try await Self.generateGreedyEvidenceOutput(
            context: context,
            prepared: try await context.prepare(ModelInput(
                chat: [.user([.text(RealOutputAssertionSupport.strictCapitalPrompt)])],
                promptOptions: PromptPreparationOptions(isThinkingEnabled: false)
            )),
            label: "Qwen3.5 evidence"
        )
        let text = RealOutputAssertionSupport.normalized(context.decode(output.visibleTokenIDs))

        Self.printOutputHeadDiagnostics(output.outputHeadDiagnostics, label: "Qwen3.5 evidence")
        print("[Qwen3.5 evidence top logits]")
        Self.printTopLogits(output.topLogits)
        print("[Qwen3.5 evidence raw token ids] \(output.rawTokenIDs)")
        print("[Qwen3.5 evidence visible token ids] \(output.visibleTokenIDs)")
        print("[Qwen3.5 evidence greedy text] \(text)")
        RealOutputAssertionSupport.assertStartsWithTokyo(
            text,
            label: "Qwen3.5 evidence greedy"
        )
    }

    @Test("LFM emits Tokyo for the capital completion prompt", .timeLimit(.minutes(2)))
    func lfmStrictCapitalPromptOutput() async throws {
        guard let directory = ReleaseSmokeTestSupport.readableLocalModelDirectoryOrSkip() else {
            return
        }

        print("[LFM evidence model directory] \(directory.path)")
        let loaded = try await ModelBundleLoader().load(directory: directory)
        let context = try LanguageModelContext(loaded)
        let output = try await Self.generateGreedyEvidenceOutput(
            context: context,
            prepared: RealOutputAssertionSupport.directTextPrompt(
                RealOutputAssertionSupport.capitalCompletionPrompt,
                using: context
            ),
            label: "LFM evidence"
        )
        let text = RealOutputAssertionSupport.normalized(context.decode(output.visibleTokenIDs))

        Self.printOutputHeadDiagnostics(output.outputHeadDiagnostics, label: "LFM evidence")
        print("[LFM evidence top logits]")
        Self.printTopLogits(output.topLogits)
        print("[LFM evidence raw token ids] \(output.rawTokenIDs)")
        print("[LFM evidence visible token ids] \(output.visibleTokenIDs)")
        print("[LFM evidence greedy text] \(text)")
        RealOutputAssertionSupport.assertStartsWithTokyo(
            text,
            label: "LFM evidence greedy"
        )
    }

    @Test("Gemma4 answers the strict capital prompt with Tokyo", .timeLimit(.minutes(2)))
    func gemma4StrictCapitalPromptOutput() async throws {
        guard let directory = try Gemma4TestSupport.optionalRealGemma4Directory() else {
            print("[Skip] No local official Gemma4 E2B snapshot found")
            return
        }

        print("[Gemma4 evidence model directory] \(directory.path)")
        let loaded = try await ModelBundleLoader().load(
            directory: directory,
            inferencePolicy: InferencePolicy(maximumSequenceLength: 256, kvCache: .automatic)
        )
        let context = try LanguageModelContext(loaded)
        let promptTokenIDs = [2] + context.encode(
            RealOutputAssertionSupport.capitalCompletionPrompt,
            addSpecialTokens: false
        )
        let prepared = PreparedPrompt(
            renderedText: "<bos>" + RealOutputAssertionSupport.capitalCompletionPrompt,
            tokenIDs: promptTokenIDs
        )
        print("[Gemma4 evidence rendered prompt suffix] \(String(prepared.renderedText.suffix(400)))")
        print("[Gemma4 evidence prepared token count] \(prepared.tokenIDs.count)")
        print("[Gemma4 evidence prepared token ids] \(prepared.tokenIDs)")
        let executable = try ExecutablePrompt(preparedPrompt: prepared, using: context)
        print("[Gemma4 evidence executable token count] \(executable.tokenIDs.count)")
        let tokenIDs = try context.debugRawGeneratedTokenIDs(
            prompt: executable,
            parameters: RealOutputAssertionSupport.greedyParameters(maxTokens: 1)
        )
        let text = RealOutputAssertionSupport.normalized(context.decode(tokenIDs))

        print("[Gemma4 evidence current top logits]")
        Self.printTopLogits(context.debugCurrentDecodeTopLogits(topK: 10))
        print("[Gemma4 evidence visible token ids] \(tokenIDs)")
        print("[Gemma4 evidence greedy text] \(text)")
        RealOutputAssertionSupport.assertStartsWithTokyo(
            text,
            label: "Gemma4 evidence greedy"
        )
    }

    private static func generateGreedyEvidenceOutput(
        context: LanguageModelContext,
        prepared: PreparedPrompt,
        label: String
    ) async throws -> (
        outputHeadDiagnostics: (
            topLogits: [(tokenID: Int, logit: Float, decoded: String)],
            inputLayout: (offset: Int, bufferLength: Int, hiddenCount: Int, readableCount: Int, bufferKind: String),
            transferLayout: (
                sourceOffset: Int,
                sourceBufferLength: Int,
                sourceReadableCount: Int,
                sourceBufferKind: String,
                destinationOffset: Int,
                destinationBufferLength: Int,
                destinationReadableCount: Int,
                destinationBufferKind: String,
                transferCount: Int
            ),
            transferSource: [Float],
            transferDestination: [Float]
        ),
        finalHidden: [Float],
        topLogits: [(tokenID: Int, logit: Float, decoded: String)],
        rawTokenIDs: [Int],
        visibleTokenIDs: [Int]
    ) {
        print("[\(label) rendered prompt suffix] \(String(prepared.renderedText.suffix(400)))")
        let executable = try ExecutablePrompt(preparedPrompt: prepared, using: context)
        let parameters = RealOutputAssertionSupport.greedyParameters(maxTokens: 6)

        let outputHeadDiagnostics = try context.debugPrefillOutputHeadDiagnostics(
            prompt: executable,
            topK: 10
        )
        let finalHidden = try context.debugPrefillFinalHidden(prompt: executable)
        let topLogits = try context.debugPrefillTopLogits(prompt: executable, topK: 10)

        context.resetState()
        let rawTokenIDs = try context.debugRawGeneratedTokenIDs(
            prompt: executable,
            parameters: parameters
        )

        context.resetState()
        let visibleTokenIDs = try context.debugGeneratedTokenIDs(
            prompt: executable,
            parameters: parameters
        )
        #expect(!rawTokenIDs.isEmpty, "Greedy raw output should emit at least one token")
        #expect(!visibleTokenIDs.isEmpty, "Greedy visible output should emit at least one token")
        print("[evidence decode-plan final hidden summary] \(Self.summary(finalHidden))")
        return (outputHeadDiagnostics, finalHidden, topLogits, rawTokenIDs, visibleTokenIDs)
    }

    private static func printOutputHeadDiagnostics(
        _ diagnostics: (
            topLogits: [(tokenID: Int, logit: Float, decoded: String)],
            inputLayout: (offset: Int, bufferLength: Int, hiddenCount: Int, readableCount: Int, bufferKind: String),
            transferLayout: (
                sourceOffset: Int,
                sourceBufferLength: Int,
                sourceReadableCount: Int,
                sourceBufferKind: String,
                destinationOffset: Int,
                destinationBufferLength: Int,
                destinationReadableCount: Int,
                destinationBufferKind: String,
                transferCount: Int
            ),
            transferSource: [Float],
            transferDestination: [Float]
        ),
        label: String
    ) {
        print("[\(label) output head input layout] \(diagnostics.inputLayout)")
        print("[\(label) output head transfer layout] \(diagnostics.transferLayout)")
        print("[\(label) transfer source summary] \(Self.summary(diagnostics.transferSource))")
        print("[\(label) transfer destination summary] \(Self.summary(diagnostics.transferDestination))")
    }

    private static func summary(_ values: [Float]) -> String {
        let finite = values.filter(\.isFinite)
        let nonZero = finite.filter { $0 != 0 }.count
        let maxAbs = finite.map { abs($0) }.max() ?? 0
        return "count=\(values.count) finite=\(finite.count) nonZero=\(nonZero) maxAbs=\(maxAbs)"
    }

    private static func printTopLogits(_ topLogits: [(tokenID: Int, logit: Float, decoded: String)]) {
        for entry in topLogits {
            let formattedLogit = String(format: "%.4f", entry.logit)
            print("  id=\(entry.tokenID) logit=\(formattedLogit) token=\(String(reflecting: entry.decoded))")
        }
    }
}
