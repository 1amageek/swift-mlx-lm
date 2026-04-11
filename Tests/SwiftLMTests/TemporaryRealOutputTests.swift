import Foundation
import Metal
import Testing
@testable import MetalCompiler
@testable import SwiftLM

private struct DiagnosticError: Error, CustomStringConvertible {
    let message: String

    var description: String { message }
}

@Suite("Temporary Real Output", .serialized)
struct TemporaryRealOutputTests {
    private let promptText = "What is the capital of Japan? Answer briefly."

    @Test("Inspect Gemma4 real output", .timeLimit(.minutes(10)))
    func inspectGemma4RealOutput() async throws {
        guard let container = try await Gemma4TestSupport.realGemma4Container() else {
            print("[Skip] No local Gemma4 snapshot")
            return
        }
        container.resetState()
        let prepared = try await container.prepare( ModelInput(prompt: promptText)
        )
        print("[Gemma4 prepared]")
        print(prepared.renderedText)
        let prompt = try container.makeExecutablePrompt(from: prepared)
        print("[Gemma prefill steps]")
        for summary in container.debugPrefillStepSummaries().prefix(32) {
            let layerLabel = summary.layerIndex.map(String.init) ?? "-"
            print("  step=\(summary.index) layer=\(layerLabel) kernel=\(summary.kernelName)")
        }
        container.resetState()
        let promptState = try container.makePromptSnapshot(from: prompt)
        let firstToken = Int(promptState.metalState.firstToken)
        let decoded = container.tokenizer.decode(tokens: [firstToken], skipSpecialTokens: false)
        print("[Gemma4 first token] \(firstToken) -> \(String(reflecting: decoded))")
        let restoreDiagnostics = try container.debugPromptStateRestoreDiagnostics(
            prompt: prompt,
            topK: 10
        )
        print("[Gemma4 direct tokenOut] \(restoreDiagnostics.directTokenOut)")
        print("[Gemma4 promptState first token] \(restoreDiagnostics.promptStateFirstToken)")
        print("[Gemma4 restored tokenOut] \(restoreDiagnostics.restoredTokenOut)")
        print("[Gemma4 direct top logits]")
        for entry in restoreDiagnostics.directTopLogits {
            print("  id=\(entry.tokenID) logit=\(String(format: "%.4f", entry.logit)) token=\(String(reflecting: entry.decoded))")
        }
        print("[Gemma4 restored top logits]")
        for entry in restoreDiagnostics.restoredTopLogits {
            print("  id=\(entry.tokenID) logit=\(String(format: "%.4f", entry.logit)) token=\(String(reflecting: entry.decoded))")
        }
        let repeatedPrefill = try container.debugRepeatedPrefillSampledFirstTokens(
            prompt: prompt,
            parameters: GenerationParameters(
                maxTokens: 1,
                streamChunkTokenCount: 1,
                temperature: 0
            )
        )
        print("[Gemma4 repeated prefill]")
        print(
            "  first=\(repeatedPrefill.first) second=\(repeatedPrefill.second) maxAbsDiff=\(String(format: "%.6f", repeatedPrefill.maxAbsDiff)) differingCount=\(repeatedPrefill.differingCount)"
        )
        print(
            "  fingerprints=(\(repeatedPrefill.firstLogitFingerprint), \(repeatedPrefill.secondLogitFingerprint)) nanCounts=(\(repeatedPrefill.firstNaNCount), \(repeatedPrefill.secondNaNCount))"
        )
        print("  firstTopLogits=")
        for entry in repeatedPrefill.firstTopLogits {
            print("    id=\(entry.tokenID) logit=\(String(format: "%.4f", entry.logit)) token=\(String(reflecting: entry.decoded))")
        }
        print("  secondTopLogits=")
        for entry in repeatedPrefill.secondTopLogits {
            print("    id=\(entry.tokenID) logit=\(String(format: "%.4f", entry.logit)) token=\(String(reflecting: entry.decoded))")
        }
        print("  firstDifferingEntries=")
        for entry in repeatedPrefill.firstDifferingEntries.prefix(10) {
            print(
                "    id=\(entry.tokenID) first=\(String(format: "%.4f", entry.first)) second=\(String(format: "%.4f", entry.second))"
            )
        }
        try printGemmaEmbeddingDiagnostics(
            container: container,
            prompt: prompt
        )
        try printGemmaResidualDiagnostics(
            container: container,
            prompt: prompt
        )
        try printGemmaRepeatedResidualDiagnostics(
            container: container,
            prompt: prompt
        )
#if ENABLE_METAL_PROBES
        try printGemmaAttentionProbeDiagnostics(
            container: container,
            prompt: prompt
        )
#endif
    }

    @Test("Inspect Qwen3.5 real output", .timeLimit(.minutes(10)))
    func inspectQwenRealOutput() async throws {
        guard let container = try await QwenVisionTestSupport.realQwen3VLContainer() else {
            print("[Skip] No local Qwen3.5 snapshot")
            return
        }
        container.resetState()
        let textPrepared = try await container.prepare( ModelInput(prompt: promptText)
        )
        print("[Qwen text prepared]")
        print(textPrepared.renderedText)
        print("[Qwen text prepared token count] \(textPrepared.tokenIDs.count)")
        try printPromptStateDiagnostics(
            label: "Qwen text",
            container: container,
            prompt: try container.makeExecutablePrompt(from: textPrepared)
        )
        let textStream = try container.generate(from: try container.makeExecutablePrompt(from: textPrepared),
            parameters: GenerationParameters(
                maxTokens: 12,
                streamChunkTokenCount: 8,
                temperature: 0.7,
                topP: 0.8,
                topK: 20,
                presencePenalty: 1.5
            )
        )
        let textResult = await QwenVisionTestSupport.collectGeneration(from: textStream)
        print("[Qwen text chunks]")
        for chunk in textResult.chunks { print(chunk) }
        print("[Qwen text joined]")
        print(textResult.chunks.joined())
        let textTokenIDs = try container.debugGeneratedTokenIDs(
            prompt: try container.makeExecutablePrompt(from: textPrepared),
            parameters: GenerationParameters(
                maxTokens: 12,
                streamChunkTokenCount: 8,
                temperature: 0.7,
                topP: 0.8,
                topK: 20,
                presencePenalty: 1.5
            )
        )
        print("[Qwen text token ids]")
        print(textTokenIDs)
        try printQwenEmbeddingDiagnostics(
            container: container,
            prompt: try container.makeExecutablePrompt(from: textPrepared)
        )

        container.resetState()
        let chatPrepared = try await container.prepare( ModelInput(chat: [
                .user([.text(promptText)])
            ])
        )
        print("[Qwen chat prepared]")
        print(chatPrepared.renderedText)
        print("[Qwen chat prepared token count] \(chatPrepared.tokenIDs.count)")
        try printPromptStateDiagnostics(
            label: "Qwen chat",
            container: container,
            prompt: try container.makeExecutablePrompt(from: chatPrepared)
        )
        let chatStream = try container.generate(from: try container.makeExecutablePrompt(from: chatPrepared),
            parameters: GenerationParameters(
                maxTokens: 12,
                streamChunkTokenCount: 8,
                temperature: 0.7,
                topP: 0.8,
                topK: 20,
                presencePenalty: 1.5
            )
        )
        let chatResult = await QwenVisionTestSupport.collectGeneration(from: chatStream)
        print("[Qwen chat chunks]")
        for chunk in chatResult.chunks { print(chunk) }
        print("[Qwen chat joined]")
        print(chatResult.chunks.joined())
        let chatTokenIDs = try container.debugGeneratedTokenIDs(
            prompt: try container.makeExecutablePrompt(from: chatPrepared),
            parameters: GenerationParameters(
                maxTokens: 12,
                streamChunkTokenCount: 8,
                temperature: 0.7,
                topP: 0.8,
                topK: 20,
                presencePenalty: 1.5
            )
        )
        print("[Qwen chat token ids]")
        print(chatTokenIDs)

        container.resetState()
        let rawPrompt = ExecutablePrompt(tokenIDs: container.encode(promptText))
        print("[Qwen raw prompt tokens] \(rawPrompt.tokenIDs.prefix(24))")
        try printPromptStateDiagnostics(
            label: "Qwen raw",
            container: container,
            prompt: rawPrompt
        )
        let rawStream = try container.generate(from: rawPrompt,
            parameters: GenerationParameters(
                maxTokens: 12,
                streamChunkTokenCount: 8,
                temperature: 0
            )
        )
        let rawResult = await QwenVisionTestSupport.collectGeneration(from: rawStream)
        print("[Qwen raw chunks]")
        for chunk in rawResult.chunks { print(chunk) }
        print("[Qwen raw joined]")
        print(rawResult.chunks.joined())
        let rawTokenIDs = try container.debugGeneratedTokenIDs(
            prompt: rawPrompt,
            parameters: GenerationParameters(
                maxTokens: 12,
                streamChunkTokenCount: 8,
                temperature: 0
            )
        )
        print("[Qwen raw token ids]")
        print(rawTokenIDs)
    }

    private func printPromptStateDiagnostics(
        label: String,
        container: LanguageModelContext,
        prompt: ExecutablePrompt
    ) throws {
        container.resetState()
        let topLogits = try container.debugPrefillTopLogits(prompt: prompt, topK: 10)
        print("[\(label) top logits]")
        for entry in topLogits {
            let formattedLogit = String(format: "%.4f", entry.logit)
            print("  id=\(entry.tokenID) logit=\(formattedLogit) token=\(String(reflecting: entry.decoded))")
        }

        let summaries = container.debugPrefillStepSummaries()
        let earlySteps = summaries.prefix(40)
        print("[\(label) prefill steps]")
        for summary in earlySteps {
            let layerLabel = summary.layerIndex.map(String.init) ?? "-"
            print("  step=\(summary.index) layer=\(layerLabel) kernel=\(summary.kernelName)")
        }
        let attentionSteps = summaries.filter { summary in
            summary.kernelName.contains("qk_rms_norm")
                || summary.kernelName.contains("packed_query_extract")
                || summary.kernelName.contains("packed_sigmoid_gate")
                || summary.kernelName.contains("flash")
                || summary.kernelName.contains("rope")
        }
        print("[\(label) attention steps]")
        for summary in attentionSteps {
            let layerLabel = summary.layerIndex.map(String.init) ?? "-"
            print("  step=\(summary.index) layer=\(layerLabel) kernel=\(summary.kernelName)")
        }

#if ENABLE_METAL_PROBES
        let decodeSummaries = container.debugDecodeStepSummaries()
        print("[\(label) decode steps]")
        for summary in decodeSummaries.prefix(80) {
            let layerLabel = summary.layerIndex.map(String.init) ?? "-"
            print("  step=\(summary.index) layer=\(layerLabel) kernel=\(summary.kernelName)")
        }
#endif

        if label == "Qwen text" {
            let hiddenSnapshots = try container.debugPrefillLastTokenHiddenSnapshots(
                prompt: prompt,
                stepIndices: Set([48, 50])
            )
            print("[\(label) pre-attention hidden]")
            for step in [48, 50] {
                let values = hiddenSnapshots[step, default: []].prefix(8).map { String(format: "%.4f", $0) }
                print("  step=\(step) hidden=\(values)")
            }
            let firstAttentionSteps = Set([52, 55, 57, 59, 60])
            let scratch0 = try container.debugPrefillLastTokenScratchSnapshots(
                prompt: prompt,
                stepIndices: firstAttentionSteps,
                slotIndex: 0,
                rowStride: 2048,
                count: 8
            )
            let scratch1 = try container.debugPrefillLastTokenScratchSnapshots(
                prompt: prompt,
                stepIndices: firstAttentionSteps,
                slotIndex: 1,
                rowStride: 4096,
                count: 8
            )
            let scratch4 = try container.debugPrefillLastTokenScratchSnapshots(
                prompt: prompt,
                stepIndices: firstAttentionSteps,
                slotIndex: 4,
                rowStride: 2048,
                count: 8
            )
            let scratch2 = try container.debugPrefillLastTokenScratchSnapshots(
                prompt: prompt,
                stepIndices: firstAttentionSteps,
                slotIndex: 2,
                rowStride: 512,
                count: 8
            )
            let scratch3 = try container.debugPrefillLastTokenScratchSnapshots(
                prompt: prompt,
                stepIndices: firstAttentionSteps,
                slotIndex: 3,
                rowStride: 512,
                count: 8
            )
            print("[\(label) first attention scratch]")
            for step in firstAttentionSteps.sorted() {
                let out0 = scratch0[step, default: []].map { String(format: "%.4f", $0) }
                let packed = scratch1[step, default: []].map { String(format: "%.4f", $0) }
                let query = scratch4[step, default: []].map { String(format: "%.4f", $0) }
                let key = scratch2[step, default: []].map { String(format: "%.4f", $0) }
                let value = scratch3[step, default: []].map { String(format: "%.4f", $0) }
                print("  step=\(step) out0=\(out0) packed1=\(packed) key2=\(key) value3=\(value) query4=\(query)")
            }
        }

        container.resetState()
        let promptState = try container.makePromptSnapshot(from: prompt)
        let firstToken = Int(promptState.metalState.firstToken)
        let decoded = container.tokenizer.decode(tokens: [firstToken], skipSpecialTokens: false)
        print("[\(label) first token] \(firstToken) -> \(String(reflecting: decoded))")
        if label == "Qwen text" || label == "Qwen chat" {
            let continuation = try container.debugContinuationLogitComparison(
                prompt: prompt,
                appendedTokenID: firstToken,
                topK: 10
            )
            print("[\(label) continuation comparison]")
            print(
                "  fingerprints=(\(continuation.prefillLogitsFingerprint), \(continuation.decodeLogitsFingerprint)) maxAbsDiff=\(String(format: "%.6f", continuation.maxAbsDiff)) differingCount=\(continuation.differingCount)"
            )
            print("  prefill+token top logits:")
            for entry in continuation.prefillTopLogits {
                print("    id=\(entry.tokenID) logit=\(String(format: "%.4f", entry.logit)) token=\(String(reflecting: entry.decoded))")
            }
            print("  prefill->decode top logits:")
            for entry in continuation.decodeTopLogits {
                print("    id=\(entry.tokenID) logit=\(String(format: "%.4f", entry.logit)) token=\(String(reflecting: entry.decoded))")
            }

            let forcedBarrierContainer = try container.debugClone(
                compiledModel: forceDecodeBarrierPolicy(
                    on: container.debugCompiledModel,
                    barrierPolicy: .bufferBarrier
                )
            )
            let forcedBarrierContinuation = try forcedBarrierContainer.debugContinuationLogitComparison(
                prompt: prompt,
                appendedTokenID: firstToken,
                topK: 10
            )
            print("[\(label) continuation comparison forced barriers]")
            print(
                "  fingerprints=(\(forcedBarrierContinuation.prefillLogitsFingerprint), \(forcedBarrierContinuation.decodeLogitsFingerprint)) maxAbsDiff=\(String(format: "%.6f", forcedBarrierContinuation.maxAbsDiff)) differingCount=\(forcedBarrierContinuation.differingCount)"
            )
            print("  prefill+token top logits:")
            for entry in forcedBarrierContinuation.prefillTopLogits {
                print("    id=\(entry.tokenID) logit=\(String(format: "%.4f", entry.logit)) token=\(String(reflecting: entry.decoded))")
            }
            print("  forced decode top logits:")
            for entry in forcedBarrierContinuation.decodeTopLogits {
                print("    id=\(entry.tokenID) logit=\(String(format: "%.4f", entry.logit)) token=\(String(reflecting: entry.decoded))")
            }

#if ENABLE_METAL_PROBES
            if label == "Qwen text" {
                try printQwenContinuationAttentionProbeDiagnostics(
                    container: container,
                    prompt: prompt,
                    appendedTokenID: Int32(firstToken)
                )
            }
#endif
        }
    }

    private func forceDecodeBarrierPolicy(
        on compiledModel: MetalCompiledModel,
        barrierPolicy: MetalBarrierPolicy
    ) -> MetalCompiledModel {
        let steps = compiledModel.decodePlan.steps.enumerated().map { index, step in
            guard index > 0 else { return step }
            let descriptor = MetalDispatchDescriptor(
                pipeline: step.pipeline,
                gridSize: step.gridSize,
                threadgroupSize: step.threadgroupSize,
                threadgroupMemoryLength: step.threadgroupMemoryLength,
                barrierPolicy: barrierPolicy
            )
            return MetalDispatchStep(
                descriptor: descriptor,
                bindings: step.bindings,
                bufferAccesses: step.bufferAccesses,
                metadata: step.metadata
            )
        }
        return MetalCompiledModel(
            decodePlan: MetalDispatchPlan(
                steps: steps,
                buffers: compiledModel.decodePlan.buffers,
                unfusedEntryCount: compiledModel.decodePlan.unfusedEntryCount,
                fusedEntryCount: compiledModel.decodePlan.fusedEntryCount,
                supplementalResidencyBuffers: compiledModel.decodePlan.supplementalResidencyBuffers
            ),
            prefillPlan: compiledModel.prefillPlan,
            auxiliaryPipelines: compiledModel.auxiliaryPipelines
        )
    }

#if ENABLE_METAL_PROBES
    private func printQwenContinuationAttentionProbeDiagnostics(
        container: LanguageModelContext,
        prompt: ExecutablePrompt,
        appendedTokenID: Int32
    ) throws {
        let extendedPrompt = ExecutablePrompt(
            tokenIDs: prompt.tokenIDs + [Int(appendedTokenID)],
            attentionMask: prompt.attentionMask,
            visualContext: prompt.visualContext
        )

        let prefillProbes: [MetalInferenceModel.DebugPrefillBindingProbe] = [
            .init(
                label: "prefill-hidden-in",
                stepIndex: 52,
                bindingIndex: 0,
                phase: .beforeStep,
                rowStride: 2048,
                count: 8,
                precision: .float32
            ),
            .init(
                label: "prefill-packed-before",
                stepIndex: 54,
                bindingIndex: 0,
                phase: .beforeStep,
                rowStride: 4096,
                count: 8,
                precision: .float32
            ),
            .init(
                label: "prefill-query-after-extract",
                stepIndex: 54,
                bindingIndex: 1,
                phase: .afterStep,
                rowStride: 2048,
                count: 8,
                precision: .float32
            ),
            .init(
                label: "prefill-query-after-qk-norm",
                stepIndex: 55,
                bindingIndex: 0,
                phase: .afterStep,
                rowStride: 2048,
                count: 8,
                precision: .float32
            ),
            .init(
                label: "prefill-query-after-rope",
                stepIndex: 57,
                bindingIndex: 0,
                phase: .afterStep,
                rowStride: 2048,
                count: 8,
                precision: .float32
            ),
            .init(
                label: "prefill-key-after-rope",
                stepIndex: 57,
                bindingIndex: 1,
                phase: .afterStep,
                rowStride: 512,
                count: 8,
                precision: .float32
            ),
            .init(
                label: "prefill-flash-out",
                stepIndex: 59,
                bindingIndex: 3,
                phase: .afterStep,
                rowStride: 2048,
                count: 8,
                precision: .float32
            ),
            .init(
                label: "prefill-gate-out",
                stepIndex: 60,
                bindingIndex: 2,
                phase: .afterStep,
                rowStride: 2048,
                count: 8,
                precision: .float32
            ),
        ]

        let decodeProbes: [MetalInferenceModel.DebugDecodeBindingProbe] = [
            .init(
                label: "decode-hidden-in",
                stepIndex: 36,
                bindingIndex: 0,
                phase: .beforeStep,
                count: 8,
                precision: .bfloat16
            ),
            .init(
                label: "decode-packed-before",
                stepIndex: 37,
                bindingIndex: 0,
                phase: .beforeStep,
                count: 8,
                precision: .bfloat16
            ),
            .init(
                label: "decode-query-after-extract",
                stepIndex: 37,
                bindingIndex: 1,
                phase: .afterStep,
                count: 8,
                precision: .bfloat16
            ),
            .init(
                label: "decode-query-after-qk-norm",
                stepIndex: 38,
                bindingIndex: 0,
                phase: .afterStep,
                count: 8,
                precision: .bfloat16
            ),
            .init(
                label: "decode-query-after-rope",
                stepIndex: 39,
                bindingIndex: 0,
                phase: .beforeStep,
                count: 8,
                precision: .bfloat16
            ),
            .init(
                label: "decode-key-after-rope",
                stepIndex: 39,
                bindingIndex: 1,
                phase: .beforeStep,
                count: 8,
                precision: .bfloat16
            ),
            .init(
                label: "decode-flash-out",
                stepIndex: 39,
                bindingIndex: 5,
                phase: .afterStep,
                count: 8,
                precision: .bfloat16
            ),
            .init(
                label: "decode-gate-out",
                stepIndex: 40,
                bindingIndex: 2,
                phase: .afterStep,
                count: 8,
                precision: .bfloat16
            ),
        ]

        let prefill = try container.debugPrefillBindingProbes(
            prompt: extendedPrompt,
            stepIndex: 60,
            probes: prefillProbes
        )
        let decode = try container.debugDecodeBindingProbes(
            prompt: prompt,
            tokenID: Int(appendedTokenID),
            probes: decodeProbes
        )

        let comparisons: [(String, String, String)] = [
            ("hidden-in", "prefill-hidden-in", "decode-hidden-in"),
            ("packed-before-extract", "prefill-packed-before", "decode-packed-before"),
            ("query-after-extract", "prefill-query-after-extract", "decode-query-after-extract"),
            ("query-after-qk-norm", "prefill-query-after-qk-norm", "decode-query-after-qk-norm"),
            ("query-after-rope", "prefill-query-after-rope", "decode-query-after-rope"),
            ("key-after-rope", "prefill-key-after-rope", "decode-key-after-rope"),
            ("flash-out", "prefill-flash-out", "decode-flash-out"),
            ("gate-out", "prefill-gate-out", "decode-gate-out"),
        ]

        print("[Qwen continuation attention probes]")
        for (label, prefillKey, decodeKey) in comparisons {
            let prefillValues = prefill[prefillKey, default: []]
            let decodeValues = decode[decodeKey, default: []]
            let maxDiff = zip(prefillValues, decodeValues).reduce(Float.zero) { partial, pair in
                max(partial, abs(pair.0 - pair.1))
            }
            let prefillSample = prefillValues.map { String(format: "%.4f", $0) }
            let decodeSample = decodeValues.map { String(format: "%.4f", $0) }
            print(
                "  \(label) maxDiff=\(String(format: "%.6f", maxDiff)) prefill=\(prefillSample) decode=\(decodeSample)"
            )
        }

        let residualPrefillStepIndices = [9, 16, 25, 32, 41, 48]
        let residualDecodeStepIndices = [6, 11, 17, 22, 28, 33]
        let residualDecodeProbes = residualDecodeStepIndices.enumerated().map { index, stepIndex in
            MetalInferenceModel.DebugDecodeBindingProbe(
                label: "decode-residual-\(index)",
                stepIndex: stepIndex,
                bindingIndex: 2,
                phase: .afterStep,
                count: 8,
                precision: .bfloat16
            )
        }
        let residualPrefill = try container.debugPrefillLastTokenHiddenSnapshots(
            prompt: extendedPrompt,
            stepIndices: Set(residualPrefillStepIndices)
        )
        let residualDecode = try container.debugDecodeBindingProbes(
            prompt: prompt,
            tokenID: Int(appendedTokenID),
            probes: residualDecodeProbes
        )
        print("[Qwen continuation residual probes]")
        for index in residualPrefillStepIndices.indices {
            let prefillValues = Array(residualPrefill[residualPrefillStepIndices[index], default: []].prefix(8))
            let decodeValues = residualDecode["decode-residual-\(index)", default: []]
            let maxDiff = zip(prefillValues, decodeValues).reduce(Float.zero) { partial, pair in
                max(partial, abs(pair.0 - pair.1))
            }
            let prefillSample = prefillValues.map { String(format: "%.4f", $0) }
            let decodeSample = decodeValues.map { String(format: "%.4f", $0) }
            print(
                "  layerBlock=\(index) maxDiff=\(String(format: "%.6f", maxDiff)) prefill=\(prefillSample) decode=\(decodeSample)"
            )
        }
    }
#endif

    private func printQwenEmbeddingDiagnostics(
        container: LanguageModelContext,
        prompt: ExecutablePrompt
    ) throws {
        guard let directory = try QwenVisionTestSupport.optionalRealQwen3VLDirectory() else {
            return
        }
        let finalHidden = try container.debugPrefillFinalHidden(prompt: prompt)
        guard !finalHidden.isEmpty else {
            print("[Qwen final hidden] unavailable")
            return
        }

        let sample = finalHidden.prefix(8).map { String(format: "%.4f", $0) }
        let norm = sqrt(finalHidden.reduce(Float.zero) { $0 + $1 * $1 })
        print("[Qwen final hidden]")
        print("  sample=\(sample)")
        print("  norm=\(String(format: "%.4f", norm))")

        let summaries = container.debugPrefillStepSummaries()
        let residualSteps = summaries.enumerated()
            .filter { _, summary in summary.kernelName.contains("residual_add_seq") }
            .map(\.offset)
        let captureSteps = Set(residualSteps)
        let snapshots = try container.debugPrefillLastTokenHiddenSnapshots(
            prompt: prompt,
            stepIndices: captureSteps
        )
        print("[Qwen layerwise hidden snapshots]")
        for (residualIndex, stepIndex) in residualSteps.enumerated() where residualIndex % 2 == 1 {
            guard let snapshot = snapshots[stepIndex] else { continue }
            let layerNorm = sqrt(snapshot.reduce(Float.zero) { $0 + $1 * $1 })
            let layerSample = snapshot.prefix(4).map { String(format: "%.4f", $0) }
            print("  layer=\(residualIndex / 2) step=\(stepIndex) norm=\(String(format: "%.4f", layerNorm)) sample=\(layerSample)")
        }

        let device = try #require(MTLCreateSystemDefaultDevice())
        let safetensorURLs = try FileManager.default.contentsOfDirectory(
            at: directory,
            includingPropertiesForKeys: nil
        )
        .filter { $0.pathExtension == "safetensors" }
        .sorted { $0.lastPathComponent < $1.lastPathComponent }
        let weights = try SafetensorsLoader().loadAll(urls: safetensorURLs, device: device)
        let stafURL = directory.appendingPathComponent("model.staf")
        let stafWeights = try STAFLoader().load(at: stafURL, device: device)
        let tensorName = "model.language_model.embed_tokens.weight"
        let tensor = try #require(weights.tensor(for: tensorName))
        let finalNormTensor = try #require(weights.tensor(for: "model.language_model.norm.weight"))
        let hiddenSize = finalHidden.count
        let candidateTokenIDs = [271, 198, 220, 51076, 74482, 25358, 59441, 57666]
        let specialTokenIDs = [248045, 248046, 248068, 248069, 3710]

        print("[Qwen STAF embedding row diffs]")
        for tokenID in specialTokenIDs {
            let safetensorsRow = readTensorRow(
                tokenID: tokenID,
                tensor: tensor,
                hiddenSize: hiddenSize
            )
            let stafRow = try readSTAFRow(
                tokenID: tokenID,
                tensorName: tensorName,
                hiddenSize: hiddenSize,
                store: stafWeights
            )
            let maxDiff = zip(safetensorsRow, stafRow).reduce(Float.zero) { partial, pair in
                max(partial, abs(pair.0 - pair.1))
            }
            let sample = zip(safetensorsRow.prefix(4), stafRow.prefix(4)).map { lhs, rhs in
                "\(String(format: "%.4f", lhs))/\(String(format: "%.4f", rhs))"
            }
            print("  id=\(tokenID) maxDiff=\(String(format: "%.6f", maxDiff)) sample=\(sample)")
        }

        print("[Qwen embedding logits from Metal final hidden]")
        for tokenID in candidateTokenIDs {
            let logit = dotEmbeddingRow(
                tokenID: tokenID,
                hidden: finalHidden,
                tensor: tensor,
                hiddenSize: hiddenSize
            )
            let decoded = container.tokenizer.decode(tokens: [tokenID], skipSpecialTokens: false)
            print("  id=\(tokenID) logit=\(String(format: "%.4f", logit)) token=\(String(reflecting: decoded))")
        }

        let manualTopLogits = topEmbeddingLogits(
            hidden: finalHidden,
            tensor: tensor,
            topK: 10
        )
        print("[Qwen manual top logits from Metal final hidden]")
        for entry in manualTopLogits {
            let decoded = container.tokenizer.decode(tokens: [entry.tokenID], skipSpecialTokens: false)
            print("  id=\(entry.tokenID) logit=\(String(format: "%.4f", entry.logit)) token=\(String(reflecting: decoded))")
        }

        if let lastStepIndex = residualSteps.last, let preFinalHidden = snapshots[lastStepIndex] {
            let cpuNormalized = applyQwenFinalRMSNorm(
                hidden: preFinalHidden,
                weightTensor: finalNormTensor
            )
            print("[Qwen CPU final norm logits from last decoder hidden]")
            for tokenID in candidateTokenIDs {
                let logit = dotEmbeddingRow(
                    tokenID: tokenID,
                    hidden: cpuNormalized,
                    tensor: tensor,
                    hiddenSize: hiddenSize
                )
                let decoded = container.tokenizer.decode(tokens: [tokenID], skipSpecialTokens: false)
                print("  id=\(tokenID) logit=\(String(format: "%.4f", logit)) token=\(String(reflecting: decoded))")
            }

            let manualNormalizedTopLogits = topEmbeddingLogits(
                hidden: cpuNormalized,
                tensor: tensor,
                topK: 10
            )
            print("[Qwen manual top logits from CPU final norm]")
            for entry in manualNormalizedTopLogits {
                let decoded = container.tokenizer.decode(tokens: [entry.tokenID], skipSpecialTokens: false)
                print("  id=\(entry.tokenID) logit=\(String(format: "%.4f", entry.logit)) token=\(String(reflecting: decoded))")
            }
        }
    }

    private func printGemmaEmbeddingDiagnostics(
        container: LanguageModelContext,
        prompt: ExecutablePrompt
    ) throws {
        let diagnostics = try container.debugPrefillOutputHeadDiagnostics(prompt: prompt, topK: 10)
        let layout = diagnostics.inputLayout
        print("[Gemma output-head input layout]")
        print(
            "  buffer=\(layout.bufferKind) offset=\(layout.offset) length=\(layout.bufferLength) hiddenCount=\(layout.hiddenCount) readableCount=\(layout.readableCount)"
        )
        let transferLayout = diagnostics.transferLayout
        print("[Gemma transfer layout]")
        print(
            "  source=\(transferLayout.sourceBufferKind) offset=\(transferLayout.sourceOffset) length=\(transferLayout.sourceBufferLength) readable=\(transferLayout.sourceReadableCount)"
        )
        print(
            "  destination=\(transferLayout.destinationBufferKind) offset=\(transferLayout.destinationOffset) length=\(transferLayout.destinationBufferLength) readable=\(transferLayout.destinationReadableCount) transferCount=\(transferLayout.transferCount)"
        )
        let transferVectors = (
            source: diagnostics.transferSource,
            destination: diagnostics.transferDestination
        )
        let transferMaxDiff = zip(transferVectors.source, transferVectors.destination).reduce(Float.zero) { partial, pair in
            max(partial, abs(pair.0 - pair.1))
        }
        let sourceSample = transferVectors.source.prefix(8).map { String(format: "%.4f", $0) }
        let destinationSample = transferVectors.destination.prefix(8).map { String(format: "%.4f", $0) }
        print("[Gemma transfer vectors]")
        print("  sourceSample=\(sourceSample)")
        print("  destinationSample=\(destinationSample)")
        print("  maxDiff=\(String(format: "%.6f", transferMaxDiff))")

        let finalHidden = diagnostics.transferDestination
        guard !finalHidden.isEmpty else {
            print("[Gemma final hidden] unavailable")
            return
        }

        let sample = finalHidden.prefix(8).map { String(format: "%.4f", $0) }
        let norm = sqrt(finalHidden.reduce(Float.zero) { $0 + $1 * $1 })
        print("[Gemma final hidden]")
        print("  sample=\(sample)")
        print("  norm=\(String(format: "%.4f", norm))")
        print("[Gemma GPU top logits]")
        for entry in diagnostics.topLogits {
            let decoded = container.tokenizer.decode(tokens: [entry.tokenID], skipSpecialTokens: false)
            print("  id=\(entry.tokenID) logit=\(String(format: "%.4f", entry.logit)) token=\(String(reflecting: decoded))")
        }

        guard let directory = try Gemma4TestSupport.optionalRealGemma4Directory(),
              let device = MTLCreateSystemDefaultDevice() else {
            return
        }
        let safetensorURLs = try FileManager.default.contentsOfDirectory(
            at: directory,
            includingPropertiesForKeys: nil
        )
        .filter { $0.pathExtension == "safetensors" }
        .sorted { $0.lastPathComponent < $1.lastPathComponent }
        let weights = try SafetensorsLoader().loadAll(urls: safetensorURLs, device: device)
        guard let tensor = weights.tensor(for: "model.language_model.embed_tokens.weight") else {
            return
        }

        var candidateTokenIDs = diagnostics.topLogits.map(\.tokenID)
        let tokyoTokenIDs = [
            container.tokenizer.encode(text: "Tokyo", addSpecialTokens: false),
            container.tokenizer.encode(text: " Tokyo", addSpecialTokens: false),
        ].flatMap { $0 }
        candidateTokenIDs.append(contentsOf: tokyoTokenIDs)
        candidateTokenIDs = Array(Set(candidateTokenIDs)).sorted()

        print("[Gemma manual logits from Metal final hidden]")
        for tokenID in candidateTokenIDs {
            let logit = dotEmbeddingRow(
                tokenID: tokenID,
                hidden: finalHidden,
                tensor: tensor,
                hiddenSize: finalHidden.count
            )
            let decoded = container.tokenizer.decode(tokens: [tokenID], skipSpecialTokens: false)
            print("  id=\(tokenID) logit=\(String(format: "%.4f", logit)) token=\(String(reflecting: decoded))")
        }
    }

    private func printGemmaResidualDiagnostics(
        container: LanguageModelContext,
        prompt: ExecutablePrompt
    ) throws {
        let summaries = container.debugPrefillStepSummaries()
        let residualSteps = summaries.enumerated()
            .filter { _, summary in summary.kernelName.contains("residual_add_seq") }
            .map(\.offset)
        print("[Gemma residual step count] \(residualSteps.count)")
        let snapshots = try container.debugPrefillLastTokenHiddenSnapshots(
            prompt: prompt,
            stepIndices: Set(residualSteps)
        )
        print("[Gemma residual hidden norms]")
        var firstZeroStep: Int?
        for (residualIndex, stepIndex) in residualSteps.enumerated() {
            guard let snapshot = snapshots[stepIndex] else { continue }
            let norm = sqrt(snapshot.reduce(Float.zero) { $0 + $1 * $1 })
            let sample = snapshot.prefix(4).map { String(format: "%.4f", $0) }
            print("  residual=\(residualIndex) step=\(stepIndex) norm=\(String(format: "%.4f", norm)) sample=\(sample)")
            if firstZeroStep == nil, norm == 0 {
                firstZeroStep = stepIndex
            }
        }
        if let firstZeroStep {
            let start = max(firstZeroStep - 16, 0)
            let end = min(firstZeroStep + 8, summaries.count - 1)
            print("[Gemma zero onset steps]")
            for stepIndex in start...end {
                let summary = summaries[stepIndex]
                let layerLabel = summary.layerIndex.map(String.init) ?? "-"
                print("  step=\(stepIndex) layer=\(layerLabel) kernel=\(summary.kernelName)")
            }
            if let hiddenSize = residualSteps.first.flatMap({ snapshots[$0]?.count }) {
                let detailedSteps = Set(start...min(firstZeroStep + 4, summaries.count - 1))
                let hiddenAroundZero = try container.debugPrefillLastTokenHiddenSnapshots(
                    prompt: prompt,
                    stepIndices: detailedSteps
                )
                _ = hiddenSize
                print("[Gemma zero onset hidden]")
                for stepIndex in (start...min(firstZeroStep + 4, summaries.count - 1)) {
                    let hidden = hiddenAroundZero[stepIndex, default: []].prefix(4).map { String(format: "%.4f", $0) }
                    print("  step=\(stepIndex) hidden=\(hidden)")
                }
            }
        }
    }

    private func printGemmaRepeatedResidualDiagnostics(
        container: LanguageModelContext,
        prompt: ExecutablePrompt
    ) throws {
        let summaries = container.debugPrefillStepSummaries()
        let residualSteps = summaries.enumerated()
            .filter { _, summary in summary.kernelName.contains("residual_add_seq") }
            .map(\.offset)
        guard !residualSteps.isEmpty else {
            print("[Gemma repeated residuals] no residual steps")
            return
        }
        let first = try container.debugPrefillLastTokenHiddenSnapshots(
            prompt: prompt,
            stepIndices: Set(residualSteps)
        )
        container.resetState()
        let second = try container.debugPrefillLastTokenHiddenSnapshots(
            prompt: prompt,
            stepIndices: Set(residualSteps)
        )
        print("[Gemma repeated residuals]")
        for stepIndex in residualSteps {
            let firstValues = first[stepIndex, default: []]
            let secondValues = second[stepIndex, default: []]
            guard !firstValues.isEmpty, firstValues.count == secondValues.count else { continue }
            let maxDiff = zip(firstValues, secondValues).reduce(Float.zero) { partial, pair in
                max(partial, abs(pair.0 - pair.1))
            }
            let firstSample = firstValues.prefix(4).map { String(format: "%.4f", $0) }
            let secondSample = secondValues.prefix(4).map { String(format: "%.4f", $0) }
            let layerLabel = summaries[stepIndex].layerIndex.map(String.init) ?? "-"
            print(
                "  step=\(stepIndex) layer=\(layerLabel) maxDiff=\(String(format: "%.6f", maxDiff)) first=\(firstSample) second=\(secondSample)"
            )
        }
    }

#if ENABLE_METAL_PROBES
    private func printGemmaAttentionProbeDiagnostics(
        container: LanguageModelContext,
        prompt: ExecutablePrompt
    ) throws {
        let summaries = container.debugPrefillStepSummaries()
        let steps = container.debugPrefillSteps()
        let residualSteps = summaries.enumerated()
            .filter { _, summary in summary.kernelName.contains("residual_add_seq") }
            .map(\.offset)
        let residualSnapshots = try container.debugPrefillLastTokenHiddenSnapshots(
            prompt: prompt,
            stepIndices: Set(residualSteps)
        )
        guard let firstZeroResidualStep = residualSteps.first(where: { stepIndex in
            guard let snapshot = residualSnapshots[stepIndex] else { return false }
            let norm = sqrt(snapshot.reduce(Float.zero) { $0 + $1 * $1 })
            return norm == 0
        }) else {
            print("[Gemma attention probe] no zero residual step")
            return
        }

        guard let flashStepIndex = (0..<firstZeroResidualStep).last(where: {
            summaries[$0].kernelName.contains("flash_attn_batch")
        }) else {
            print("[Gemma attention probe] flash step not found before residual step \(firstZeroResidualStep)")
            return
        }
        guard let outProjStepIndex = ((flashStepIndex + 1)..<firstZeroResidualStep).first(where: {
            summaries[$0].kernelName.contains("gemm")
        }) else {
            print("[Gemma attention probe] out-proj GEMM not found after flash step \(flashStepIndex)")
            return
        }
        guard let postAttentionNormStepIndex = ((outProjStepIndex + 1)..<firstZeroResidualStep).first(where: {
            summaries[$0].kernelName.contains("rms_norm")
        }) else {
            print("[Gemma attention probe] post-attention RMSNorm not found after out-proj step \(outProjStepIndex)")
            return
        }
        let residualCopyStepIndex = max(firstZeroResidualStep - 1, 0)

        let flashStep = steps[flashStepIndex]
        let outProjStep = steps[outProjStepIndex]
        let postAttentionNormStep = steps[postAttentionNormStepIndex]
        let residualCopyStep = steps[residualCopyStepIndex]
        func uint32Value(
            in step: MetalPrefillStep,
            at bindingIndex: Int
        ) throws -> Int {
            if let inlineBinding = step.bytesBindings.first(where: { $0.index == bindingIndex }) {
                guard inlineBinding.value.count == MemoryLayout<UInt32>.size else {
                    throw DiagnosticError(message: "Unexpected byte count for binding[\(bindingIndex)]")
                }
                return inlineBinding.value.withUnsafeBytes { bytes in
                    Int(bytes.load(as: UInt32.self))
                }
            }

            guard let constant = step.bindings.constants.first(where: { $0.index == bindingIndex }) else {
                throw DiagnosticError(message: "Missing uint32 binding[\(bindingIndex)]")
            }

            switch constant {
            case .inline(let binding):
                guard binding.value.count == MemoryLayout<UInt32>.size else {
                    throw DiagnosticError(message: "Unexpected byte count for binding[\(bindingIndex)]")
                }
                return binding.value.withUnsafeBytes { bytes in
                    Int(bytes.load(as: UInt32.self))
                }
            case .buffer(let binding):
                guard binding.length == MemoryLayout<UInt32>.size else {
                    throw DiagnosticError(message: "Unexpected resident constant length for binding[\(bindingIndex)]")
                }
                return Int(
                    binding.buffer.contents()
                        .advanced(by: binding.offset)
                        .bindMemory(to: UInt32.self, capacity: 1)
                        .pointee
                )
            }
        }

        let headCount = try uint32Value(in: flashStep, at: 4)
        let headDimension = try uint32Value(in: flashStep, at: 6)
        let flashOutputDimension = headCount * headDimension
        let outProjInputDimension = try uint32Value(in: outProjStep, at: 3)
        let outProjOutputDimension = try uint32Value(in: outProjStep, at: 4)
        guard let outProjWeightBinding = outProjStep.bindings.buffers.first(where: { $0.index == 1 }) else {
            throw DiagnosticError(message: "Missing out-proj weight binding")
        }
        guard let flashOutputBinding = flashStep.bindings.buffers.first(where: { $0.index == 3 }) else {
            throw DiagnosticError(message: "Missing flash output binding")
        }
        guard let outProjInputBindingIndex = outProjStep.bindings.buffers.first(where: { binding in
            binding.buffer === flashOutputBinding.buffer && binding.offset == flashOutputBinding.offset
        })?.index else {
            throw DiagnosticError(
                message: "Missing out-proj input binding for flash output at step \(outProjStepIndex)"
            )
        }
        guard let outProjOutputBinding = outProjStep.bindings.buffers.first(where: { $0.index == 2 }) else {
            throw DiagnosticError(message: "Missing out-proj output binding")
        }
        guard let normInputBindingIndex = postAttentionNormStep.bindings.buffers.first(where: { binding in
            binding.buffer === outProjOutputBinding.buffer && binding.offset == outProjOutputBinding.offset
        })?.index else {
            throw DiagnosticError(
                message: "Missing post-attention norm input binding for out-proj output at step \(postAttentionNormStepIndex)"
            )
        }
        guard let normOutputBinding = postAttentionNormStep.bindings.buffers.first(where: { $0.index == 2 }) else {
            throw DiagnosticError(message: "Missing post-attention norm output binding")
        }
        guard let residualDeltaBindingIndex = steps[firstZeroResidualStep].bindings.buffers.first(where: { binding in
            binding.buffer === normOutputBinding.buffer && binding.offset == normOutputBinding.offset
        })?.index else {
            throw DiagnosticError(
                message: "Missing residual delta binding for post-attention norm output at step \(firstZeroResidualStep)"
            )
        }
        guard let residualSourceBinding = residualCopyStep.bindings.buffers.first(where: { $0.index == 0 }) else {
            throw DiagnosticError(message: "Missing residual copy source binding")
        }
        guard let residualHiddenBindingIndex = steps[firstZeroResidualStep].bindings.buffers.first(where: { binding in
            binding.buffer === residualSourceBinding.buffer && binding.offset == residualSourceBinding.offset
        })?.index else {
            throw DiagnosticError(
                message: "Missing residual hidden binding for residual copy source at step \(firstZeroResidualStep)"
            )
        }

        let segmentCount = min(flashOutputDimension, 32)
        let probes: [MetalInferenceModel.DebugPrefillBindingProbe] = [
            .init(
                label: "flash-query-head",
                stepIndex: flashStepIndex,
                bindingIndex: 0,
                phase: .beforeStep,
                rowStride: flashOutputDimension,
                count: segmentCount
            ),
            .init(
                label: "flash-out-head",
                stepIndex: flashStepIndex,
                bindingIndex: 3,
                phase: .afterStep,
                rowStride: flashOutputDimension,
                count: segmentCount
            ),
            .init(
                label: "out-proj-in-head",
                stepIndex: outProjStepIndex,
                bindingIndex: outProjInputBindingIndex,
                phase: .beforeStep,
                rowStride: flashOutputDimension,
                count: segmentCount
            ),
            .init(
                label: "out-proj-in-full",
                stepIndex: outProjStepIndex,
                bindingIndex: outProjInputBindingIndex,
                phase: .beforeStep,
                rowStride: flashOutputDimension,
                count: flashOutputDimension
            ),
            .init(
                label: "out-proj-out-head",
                stepIndex: outProjStepIndex,
                bindingIndex: 2,
                phase: .afterStep,
                rowStride: outProjOutputDimension,
                count: segmentCount
            ),
            .init(
                label: "post-attn-norm-in-head",
                stepIndex: postAttentionNormStepIndex,
                bindingIndex: normInputBindingIndex,
                phase: .beforeStep,
                rowStride: outProjOutputDimension,
                count: segmentCount
            ),
            .init(
                label: "post-attn-norm-out-head",
                stepIndex: postAttentionNormStepIndex,
                bindingIndex: 2,
                phase: .afterStep,
                rowStride: outProjOutputDimension,
                count: segmentCount
            ),
            .init(
                label: "residual-copy-source-head",
                stepIndex: residualCopyStepIndex,
                bindingIndex: 0,
                phase: .beforeStep,
                rowStride: outProjOutputDimension,
                count: segmentCount
            ),
            .init(
                label: "residual-hidden-head",
                stepIndex: firstZeroResidualStep,
                bindingIndex: residualHiddenBindingIndex,
                phase: .beforeStep,
                rowStride: outProjOutputDimension,
                count: segmentCount
            ),
            .init(
                label: "residual-delta-head",
                stepIndex: firstZeroResidualStep,
                bindingIndex: residualDeltaBindingIndex,
                phase: .beforeStep,
                rowStride: outProjOutputDimension,
                count: segmentCount
            ),
            .init(
                label: "residual-out-head",
                stepIndex: firstZeroResidualStep,
                bindingIndex: 2,
                phase: .afterStep,
                rowStride: outProjOutputDimension,
                count: segmentCount
            ),
        ]

        let replaySnapshots = try container.debugPrefillBindingProbes(
            prompt: prompt,
            stepIndex: flashStepIndex,
            probes: probes
        )
        let actualSnapshots = try container.debugActualPrefillBindingProbes(
            prompt: prompt,
            stepIndex: flashStepIndex,
            probes: probes
        )
        print("[Gemma attention probe]")
        print("  firstZeroResidualStep=\(firstZeroResidualStep)")
        print("  flashStep=\(flashStepIndex) kernel=\(summaries[flashStepIndex].kernelName)")
        print("  outProjStep=\(outProjStepIndex) kernel=\(summaries[outProjStepIndex].kernelName)")
        print("  postAttentionNormStep=\(postAttentionNormStepIndex) kernel=\(summaries[postAttentionNormStepIndex].kernelName)")
        print("  residualCopyStep=\(residualCopyStepIndex) kernel=\(summaries[residualCopyStepIndex].kernelName)")
        print("  residualStep=\(firstZeroResidualStep) kernel=\(summaries[firstZeroResidualStep].kernelName)")
        print("  outProjWeightTensor=\(outProjStep.metadata.weightTensorName ?? "<unknown>")")
        print("  outProjWeightBufferLabel=\(outProjWeightBinding.buffer.label ?? "<nil>") offset=\(outProjWeightBinding.offset)")
        for label in [
            "flash-query-head",
            "flash-out-head",
            "out-proj-in-head",
            "out-proj-out-head",
            "post-attn-norm-in-head",
            "post-attn-norm-out-head",
            "residual-copy-source-head",
            "residual-hidden-head",
            "residual-delta-head",
            "residual-out-head",
        ] {
            let replayValues = replaySnapshots[label, default: []]
            let replayMaximum = replayValues.reduce(Float.zero) { max($0, abs($1)) }
            let replaySample = replayValues.prefix(8).map { String(format: "%.4f", $0) }
            let actualValues = actualSnapshots[label, default: []]
            let actualMaximum = actualValues.reduce(Float.zero) { max($0, abs($1)) }
            let actualSample = actualValues.prefix(8).map { String(format: "%.4f", $0) }
            print("  \(label) replay.max=\(String(format: "%.4f", replayMaximum)) replay.sample=\(replaySample)")
            print("  \(label) actual.max=\(String(format: "%.4f", actualMaximum)) actual.sample=\(actualSample)")
        }

        let outProjInputFull = actualSnapshots["out-proj-in-full", default: []]
        if outProjInputFull.count == flashOutputDimension {
            let cpuReference = computeBF16ProjectionHead(
                input: outProjInputFull,
                weightBinding: outProjWeightBinding,
                inputDimension: outProjInputDimension,
                outputCount: 8
            )
            let cpuMaximum = cpuReference.reduce(Float.zero) { max($0, abs($1)) }
            let cpuSample = cpuReference.map { String(format: "%.4f", $0) }
            print("  out-proj-cpu-head max=\(String(format: "%.4f", cpuMaximum)) sample=\(cpuSample)")
            let weightSample = readBF16WeightSample(
                weightBinding: outProjWeightBinding,
                count: 8
            ).map { String(format: "%.4f", $0) }
            print("  out-proj-weight-sample=\(weightSample)")

            if let directory = try Gemma4TestSupport.optionalRealGemma4Directory() {
                let device = try #require(MTLCreateSystemDefaultDevice())
                let safetensorURLs = try FileManager.default.contentsOfDirectory(
                    at: directory,
                    includingPropertiesForKeys: nil
                )
                .filter { $0.pathExtension == "safetensors" }
                .sorted { $0.lastPathComponent < $1.lastPathComponent }
                let safetensors = try SafetensorsLoader().loadAll(urls: safetensorURLs, device: device)
                let stafStore = try STAFLoader().load(
                    at: directory.appendingPathComponent("model.staf"),
                    device: device
                )
                if let tensorName = outProjStep.metadata.weightTensorName {
                    let safetensorsTensor = try #require(safetensors.tensor(for: tensorName))
                    let stafEntry = try #require(stafStore.entries[tensorName])
                    let stafAccess = try #require(stafStore.bufferAccess(for: tensorName, layout: .rowMajor))
                    let safetensorsRowMajor = computeTensorProjectionHead(
                        input: outProjInputFull,
                        tensor: safetensorsTensor,
                        inputDimension: outProjInputDimension,
                        outputCount: 8,
                        transpose: false
                    )
                    let safetensorsTransposed = computeTensorProjectionHead(
                        input: outProjInputFull,
                        tensor: safetensorsTensor,
                        inputDimension: outProjInputDimension,
                        outputCount: 8,
                        transpose: true
                    )
                    let stafRowMajor = computeBufferAccessProjectionHead(
                        input: outProjInputFull,
                        access: stafAccess,
                        scheme: stafEntry.schemeIdentifier,
                        inputDimension: outProjInputDimension,
                        outputCount: 8
                    )
                    let safetensorsWeightSample = readTensorSample(
                        tensor: safetensorsTensor,
                        count: 8
                    ).map { String(format: "%.4f", $0) }
                    let stafRowSample = readBufferAccessSample(
                        access: stafAccess,
                        scheme: stafEntry.schemeIdentifier,
                        count: 8
                    ).map { String(format: "%.4f", $0) }
                    let safetensorsRowMajorSample = safetensorsRowMajor.map { String(format: "%.4f", $0) }
                    let safetensorsTransposedSample = safetensorsTransposed.map { String(format: "%.4f", $0) }
                    let stafRowMajorSample = stafRowMajor.map { String(format: "%.4f", $0) }
                    print("  out-proj-safetensors-shape=\(safetensorsTensor.shape) dtype=\(safetensorsTensor.dtype)")
                    print(
                        "  out-proj-staf-shape=\(stafEntry.shape) scheme=\(stafEntry.schemeIdentifier) bufferOffset=\(stafEntry.bufferOffset) accessOffset=\(stafAccess.offset)"
                    )
                    print("  out-proj-safetensors-weight-sample=\(safetensorsWeightSample)")
                    print("  out-proj-staf-row-weight-sample=\(stafRowSample)")
                    print("  out-proj-safetensors-row-major-head=\(safetensorsRowMajorSample)")
                    print("  out-proj-safetensors-transposed-head=\(safetensorsTransposedSample)")
                    print("  out-proj-staf-row-major-head=\(stafRowMajorSample)")
                }
            }
        } else {
            print("  out-proj-cpu-head unavailable")
        }
    }

    private func computeBF16ProjectionHead(
        input: [Float],
        weightBinding: MetalBufferBinding,
        inputDimension: Int,
        outputCount: Int
    ) -> [Float] {
        let pointer = weightBinding.buffer.contents()
            .advanced(by: weightBinding.offset)
            .bindMemory(to: UInt16.self, capacity: inputDimension * max(outputCount, 1))
        return (0..<outputCount).map { row in
            var sum: Float = 0
            let rowBase = row * inputDimension
            for column in 0..<inputDimension {
                let bf16 = pointer[rowBase + column]
                let weight = Float(bitPattern: UInt32(bf16) << 16)
                sum += input[column] * weight
            }
            return sum
        }
    }

    private func readBF16WeightSample(
        weightBinding: MetalBufferBinding,
        count: Int
    ) -> [Float] {
        let pointer = weightBinding.buffer.contents()
            .advanced(by: weightBinding.offset)
            .bindMemory(to: UInt16.self, capacity: max(count, 1))
        return (0..<count).map { index in
            Float(bitPattern: UInt32(pointer[index]) << 16)
        }
    }

    private func computeTensorProjectionHead(
        input: [Float],
        tensor: MetalTensor,
        inputDimension: Int,
        outputCount: Int,
        transpose: Bool
    ) -> [Float] {
        guard tensor.shape.count >= 2 else {
            return Array(repeating: .zero, count: outputCount)
        }
        let rows = tensor.shape[0]
        let columns = tensor.shape[1]
        guard rows > 0, columns > 0 else {
            return Array(repeating: .zero, count: outputCount)
        }
        let outputLimit = min(outputCount, transpose ? columns : rows)
        return (0..<outputLimit).map { row in
            var sum: Float = 0
            for column in 0..<inputDimension {
                let weight: Float
                if transpose {
                    weight = readTensorElement(
                        tensor: tensor,
                        row: column,
                        column: row,
                        rowStride: columns
                    )
                } else {
                    weight = readTensorElement(
                        tensor: tensor,
                        row: row,
                        column: column,
                        rowStride: columns
                    )
                }
                sum += input[column] * weight
            }
            return sum
        }
    }

    private func computeBufferAccessProjectionHead(
        input: [Float],
        access: STAFWeightBufferAccess,
        scheme: QuantizationSchemeIdentifier,
        inputDimension: Int,
        outputCount: Int
    ) -> [Float] {
        let basePointer = access.buffer.contents().advanced(by: access.offset)
        return (0..<outputCount).map { row in
            var sum: Float = 0
            let rowBase = row * inputDimension
            for column in 0..<inputDimension {
                let index = rowBase + column
                let weight: Float
                switch scheme {
                case .bf16RowMajor:
                    let pointer = basePointer.bindMemory(
                        to: UInt16.self,
                        capacity: rowBase + inputDimension
                    )
                    weight = Float(bitPattern: UInt32(pointer[index]) << 16)
                case .fp16RowMajor:
                    let pointer = basePointer.bindMemory(
                        to: UInt16.self,
                        capacity: rowBase + inputDimension
                    )
                    weight = Float(Float16(bitPattern: pointer[index]))
                case .fp32RowMajor:
                    let pointer = basePointer.bindMemory(
                        to: Float.self,
                        capacity: rowBase + inputDimension
                    )
                    weight = pointer[index]
                default:
                    return .zero
                }
                sum += input[column] * weight
            }
            return sum
        }
    }

    private func readTensorSample(
        tensor: MetalTensor,
        count: Int
    ) -> [Float] {
        let basePointer = tensor.buffer.contents().advanced(by: tensor.offset)
        switch tensor.dtype {
        case .float16:
            let pointer = basePointer.bindMemory(to: UInt16.self, capacity: max(count, 1))
            return (0..<count).map { index in Float(Float16(bitPattern: pointer[index])) }
        case .bfloat16:
            let pointer = basePointer.bindMemory(to: UInt16.self, capacity: max(count, 1))
            return (0..<count).map { index in Float(bitPattern: UInt32(pointer[index]) << 16) }
        case .float32:
            let pointer = basePointer.bindMemory(to: Float.self, capacity: max(count, 1))
            return (0..<count).map { index in pointer[index] }
        case .quantized:
            return Array(repeating: .zero, count: count)
        }
    }

    private func readBufferAccessSample(
        access: STAFWeightBufferAccess,
        scheme: QuantizationSchemeIdentifier,
        count: Int
    ) -> [Float] {
        let basePointer = access.buffer.contents().advanced(by: access.offset)
        switch scheme {
        case .bf16RowMajor:
            let pointer = basePointer.bindMemory(to: UInt16.self, capacity: max(count, 1))
            return (0..<count).map { index in Float(bitPattern: UInt32(pointer[index]) << 16) }
        case .fp16RowMajor:
            let pointer = basePointer.bindMemory(to: UInt16.self, capacity: max(count, 1))
            return (0..<count).map { index in Float(Float16(bitPattern: pointer[index])) }
        case .fp32RowMajor:
            let pointer = basePointer.bindMemory(to: Float.self, capacity: max(count, 1))
            return (0..<count).map { index in pointer[index] }
        default:
            return Array(repeating: .zero, count: count)
        }
    }

    private func readTensorElement(
        tensor: MetalTensor,
        row: Int,
        column: Int,
        rowStride: Int
    ) -> Float {
        let index = row * rowStride + column
        let basePointer = tensor.buffer.contents().advanced(by: tensor.offset)
        switch tensor.dtype {
        case .float16:
            let pointer = basePointer.bindMemory(to: UInt16.self, capacity: index + 1)
            return Float(Float16(bitPattern: pointer[index]))
        case .bfloat16:
            let pointer = basePointer.bindMemory(to: UInt16.self, capacity: index + 1)
            return Float(bitPattern: UInt32(pointer[index]) << 16)
        case .float32:
            let pointer = basePointer.bindMemory(to: Float.self, capacity: index + 1)
            return pointer[index]
        case .quantized:
            return .zero
        }
    }
#endif

    private func readTensorRow(
        tokenID: Int,
        tensor: MetalTensor,
        hiddenSize: Int
    ) -> [Float] {
        let basePointer = tensor.buffer.contents().advanced(by: tensor.offset)
        switch tensor.dtype {
        case .float16:
            let pointer = basePointer.bindMemory(to: UInt16.self, capacity: tensor.shape.reduce(1, *))
            let start = tokenID * hiddenSize
            return (0..<hiddenSize).map { index in
                Float(Float16(bitPattern: pointer[start + index]))
            }
        case .bfloat16:
            let pointer = basePointer.bindMemory(to: UInt16.self, capacity: tensor.shape.reduce(1, *))
            let start = tokenID * hiddenSize
            return (0..<hiddenSize).map { index in
                Float(bitPattern: UInt32(pointer[start + index]) << 16)
            }
        case .float32:
            let pointer = basePointer.bindMemory(to: Float.self, capacity: tensor.shape.reduce(1, *))
            let start = tokenID * hiddenSize
            return (0..<hiddenSize).map { index in
                pointer[start + index]
            }
        case .quantized:
            Issue.record("Quantized embedding tensor is unsupported in this diagnostic")
            return Array(repeating: .zero, count: hiddenSize)
        }
    }

    private func readSTAFRow(
        tokenID: Int,
        tensorName: String,
        hiddenSize: Int,
        store: STAFWeightStore
    ) throws -> [Float] {
        guard let entry = store.entries[tensorName],
              let access = store.bufferAccess(for: tensorName),
              let format = QuantizationFormatRegistry.format(for: entry.schemeIdentifier) else {
            throw DiagnosticError(message: "Missing STAF embedding tensor \(tensorName)")
        }
        let basePointer = access.buffer.contents().advanced(by: access.offset)
        let start = tokenID * hiddenSize
        switch format.schemeIdentifier {
        case .fp16RowMajor:
            let pointer = basePointer.bindMemory(to: UInt16.self, capacity: entry.shape.reduce(1, *))
            return (0..<hiddenSize).map { index in
                Float(Float16(bitPattern: pointer[start + index]))
            }
        case .bf16RowMajor:
            let pointer = basePointer.bindMemory(to: UInt16.self, capacity: entry.shape.reduce(1, *))
            return (0..<hiddenSize).map { index in
                Float(bitPattern: UInt32(pointer[start + index]) << 16)
            }
        case .fp32RowMajor:
            let pointer = basePointer.bindMemory(to: Float.self, capacity: entry.shape.reduce(1, *))
            return (0..<hiddenSize).map { index in
                pointer[start + index]
            }
        default:
            throw DiagnosticError(message: "Unsupported STAF embedding format \(format.schemeIdentifier)")
        }
    }

    private func dotEmbeddingRow(
        tokenID: Int,
        hidden: [Float],
        tensor: MetalTensor,
        hiddenSize: Int
    ) -> Float {
        let basePointer = tensor.buffer.contents().advanced(by: tensor.offset)
        switch tensor.dtype {
        case .float16:
            let pointer = basePointer.bindMemory(to: UInt16.self, capacity: tensor.shape.reduce(1, *))
            let start = tokenID * hiddenSize
            return zip(0..<hiddenSize, hidden).reduce(Float.zero) { partial, pair in
                partial + Float(Float16(bitPattern: pointer[start + pair.0])) * pair.1
            }
        case .bfloat16:
            let pointer = basePointer.bindMemory(to: UInt16.self, capacity: tensor.shape.reduce(1, *))
            let start = tokenID * hiddenSize
            return zip(0..<hiddenSize, hidden).reduce(Float.zero) { partial, pair in
                partial + Float(bitPattern: UInt32(pointer[start + pair.0]) << 16) * pair.1
            }
        case .float32:
            let pointer = basePointer.bindMemory(to: Float.self, capacity: tensor.shape.reduce(1, *))
            let start = tokenID * hiddenSize
            return zip(0..<hiddenSize, hidden).reduce(Float.zero) { partial, pair in
                partial + pointer[start + pair.0] * pair.1
            }
        case .quantized:
            Issue.record("Quantized embedding tensor is unsupported in this diagnostic")
            return .zero
        }
    }

    private func topEmbeddingLogits(
        hidden: [Float],
        tensor: MetalTensor,
        topK: Int
    ) -> [(tokenID: Int, logit: Float)] {
        guard tensor.shape.count >= 2 else {
            Issue.record("Embedding tensor rank is unsupported in this diagnostic")
            return []
        }

        let vocabularySize = tensor.shape[0]
        let hiddenSize = hidden.count
        let limit = max(0, min(topK, vocabularySize))
        guard vocabularySize > 0, hiddenSize > 0, limit > 0 else {
            return []
        }

        var ranked: [(tokenID: Int, logit: Float)] = []
        ranked.reserveCapacity(limit)
        for tokenID in 0..<vocabularySize {
            let logit = dotEmbeddingRow(
                tokenID: tokenID,
                hidden: hidden,
                tensor: tensor,
                hiddenSize: hiddenSize
            )
            if ranked.count < limit {
                ranked.append((tokenID, logit))
                ranked.sort(by: compareLogits)
                continue
            }
            guard let last = ranked.last else { continue }
            let shouldInsert = logit > last.logit || (logit == last.logit && tokenID < last.tokenID)
            guard shouldInsert else { continue }
            ranked.removeLast()
            ranked.append((tokenID, logit))
            ranked.sort(by: compareLogits)
        }
        return ranked
    }

    private func compareLogits(
        lhs: (tokenID: Int, logit: Float),
        rhs: (tokenID: Int, logit: Float)
    ) -> Bool {
        if lhs.logit == rhs.logit {
            return lhs.tokenID < rhs.tokenID
        }
        return lhs.logit > rhs.logit
    }

    private func applyQwenFinalRMSNorm(
        hidden: [Float],
        weightTensor: MetalTensor,
        epsilon: Float = 1e-6
    ) -> [Float] {
        let variance = hidden.reduce(Float.zero) { partial, value in
            partial + value * value
        } / Float(hidden.count)
        let scale = 1 / sqrt(variance + epsilon)
        let basePointer = weightTensor.buffer.contents().advanced(by: weightTensor.offset)

        switch weightTensor.dtype {
        case .float16:
            let pointer = basePointer.bindMemory(to: UInt16.self, capacity: hidden.count)
            return hidden.enumerated().map { index, value in
                let weight = Float(Float16(bitPattern: pointer[index]))
                return value * scale * (1 + weight)
            }
        case .bfloat16:
            let pointer = basePointer.bindMemory(to: UInt16.self, capacity: hidden.count)
            return hidden.enumerated().map { index, value in
                let weight = Float(bitPattern: UInt32(pointer[index]) << 16)
                return value * scale * (1 + weight)
            }
        case .float32:
            let pointer = basePointer.bindMemory(to: Float.self, capacity: hidden.count)
            return hidden.enumerated().map { index, value in
                value * scale * (1 + pointer[index])
            }
        case .quantized:
            Issue.record("Quantized final norm tensor is unsupported in this diagnostic")
            return hidden
        }
    }

    private func applyGemmaFinalRMSNorm(
        hidden: [Float],
        weightTensor: MetalTensor,
        epsilon: Float = 1e-6
    ) -> [Float] {
        let variance = hidden.reduce(Float.zero) { partial, value in
            partial + value * value
        } / Float(hidden.count)
        let scale = 1 / sqrt(variance + epsilon)
        let basePointer = weightTensor.buffer.contents().advanced(by: weightTensor.offset)

        switch weightTensor.dtype {
        case .float16:
            let pointer = basePointer.bindMemory(to: UInt16.self, capacity: hidden.count)
            return hidden.enumerated().map { index, value in
                value * scale * Float(Float16(bitPattern: pointer[index]))
            }
        case .bfloat16:
            let pointer = basePointer.bindMemory(to: UInt16.self, capacity: hidden.count)
            return hidden.enumerated().map { index, value in
                value * scale * Float(bitPattern: UInt32(pointer[index]) << 16)
            }
        case .float32:
            let pointer = basePointer.bindMemory(to: Float.self, capacity: hidden.count)
            return hidden.enumerated().map { index, value in
                value * scale * pointer[index]
            }
        case .quantized:
            Issue.record("Quantized final norm tensor is unsupported in this diagnostic")
            return hidden
        }
    }
}
