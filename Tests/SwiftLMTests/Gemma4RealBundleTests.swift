import Foundation
import Testing
@testable import SwiftLM
#if ENABLE_METAL_PROBES
import MetalCompiler
#endif

@Suite("Gemma4 Real Bundle", .serialized)
struct Gemma4RealBundleTests {
    @Test("Real Gemma4 bundle starts a simple factual answer with Tokyo", .timeLimit(.minutes(10)))
    func realBundleTextPrompt() async throws {
        guard let container = try await Gemma4TestSupport.realGemma4Container() else {
            print("[Skip] No local official Gemma4 E2B snapshot found")
            return
        }

        container.resetCaches()
        let prepared = try await container.prepare(
            input: ModelInput(prompt: RealOutputAssertionSupport.strictCapitalPrompt)
        )
        print("[Gemma4 rendered text]")
        print(prepared.renderedText)
        print("[Gemma4 token count] \(prepared.tokenIDs.count)")
        let prompt = try container.makeExecutablePrompt(from: prepared)
        if let gemma4PromptContext = prompt.gemma4PromptContext {
            if let lastPromptEmbedding = gemma4PromptContext.promptEmbeddings.last {
                print("[Gemma4 prompt embedding prefix] \(Array(lastPromptEmbedding.prefix(16)))")
            }
            if let layer0Inputs = gemma4PromptContext.perLayerInputs.first,
               let lastLayer0Input = layer0Inputs.last {
                print("[Gemma4 layer0 per-layer input prefix] \(Array(lastLayer0Input.prefix(16)))")
            }
        }
        let outputHeadDiagnostics = try container.debugPrefillOutputHeadDiagnostics(
            prompt: prompt,
            topK: 10
        )
        print("[Gemma4 output-head input layout]")
        print(
            "  buffer=\(outputHeadDiagnostics.inputLayout.bufferKind) "
                + "offset=\(outputHeadDiagnostics.inputLayout.offset) "
                + "length=\(outputHeadDiagnostics.inputLayout.bufferLength) "
                + "hiddenCount=\(outputHeadDiagnostics.inputLayout.hiddenCount) "
                + "readableCount=\(outputHeadDiagnostics.inputLayout.readableCount)"
        )
        print("[Gemma4 transfer layout]")
        print(
            "  source=\(outputHeadDiagnostics.transferLayout.sourceBufferKind) "
                + "offset=\(outputHeadDiagnostics.transferLayout.sourceOffset) "
                + "length=\(outputHeadDiagnostics.transferLayout.sourceBufferLength) "
                + "readable=\(outputHeadDiagnostics.transferLayout.sourceReadableCount)"
        )
        print(
            "  destination=\(outputHeadDiagnostics.transferLayout.destinationBufferKind) "
                + "offset=\(outputHeadDiagnostics.transferLayout.destinationOffset) "
                + "length=\(outputHeadDiagnostics.transferLayout.destinationBufferLength) "
                + "readable=\(outputHeadDiagnostics.transferLayout.destinationReadableCount) "
                + "transferCount=\(outputHeadDiagnostics.transferLayout.transferCount)"
        )
        let transferMaxDiff = zip(
            outputHeadDiagnostics.transferSource,
            outputHeadDiagnostics.transferDestination
        ).reduce(Float.zero) { partial, pair in
            max(partial, abs(pair.0 - pair.1))
        }
        print("[Gemma4 transfer vectors]")
        print("  source=\(Array(outputHeadDiagnostics.transferSource.prefix(8)))")
        print("  destination=\(Array(outputHeadDiagnostics.transferDestination.prefix(8)))")
        print("  maxDiff=\(transferMaxDiff)")
        print("[Gemma4 GPU top logits]")
        for entry in outputHeadDiagnostics.topLogits {
            print(
                "  id=\(entry.tokenID) logit=\(entry.logit) "
                    + "token=\(String(reflecting: entry.decoded))"
            )
        }
        let selectedLayers = Set([0, 4, 14, 34])
        let stepIndices: [Int: Int] = container.debugPrefillStepSummaries().reduce(into: [:]) {
            partialResult, summary in
            guard let layerIndex = summary.layerIndex, selectedLayers.contains(layerIndex) else {
                return
            }
            partialResult[layerIndex] = summary.index
        }
        let hiddenSnapshots = try container.debugPrefillLastTokenHiddenSnapshots(
            prompt: prompt,
            stepIndices: Set(stepIndices.values)
        )
        for layerIndex in selectedLayers.sorted() {
            guard let stepIndex = stepIndices[layerIndex],
                  let snapshot = hiddenSnapshots[stepIndex] else {
                continue
            }
            print("[Gemma4 layer \(layerIndex) hidden prefix] \(Array(snapshot.prefix(16)))")
        }
        let layer4Steps = container.debugPrefillStepSummaries().filter { summary in
            summary.layerIndex == 4
        }
        let layer4StepIndices = Set(layer4Steps.map(\.index))
        let layer4StepDescriptions = container.debugDescribePrefillSteps(indices: layer4StepIndices)
        print("[Gemma4 layer 4 steps]")
        for summary in layer4Steps {
            print("  step=\(summary.index) kernel=\(summary.kernelName)")
            if let description = layer4StepDescriptions[summary.index] {
                print("    \(description)")
            }
        }
#if ENABLE_METAL_PROBES
        let lastRowIndex = prompt.tokenIDs.count - 1
        let probeValues = try container.debugActualPrefillBindingProbes(
            prompt: prompt,
            stepIndex: 143,
            probes: [
                .init(
                    label: "step143.hidden.before",
                    stepIndex: 143,
                    bindingIndex: 0,
                    phase: .beforeStep,
                    rowIndex: lastRowIndex,
                    rowStride: 1536,
                    count: 8,
                    precision: .float32
                ),
                .init(
                    label: "step143.residual.before",
                    stepIndex: 143,
                    bindingIndex: 1,
                    phase: .beforeStep,
                    rowIndex: lastRowIndex,
                    rowStride: 1536,
                    count: 8,
                    precision: .float32
                ),
                .init(
                    label: "step143.hidden.after",
                    stepIndex: 143,
                    bindingIndex: 0,
                    phase: .afterStep,
                    rowIndex: lastRowIndex,
                    rowStride: 1536,
                    count: 8,
                    precision: .float32
                ),
            ]
        )
        print("[Gemma4 layer 4 probes]")
        for key in probeValues.keys.sorted() {
            print("  \(key)=\(probeValues[key, default: []])")
        }
        try printActualLayerOutputs(
            container: container,
            prompt: prompt,
            selectedLayers: selectedLayers
        )
#endif
        let comparison = try RealOutputAssertionSupport.assertGreedyDirectMatchesPromptState(
            container: container,
            prompt: prompt,
            label: "Gemma4 real greedy"
        )
        RealOutputAssertionSupport.assertStartsWithTokyo(
            comparison.directText,
            label: "Gemma4 real greedy"
        )
    }

    @Test("Real Gemma4 bundle can prepare and generate from an image prompt", .timeLimit(.minutes(10)))
    func realBundleImagePrompt() async throws {
        guard let container = try await Gemma4TestSupport.realGemma4Container() else {
            print("[Skip] No local official Gemma4 E2B snapshot found")
            return
        }
        guard container.configuration.executionCapabilities.supportsImageExecution else {
            print("[Skip] Loaded Gemma4 bundle does not execute image prompts")
            return
        }

        container.resetCaches()
        let imageData = try TestImageFixtures.makeOnePixelPNGData()
        let prepared = try await container.prepare(
            input: ModelInput(chat: [
                .user([
                    .text("Describe the image"),
                    .image(InputImage(data: imageData, mimeType: "image/png")),
                ])
            ])
        )
        let prompt = try container.makeExecutablePrompt(from: prepared)
        let stream = try container.generate(
            prompt: prompt,
            parameters: GenerateParameters(maxTokens: 1, streamChunkTokenCount: 1)
        )
        var sawOutput = false
        for await generation in stream {
            if generation.chunk != nil || generation.info != nil {
                sawOutput = true
                break
            }
        }

        #expect(prompt.gemma4PromptContext != nil)
        #expect(sawOutput)
    }

#if ENABLE_METAL_PROBES
    private func printActualLayerOutputs(
        container: ModelContainer,
        prompt: ExecutablePrompt,
        selectedLayers: Set<Int>
    ) throws {
        let summaries = container.debugPrefillStepSummaries()
        let finalStepByLayer = summaries.reduce(into: [Int: Int]()) { partialResult, summary in
            guard let layerIndex = summary.layerIndex, selectedLayers.contains(layerIndex) else {
                return
            }
            if summary.kernelName == "scalar_multiply_seq_bf16" {
                partialResult[layerIndex] = summary.index
                return
            }
            if summary.kernelName == "residual_add_inplace_seq_f32",
               partialResult[layerIndex] == nil {
                partialResult[layerIndex] = summary.index
            }
        }
        let lastRowIndex = prompt.tokenIDs.count - 1
        for layerIndex in selectedLayers.sorted() {
            guard let stepIndex = finalStepByLayer[layerIndex] else {
                continue
            }
            let label = "layer-\(layerIndex)-hidden-after"
            let values = try container.debugActualPrefillBindingProbes(
                prompt: prompt,
                stepIndex: stepIndex,
                probes: [
                    .init(
                        label: label,
                        stepIndex: stepIndex,
                        bindingIndex: 0,
                        phase: .afterStep,
                        rowIndex: lastRowIndex,
                        rowStride: 1536,
                        count: 16,
                        precision: .float32
                    )
                ]
            )[label, default: []]
            let norm = sqrt(values.reduce(Float.zero) { $0 + $1 * $1 })
            print("[Gemma4 actual layer \(layerIndex) hidden prefix] \(values)")
            print("[Gemma4 actual layer \(layerIndex) hidden probed-prefix norm] \(norm)")
        }
    }
#endif
}
