import Foundation
import Metal
import Testing
@testable import SwiftLM
@testable import MetalCompiler

@Suite("Embedding GPU vs CPU Parity", .serialized)
struct EmbeddingGpuCpuParityTests {

    @Test("GPU embedding matches CPU embedding for Q4 model", .timeLimit(.minutes(5)))
    func gpuMatchesCpuQ4() async throws {
        guard let container = try await EmbeddingGemmaTestSupport.realEmbeddingGemmaContainer(
            variant: .community4Bit
        ) else {
            print("[GpuCpuParity.Skip] No Q4 EmbeddingGemma model found")
            return
        }

        // Diagnose buffer layout
        let plan = container.prefillPlan
        let fhb = plan.finalHiddenBuffer
        let elementSize = max(plan.buffers.bufferPrecision.byteSize, 1)
        let hiddenDimFromHidden = plan.buffers.hidden.length
            / max(plan.maximumSequenceLength, 1)
            / elementSize
        let hiddenDimFromFHB = fhb.length
            / max(plan.maximumSequenceLength, 1)
            / elementSize
        print(
            "[GpuCpuParity.Config] "
                + "finalHiddenBuffer.storageMode=\(fhb.storageMode.rawValue) "
                + "finalHiddenBuffer.length=\(fhb.length) "
                + "hidden.length=\(plan.buffers.hidden.length) "
                + "scratch.length=\(plan.buffers.scratch.length) "
                + "finalHiddenBaseOffset=\(plan.finalHiddenBaseOffset) "
                + "finalHiddenRowStride=\(plan.finalHiddenRowStride) "
                + "maxSeqLen=\(plan.maximumSequenceLength) "
                + "elementSize=\(elementSize) "
                + "hiddenDimFromHidden=\(hiddenDimFromHidden) "
                + "hiddenDimFromFHB=\(hiddenDimFromFHB) "
                + "slotDimension=\(plan.slotDimension) "
                + "fhbIsHidden=\(fhb === plan.buffers.hidden) "
                + "fhbIsScratch=\(fhb === plan.buffers.scratch)"
        )

        try compareGpuVsCpu(container: container, label: "Q4")
    }

    private func compareGpuVsCpu(container: TextEmbeddingContainer, label: String) throws {
        let inputs: [TextEmbeddingInput] = [
            TextEmbeddingInput("SwiftLM performs text embeddings.", promptName: "document"),
        ]

        // CPU path: create context WITHOUT post-processor
        let cpuContainer = TextEmbeddingContainer(
            prefillPlan: container.prefillPlan,
            device: container.device,
            tokenizer: container.tokenizer,
            runtime: container.runtime,
            configuration: container.modelConfiguration,
            postProcessor: nil  // force CPU path
        )
        let cpuContext = try TextEmbeddingContext(cpuContainer)

        // GPU path: create context WITH post-processor
        let gpuContext = try TextEmbeddingContext(container)

        // Get raw hidden states via a separate prefill model (known working path)
        let isolatedPlan = try container.prefillPlan.makeRuntimeIsolatedCopy(device: container.device)
        var prefillModel = try MetalPrefillModel(plan: isolatedPlan, device: container.device)
        let prepared = try container.runtime.prepare(
            text: inputs[0].text,
            promptName: inputs[0].promptName,
            tokenizer: container.tokenizer
        )
        let tokenIDs = prepared.tokenIDs.map(Int32.init)
        let hiddenStates = try prefillModel.finalHiddenStates(tokens: tokenIDs)
        let firstRow = hiddenStates.first ?? []
        let lastRow = hiddenStates.last ?? []
        let firstRowHead = firstRow.prefix(5).map { String(format: "%.4f", $0) }.joined(separator: ", ")
        let lastRowHead = lastRow.prefix(5).map { String(format: "%.4f", $0) }.joined(separator: ", ")
        let firstRowNorm = sqrt(firstRow.reduce(0) { $0 + $1 * $1 })
        let lastRowNorm = sqrt(lastRow.reduce(0) { $0 + $1 * $1 })
        print(
            "[GpuCpuParity.\(label).Hidden] rows=\(hiddenStates.count) "
                + "dim=\(firstRow.count) "
                + "firstRowNorm=\(String(format: "%.4f", firstRowNorm)) "
                + "lastRowNorm=\(String(format: "%.4f", lastRowNorm)) "
                + "first=[\(firstRowHead)] last=[\(lastRowHead)]"
        )

        // Run GPU embedding once and dump workspace buffers
        let gpuFirstVector = try gpuContext.embed(inputs[0])
        if let ws = gpuContext.debugWorkspace {
            print("[GpuCpuParity.\(label).Buffers]\n\(ws.debugDumpBuffers(hiddenDimension: 768))")
        }

        // Reset GPU context for clean comparison
        let gpuContext2 = try TextEmbeddingContext(container)

        for (index, input) in inputs.enumerated() {
            let cpuVector = try cpuContext.embed(input)
            let gpuVector = try gpuContext2.embed(input)

            #expect(cpuVector.count == gpuVector.count)
            let dimension = cpuVector.count

            let cpuHead = cpuVector.prefix(8).map { String(format: "%.6f", $0) }.joined(separator: ", ")
            let gpuHead = gpuVector.prefix(8).map { String(format: "%.6f", $0) }.joined(separator: ", ")
            let cpuNorm = sqrt(cpuVector.reduce(0) { $0 + $1 * $1 })
            let gpuNorm = sqrt(gpuVector.reduce(0) { $0 + $1 * $1 })

            var dotProduct: Float = 0
            for i in 0..<dimension {
                dotProduct += cpuVector[i] * gpuVector[i]
            }
            let cosine = (cpuNorm > 0 && gpuNorm > 0) ? dotProduct / (cpuNorm * gpuNorm) : 0

            var maxDiff: Float = 0
            for i in 0..<dimension {
                maxDiff = max(maxDiff, abs(cpuVector[i] - gpuVector[i]))
            }

            // Count non-zero elements
            let cpuNonZero = cpuVector.filter { $0 != 0 }.count
            let gpuNonZero = gpuVector.filter { $0 != 0 }.count

            print(
                "[GpuCpuParity.\(label).Input\(index)] dim=\(dimension) "
                    + "cpuNorm=\(String(format: "%.6f", cpuNorm)) "
                    + "gpuNorm=\(String(format: "%.6f", gpuNorm)) "
                    + "cosine=\(String(format: "%.6f", cosine)) "
                    + "maxDiff=\(String(format: "%.6f", maxDiff)) "
                    + "cpuNonZero=\(cpuNonZero) gpuNonZero=\(gpuNonZero)"
            )
            print("[GpuCpuParity.\(label).Input\(index).cpu] \(cpuHead)")
            print("[GpuCpuParity.\(label).Input\(index).gpu] \(gpuHead)")

            #expect(cosine > 0.99, "Cosine similarity too low: \(cosine)")
            #expect(gpuNorm > 0.1, "GPU norm too small: \(gpuNorm)")
        }
    }
}

