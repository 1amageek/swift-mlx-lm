import Foundation
import Metal
import Testing
@testable import SwiftLM
@testable import MetalCompiler

/// Test that verifies the post-processing kernel works when given
/// the exact same hidden states that the real model produces,
/// using a fresh MetalSubmissionContext.
@Suite("Embedding PostProcess Isolated", .serialized)
struct EmbeddingPostProcessIsolatedTests {

    @Test("Post-process real hidden states with fresh submission context", .timeLimit(.minutes(5)))
    func postProcessRealHiddenStates() async throws {
        guard let container = try await EmbeddingGemmaTestSupport.realEmbeddingGemmaContainer(
            variant: .community4Bit
        ) else {
            print("[PostProcessIsolated.Skip] No Q4 EmbeddingGemma model found")
            return
        }

        guard let postProcessor = container.postProcessor else {
            print("[PostProcessIsolated.Skip] No post-processor available")
            return
        }

        let input = TextEmbeddingInput("SwiftLM performs text embeddings.", promptName: "document")
        let prepared = try container.runtime.prepare(
            text: input.text,
            promptName: input.promptName,
            tokenizer: container.tokenizer
        )
        let tokenIDs = prepared.tokenIDs.map(Int32.init)

        // Step 1: Get hidden states via proven CPU path
        let isolatedPlan = try container.prefillPlan.makeRuntimeIsolatedCopy(device: container.device)
        var prefillModel = try MetalPrefillModel(plan: isolatedPlan, device: container.device)
        let hiddenStates = try prefillModel.finalHiddenStates(tokens: tokenIDs)

        let seqLen = hiddenStates.count
        let hiddenDim = hiddenStates.first?.count ?? 0
        print("[PostProcessIsolated] seqLen=\(seqLen) hiddenDim=\(hiddenDim)")
        #expect(seqLen > 0 && hiddenDim > 0)

        // Step 2: Write hidden states into a fresh shared buffer (exactly like isolation test)
        let device = container.device
        let bufferSize = seqLen * hiddenDim * MemoryLayout<Float>.stride
        guard let hiddenBuffer = device.makeBuffer(length: bufferSize, options: .storageModeShared) else {
            throw MetalCompilerError.deviceSetupFailed("Failed to allocate hidden buffer")
        }
        let hiddenPtr = hiddenBuffer.contents().bindMemory(to: Float.self, capacity: seqLen * hiddenDim)
        for (row, rowData) in hiddenStates.enumerated() {
            for (col, value) in rowData.enumerated() {
                hiddenPtr[row * hiddenDim + col] = value
            }
        }

        // Verify the data is there
        let row0Norm = sqrt(hiddenStates[0].reduce(0) { $0 + $1 * $1 })
        print("[PostProcessIsolated] row0Norm=\(String(format: "%.4f", row0Norm))")

        // Step 3: Create fresh workspace and fresh submission context
        let workspace = try postProcessor.makeWorkspace(device: device)
        var submission = try MetalSubmissionContext(device: device)
        let residency = try MetalResidencyLease.required(
            device: device,
            label: "test.postprocess",
            buffers: [hiddenBuffer] + workspace.residencyBuffers + postProcessor.residencyBuffers
        )
        residency.add(to: submission.queue)

        // Step 4: Dispatch post-processing on fresh context
        let hiddenRowStride = hiddenDim * MemoryLayout<Float>.stride
        try submission.withCompute(ephemeralResidency: residency) { encoder, argumentTable in
            workspace.encode(
                encoder: encoder,
                argumentTable: argumentTable,
                hiddenBuffer: hiddenBuffer,
                hiddenBaseOffset: 0,
                hiddenRowStride: hiddenRowStride,
                hiddenDimension: hiddenDim,
                sequenceLength: seqLen,
                promptTokenCount: prepared.promptTokenCount
            )
        }

        // Step 5: Read result and verify
        let gpuResult = workspace.readResult(hiddenDimension: hiddenDim)
        let gpuNorm = sqrt(gpuResult.reduce(0) { $0 + $1 * $1 })
        let gpuHead = gpuResult.prefix(8).map { String(format: "%.6f", $0) }.joined(separator: ", ")
        print("[PostProcessIsolated.FreshCtx] norm=\(String(format: "%.6f", gpuNorm)) head=[\(gpuHead)]")
        print("[PostProcessIsolated.FreshCtx.Buffers]\n\(workspace.debugDumpBuffers(hiddenDimension: hiddenDim))")

        #expect(gpuNorm > 0.1, "Fresh context GPU norm is zero")

        // Step 6: Compare with CPU path
        let cpuContainer = TextEmbeddingContainer(
            prefillPlan: container.prefillPlan,
            device: container.device,
            tokenizer: container.tokenizer,
            runtime: container.runtime,
            configuration: container.modelConfiguration,
            postProcessor: nil
        )
        let cpuContext = try TextEmbeddingContext(cpuContainer)
        let cpuResult = try cpuContext.embed(input)
        let cpuNorm = sqrt(cpuResult.reduce(0) { $0 + $1 * $1 })
        let cpuHead = cpuResult.prefix(8).map { String(format: "%.6f", $0) }.joined(separator: ", ")
        print("[PostProcessIsolated.CPU] norm=\(String(format: "%.6f", cpuNorm)) head=[\(cpuHead)]")

        if gpuNorm > 0 && cpuNorm > 0 {
            var dot: Float = 0
            for i in 0..<min(gpuResult.count, cpuResult.count) {
                dot += gpuResult[i] * cpuResult[i]
            }
            let cosine = dot / (gpuNorm * cpuNorm)
            print("[PostProcessIsolated] cosine=\(String(format: "%.6f", cosine))")
            #expect(cosine > 0.99, "Cosine too low: \(cosine)")
        }
    }

    @Test("Post-process using MetalPrefillModel submission context", .timeLimit(.minutes(5)))
    func postProcessViaPrefillModelContext() async throws {
        guard let container = try await EmbeddingGemmaTestSupport.realEmbeddingGemmaContainer(
            variant: .community4Bit
        ) else {
            print("[PostProcessViaModel.Skip] No Q4 EmbeddingGemma model found")
            return
        }

        guard let postProcessor = container.postProcessor else {
            print("[PostProcessViaModel.Skip] No post-processor available")
            return
        }

        let input = TextEmbeddingInput("SwiftLM performs text embeddings.", promptName: "document")
        let prepared = try container.runtime.prepare(
            text: input.text,
            promptName: input.promptName,
            tokenizer: container.tokenizer
        )
        let tokenIDs = prepared.tokenIDs.map(Int32.init)

        // Step 1: Run full captureEmbeddingVector GPU path
        let workspace = try postProcessor.makeWorkspace(device: container.device)
        let isolatedPlan = try container.prefillPlan.makeRuntimeIsolatedCopy(device: container.device)
        var prefillModel = try MetalPrefillModel(plan: isolatedPlan, device: container.device)
        let gpuResult = try prefillModel.captureEmbeddingVector(
            tokens: tokenIDs,
            workspace: workspace,
            promptTokenCount: prepared.promptTokenCount
        )
        let gpuNorm = sqrt(gpuResult.reduce(0) { $0 + $1 * $1 })
        let gpuHead = gpuResult.prefix(8).map { String(format: "%.6f", $0) }.joined(separator: ", ")
        print("[PostProcessViaModel] captureEmbeddingVector norm=\(String(format: "%.6f", gpuNorm)) head=[\(gpuHead)]")

        // Step 2: Now get hidden states via CPU path using the SAME prefillModel
        let hiddenStates = try prefillModel.finalHiddenStates(tokens: tokenIDs)
        let seqLen = hiddenStates.count
        let hiddenDim = hiddenStates.first?.count ?? 0
        let row0Norm = sqrt(hiddenStates[0].reduce(0) { $0 + $1 * $1 })
        print("[PostProcessViaModel] finalHiddenStates: seqLen=\(seqLen) hiddenDim=\(hiddenDim) row0Norm=\(String(format: "%.4f", row0Norm))")

        // Step 3: Write hidden states to a fresh buffer
        let device = container.device
        let bufferSize = seqLen * hiddenDim * MemoryLayout<Float>.stride
        guard let freshHiddenBuffer = device.makeBuffer(length: bufferSize, options: .storageModeShared) else {
            throw MetalCompilerError.deviceSetupFailed("Failed to allocate hidden buffer")
        }
        let hiddenPtr = freshHiddenBuffer.contents().bindMemory(to: Float.self, capacity: seqLen * hiddenDim)
        for (row, rowData) in hiddenStates.enumerated() {
            for (col, value) in rowData.enumerated() {
                hiddenPtr[row * hiddenDim + col] = value
            }
        }

        // Step 4: Create a FRESH workspace and fresh submission
        let freshWorkspace = try postProcessor.makeWorkspace(device: device)
        var freshSubmission = try MetalSubmissionContext(device: device)
        let residency = try MetalResidencyLease.required(
            device: device,
            label: "test.fresh",
            buffers: [freshHiddenBuffer] + freshWorkspace.residencyBuffers + postProcessor.residencyBuffers
        )
        residency.add(to: freshSubmission.queue)

        let hiddenRowStride = hiddenDim * MemoryLayout<Float>.stride
        try freshSubmission.withCompute(ephemeralResidency: residency) { encoder, argumentTable in
            freshWorkspace.encode(
                encoder: encoder,
                argumentTable: argumentTable,
                hiddenBuffer: freshHiddenBuffer,
                hiddenBaseOffset: 0,
                hiddenRowStride: hiddenRowStride,
                hiddenDimension: hiddenDim,
                sequenceLength: seqLen,
                promptTokenCount: prepared.promptTokenCount
            )
        }

        let freshResult = freshWorkspace.readResult(hiddenDimension: hiddenDim)
        let freshNorm = sqrt(freshResult.reduce(0) { $0 + $1 * $1 })
        let freshHead = freshResult.prefix(8).map { String(format: "%.6f", $0) }.joined(separator: ", ")
        print("[PostProcessViaModel.FreshCtx] norm=\(String(format: "%.6f", freshNorm)) head=[\(freshHead)]")

        #expect(freshNorm > 0.1, "Fresh context should produce non-zero output")
        #expect(gpuNorm > 0.1, "captureEmbeddingVector should produce non-zero output")
    }
}
