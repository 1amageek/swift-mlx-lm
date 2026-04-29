import Foundation
import Testing
@testable import MetalCompiler

@Suite("LFM Sequence Prefill Ingestion", .serialized)
struct LFMSequencePrefillIngestionTests {
    @Test("BF16 stateful sequence prefill matches decode-equivalent ingestion")
    func bf16StatefulSequencePrefillMatchesDecodeEquivalentTrace() throws {
        let bundlePath = BenchmarkSupport.lfmBundlePath
        guard !bundlePath.isEmpty else {
            print("[Skip] LFM2.5-1.2B-Thinking bundle not cached. Run `huggingface-cli download LiquidAI/LFM2.5-1.2B-Thinking`.")
            return
        }

        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }
        BenchmarkSupport.settleGPU()

        let promptTokens: [Int32] = [1, 1, 6, 6423, 708]
        try autoreleasepool {
            let (model, _, _) = try BenchmarkSupport.setupFromBundle(
                bundlePath: bundlePath,
                inferencePolicy: InferencePolicy(maximumSequenceLength: 64)
            )

            let prefillPlan = try #require(model.prefillPlan)
            #expect(!prefillPlan.requiresSequentialPromptIngestion)
            #expect(prefillPlan.sequencePrefillFallbackReason == nil)

            var prefillModel = try Self.makeRuntimeIsolatedModel(from: model)
            prefillModel.resetState()
            let prefillTrace = BenchmarkSupport.decodeTokenTrace(
                model: &prefillModel,
                promptTokens: promptTokens,
                decodeSteps: 8
            )

            var sequentialModel = try Self.makeRuntimeIsolatedModel(from: model)
            sequentialModel.prefillPlan = nil
            sequentialModel.resetState()
            let sequentialTrace = BenchmarkSupport.decodeTokenTrace(
                model: &sequentialModel,
                promptTokens: promptTokens,
                decodeSteps: 8
            )

            #expect(
                prefillTrace == sequentialTrace,
                "LFM BF16 sequence prefill must match decode-equivalent prompt ingestion. prefill=\(prefillTrace), sequential=\(sequentialTrace)"
            )
        }
    }

    private static func makeRuntimeIsolatedModel(from model: MetalInferenceModel) throws -> MetalInferenceModel {
        let isolated = try model.compiledModel.makeRuntimeIsolatedCopy(device: model.device)
        return try MetalInferenceModel(compiledModel: isolated, device: model.device)
    }
}
