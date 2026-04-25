import Foundation
import Testing
@testable import MetalCompiler

#if ENABLE_METAL_PROBES
@Suite("Qwen35 Prompt Ingestion", .serialized)
struct Qwen35PromptIngestionTests {

    @Test("BF16 prefill ingestion matches decode-equivalent ingestion")
    func bf16PrefillIngestionMatchesDecodeEquivalentTrace() throws {
        guard let bundlePath = try Self.resolveBundle(repoName: "mlx-community--Qwen3.5-0.8B-MLX-bf16") else {
            Issue.record("BF16 bundle not cached. Expected ~/.cache/huggingface/hub/models--mlx-community--Qwen3.5-0.8B-MLX-bf16")
            return
        }

        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }
        BenchmarkSupport.settleGPU()

        let promptTokens = Self.japanPromptTokens
        try autoreleasepool {
            let (model, _, _) = try BenchmarkSupport.setupFromBundle(
                bundlePath: bundlePath,
                inferencePolicy: InferencePolicy(maximumSequenceLength: 64)
            )

            let prefillPlan = try #require(model.prefillPlan)
            #expect(prefillPlan.requiresSequentialPromptIngestion)

            var prefillModel = model
            prefillModel.resetState()
            let prefillTrace = BenchmarkSupport.decodeTokenTrace(
                model: &prefillModel,
                promptTokens: promptTokens,
                decodeSteps: 12
            )

            var sequentialModel = model
            sequentialModel.prefillPlan = nil
            sequentialModel.resetState()
            let sequentialTrace = BenchmarkSupport.decodeTokenTrace(
                model: &sequentialModel,
                promptTokens: promptTokens,
                decodeSteps: 12
            )

            #expect(
                prefillTrace == sequentialTrace,
                "BF16 prefill transfer must match decode-equivalent prompt ingestion. prefill=\(prefillTrace), sequential=\(sequentialTrace)"
            )
        }
    }

    @Test("Q3 prefill ingestion matches decode-equivalent ingestion")
    func q3PrefillIngestionMatchesDecodeEquivalentTrace() throws {
        guard let bundlePath = try Self.resolveBundle(repoName: "mlx-community--Qwen3.5-0.8B-3bit") else {
            Issue.record("Q3 bundle not cached. Expected ~/.cache/huggingface/hub/models--mlx-community--Qwen3.5-0.8B-3bit")
            return
        }

        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }
        BenchmarkSupport.settleGPU()

        let promptTokens = Self.japanPromptTokens

        try autoreleasepool {
            let (model, _, _) = try BenchmarkSupport.setupFromBundle(
                bundlePath: bundlePath,
                inferencePolicy: InferencePolicy(maximumSequenceLength: 64)
            )

            let prefillPlan = try #require(model.prefillPlan)
            #expect(prefillPlan.requiresSequentialPromptIngestion)

            var prefillModel = model
            prefillModel.resetState()
            let prefillTrace = BenchmarkSupport.decodeTokenTrace(
                model: &prefillModel,
                promptTokens: promptTokens,
                decodeSteps: 8
            )

            var sequentialModel = model
            sequentialModel.prefillPlan = nil
            sequentialModel.resetState()
            let sequentialTrace = BenchmarkSupport.decodeTokenTrace(
                model: &sequentialModel,
                promptTokens: promptTokens,
                decodeSteps: 8
            )

            #expect(
                Array(prefillTrace.prefix(9)) == Array(sequentialTrace.prefix(9)),
                "Q3 prompt ingestion must stay decode-equivalent. prefill=\(prefillTrace), sequential=\(sequentialTrace)"
            )
        }
    }

    private static let japanPromptTokens: [Int32] = [
        248045, 846, 198, 3710, 369, 279, 6511, 314, 6124, 30,
        248046, 198, 248045, 74455, 198, 248068, 271, 248069, 271,
    ]

    private static func resolveBundle(repoName: String) throws -> String? {
        let hubRoot = NSString(string: "~/.cache/huggingface/hub").expandingTildeInPath
        let snapshotsDir = "\(hubRoot)/models--\(repoName)/snapshots"
        guard FileManager.default.fileExists(atPath: snapshotsDir) else { return nil }
        let entries = try FileManager.default.contentsOfDirectory(atPath: snapshotsDir).sorted()
        for entry in entries {
            let candidate = "\(snapshotsDir)/\(entry)"
            if FileManager.default.fileExists(atPath: "\(candidate)/config.json") {
                return candidate
            }
        }
        return nil
    }
}
#endif
