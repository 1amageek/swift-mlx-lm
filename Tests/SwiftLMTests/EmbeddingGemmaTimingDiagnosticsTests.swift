import Foundation
import Testing
@testable import SwiftLM
@testable import MetalCompiler

@Suite("EmbeddingGemma Timing Diagnostics", .serialized)
struct EmbeddingGemmaTimingDiagnosticsTests {

    @Test("Per-step timing breakdown of embed()", .timeLimit(.minutes(5)))
    func perStepTimingBreakdown() async throws {
        guard let container = try await EmbeddingGemmaTestSupport.realEmbeddingGemmaContainer(
            variant: .community4Bit
        ) else {
            print("[TimingDiag.Skip] No Q4 EmbeddingGemma model found")
            return
        }

        let context = try TextEmbeddingContext(container)
        let tokenizer = container.tokenizer
        let runtime = container.runtime
        let gpuPath = context.debugWorkspace != nil ? "gpu" : "cpuFallback"

        let inputs: [(text: String, promptName: String?)] = [
            ("SwiftLM performs text embeddings on Apple Silicon with Metal.", "document"),
            ("quantized embedding throughput on macOS", "query"),
            ("Sentence-transformer style retrieval depends on stable normalized vectors.", "document"),
        ]

        let clock = ContinuousClock()

        // Warmup: run full GPU pipeline once for each input
        for input in inputs {
            try autoreleasepool {
                let _ = try context.embed(
                    TextEmbeddingInput(input.text, promptName: input.promptName)
                )
            }
        }

        let stepCount = context.debugPrefillPlan.steps.count
        print("[TimingDiag.Config] variant=Q4 inputs=\(inputs.count) stepCount=\(stepCount) path=\(gpuPath)")

        // Measure each input independently
        for (index, input) in inputs.enumerated() {
            try autoreleasepool {
                let embeddingInput = TextEmbeddingInput(input.text, promptName: input.promptName)

                // Step 1: Tokenization only
                let tokenizeDuration = try clock.measure {
                    let _ = try runtime.prepare(
                        text: input.text,
                        promptName: input.promptName,
                        tokenizer: tokenizer
                    )
                }

                // Step 2: Full pipeline (tokenize + GPU prefill + post-process)
                var embedding: [Float]!
                let e2eDuration = try clock.measure {
                    embedding = try context.embed(embeddingInput)
                }

                let tokenizeMs = durationToMs(tokenizeDuration)
                let e2eMs = durationToMs(e2eDuration)
                let gpuMs = max(e2eMs - tokenizeMs, 0)

                let tokenCount = try runtime.prepare(
                    text: input.text,
                    promptName: input.promptName,
                    tokenizer: tokenizer
                ).tokenIDs.count

                print(
                    "[TimingDiag.Input\(index)] tokens=\(tokenCount) "
                        + "tokenize=\(String(format: "%.3f", tokenizeMs))ms "
                        + "gpu=\(String(format: "%.3f", gpuMs))ms "
                        + "e2e=\(String(format: "%.3f", e2eMs))ms "
                        + "dim=\(embedding.count)"
                )

                #expect(embedding.count == 768)
            }
        }

        // Measure averages over multiple iterations
        let iterations = 5
        var totalTokenize = Duration.zero
        var totalE2E = Duration.zero
        let totalEmbeddings = iterations * inputs.count

        for _ in 0..<iterations {
            for input in inputs {
                try autoreleasepool {
                    let embeddingInput = TextEmbeddingInput(input.text, promptName: input.promptName)

                    let t0 = clock.now
                    let _ = try runtime.prepare(
                        text: input.text,
                        promptName: input.promptName,
                        tokenizer: tokenizer
                    )
                    let t1 = clock.now
                    totalTokenize += t1 - t0

                    let t2 = clock.now
                    let _ = try context.embed(embeddingInput)
                    let t3 = clock.now
                    totalE2E += t3 - t2
                }
            }
        }

        let avgTokenize = durationToMs(totalTokenize) / Double(totalEmbeddings)
        let avgE2E = durationToMs(totalE2E) / Double(totalEmbeddings)
        let avgGpu = max(avgE2E - avgTokenize, 0)
        let embPerSec = 1000.0 / avgE2E

        print(
            "[TimingDiag.Average] iterations=\(iterations) embeddings=\(totalEmbeddings) "
                + "avgTokenize=\(String(format: "%.3f", avgTokenize))ms "
                + "avgGpu=\(String(format: "%.3f", avgGpu))ms "
                + "avgE2E=\(String(format: "%.3f", avgE2E))ms "
                + "embPerSec=\(String(format: "%.3f", embPerSec))"
        )

        print(
            "[TimingDiag.Breakdown] "
                + "tokenize=\(String(format: "%.1f", avgTokenize / avgE2E * 100))% "
                + "gpu=\(String(format: "%.1f", avgGpu / avgE2E * 100))%"
        )

        #expect(avgE2E > 0)
    }

    private func durationToMs(_ duration: Duration) -> Double {
        Double(duration.components.seconds) * 1000.0
            + Double(duration.components.attoseconds) / 1_000_000_000_000_000
    }
}
