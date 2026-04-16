import Foundation
import Testing
@testable import SwiftLM
@testable import MetalCompiler

@Suite("EmbeddingGemma Performance", .serialized)
struct EmbeddingGemmaPerformanceTests {
    @Test("Community EmbeddingGemma 4bit reports reusable-context embedding throughput", .timeLimit(.minutes(10)))
    func community4BitEmbeddingGemmaThroughput() async throws {
        try await assertThroughput(for: .community4Bit)
    }

    @Test("Community EmbeddingGemma bf16 reports reusable-context embedding throughput", .timeLimit(.minutes(10)))
    func communityBF16EmbeddingGemmaThroughput() async throws {
        try await assertThroughput(for: .communityBF16)
    }

    private func benchmarkInputs() -> [TextEmbeddingInput] {
        [
            TextEmbeddingInput(
                "SwiftLM performs text embeddings on Apple Silicon with Metal.",
                promptName: "document"
            ),
            TextEmbeddingInput(
                "quantized embedding throughput on macOS",
                promptName: "query"
            ),
            TextEmbeddingInput(
                "Sentence-transformer style retrieval depends on stable normalized vectors.",
                promptName: "document"
            ),
        ]
    }

    private func benchmarkIterations() -> Int {
        let environment = ProcessInfo.processInfo.environment
        let rawValue = environment["SWIFTLM_EMBEDDING_BENCH_ITERATIONS"] ?? "2"
        guard let iterations = Int(rawValue), iterations > 0 else {
            return 2
        }
        return iterations
    }

    private func summarizeKernels(in container: TextEmbeddingContainer) -> String {
        let labels = container.prefillPlan.steps.compactMap(\.pipeline.label)
        let lookup = container.prefillPlan.quantizationKernelFamilies(path: "embeddingLookup")
        let projections = container.prefillPlan.quantizationKernelFamilies(path: "prefillProjection")
        let summary = container.prefillPlan.quantizationSummary(limit: 6)
            .replacingOccurrences(of: "\n", with: " | ")
        return "steps=\(labels.count),lookupEntries=\(lookup.count),projectionEntries=\(projections.count),lookupKernels=\(lookup.joined(separator: ",")),projectionKernels=\(projections.joined(separator: ",")),summary=\(summary)"
    }

    private func printKernelFrequency(in container: TextEmbeddingContainer) {
        var frequency: [String: Int] = [:]
        for step in container.prefillPlan.steps {
            let name = step.metadata.kernelName ?? step.pipeline.label ?? "<unknown>"
            frequency[name, default: 0] += 1
        }
        let sorted = frequency.sorted { $0.value > $1.value }
        print("[KernelFrequency] total=\(container.prefillPlan.steps.count)")
        for (name, count) in sorted {
            print("  \(count)x \(name)")
        }
    }

    private func assertThroughput(for variant: EmbeddingGemmaVariant) async throws {
        guard let container = try await EmbeddingGemmaTestSupport.realEmbeddingGemmaContainer(variant: variant) else {
            print("[Skip] No \(variant.rawValue) EmbeddingGemma snapshot found")
            return
        }

        let source = try EmbeddingGemmaTestSupport.sourceDescription(for: variant)
        let context = try TextEmbeddingContext(container)
        let inputs = benchmarkInputs()
        let iterations = benchmarkIterations()
        let kernelSummary = summarizeKernels(in: container)
        let rawIterationOverride = ProcessInfo.processInfo.environment["SWIFTLM_EMBEDDING_BENCH_ITERATIONS"] ?? "nil"

        print(
            "[EmbeddingGemma.Perf.Config] variant=\(variant.rawValue) source=\(source) "
                + "rawIterations=\(rawIterationOverride) "
                + "effectiveIterations=\(iterations) "
                + "usesMPP=\(container.prefillPlan.usesMPP) "
                + "kernels=\(kernelSummary)"
        )
        printKernelFrequency(in: container)

        for input in inputs {
            try autoreleasepool {
                let vector = try context.embed(input)
                #expect(vector.count == 768)
            }
        }

        let clock = ContinuousClock()
        var checksum = Float.zero
        let duration = try clock.measure {
            for _ in 0..<iterations {
                for input in inputs {
                    try autoreleasepool {
                        let vector = try context.embed(input)
                        checksum += vector[0]
                    }
                }
            }
        }

        let embeddingCount = iterations * inputs.count
        let seconds = Double(duration.components.seconds)
            + (Double(duration.components.attoseconds) / 1_000_000_000_000_000_000)
        let embeddingsPerSecond = Double(embeddingCount) / seconds
        let secondsString = String(format: "%.3f", seconds)
        let throughputString = String(format: "%.3f", embeddingsPerSecond)
        let checksumString = String(format: "%.6f", checksum)
        print(
            "[EmbeddingGemma.Perf] variant=\(variant.rawValue) source=\(source) " +
                "embeddings=\(embeddingCount) " +
                "seconds=\(secondsString) " +
                "embeddingsPerSecond=\(throughputString) " +
                "checksum=\(checksumString)"
        )

        #expect(checksum.isFinite)
        #expect(embeddingsPerSecond > 0)
    }
}
