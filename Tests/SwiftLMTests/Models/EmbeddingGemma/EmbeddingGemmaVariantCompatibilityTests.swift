import Foundation
import Testing
@testable import SwiftLM
@testable import MetalCompiler

@Suite("EmbeddingGemma Variant Compatibility", .serialized)
struct EmbeddingGemmaVariantCompatibilityTests {
    @Test("Community EmbeddingGemma 4bit returns normalized retrieval-friendly embeddings", .timeLimit(.minutes(10)))
    func community4BitCompatibility() async throws {
        try await assertVariantCompatibility(for: .community4Bit)
    }

    @Test("Community EmbeddingGemma bf16 returns normalized retrieval-friendly embeddings", .timeLimit(.minutes(10)))
    func communityBF16Compatibility() async throws {
        try await assertVariantCompatibility(for: .communityBF16)
    }

    private func assertVariantCompatibility(for variant: EmbeddingGemmaVariant) async throws {
        guard let container = try await EmbeddingGemmaTestSupport.realEmbeddingGemmaContainer(variant: variant) else {
            print("[Skip] No \(variant.rawValue) EmbeddingGemma snapshot found")
            return
        }

        let source = try EmbeddingGemmaTestSupport.sourceDescription(for: variant)
        let quantizationSummary = container.prefillPlan.quantizationSummary(limit: 6)
        let embeddingKernelFamilies = Set(
            container.prefillPlan.quantizationKernelFamilies(path: "embeddingLookup")
        )
        let query = try container.embed("swift metal inference", promptName: "query")
        let relevant = try container.embed(
            "SwiftLM performs Metal inference on Apple Silicon.",
            promptName: "document"
        )
        let unrelated = try container.embed(
            "A ripe banana is yellow and curved.",
            promptName: "document"
        )

        let queryNorm = l2Norm(query)
        let relevantNorm = l2Norm(relevant)
        let unrelatedNorm = l2Norm(unrelated)
        let relevantScore = cosineSimilarity(query, relevant)
        let unrelatedScore = cosineSimilarity(query, unrelated)

        print(
            "[EmbeddingGemma.Smoke] variant=\(variant.rawValue) source=\(source) "
                + "embeddingEntries=\(embeddingKernelFamilies.count) "
                + "embeddingKernels=\(embeddingKernelFamilies.sorted().joined(separator: ",")) "
                + "queryNorm=\(String(format: "%.4f", queryNorm)) "
                + "relevantNorm=\(String(format: "%.4f", relevantNorm)) "
                + "unrelatedNorm=\(String(format: "%.4f", unrelatedNorm)) "
                + "relevantScore=\(String(format: "%.4f", relevantScore)) "
                + "unrelatedScore=\(String(format: "%.4f", unrelatedScore)) "
                + "quantization=\(quantizationSummary.replacingOccurrences(of: "\n", with: " | "))"
        )

        #expect(query.count == 768)
        #expect(relevant.count == 768)
        #expect(unrelated.count == 768)
        #expect(abs(queryNorm - 1) < 0.01)
        #expect(abs(relevantNorm - 1) < 0.01)
        #expect(abs(unrelatedNorm - 1) < 0.01)
        #expect(relevantScore.isFinite)
        #expect(unrelatedScore.isFinite)
        #expect(relevantScore > unrelatedScore)
        #expect(embeddingKernelFamilies.isEmpty == false)

        switch variant {
        case .community4Bit:
            #expect(
                embeddingKernelFamilies.contains("q4G64EmbeddingLookup")
                    || embeddingKernelFamilies.contains("q4G128EmbeddingLookup")
            )
        case .official:
            break
        case .communityBF16:
            #expect(
                embeddingKernelFamilies.contains("denseEmbeddingLookup")
                    || embeddingKernelFamilies.contains("bf16EmbeddingLookup")
                    || embeddingKernelFamilies.contains("fp32EmbeddingLookup")
            )
        }
    }

    private func l2Norm(_ values: [Float]) -> Float {
        values.reduce(into: Float.zero) { partial, value in
            partial += value * value
        }.squareRoot()
    }

    private func cosineSimilarity(_ lhs: [Float], _ rhs: [Float]) -> Float {
        precondition(lhs.count == rhs.count)
        return zip(lhs, rhs).reduce(into: Float.zero) { partial, pair in
            partial += pair.0 * pair.1
        }
    }
}
