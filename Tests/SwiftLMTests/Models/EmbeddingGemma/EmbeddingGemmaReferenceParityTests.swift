import Foundation
import Testing
@testable import SwiftLM

@Suite("EmbeddingGemma Reference Parity", .serialized)
struct EmbeddingGemmaReferenceParityTests {
    @Test("EmbeddingGemma stays close to the provided reference embeddings", .timeLimit(.minutes(10)))
    func referenceParity() async throws {
        guard let snapshot = try EmbeddingGemmaTestSupport.referenceSnapshot() else {
            print("[Skip] No EmbeddingGemma reference snapshot configured")
            return
        }
        guard let container = try await EmbeddingGemmaTestSupport.realEmbeddingGemmaContainer() else {
            print("[Skip] No local or configured EmbeddingGemma snapshot found")
            return
        }

        let thresholds = try TextEmbeddingReferenceSupport.acceptanceThresholds()
        let comparison = try TextEmbeddingReferenceSupport.compare(
            container: container,
            snapshot: snapshot
        )

        print(
            "[EmbeddingRef] dataset=\(comparison.datasetName) "
                + "docs=\(comparison.documentCount) queries=\(comparison.queryCount) "
                + "docCos=\(String(format: "%.3f", comparison.meanDocumentCosine)) "
                + "queryCos=\(String(format: "%.3f", comparison.meanQueryCosine)) "
                + "top1Agree=\(String(format: "%.3f", comparison.top1Agreement)) "
                + "swiftTop1=\(String(format: "%.3f", comparison.swiftMetrics.top1Accuracy)) "
                + "refTop1=\(String(format: "%.3f", comparison.referenceMetrics.top1Accuracy))"
        )
        print(
            "[EmbeddingRef.Acceptance] "
                + "minDocCos=\(String(format: "%.3f", thresholds.minimumMeanDocumentCosine)) "
                + "minQueryCos=\(String(format: "%.3f", thresholds.minimumMeanQueryCosine)) "
                + "minTop1Agree=\(String(format: "%.3f", thresholds.minimumTop1Agreement)) "
                + "maxTop1Delta=\(String(format: "%.3f", thresholds.maximumTop1AccuracyDelta)) "
                + "maxMRRDelta=\(String(format: "%.3f", thresholds.maximumMeanReciprocalRankDelta)) "
                + "maxRecallAt3Delta=\(String(format: "%.3f", thresholds.maximumMeanRecallAt3Delta)) "
                + "maxMarginDelta=\(String(format: "%.3f", thresholds.maximumMeanRelevantMarginDelta))"
        )
        for mismatch in comparison.topDocumentMismatches {
            print(
                "[EmbeddingRef.Mismatch] query=\(mismatch.queryID) "
                    + "referenceTop=\(mismatch.referenceTopDocumentID) "
                    + "swiftTop=\(mismatch.swiftTopDocumentID)"
            )
        }

        #expect(comparison.documentCount == snapshot.dataset.documents.count)
        #expect(comparison.queryCount == snapshot.dataset.queries.count)
        #expect(comparison.meanDocumentCosine.isFinite)
        #expect(comparison.meanQueryCosine.isFinite)
        #expect(comparison.minimumDocumentCosine.isFinite)
        #expect(comparison.minimumQueryCosine.isFinite)
        #expect(comparison.top1Agreement.isFinite)
        #expect(comparison.top1Agreement >= 0)
        #expect(comparison.top1Agreement <= 1)

        #expect(comparison.meanDocumentCosine >= thresholds.minimumMeanDocumentCosine)
        #expect(comparison.meanQueryCosine >= thresholds.minimumMeanQueryCosine)
        #expect(comparison.top1Agreement >= thresholds.minimumTop1Agreement)
        #expect(
            abs(comparison.swiftMetrics.top1Accuracy - comparison.referenceMetrics.top1Accuracy)
                <= thresholds.maximumTop1AccuracyDelta
        )
        #expect(
            abs(comparison.swiftMetrics.meanReciprocalRank - comparison.referenceMetrics.meanReciprocalRank)
                <= thresholds.maximumMeanReciprocalRankDelta
        )
        #expect(
            abs(comparison.swiftMetrics.meanRecallAt3 - comparison.referenceMetrics.meanRecallAt3)
                <= thresholds.maximumMeanRecallAt3Delta
        )
        #expect(
            abs(comparison.swiftMetrics.meanRelevantMargin - comparison.referenceMetrics.meanRelevantMargin)
                <= thresholds.maximumMeanRelevantMarginDelta
        )
    }
}
