import Foundation
import Testing
@testable import SwiftLM

@Suite("EmbeddingGemma Retrieval Evaluation", .serialized)
struct EmbeddingGemmaRetrievalEvaluationTests {
    @Test("EmbeddingGemma retrieval evaluation reports ranked retrieval metrics", .timeLimit(.minutes(10)))
    func retrievalEvaluation() async throws {
        guard let container = try await EmbeddingGemmaTestSupport.realEmbeddingGemmaContainer() else {
            print("[Skip] No local or configured EmbeddingGemma snapshot found")
            return
        }

        let dataset = try TextEmbeddingEvaluationSupport.loadDataset()
        let thresholds = try TextEmbeddingEvaluationSupport.acceptanceThresholds()
        let metrics = try TextEmbeddingEvaluationSupport.evaluate(
            container: container,
            dataset: dataset,
            queryPromptName: dataset.queryPromptName ?? "query",
            documentPromptName: dataset.documentPromptName ?? "document"
        )

        print(
            "[EmbeddingEval] dataset=\(metrics.datasetName) "
                + "queries=\(metrics.queryCount) docs=\(metrics.documentCount) "
                + "top1=\(String(format: "%.3f", metrics.top1Accuracy)) "
                + "mrr=\(String(format: "%.3f", metrics.meanReciprocalRank)) "
                + "recall@3=\(String(format: "%.3f", metrics.meanRecallAt3)) "
                + "margin=\(String(format: "%.3f", metrics.meanRelevantMargin))"
        )
        print(
            "[EmbeddingEval.Acceptance] source=\(TextEmbeddingEvaluationSupport.usesExternalDataset ? "external" : "built-in") "
                + "minTop1=\(thresholds.map { String(describing: $0.minimumTop1Accuracy) } ?? "nil") "
                + "minMRR=\(thresholds.map { String(describing: $0.minimumMeanReciprocalRank) } ?? "nil") "
                + "minRecallAt3=\(thresholds.map { String(describing: $0.minimumMeanRecallAt3) } ?? "nil") "
                + "minMargin=\(thresholds.map { String(describing: $0.minimumMeanRelevantMargin) } ?? "nil")"
        )
        for result in metrics.queryResults where result.top1Hit == false {
            print(
                "[EmbeddingEval.Miss] query=\(result.queryID) "
                    + "rank=\(result.firstRelevantRank) "
                    + "top=\(result.topDocumentID) "
                    + "topScore=\(String(format: "%.3f", result.topDocumentScore)) "
                    + "margin=\(String(format: "%.3f", result.relevantMargin))"
            )
        }

        #expect(metrics.queryCount == dataset.queries.count)
        #expect(metrics.documentCount == dataset.documents.count)
        #expect(metrics.top1Accuracy.isFinite)
        #expect(metrics.top1Accuracy >= 0)
        #expect(metrics.top1Accuracy <= 1)
        #expect(metrics.meanReciprocalRank.isFinite)
        #expect(metrics.meanReciprocalRank >= 0)
        #expect(metrics.meanReciprocalRank <= 1)
        #expect(metrics.meanRecallAt3.isFinite)
        #expect(metrics.meanRecallAt3 >= 0)
        #expect(metrics.meanRecallAt3 <= 1)
        #expect(metrics.meanRelevantMargin.isFinite)

        if let minimumTop1Accuracy = thresholds?.minimumTop1Accuracy {
            #expect(metrics.top1Accuracy >= minimumTop1Accuracy)
        }
        if let minimumMeanReciprocalRank = thresholds?.minimumMeanReciprocalRank {
            #expect(metrics.meanReciprocalRank >= minimumMeanReciprocalRank)
        }
        if let minimumMeanRecallAt3 = thresholds?.minimumMeanRecallAt3 {
            #expect(metrics.meanRecallAt3 >= minimumMeanRecallAt3)
        }
        if let minimumMeanRelevantMargin = thresholds?.minimumMeanRelevantMargin {
            #expect(metrics.meanRelevantMargin > minimumMeanRelevantMargin)
        }
    }
}
