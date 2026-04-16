import Foundation
@testable import SwiftLM

enum TextEmbeddingReferenceSupport {
    struct MetricsSnapshot: Codable, Sendable {
        let top1Accuracy: Float
        let meanReciprocalRank: Float
        let meanRecallAt3: Float
        let meanRelevantMargin: Float
    }

    struct Snapshot: Codable, Sendable {
        let dataset: TextEmbeddingEvaluationSupport.Dataset
        let embeddingDimension: Int
        let documentEmbeddings: [String: [Float]]
        let queryEmbeddings: [String: [Float]]
        let metrics: MetricsSnapshot
    }

    struct AcceptanceThresholds: Sendable {
        let minimumMeanDocumentCosine: Float
        let minimumMeanQueryCosine: Float
        let minimumTop1Agreement: Float
        let maximumTop1AccuracyDelta: Float
        let maximumMeanReciprocalRankDelta: Float
        let maximumMeanRecallAt3Delta: Float
        let maximumMeanRelevantMarginDelta: Float
    }

    struct VectorComparison: Sendable {
        let id: String
        let cosine: Float
    }

    struct TopDocumentMismatch: Sendable {
        let queryID: String
        let referenceTopDocumentID: String
        let swiftTopDocumentID: String
    }

    struct ComparisonMetrics: Sendable {
        let datasetName: String
        let documentCount: Int
        let queryCount: Int
        let referenceMetrics: TextEmbeddingEvaluationSupport.Metrics
        let swiftMetrics: TextEmbeddingEvaluationSupport.Metrics
        let meanDocumentCosine: Float
        let meanQueryCosine: Float
        let minimumDocumentCosine: Float
        let minimumQueryCosine: Float
        let top1Agreement: Float
        let documentComparisons: [VectorComparison]
        let queryComparisons: [VectorComparison]
        let topDocumentMismatches: [TopDocumentMismatch]
    }

    enum ReferenceError: Error, CustomStringConvertible {
        case malformedSnapshot(String)
        case missingReferenceDocument(id: String)
        case missingReferenceQuery(id: String)

        var description: String {
            switch self {
            case let .malformedSnapshot(message):
                return "Malformed text embedding reference snapshot: \(message)"
            case let .missingReferenceDocument(id):
                return "Missing reference document embedding for \(id)"
            case let .missingReferenceQuery(id):
                return "Missing reference query embedding for \(id)"
            }
        }
    }

    static func loadSnapshot() throws -> Snapshot? {
        guard let snapshotURL = optionalSnapshotURL() else {
            return nil
        }
        let data = try Data(contentsOf: snapshotURL)
        return try JSONDecoder().decode(Snapshot.self, from: data)
    }

    static func acceptanceThresholds() throws -> AcceptanceThresholds {
        let minimumMeanDocumentCosine = try TextEmbeddingEvaluationSupport.optionalFloatEnvironmentValue(
            primaryKey: "SWIFTLM_TEXT_EMBEDDING_REFERENCE_MIN_DOC_COS",
            fallbackKey: "SWIFTLM_EMBEDDING_REFERENCE_MIN_DOC_COS"
        ) ?? 0.99
        let minimumMeanQueryCosine = try TextEmbeddingEvaluationSupport.optionalFloatEnvironmentValue(
            primaryKey: "SWIFTLM_TEXT_EMBEDDING_REFERENCE_MIN_QUERY_COS",
            fallbackKey: "SWIFTLM_EMBEDDING_REFERENCE_MIN_QUERY_COS"
        ) ?? 0.99
        let minimumTop1Agreement = try TextEmbeddingEvaluationSupport.optionalFloatEnvironmentValue(
            primaryKey: "SWIFTLM_TEXT_EMBEDDING_REFERENCE_MIN_TOP1_AGREEMENT",
            fallbackKey: "SWIFTLM_EMBEDDING_REFERENCE_MIN_TOP1_AGREEMENT"
        ) ?? 0.875
        let maximumTop1AccuracyDelta = try TextEmbeddingEvaluationSupport.optionalFloatEnvironmentValue(
            primaryKey: "SWIFTLM_TEXT_EMBEDDING_REFERENCE_MAX_TOP1_DELTA",
            fallbackKey: "SWIFTLM_EMBEDDING_REFERENCE_MAX_TOP1_DELTA"
        ) ?? 0.05
        let maximumMeanReciprocalRankDelta = try TextEmbeddingEvaluationSupport.optionalFloatEnvironmentValue(
            primaryKey: "SWIFTLM_TEXT_EMBEDDING_REFERENCE_MAX_MRR_DELTA",
            fallbackKey: "SWIFTLM_EMBEDDING_REFERENCE_MAX_MRR_DELTA"
        ) ?? 0.05
        let maximumMeanRecallAt3Delta = try TextEmbeddingEvaluationSupport.optionalFloatEnvironmentValue(
            primaryKey: "SWIFTLM_TEXT_EMBEDDING_REFERENCE_MAX_RECALL_AT3_DELTA",
            fallbackKey: "SWIFTLM_EMBEDDING_REFERENCE_MAX_RECALL_AT3_DELTA"
        ) ?? 0.05
        let maximumMeanRelevantMarginDelta = try TextEmbeddingEvaluationSupport.optionalFloatEnvironmentValue(
            primaryKey: "SWIFTLM_TEXT_EMBEDDING_REFERENCE_MAX_MARGIN_DELTA",
            fallbackKey: "SWIFTLM_EMBEDDING_REFERENCE_MAX_MARGIN_DELTA"
        ) ?? 0.1

        return AcceptanceThresholds(
            minimumMeanDocumentCosine: minimumMeanDocumentCosine,
            minimumMeanQueryCosine: minimumMeanQueryCosine,
            minimumTop1Agreement: minimumTop1Agreement,
            maximumTop1AccuracyDelta: maximumTop1AccuracyDelta,
            maximumMeanReciprocalRankDelta: maximumMeanReciprocalRankDelta,
            maximumMeanRecallAt3Delta: maximumMeanRecallAt3Delta,
            maximumMeanRelevantMarginDelta: maximumMeanRelevantMarginDelta
        )
    }

    static func compare(
        container: TextEmbeddingContainer,
        snapshot: Snapshot
    ) throws -> ComparisonMetrics {
        let referenceMetrics = try TextEmbeddingEvaluationSupport.evaluate(
            dataset: snapshot.dataset,
            queryEmbeddings: snapshot.queryEmbeddings,
            documentEmbeddings: snapshot.documentEmbeddings
        )
        try validateStoredMetrics(snapshot.metrics, against: referenceMetrics)
        try validateEmbeddingDimension(snapshot)

        let queryPromptName = snapshot.dataset.queryPromptName ?? "query"
        let documentPromptName = snapshot.dataset.documentPromptName ?? "document"

        var swiftDocumentEmbeddings: [String: [Float]] = [:]
        swiftDocumentEmbeddings.reserveCapacity(snapshot.dataset.documents.count)
        for document in snapshot.dataset.documents {
            swiftDocumentEmbeddings[document.id] = try container.embed(
                document.text,
                promptName: documentPromptName
            )
        }

        var swiftQueryEmbeddings: [String: [Float]] = [:]
        swiftQueryEmbeddings.reserveCapacity(snapshot.dataset.queries.count)
        for query in snapshot.dataset.queries {
            swiftQueryEmbeddings[query.id] = try container.embed(
                query.text,
                promptName: queryPromptName
            )
        }

        let swiftMetrics = try TextEmbeddingEvaluationSupport.evaluate(
            dataset: snapshot.dataset,
            queryEmbeddings: swiftQueryEmbeddings,
            documentEmbeddings: swiftDocumentEmbeddings
        )

        let documentComparisons = try snapshot.dataset.documents.map { document in
            guard let referenceEmbedding = snapshot.documentEmbeddings[document.id] else {
                throw ReferenceError.missingReferenceDocument(id: document.id)
            }
            guard let swiftEmbedding = swiftDocumentEmbeddings[document.id] else {
                throw ReferenceError.malformedSnapshot("Missing Swift document embedding for \(document.id)")
            }
            guard swiftEmbedding.count == snapshot.embeddingDimension else {
                throw ReferenceError.malformedSnapshot(
                    "Swift document embedding \(document.id) has dimension \(swiftEmbedding.count), expected \(snapshot.embeddingDimension)"
                )
            }
            return VectorComparison(
                id: document.id,
                cosine: TextEmbeddingEvaluationSupport.cosineSimilarity(referenceEmbedding, swiftEmbedding)
            )
        }
        let queryComparisons = try snapshot.dataset.queries.map { query in
            guard let referenceEmbedding = snapshot.queryEmbeddings[query.id] else {
                throw ReferenceError.missingReferenceQuery(id: query.id)
            }
            guard let swiftEmbedding = swiftQueryEmbeddings[query.id] else {
                throw ReferenceError.malformedSnapshot("Missing Swift query embedding for \(query.id)")
            }
            guard swiftEmbedding.count == snapshot.embeddingDimension else {
                throw ReferenceError.malformedSnapshot(
                    "Swift query embedding \(query.id) has dimension \(swiftEmbedding.count), expected \(snapshot.embeddingDimension)"
                )
            }
            return VectorComparison(
                id: query.id,
                cosine: TextEmbeddingEvaluationSupport.cosineSimilarity(referenceEmbedding, swiftEmbedding)
            )
        }

        let referenceTopDocuments = Dictionary(
            uniqueKeysWithValues: referenceMetrics.queryResults.map { ($0.queryID, $0.topDocumentID) }
        )
        let swiftTopDocuments = Dictionary(
            uniqueKeysWithValues: swiftMetrics.queryResults.map { ($0.queryID, $0.topDocumentID) }
        )
        let topDocumentMismatches = snapshot.dataset.queries.compactMap { query -> TopDocumentMismatch? in
            guard let referenceTopDocumentID = referenceTopDocuments[query.id],
                  let swiftTopDocumentID = swiftTopDocuments[query.id],
                  referenceTopDocumentID != swiftTopDocumentID else {
                return nil
            }
            return TopDocumentMismatch(
                queryID: query.id,
                referenceTopDocumentID: referenceTopDocumentID,
                swiftTopDocumentID: swiftTopDocumentID
            )
        }
        let queryCount = max(snapshot.dataset.queries.count, 1)
        let top1Agreement = Float(snapshot.dataset.queries.count - topDocumentMismatches.count) / Float(queryCount)

        return ComparisonMetrics(
            datasetName: snapshot.dataset.name,
            documentCount: snapshot.dataset.documents.count,
            queryCount: snapshot.dataset.queries.count,
            referenceMetrics: referenceMetrics,
            swiftMetrics: swiftMetrics,
            meanDocumentCosine: meanCosine(documentComparisons),
            meanQueryCosine: meanCosine(queryComparisons),
            minimumDocumentCosine: documentComparisons.map(\.cosine).min() ?? 0,
            minimumQueryCosine: queryComparisons.map(\.cosine).min() ?? 0,
            top1Agreement: top1Agreement,
            documentComparisons: documentComparisons,
            queryComparisons: queryComparisons,
            topDocumentMismatches: topDocumentMismatches
        )
    }

    private static func optionalSnapshotURL() -> URL? {
        let candidates = [
            ProcessInfo.processInfo.environment["SWIFTLM_TEXT_EMBEDDING_REFERENCE"],
            ProcessInfo.processInfo.environment["SWIFTLM_EMBEDDING_REFERENCE"],
        ].compactMap { value -> String? in
            guard let value else { return nil }
            let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
            return trimmed.isEmpty ? nil : trimmed
        }
        guard let first = candidates.first else {
            return nil
        }
        return URL(fileURLWithPath: NSString(string: first).expandingTildeInPath)
    }

    private static func meanCosine(_ values: [VectorComparison]) -> Float {
        guard values.isEmpty == false else {
            return 0
        }
        let total = values.reduce(into: Float(0)) { partial, comparison in
            partial += comparison.cosine
        }
        return total / Float(values.count)
    }

    private static func validateStoredMetrics(
        _ snapshotMetrics: MetricsSnapshot,
        against computedMetrics: TextEmbeddingEvaluationSupport.Metrics
    ) throws {
        let tolerance: Float = 0.0005
        guard abs(snapshotMetrics.top1Accuracy - computedMetrics.top1Accuracy) <= tolerance,
              abs(snapshotMetrics.meanReciprocalRank - computedMetrics.meanReciprocalRank) <= tolerance,
              abs(snapshotMetrics.meanRecallAt3 - computedMetrics.meanRecallAt3) <= tolerance,
              abs(snapshotMetrics.meanRelevantMargin - computedMetrics.meanRelevantMargin) <= tolerance else {
            throw ReferenceError.malformedSnapshot(
                "Stored retrieval metrics do not match the provided reference embeddings"
            )
        }
    }

    private static func validateEmbeddingDimension(_ snapshot: Snapshot) throws {
        for (id, embedding) in snapshot.documentEmbeddings where embedding.count != snapshot.embeddingDimension {
            throw ReferenceError.malformedSnapshot(
                "Document \(id) has dimension \(embedding.count), expected \(snapshot.embeddingDimension)"
            )
        }
        for (id, embedding) in snapshot.queryEmbeddings where embedding.count != snapshot.embeddingDimension {
            throw ReferenceError.malformedSnapshot(
                "Query \(id) has dimension \(embedding.count), expected \(snapshot.embeddingDimension)"
            )
        }
    }
}
