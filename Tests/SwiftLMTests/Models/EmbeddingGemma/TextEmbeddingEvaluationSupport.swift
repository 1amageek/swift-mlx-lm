import Foundation
@testable import SwiftLM

enum TextEmbeddingEvaluationSupport {
    struct AcceptanceThresholds: Sendable {
        let minimumTop1Accuracy: Float?
        let minimumMeanReciprocalRank: Float?
        let minimumMeanRecallAt3: Float?
        let minimumMeanRelevantMargin: Float?
    }

    struct Document: Codable, Sendable {
        let id: String
        let text: String
    }

    struct Query: Codable, Sendable {
        let id: String
        let text: String
        let relevantDocumentIDs: [String]
    }

    struct Dataset: Codable, Sendable {
        let name: String
        let queryPromptName: String?
        let documentPromptName: String?
        let documents: [Document]
        let queries: [Query]
    }

    struct QueryResult: Sendable {
        let queryID: String
        let firstRelevantRank: Int
        let topDocumentID: String
        let topDocumentScore: Float
        let top1Hit: Bool
        let recallAt3: Float
        let bestRelevantScore: Float
        let bestIrrelevantScore: Float

        var relevantMargin: Float {
            bestRelevantScore - bestIrrelevantScore
        }

        var reciprocalRank: Float {
            1 / Float(firstRelevantRank)
        }
    }

    struct Metrics: Sendable {
        let datasetName: String
        let documentCount: Int
        let queryCount: Int
        let top1Accuracy: Float
        let meanReciprocalRank: Float
        let meanRecallAt3: Float
        let meanRelevantMargin: Float
        let queryResults: [QueryResult]
    }

    enum EnvironmentError: Error, CustomStringConvertible {
        case invalidFloat(key: String, value: String)

        var description: String {
            switch self {
            case let .invalidFloat(key, value):
                return "Invalid floating-point environment value for \(key): \(value)"
            }
        }
    }

    enum EvaluationError: Error, CustomStringConvertible {
        case missingDocumentEmbedding(id: String)
        case missingQueryEmbedding(id: String)
        case embeddingDimensionMismatch(kind: String, id: String, expected: Int, actual: Int)

        var description: String {
            switch self {
            case let .missingDocumentEmbedding(id):
                return "Missing document embedding for \(id)"
            case let .missingQueryEmbedding(id):
                return "Missing query embedding for \(id)"
            case let .embeddingDimensionMismatch(kind, id, expected, actual):
                return "\(kind) embedding dimension mismatch for \(id): expected \(expected), got \(actual)"
            }
        }
    }

    static var usesExternalDataset: Bool {
        optionalDatasetPath() != nil
    }

    static func loadDataset() throws -> Dataset {
        if let datasetPath = optionalDatasetPath() {
            let data = try Data(contentsOf: datasetPath)
            return try JSONDecoder().decode(Dataset.self, from: data)
        }
        return try defaultDataset()
    }

    static func acceptanceThresholds() throws -> AcceptanceThresholds? {
        if usesExternalDataset {
            return try externalAcceptanceThresholds()
        }
        return AcceptanceThresholds(
            minimumTop1Accuracy: 0.875,
            minimumMeanReciprocalRank: 0.90,
            minimumMeanRecallAt3: 1.0,
            minimumMeanRelevantMargin: 0.02
        )
    }

    static func evaluate(
        container: TextEmbeddingContainer,
        dataset: Dataset,
        queryPromptName: String,
        documentPromptName: String
    ) throws -> Metrics {
        precondition(dataset.documents.isEmpty == false, "Dataset must contain documents")
        precondition(dataset.queries.isEmpty == false, "Dataset must contain queries")

        var documentEmbeddings: [String: [Float]] = [:]
        documentEmbeddings.reserveCapacity(dataset.documents.count)
        for document in dataset.documents {
            documentEmbeddings[document.id] = try container.embed(
                document.text,
                promptName: documentPromptName
            )
        }

        var queryEmbeddings: [String: [Float]] = [:]
        queryEmbeddings.reserveCapacity(dataset.queries.count)
        for query in dataset.queries {
            queryEmbeddings[query.id] = try container.embed(
                query.text,
                promptName: queryPromptName
            )
        }

        return try evaluate(
            dataset: dataset,
            queryEmbeddings: queryEmbeddings,
            documentEmbeddings: documentEmbeddings
        )
    }

    static func evaluate(
        dataset: Dataset,
        queryEmbeddings: [String: [Float]],
        documentEmbeddings: [String: [Float]]
    ) throws -> Metrics {
        precondition(dataset.documents.isEmpty == false, "Dataset must contain documents")
        precondition(dataset.queries.isEmpty == false, "Dataset must contain queries")

        let expectedDimension = try embeddingDimension(
            dataset: dataset,
            queryEmbeddings: queryEmbeddings,
            documentEmbeddings: documentEmbeddings
        )

        var results: [QueryResult] = []
        results.reserveCapacity(dataset.queries.count)

        for query in dataset.queries {
            let relevantIDs = Set(query.relevantDocumentIDs)
            precondition(relevantIDs.isEmpty == false, "Each query must define at least one relevant document")
            let queryEmbedding = try requireQueryEmbedding(
                id: query.id,
                queryEmbeddings: queryEmbeddings,
                expectedDimension: expectedDimension
            )
            let rankedDocuments = try dataset.documents.map { document -> (id: String, score: Float) in
                let documentEmbedding = try requireDocumentEmbedding(
                    id: document.id,
                    documentEmbeddings: documentEmbeddings,
                    expectedDimension: expectedDimension
                )
                return (
                    id: document.id,
                    score: cosineSimilarity(queryEmbedding, documentEmbedding)
                )
            }.sorted { lhs, rhs in
                if lhs.score == rhs.score {
                    return lhs.id < rhs.id
                }
                return lhs.score > rhs.score
            }

            guard let top = rankedDocuments.first else {
                continue
            }
            guard let firstRelevantIndex = rankedDocuments.firstIndex(where: { relevantIDs.contains($0.id) }) else {
                preconditionFailure("Query \(query.id) has no relevant documents in the corpus")
            }

            let top3 = Array(rankedDocuments.prefix(3))
            let relevantInTop3 = top3.reduce(into: 0) { count, item in
                if relevantIDs.contains(item.id) {
                    count += 1
                }
            }
            let bestRelevantScore = rankedDocuments
                .filter { relevantIDs.contains($0.id) }
                .map(\.score)
                .max() ?? -.infinity
            let bestIrrelevantScore = rankedDocuments
                .filter { !relevantIDs.contains($0.id) }
                .map(\.score)
                .max() ?? -.infinity

            results.append(
                QueryResult(
                    queryID: query.id,
                    firstRelevantRank: firstRelevantIndex + 1,
                    topDocumentID: top.id,
                    topDocumentScore: top.score,
                    top1Hit: relevantIDs.contains(top.id),
                    recallAt3: Float(relevantInTop3) / Float(relevantIDs.count),
                    bestRelevantScore: bestRelevantScore,
                    bestIrrelevantScore: bestIrrelevantScore
                )
            )
        }

        let queryCount = max(results.count, 1)
        let top1Accuracy = results.reduce(into: Float(0)) { partial, result in
            partial += result.top1Hit ? 1 : 0
        } / Float(queryCount)
        let meanReciprocalRank = results.reduce(into: Float(0)) { partial, result in
            partial += result.reciprocalRank
        } / Float(queryCount)
        let meanRecallAt3 = results.reduce(into: Float(0)) { partial, result in
            partial += result.recallAt3
        } / Float(queryCount)
        let meanRelevantMargin = results.reduce(into: Float(0)) { partial, result in
            partial += result.relevantMargin
        } / Float(queryCount)

        return Metrics(
            datasetName: dataset.name,
            documentCount: dataset.documents.count,
            queryCount: results.count,
            top1Accuracy: top1Accuracy,
            meanReciprocalRank: meanReciprocalRank,
            meanRecallAt3: meanRecallAt3,
            meanRelevantMargin: meanRelevantMargin,
            queryResults: results
        )
    }

    private static func optionalDatasetPath() -> URL? {
        let candidates = [
            ProcessInfo.processInfo.environment["SWIFTLM_TEXT_EMBEDDING_EVAL_DATASET"],
            ProcessInfo.processInfo.environment["SWIFTLM_EMBEDDING_EVAL_DATASET"],
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

    private static func externalAcceptanceThresholds() throws -> AcceptanceThresholds? {
        let minimumTop1Accuracy = try optionalFloatEnvironmentValue(
            primaryKey: "SWIFTLM_TEXT_EMBEDDING_EVAL_MIN_TOP1",
            fallbackKey: "SWIFTLM_EMBEDDING_EVAL_MIN_TOP1"
        )
        let minimumMeanReciprocalRank = try optionalFloatEnvironmentValue(
            primaryKey: "SWIFTLM_TEXT_EMBEDDING_EVAL_MIN_MRR",
            fallbackKey: "SWIFTLM_EMBEDDING_EVAL_MIN_MRR"
        )
        let minimumMeanRecallAt3 = try optionalFloatEnvironmentValue(
            primaryKey: "SWIFTLM_TEXT_EMBEDDING_EVAL_MIN_RECALL_AT3",
            fallbackKey: "SWIFTLM_EMBEDDING_EVAL_MIN_RECALL_AT3"
        )
        let minimumMeanRelevantMargin = try optionalFloatEnvironmentValue(
            primaryKey: "SWIFTLM_TEXT_EMBEDDING_EVAL_MIN_MARGIN",
            fallbackKey: "SWIFTLM_EMBEDDING_EVAL_MIN_MARGIN"
        )

        if minimumTop1Accuracy == nil,
           minimumMeanReciprocalRank == nil,
           minimumMeanRecallAt3 == nil,
           minimumMeanRelevantMargin == nil {
            return nil
        }

        return AcceptanceThresholds(
            minimumTop1Accuracy: minimumTop1Accuracy,
            minimumMeanReciprocalRank: minimumMeanReciprocalRank,
            minimumMeanRecallAt3: minimumMeanRecallAt3,
            minimumMeanRelevantMargin: minimumMeanRelevantMargin
        )
    }

    static func optionalFloatEnvironmentValue(
        primaryKey: String,
        fallbackKey: String
    ) throws -> Float? {
        for key in [primaryKey, fallbackKey] {
            guard let value = ProcessInfo.processInfo.environment[key] else {
                continue
            }
            let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
            guard trimmed.isEmpty == false else {
                continue
            }
            guard let parsed = Float(trimmed) else {
                throw EnvironmentError.invalidFloat(key: key, value: value)
            }
            return parsed
        }
        return nil
    }

    private static func defaultDataset() throws -> Dataset {
        let data = try Data(contentsOf: defaultDatasetURL())
        return try JSONDecoder().decode(Dataset.self, from: data)
    }

    static func cosineSimilarity(_ lhs: [Float], _ rhs: [Float]) -> Float {
        precondition(lhs.count == rhs.count)
        let lhsNorm = l2Norm(lhs)
        let rhsNorm = l2Norm(rhs)
        guard lhsNorm > 0, rhsNorm > 0 else {
            return 0
        }
        let dot = zip(lhs, rhs).reduce(into: Float(0)) { partial, pair in
            partial += pair.0 * pair.1
        }
        return dot / (lhsNorm * rhsNorm)
    }

    static func l2Norm(_ values: [Float]) -> Float {
        values.reduce(into: Float(0)) { partial, value in
            partial += value * value
        }.squareRoot()
    }

    private static func defaultDatasetURL() -> URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .appendingPathComponent("TestData", isDirectory: true)
            .appendingPathComponent("text_embedding_smoke_dataset.json")
    }

    private static func embeddingDimension(
        dataset: Dataset,
        queryEmbeddings: [String: [Float]],
        documentEmbeddings: [String: [Float]]
    ) throws -> Int {
        guard let firstDocumentID = dataset.documents.first?.id else {
            preconditionFailure("Dataset must contain documents")
        }
        let firstDocumentEmbedding = try requireDocumentEmbedding(
            id: firstDocumentID,
            documentEmbeddings: documentEmbeddings,
            expectedDimension: nil
        )
        let expectedDimension = firstDocumentEmbedding.count
        for document in dataset.documents {
            _ = try requireDocumentEmbedding(
                id: document.id,
                documentEmbeddings: documentEmbeddings,
                expectedDimension: expectedDimension
            )
        }
        for query in dataset.queries {
            _ = try requireQueryEmbedding(
                id: query.id,
                queryEmbeddings: queryEmbeddings,
                expectedDimension: expectedDimension
            )
        }
        return expectedDimension
    }

    private static func requireDocumentEmbedding(
        id: String,
        documentEmbeddings: [String: [Float]],
        expectedDimension: Int?
    ) throws -> [Float] {
        guard let embedding = documentEmbeddings[id] else {
            throw EvaluationError.missingDocumentEmbedding(id: id)
        }
        if let expectedDimension, embedding.count != expectedDimension {
            throw EvaluationError.embeddingDimensionMismatch(
                kind: "Document",
                id: id,
                expected: expectedDimension,
                actual: embedding.count
            )
        }
        return embedding
    }

    private static func requireQueryEmbedding(
        id: String,
        queryEmbeddings: [String: [Float]],
        expectedDimension: Int?
    ) throws -> [Float] {
        guard let embedding = queryEmbeddings[id] else {
            throw EvaluationError.missingQueryEmbedding(id: id)
        }
        if let expectedDimension, embedding.count != expectedDimension {
            throw EvaluationError.embeddingDimensionMismatch(
                kind: "Query",
                id: id,
                expected: expectedDimension,
                actual: embedding.count
            )
        }
        return embedding
    }
}
