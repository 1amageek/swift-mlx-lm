import Foundation
@testable import SwiftLM

enum EmbeddingGemmaVariant: String, CaseIterable, Sendable {
    case community4Bit = "community-4bit"
    case official = "official"
    case communityBF16 = "community-bf16"

    var autoDiscoveredDirectoryNames: [String] {
        switch self {
        case .community4Bit:
            ["models--mlx-community--embeddinggemma-300m-4bit"]
        case .official:
            ["models--google--embeddinggemma-300m"]
        case .communityBF16:
            [
                "models--mlx-community--embeddinggemma-300m-bf16",
                "models--mlx-community--embeddinggemma-300m",
            ]
        }
    }
}

enum EmbeddingGemmaTestSupport {
    private static let preferredAutoDiscoveredDirectoryNames = [
        "models--mlx-community--embeddinggemma-300m-4bit",
        "models--google--embeddinggemma-300m",
        "models--mlx-community--embeddinggemma-300m-bf16",
    ]

    static func realEmbeddingGemmaContainer(
        variant: EmbeddingGemmaVariant? = nil
    ) async throws -> TextEmbeddingContainer? {
        let loader = ModelBundleLoader()
        if let repo = optionalRealEmbeddingGemmaRepoID(variant: variant) {
            return try await loader.loadTextEmbeddings(repo: repo)
        }
        guard let directory = try optionalRealEmbeddingGemmaDirectory(variant: variant) else {
            return nil
        }
        return try await loader.loadTextEmbeddings(directory: directory)
    }

    static func referenceSnapshot() throws -> TextEmbeddingReferenceSupport.Snapshot? {
        if let snapshot = try TextEmbeddingReferenceSupport.loadSnapshot() {
            return snapshot
        }
        let snapshotURL = defaultReferenceSnapshotURL()
        guard FileManager.default.fileExists(atPath: snapshotURL.path) else {
            return nil
        }
        let data = try Data(contentsOf: snapshotURL)
        return try JSONDecoder().decode(TextEmbeddingReferenceSupport.Snapshot.self, from: data)
    }

    static func optionalRealEmbeddingGemmaDirectory(
        variant: EmbeddingGemmaVariant? = nil
    ) throws -> URL? {
        let envCandidates = [
            ProcessInfo.processInfo.environment["SWIFTLM_EMBEDDINGGEMMA_DIR"],
            ProcessInfo.processInfo.environment["SWIFTLM_EMBEDDING_GEMMA_DIR"],
        ].compactMap { $0 }
        for candidate in envCandidates {
            let url = URL(fileURLWithPath: NSString(string: candidate).expandingTildeInPath)
            if try isUsableModelDirectory(url) {
                return url
            }
        }

        let hubRoot = URL(
            fileURLWithPath: NSString(
                string: "~/.cache/huggingface/hub"
            ).expandingTildeInPath
        )
        guard FileManager.default.fileExists(atPath: hubRoot.path) else {
            return nil
        }
        let entries = try FileManager.default.contentsOfDirectory(
            at: hubRoot,
            includingPropertiesForKeys: nil
        )
        let preferredNames = variant?.autoDiscoveredDirectoryNames ?? preferredAutoDiscoveredDirectoryNames
        let candidates = preferredNames.compactMap { expectedName in
            entries.first { $0.lastPathComponent.lowercased() == expectedName }
        }
        for entry in candidates {
            let snapshots = try snapshotDirectories(baseURL: entry)
            for snapshot in snapshots where try isUsableModelDirectory(snapshot) {
                return snapshot
            }
        }
        return nil
    }

    static func optionalRealEmbeddingGemmaRepoID(
        variant: EmbeddingGemmaVariant? = nil
    ) -> String? {
        _ = variant
        let candidates = [
            ProcessInfo.processInfo.environment["SWIFTLM_EMBEDDINGGEMMA_REPO"],
            ProcessInfo.processInfo.environment["SWIFTLM_EMBEDDING_GEMMA_REPO"],
        ].compactMap { value -> String? in
            guard let value else { return nil }
            let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
            return trimmed.isEmpty ? nil : trimmed
        }
        return candidates.first
    }

    static func sourceDescription(for variant: EmbeddingGemmaVariant) throws -> String {
        if let repo = optionalRealEmbeddingGemmaRepoID(variant: variant) {
            return repo
        }
        if let directory = try optionalRealEmbeddingGemmaDirectory(variant: variant) {
            return directory.lastPathComponent
        }
        return variant.rawValue
    }

    private static func snapshotDirectories(baseURL: URL) throws -> [URL] {
        let snapshotsURL = baseURL.appendingPathComponent("snapshots")
        guard FileManager.default.fileExists(atPath: snapshotsURL.path) else {
            return []
        }
        return try FileManager.default.contentsOfDirectory(
            at: snapshotsURL,
            includingPropertiesForKeys: nil
        ).sorted { $0.lastPathComponent < $1.lastPathComponent }
    }

    private static func isUsableModelDirectory(_ directory: URL) throws -> Bool {
        let configPath = directory.appendingPathComponent("config.json")
        let tokenizerPath = directory.appendingPathComponent("tokenizer.json")
        guard FileManager.default.fileExists(atPath: configPath.path),
              FileManager.default.fileExists(atPath: tokenizerPath.path) else {
            return false
        }
        let contents = try FileManager.default.contentsOfDirectory(
            at: directory,
            includingPropertiesForKeys: nil
        )
        return contents.contains { $0.pathExtension == "safetensors" }
    }

    private static func defaultReferenceSnapshotURL() -> URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .appendingPathComponent("TestData", isDirectory: true)
            .appendingPathComponent("embeddinggemma_reference.json")
    }
}
