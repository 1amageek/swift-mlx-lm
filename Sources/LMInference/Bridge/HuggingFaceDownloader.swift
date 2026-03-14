import Foundation

/// Downloads model files from Hugging Face Hub with local caching.
///
/// Files are stored at `~/.cache/huggingface/hub/{repo}/{revision}/`.
/// Subsequent calls return the cached path without re-downloading unless the remote
/// file has changed (ETag-based validation).
///
/// ```swift
/// let downloader = HuggingFaceDownloader()
/// let dir = try await downloader.downloadBundle(repo: "Qwen/Qwen2.5-0.5B-Instruct")
/// ```
public struct HuggingFaceDownloader: Sendable {

    private static let baseURL = "https://huggingface.co"

    public init() {}

    // MARK: - HF Bundle Download

    /// Download all files needed for an HF directory bundle.
    ///
    /// Downloads config.json, tokenizer files, safetensors weights, and optional
    /// files (chat_template.jinja, preprocessor_config.json) to a local cache directory.
    ///
    /// - Parameters:
    ///   - repo: Repository ID (e.g., `"Qwen/Qwen2.5-0.5B-Instruct"`).
    ///   - revision: Git revision. Defaults to `"main"`.
    ///   - token: Optional Hugging Face access token.
    ///   - progress: Optional `Progress` for byte-level download tracking.
    /// - Returns: Local directory URL containing the downloaded files.
    public func downloadBundle(
        repo: String,
        revision: String = "main",
        token: String? = nil,
        progress: Progress? = nil
    ) async throws -> URL {
        let cacheDir = buildCacheDirectory(repo: repo, revision: revision)
        try FileManager.default.createDirectory(at: cacheDir, withIntermediateDirectories: true)

        // Check if already cached: all required files must exist
        let requiredCacheFiles = [
            "config.json", "tokenizer.json", "tokenizer_config.json"
        ]
        let allRequiredExist = requiredCacheFiles.allSatisfy { filename in
            FileManager.default.fileExists(
                atPath: cacheDir.appendingPathComponent(filename).path)
        }
        if allRequiredExist {
            let contents = try FileManager.default.contentsOfDirectory(
                at: cacheDir, includingPropertiesForKeys: nil)
            let hasSafetensors = contents.contains { $0.pathExtension == "safetensors" }
            if hasSafetensors {
                print("[HuggingFaceDownloader] bundle cache hit: \(repo)")
                return cacheDir
            }
        }

        // List all files in the repo
        let allFiles = try await listRepoFiles(repo: repo, token: token)

        // Required files
        let requiredFiles = ["config.json", "tokenizer.json", "tokenizer_config.json"]
        for filename in requiredFiles {
            guard allFiles.contains(filename) else {
                throw HuggingFaceDownloadError.missingBundleFile(repo: repo, filename: filename)
            }
        }

        // Determine safetensors files to download
        let safetensorsFiles: [String]
        if allFiles.contains("model.safetensors.index.json") {
            // Sharded model — download index and all shard files
            let indexURL = try await downloadFile(
                repo: repo, filename: "model.safetensors.index.json",
                revision: revision, token: token, cacheDir: cacheDir, progress: nil)
            let indexData = try Data(contentsOf: indexURL)
            guard let index = try JSONSerialization.jsonObject(with: indexData) as? [String: Any],
                  let weightMap = index["weight_map"] as? [String: String] else {
                throw HuggingFaceDownloadError.invalidResponse
            }
            safetensorsFiles = Array(Set(weightMap.values)).sorted()
        } else if allFiles.contains("model.safetensors") {
            safetensorsFiles = ["model.safetensors"]
        } else {
            // Glob for any *.safetensors
            let stFiles = allFiles.filter { $0.hasSuffix(".safetensors") }
            guard !stFiles.isEmpty else {
                throw HuggingFaceDownloadError.noBundleWeights(repo: repo)
            }
            safetensorsFiles = stFiles
        }

        // Build download list
        var filesToDownload = requiredFiles + safetensorsFiles

        // Optional files
        let optionalFiles = ["chat_template.jinja", "preprocessor_config.json",
                             "model.safetensors.index.json"]
        for filename in optionalFiles {
            if allFiles.contains(filename) && !filesToDownload.contains(filename) {
                filesToDownload.append(filename)
            }
        }

        // Set up progress tracking
        let totalFiles = filesToDownload.count
        let overallProgress: Progress?
        if let progress {
            overallProgress = Progress(
                totalUnitCount: Int64(totalFiles), parent: progress,
                pendingUnitCount: progress.totalUnitCount)
        } else {
            overallProgress = nil
        }

        print("[HuggingFaceDownloader] downloading bundle: \(repo) (\(totalFiles) files)")

        // Download each file
        for filename in filesToDownload {
            _ = try await downloadFile(
                repo: repo, filename: filename, revision: revision,
                token: token, cacheDir: cacheDir, progress: nil)
            overallProgress?.completedUnitCount += 1
        }

        // Verify all safetensors shards were downloaded
        for shard in safetensorsFiles {
            let shardURL = cacheDir.appendingPathComponent(shard)
            guard FileManager.default.fileExists(atPath: shardURL.path) else {
                throw HuggingFaceDownloadError.missingBundleFile(repo: repo, filename: shard)
            }
        }

        print("[HuggingFaceDownloader] bundle download complete: \(repo)")
        return cacheDir
    }

    /// List all files in a Hugging Face repository.
    public func listRepoFiles(
        repo: String,
        token: String? = nil
    ) async throws -> [String] {
        let url = URL(string: "\(Self.baseURL)/api/models/\(repo)")!
        var request = URLRequest(url: url)
        if let token = resolveToken(token) {
            request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        }

        let (data, response) = try await URLSession.shared.data(for: request)

        guard let httpResponse = response as? HTTPURLResponse,
              (200..<300).contains(httpResponse.statusCode) else {
            throw HuggingFaceDownloadError.httpError(
                statusCode: (response as? HTTPURLResponse)?.statusCode ?? 0)
        }

        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let siblings = json["siblings"] as? [[String: Any]] else {
            throw HuggingFaceDownloadError.invalidResponse
        }

        return siblings.compactMap { sibling -> String? in
            sibling["rfilename"] as? String
        }
    }

    /// Download a single file from a repo to the local cache directory.
    private func downloadFile(
        repo: String,
        filename: String,
        revision: String,
        token: String?,
        cacheDir: URL,
        progress: Progress?
    ) async throws -> URL {
        let localURL = cacheDir.appendingPathComponent(filename)

        // Skip if already cached
        if FileManager.default.fileExists(atPath: localURL.path) {
            return localURL
        }

        // Create subdirectories if needed (for sharded files in subdirectories)
        let fileDir = localURL.deletingLastPathComponent()
        try FileManager.default.createDirectory(at: fileDir, withIntermediateDirectories: true)

        let remoteURL = buildRemoteURL(repo: repo, filename: filename, revision: revision)
        let resolvedToken = resolveToken(token)

        let delegate = DownloadDelegate(progress: progress)
        let session = URLSession(configuration: .default, delegate: delegate, delegateQueue: nil)
        defer { session.finishTasksAndInvalidate() }

        var request = URLRequest(url: remoteURL)
        if let token = resolvedToken {
            request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        }

        let (tempURL, response) = try await session.download(for: request)

        guard let httpResponse = response as? HTTPURLResponse,
              (200..<300).contains(httpResponse.statusCode) else {
            let code = (response as? HTTPURLResponse)?.statusCode ?? 0
            throw HuggingFaceDownloadError.httpError(statusCode: code)
        }

        if FileManager.default.fileExists(atPath: localURL.path) {
            try FileManager.default.removeItem(at: localURL)
        }
        try FileManager.default.moveItem(at: tempURL, to: localURL)

        let fileSize = (try? FileManager.default.attributesOfItem(
            atPath: localURL.path)[.size] as? Int64) ?? 0
        print("[HuggingFaceDownloader] downloaded: \(filename) (\(ByteCountFormatter.string(fromByteCount: fileSize, countStyle: .file)))")

        return localURL
    }

    // MARK: - Private

    private func buildRemoteURL(repo: String, filename: String, revision: String) -> URL {
        URL(string: "\(Self.baseURL)/\(repo)/resolve/\(revision)/\(filename)")!
    }

    private func buildCacheDirectory(repo: String, revision: String) -> URL {
        let home = URL(fileURLWithPath: NSHomeDirectory())
        return home
            .appendingPathComponent(".cache")
            .appendingPathComponent("huggingface")
            .appendingPathComponent("hub")
            .appendingPathComponent(repo.replacingOccurrences(of: "/", with: "--"))
            .appendingPathComponent(revision)
    }

    private func buildCachePath(repo: String, filename: String, revision: String) -> URL {
        buildCacheDirectory(repo: repo, revision: revision)
            .appendingPathComponent(filename)
    }

    private func resolveToken(_ explicit: String?) -> String? {
        if let explicit { return explicit }

        if let envToken = ProcessInfo.processInfo.environment["HF_TOKEN"], !envToken.isEmpty {
            return envToken
        }

        #if os(macOS)
        let tokenPath = URL(fileURLWithPath: NSHomeDirectory())
            .appendingPathComponent(".cache/huggingface/token")
        if let fileToken = try? String(contentsOf: tokenPath, encoding: .utf8).trimmingCharacters(in: .whitespacesAndNewlines),
           !fileToken.isEmpty {
            return fileToken
        }
        #endif

        return nil
    }

    private func fetchEtag(url: URL, token: String?) async throws -> String? {
        var request = URLRequest(url: url)
        request.httpMethod = "HEAD"
        if let token {
            request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        }

        let (_, response) = try await URLSession.shared.data(for: request)
        return (response as? HTTPURLResponse)?.value(forHTTPHeaderField: "ETag")
    }
}

// MARK: - Download Delegate

private final class DownloadDelegate: NSObject, URLSessionDownloadDelegate, Sendable {

    let progress: Progress?

    init(progress: Progress?) {
        self.progress = progress
    }

    func urlSession(
        _ session: URLSession,
        downloadTask: URLSessionDownloadTask,
        didWriteData bytesWritten: Int64,
        totalBytesWritten: Int64,
        totalBytesExpectedToWrite: Int64
    ) {
        guard let progress, totalBytesExpectedToWrite > 0 else { return }
        if progress.totalUnitCount != totalBytesExpectedToWrite {
            progress.totalUnitCount = totalBytesExpectedToWrite
        }
        progress.completedUnitCount = totalBytesWritten
        let completed = ByteCountFormatter.string(fromByteCount: totalBytesWritten, countStyle: .file)
        let total = ByteCountFormatter.string(fromByteCount: totalBytesExpectedToWrite, countStyle: .file)
        progress.localizedDescription = String(
            format: NSLocalizedString("Downloading… %@ / %@", comment: ""),
            completed, total
        )
    }

    func urlSession(
        _ session: URLSession,
        downloadTask: URLSessionDownloadTask,
        didFinishDownloadingTo location: URL
    ) {
        // Handled by the async download(for:) call
    }
}

// MARK: - Errors

/// Errors during Hugging Face Hub file downloads.
public enum HuggingFaceDownloadError: Error, CustomStringConvertible {
    case httpError(statusCode: Int)
    case invalidResponse
    case missingBundleFile(repo: String, filename: String)
    case noBundleWeights(repo: String)

    public var description: String {
        switch self {
        case .httpError(let code):
            switch code {
            case 401: return "Authentication required. Set HF_TOKEN or pass a token."
            case 403: return "Access denied. Check your token permissions."
            case 404: return "File not found on Hugging Face Hub."
            default: return "HTTP error \(code)"
            }
        case .invalidResponse:
            return "Invalid response from Hugging Face Hub"
        case .missingBundleFile(let repo, let filename):
            return "Required file '\(filename)' not found in repository '\(repo)'"
        case .noBundleWeights(let repo):
            return "No safetensors weight files found in repository '\(repo)'"
        }
    }
}
