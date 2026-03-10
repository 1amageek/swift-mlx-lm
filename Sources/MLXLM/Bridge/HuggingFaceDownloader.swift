import Foundation
import GGUFParser

/// Downloads model files from Hugging Face Hub with local caching.
///
/// Files are stored at `~/Library/Application Support/swift-mlx-lm/huggingface/{repo}/{revision}/`.
/// Subsequent calls return the cached path without re-downloading unless the remote
/// file has changed (ETag-based validation).
///
/// ```swift
/// let downloader = HuggingFaceDownloader()
///
/// // Single GGUF in repo — just works
/// let url = try await downloader.download(repo: "someone/Model-GGUF")
///
/// // Multiple GGUFs — specify which quantization
/// let url = try await downloader.download(
///     repo: "someone/Model-GGUF",
///     filename: "model-q4_k_m.gguf"
/// )
/// ```
public struct HuggingFaceDownloader: Sendable {

    private static let baseURL = "https://huggingface.co"

    public init() {}

    /// Download a GGUF file from Hugging Face Hub.
    ///
    /// When `filename` is omitted, queries the repository to find GGUF files.
    /// If there is exactly one, it is used. If there are multiple, the best
    /// quantization (Q4_K_M preferred) is auto-selected.
    ///
    /// - Parameters:
    ///   - repo: Repository ID (e.g., `"someone/Model-GGUF"`).
    ///   - filename: Explicit filename. When `nil`, auto-resolves.
    ///   - revision: Git revision. Defaults to `"main"`.
    ///   - token: Optional Hugging Face access token.
    ///   - progress: Optional `Progress` for byte-level download tracking.
    /// - Returns: Local file URL of the downloaded (or cached) file.
    public func download(
        repo: String,
        filename: String? = nil,
        revision: String = "main",
        token: String? = nil,
        progress: Progress? = nil
    ) async throws -> URL {

        // 1. Fast path: check local cache before any network calls.
        if let filename {
            let localURL = buildCachePath(repo: repo, filename: filename, revision: revision)
            if FileManager.default.fileExists(atPath: localURL.path) {
                let fileSize = (try? FileManager.default.attributesOfItem(atPath: localURL.path)[.size] as? Int64) ?? 0
                print("[HuggingFaceDownloader] cache hit: \(filename) (\(ByteCountFormatter.string(fromByteCount: fileSize, countStyle: .file)))")
                return localURL
            }
        } else {
            let cacheDir = buildCacheDirectory(repo: repo, revision: revision)
            if let cachedURL = findCachedGGUF(in: cacheDir) {
                let fileSize = (try? FileManager.default.attributesOfItem(atPath: cachedURL.path)[.size] as? Int64) ?? 0
                print("[HuggingFaceDownloader] cache hit: \(cachedURL.lastPathComponent) (\(ByteCountFormatter.string(fromByteCount: fileSize, countStyle: .file)))")
                return cachedURL
            }
        }

        // 2. No local cache — resolve filename via API if needed.
        let resolvedFilename: String
        if let filename {
            resolvedFilename = filename
            print("[HuggingFaceDownloader] repo=\(repo) filename=\(filename) (explicit)")
        } else {
            print("[HuggingFaceDownloader] repo=\(repo) querying available GGUF files…")
            let allFiles = try await listGGUFFiles(repo: repo, token: token)
            let modelFiles = allFiles.filter { !$0.hasPrefix("mmproj") }
            print("[HuggingFaceDownloader] repo=\(repo) found \(allFiles.count) GGUF files (\(modelFiles.count) model, \(allFiles.count - modelFiles.count) mmproj)")
            switch modelFiles.count {
            case 0:
                throw HuggingFaceDownloadError.noGGUFFiles(repo: repo)
            case 1:
                resolvedFilename = modelFiles[0]
                print("[HuggingFaceDownloader] auto-selected: \(resolvedFilename) (only model file)")
            default:
                resolvedFilename = selectPreferredFile(from: modelFiles)
                print("[HuggingFaceDownloader] auto-selected: \(resolvedFilename) (preferred quantization)")
            }
        }

        // 3. Download the file.
        let remoteURL = buildRemoteURL(repo: repo, filename: resolvedFilename, revision: revision)
        let localURL = buildCachePath(repo: repo, filename: resolvedFilename, revision: revision)

        let directory = localURL.deletingLastPathComponent()
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)

        let resolvedToken = resolveToken(token)
        let hasAuth = resolvedToken != nil
        print("[HuggingFaceDownloader] downloading \(remoteURL) → \(localURL.path) auth=\(hasAuth)")
        let delegate = DownloadDelegate(progress: progress)
        let session = URLSession(configuration: .default, delegate: delegate, delegateQueue: nil)
        defer { session.finishTasksAndInvalidate() }

        var request = URLRequest(url: remoteURL)
        if let token = resolvedToken {
            request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        }

        progress?.kind = .file
        progress?.fileOperationKind = .downloading
        progress?.localizedDescription = NSLocalizedString("Downloading…", comment: "")
        let (tempURL, response) = try await session.download(for: request)

        guard let httpResponse = response as? HTTPURLResponse else {
            throw HuggingFaceDownloadError.invalidResponse
        }
        guard (200..<300).contains(httpResponse.statusCode) else {
            print("[HuggingFaceDownloader] download failed: HTTP \(httpResponse.statusCode) url=\(remoteURL)")
            throw HuggingFaceDownloadError.httpError(statusCode: httpResponse.statusCode)
        }

        if FileManager.default.fileExists(atPath: localURL.path) {
            try FileManager.default.removeItem(at: localURL)
        }
        try FileManager.default.moveItem(at: tempURL, to: localURL)

        if let etag = httpResponse.value(forHTTPHeaderField: "ETag") {
            let etagPath = localURL.appendingPathExtension("etag")
            try etag.write(to: etagPath, atomically: true, encoding: .utf8)
        }

        let downloadedSize = (try? FileManager.default.attributesOfItem(atPath: localURL.path)[.size] as? Int64) ?? 0
        print("[HuggingFaceDownloader] download complete: \(resolvedFilename) (\(ByteCountFormatter.string(fromByteCount: downloadedSize, countStyle: .file)))")

        return localURL
    }

    /// List GGUF files available in a Hugging Face repository.
    public func listGGUFFiles(
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
            guard let filename = sibling["rfilename"] as? String,
                  filename.hasSuffix(".gguf") else { return nil }
            return filename
        }
    }

    // MARK: - Private

    private func buildRemoteURL(repo: String, filename: String, revision: String) -> URL {
        URL(string: "\(Self.baseURL)/\(repo)/resolve/\(revision)/\(filename)")!
    }

    private func buildCacheDirectory(repo: String, revision: String) -> URL {
        let appSupport = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
        return appSupport
            .appendingPathComponent("swift-mlx-lm")
            .appendingPathComponent("huggingface")
            .appendingPathComponent(repo.replacingOccurrences(of: "/", with: "--"))
            .appendingPathComponent(revision)
    }

    private func buildCachePath(repo: String, filename: String, revision: String) -> URL {
        buildCacheDirectory(repo: repo, revision: revision)
            .appendingPathComponent(filename)
    }

    /// Find a cached GGUF model file (excluding mmproj) in the cache directory.
    private func findCachedGGUF(in directory: URL) -> URL? {
        guard let contents = try? FileManager.default.contentsOfDirectory(
            at: directory, includingPropertiesForKeys: [.fileSizeKey]
        ) else { return nil }

        let ggufFiles = contents.filter {
            $0.pathExtension == "gguf" && !$0.lastPathComponent.hasPrefix("mmproj")
        }
        guard !ggufFiles.isEmpty else { return nil }

        // If multiple, prefer the same ranking as selectPreferredFile
        if ggufFiles.count == 1 { return ggufFiles[0] }
        let filenames = ggufFiles.map { $0.lastPathComponent }
        let preferred = selectPreferredFile(from: filenames)
        return ggufFiles.first { $0.lastPathComponent == preferred }
    }

    private func resolveToken(_ explicit: String?) -> String? {
        if let explicit { return explicit }

        if let envToken = ProcessInfo.processInfo.environment["HF_TOKEN"], !envToken.isEmpty {
            return envToken
        }

        #if os(macOS)
        let tokenPath = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".cache/huggingface/token")
        if let fileToken = try? String(contentsOf: tokenPath, encoding: .utf8).trimmingCharacters(in: .whitespacesAndNewlines),
           !fileToken.isEmpty {
            return fileToken
        }
        #endif

        return nil
    }

    /// Select the best GGUF file from multiple candidates.
    ///
    /// Preference order balances quality and memory usage:
    /// Q4_K_M > Q4_K_S > Q5_K_M > Q5_K_S > Q6_K > Q8_0 > Q3_K_M > Q4_0 > Q4_1
    private func selectPreferredFile(from files: [String]) -> String {
        let ranked = [
            "Q4_K_M", "Q4_K_S", "Q5_K_M", "Q5_K_S", "Q6_K",
            "Q8_0", "Q3_K_M", "Q4_0", "Q4_1", "Q3_K_S",
        ]
        let uppercased = files.map { $0.uppercased() }
        for quantTag in ranked {
            if let index = uppercased.firstIndex(where: { $0.contains(quantTag) }) {
                return files[index]
            }
        }
        return files[0]
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
    case noGGUFFiles(repo: String)
    case multipleGGUFFiles(repo: String, available: [String])

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
        case .noGGUFFiles(let repo):
            return "No GGUF files found in repository '\(repo)'"
        case .multipleGGUFFiles(let repo, let available):
            let list = available.map { "  - \($0)" }.joined(separator: "\n")
            return "Multiple GGUF files in '\(repo)'. Specify one:\n\(list)"
        }
    }
}
