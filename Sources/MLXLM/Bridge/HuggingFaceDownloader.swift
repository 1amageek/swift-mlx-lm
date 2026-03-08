import Foundation
import GGUFParser

/// Downloads model files from Hugging Face Hub with local caching.
///
/// Files are cached at `~/Library/Caches/swift-mlx-lm/huggingface/{repo}/{revision}/`.
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
    /// If there is exactly one, it is used. If there are multiple, an error
    /// is thrown listing the available files so the caller can choose.
    ///
    /// - Parameters:
    ///   - repo: Repository ID (e.g., `"someone/Model-GGUF"`).
    ///   - filename: Explicit filename. When `nil`, auto-resolves if unambiguous.
    ///   - revision: Git revision. Defaults to `"main"`.
    ///   - token: Optional Hugging Face access token.
    ///   - progress: Optional download progress callback (0.0 to 1.0).
    /// - Returns: Local file URL of the downloaded (or cached) file.
    public func download(
        repo: String,
        filename: String? = nil,
        revision: String = "main",
        token: String? = nil,
        progress: (@Sendable (Double) -> Void)? = nil
    ) async throws -> URL {
        let resolvedFilename: String
        if let filename {
            resolvedFilename = filename
        } else {
            let files = try await listGGUFFiles(repo: repo, token: token)
            switch files.count {
            case 0:
                throw HuggingFaceDownloadError.noGGUFFiles(repo: repo)
            case 1:
                resolvedFilename = files[0]
            default:
                throw HuggingFaceDownloadError.multipleGGUFFiles(repo: repo, available: files)
            }
        }

        let remoteURL = buildRemoteURL(repo: repo, filename: resolvedFilename, revision: revision)
        let localURL = buildCachePath(repo: repo, filename: resolvedFilename, revision: revision)
        let etagPath = localURL.appendingPathExtension("etag")

        // Check if cached file is still valid
        let cachedEtag = try? String(contentsOf: etagPath, encoding: .utf8)
        if FileManager.default.fileExists(atPath: localURL.path), cachedEtag != nil {
            let headEtag = try? await fetchEtag(url: remoteURL, token: resolveToken(token))
            if let cached = cachedEtag, let remote = headEtag, cached == remote {
                return localURL
            }
        }

        // Create parent directory
        let directory = localURL.deletingLastPathComponent()
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)

        // Download
        let resolvedToken = resolveToken(token)
        let delegate = DownloadDelegate(progress: progress)
        let session = URLSession(configuration: .default, delegate: delegate, delegateQueue: nil)
        defer { session.finishTasksAndInvalidate() }

        var request = URLRequest(url: remoteURL)
        if let token = resolvedToken {
            request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        }

        let (tempURL, response) = try await session.download(for: request)

        guard let httpResponse = response as? HTTPURLResponse else {
            throw HuggingFaceDownloadError.invalidResponse
        }
        guard (200..<300).contains(httpResponse.statusCode) else {
            throw HuggingFaceDownloadError.httpError(statusCode: httpResponse.statusCode)
        }

        // Move to cache location (replace existing)
        if FileManager.default.fileExists(atPath: localURL.path) {
            try FileManager.default.removeItem(at: localURL)
        }
        try FileManager.default.moveItem(at: tempURL, to: localURL)

        // Save ETag for future cache validation
        if let etag = httpResponse.value(forHTTPHeaderField: "ETag") {
            try etag.write(to: etagPath, atomically: true, encoding: .utf8)
        }

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

    private func buildCachePath(repo: String, filename: String, revision: String) -> URL {
        let cacheDir = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first!
        return cacheDir
            .appendingPathComponent("swift-mlx-lm")
            .appendingPathComponent("huggingface")
            .appendingPathComponent(repo.replacingOccurrences(of: "/", with: "--"))
            .appendingPathComponent(revision)
            .appendingPathComponent(filename)
    }

    private func resolveToken(_ explicit: String?) -> String? {
        if let explicit { return explicit }

        if let envToken = ProcessInfo.processInfo.environment["HF_TOKEN"], !envToken.isEmpty {
            return envToken
        }

        let tokenPath = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".cache/huggingface/token")
        if let fileToken = try? String(contentsOf: tokenPath, encoding: .utf8).trimmingCharacters(in: .whitespacesAndNewlines),
           !fileToken.isEmpty {
            return fileToken
        }

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

    let progress: (@Sendable (Double) -> Void)?

    init(progress: (@Sendable (Double) -> Void)?) {
        self.progress = progress
    }

    func urlSession(
        _ session: URLSession,
        downloadTask: URLSessionDownloadTask,
        didWriteData bytesWritten: Int64,
        totalBytesWritten: Int64,
        totalBytesExpectedToWrite: Int64
    ) {
        guard totalBytesExpectedToWrite > 0 else { return }
        let fraction = Double(totalBytesWritten) / Double(totalBytesExpectedToWrite)
        progress?(fraction)
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
