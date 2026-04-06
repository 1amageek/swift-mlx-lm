import Foundation

/// A video payload for multimodal prompts.
public struct InputVideo: Sendable {
    public let source: Source

    public init(fileURL: URL) {
        self.source = .fileURL(fileURL)
    }

    public init(data: Data, mimeType: String? = nil) {
        self.source = .data(data, mimeType: mimeType)
    }

    public enum Source: Sendable {
        case fileURL(URL)
        case data(Data, mimeType: String?)
    }
}
