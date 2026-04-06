/// Declared input capabilities of a loaded model bundle.
///
/// This reflects what the bundle metadata says the model can accept, not what
/// the current `SwiftLM` runtime can already execute end to end.
public struct ModelInputCapabilities: Sendable, Equatable {
    public var supportsText: Bool
    public var supportsImages: Bool
    public var supportsVideo: Bool

    public init(
        supportsText: Bool = true,
        supportsImages: Bool = false,
        supportsVideo: Bool = false
    ) {
        self.supportsText = supportsText
        self.supportsImages = supportsImages
        self.supportsVideo = supportsVideo
    }

    public static let textOnly = ModelInputCapabilities()
}
