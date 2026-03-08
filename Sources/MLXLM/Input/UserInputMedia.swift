import CoreImage
import CoreMedia
@preconcurrency import MLX

/// Error during media conversion.
public enum MediaError: Error {
    case unableToLoad(URL)
    case invalidArrayDimensions(String)
}

/// Representation of an image resource for model input.
public enum InputImage: Sendable {
    case ciImage(CIImage)
    case url(URL)
    case array(MLXArray)

    /// Convert to CIImage regardless of source.
    public func asCIImage() throws -> CIImage {
        switch self {
        case .ciImage(let image):
            return image
        case .url(let url):
            guard let image = CIImage(contentsOf: url) else {
                throw MediaError.unableToLoad(url)
            }
            return image
        case .array(let array):
            guard array.ndim == 3 else {
                throw MediaError.invalidArrayDimensions(
                    "array must have 3 dimensions: \(array.ndim)")
            }
            var array = array
            if array.max().item(Float.self) <= 1.0 {
                array = array * 255
            }
            // Channels-first to channels-last
            switch array.dim(0) {
            case 3, 4:
                array = array.transposed(1, 2, 0)
            default:
                break
            }
            // Pad to 4 channels if needed
            switch array.dim(-1) {
            case 3:
                array = padded(array, widths: [0, 0, [0, 1]], value: MLXArray(255))
            case 4:
                break
            default:
                throw MediaError.invalidArrayDimensions(
                    "channel dimension must be last and 3/4: \(array.shape)")
            }
            let arrayData = array.asData()
            let (H, W, _) = array.shape3
            let cs = CGColorSpace(name: CGColorSpace.sRGB)!
            return CIImage(
                bitmapData: arrayData.data, bytesPerRow: W * 4,
                size: .init(width: W, height: H),
                format: .RGBA8, colorSpace: cs)
        }
    }
}

/// Decoded video frame with timestamp.
public struct VideoFrame: Sendable {
    public let frame: CIImage
    public let timeStamp: CMTime

    public init(frame: CIImage, timeStamp: CMTime) {
        self.frame = frame
        self.timeStamp = timeStamp
    }
}

/// Representation of a video resource for model input.
public enum InputVideo: Sendable {
    case url(URL)
    case frames([VideoFrame])
}

/// Processing options for media.
public struct MediaProcessing: Sendable {
    /// Target resize dimensions (nil = no resize).
    public var resize: CGSize?

    public init(resize: CGSize? = nil) {
        self.resize = resize
    }
}

// MARK: - UserInput Type Aliases

extension UserInput {
    public typealias Image = InputImage
    public typealias Video = InputVideo
    public typealias Processing = MediaProcessing
}
