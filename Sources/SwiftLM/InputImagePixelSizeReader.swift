import CoreGraphics
import Foundation
import ImageIO

struct InputImagePixelSizeReader {
    struct PixelSize: Sendable {
        let width: Int
        let height: Int
    }

    func read(_ image: InputImage) throws -> PixelSize {
        let source: CGImageSource?
        switch image.source {
        case .fileURL(let url):
            source = CGImageSourceCreateWithURL(url as CFURL, nil)
        case .data(let data, _):
            source = CGImageSourceCreateWithData(data as CFData, nil)
        }

        guard let source,
              let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any],
              let width = properties[kCGImagePropertyPixelWidth] as? Int,
              let height = properties[kCGImagePropertyPixelHeight] as? Int else {
            throw LanguageModelContextError.unsupportedInputForModel(
                "Could not read image dimensions from the provided input."
            )
        }

        return PixelSize(width: width, height: height)
    }
}
