import CoreGraphics
import Foundation
import ImageIO
import UniformTypeIdentifiers

enum TestImageFixtures {
    static func makeOnePixelPNGData(
        red: UInt8 = 255,
        green: UInt8 = 255,
        blue: UInt8 = 255,
        alpha: UInt8 = 255
    ) throws -> Data {
        try makePNGData(
            width: 1,
            height: 1,
            red: red,
            green: green,
            blue: blue,
            alpha: alpha
        )
    }

    static func makePNGData(
        width: Int,
        height: Int,
        red: UInt8 = 255,
        green: UInt8 = 255,
        blue: UInt8 = 255,
        alpha: UInt8 = 255
    ) throws -> Data {
        let colorSpace = CGColorSpace(name: CGColorSpace.sRGB) ?? CGColorSpaceCreateDeviceRGB()
        let bytesPerPixel = 4
        let bytesPerRow = width * bytesPerPixel
        var rgba = [UInt8](repeating: 0, count: width * height * bytesPerPixel)
        for pixel in stride(from: 0, to: rgba.count, by: bytesPerPixel) {
            rgba[pixel] = red
            rgba[pixel + 1] = green
            rgba[pixel + 2] = blue
            rgba[pixel + 3] = alpha
        }

        guard let context = CGContext(
            data: &rgba,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: bytesPerRow,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue
        ) else {
            throw CocoaError(.coderInvalidValue)
        }
        guard let image = context.makeImage() else {
            throw CocoaError(.coderInvalidValue)
        }

        let data = NSMutableData()
        guard let destination = CGImageDestinationCreateWithData(
            data,
            UTType.png.identifier as CFString,
            1,
            nil
        ) else {
            throw CocoaError(.coderInvalidValue)
        }
        CGImageDestinationAddImage(destination, image, nil)
        guard CGImageDestinationFinalize(destination) else {
            throw CocoaError(.coderInvalidValue)
        }
        return data as Data
    }

    static func writeTemporaryPNG(
        width: Int = 1,
        height: Int = 1,
        red: UInt8 = 255,
        green: UInt8 = 255,
        blue: UInt8 = 255,
        alpha: UInt8 = 255
    ) throws -> URL {
        let data = try makePNGData(
            width: width,
            height: height,
            red: red,
            green: green,
            blue: blue,
            alpha: alpha
        )
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
            .appendingPathExtension("png")
        try data.write(to: url, options: .atomic)
        return url
    }
}
