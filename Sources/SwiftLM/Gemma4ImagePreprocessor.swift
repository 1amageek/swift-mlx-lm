import CoreGraphics
import Foundation
import ImageIO

struct Gemma4ImagePreprocessor {
    private let configuration: ModelVisionConfiguration

    init(configuration: ModelVisionConfiguration) {
        self.configuration = configuration
    }

    func prepare(_ image: InputImage) throws -> PreparedInput.Multimodal.Image {
        let decodedImage = try decodeCGImage(image)
        let patchSize = configuration.patchSize ?? 16
        let poolingKernelSize = configuration.poolingKernelSize ?? 3
        let maxSoftTokens = configuration.defaultOutputLength ?? 280
        let maxPatches = maxSoftTokens * poolingKernelSize * poolingKernelSize
        let resizedSize = try aspectRatioPreservingResize(
            height: decodedImage.height,
            width: decodedImage.width,
            patchSize: patchSize,
            maxPatches: maxPatches,
            poolingKernelSize: poolingKernelSize
        )
        let normalizedPixels = try resizeAndNormalize(
            decodedImage.image,
            width: resizedSize.width,
            height: resizedSize.height,
            mean: resolvedImageMean,
            std: resolvedImageStd
        )
        let gridH = resizedSize.height / patchSize
        let gridW = resizedSize.width / patchSize
        let flattenedPatches = makeFlattenedPatches(
            normalizedPixels: normalizedPixels,
            width: resizedSize.width,
            height: resizedSize.height,
            patchSize: patchSize
        )
        let placeholderTokenCount = (gridH * gridW) / (poolingKernelSize * poolingKernelSize)

        return PreparedInput.Multimodal.Image(
            gridTHW: [1, gridH, gridW],
            placeholderTokenCount: placeholderTokenCount,
            pixelValuesShape: [
                gridH * gridW,
                3 * patchSize * patchSize,
            ],
            pixelValues: flattenedPatches,
            resizedSize: [resizedSize.height, resizedSize.width]
        )
    }

    private var resolvedImageMean: [Float] {
        let mean = configuration.imageMean
        if mean.count == 3 {
            return mean.map(Float.init)
        }
        return [0, 0, 0]
    }

    private var resolvedImageStd: [Float] {
        let std = configuration.imageStd
        if std.count == 3 {
            return std.map(Float.init)
        }
        return [1, 1, 1]
    }

    private func decodeCGImage(_ image: InputImage) throws -> (image: CGImage, width: Int, height: Int) {
        let source: CGImageSource?
        switch image.source {
        case .fileURL(let url):
            source = CGImageSourceCreateWithURL(url as CFURL, nil)
        case .data(let data, _):
            source = CGImageSourceCreateWithData(data as CFData, nil)
        }

        guard let source,
              let cgImage = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
            throw ModelContainerError.unsupportedInputForModel(
                "Could not decode image data for Gemma4 preprocessing."
            )
        }
        return (cgImage, cgImage.width, cgImage.height)
    }

    private func resizeAndNormalize(
        _ image: CGImage,
        width: Int,
        height: Int,
        mean: [Float],
        std: [Float]
    ) throws -> [Float] {
        let colorSpace = CGColorSpace(name: CGColorSpace.sRGB) ?? CGColorSpaceCreateDeviceRGB()
        let bytesPerPixel = 4
        let bytesPerRow = width * bytesPerPixel
        var rgba = [UInt8](repeating: 0, count: height * bytesPerRow)

        guard let context = CGContext(
            data: &rgba,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: bytesPerRow,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue
        ) else {
            throw ModelContainerError.unsupportedInputForModel(
                "Could not create image resize context for Gemma4 preprocessing."
            )
        }

        context.interpolationQuality = .high
        context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))

        var rgb = [Float](repeating: 0, count: width * height * 3)
        for y in 0..<height {
            for x in 0..<width {
                let rgbaIndex = y * bytesPerRow + x * bytesPerPixel
                let pixelIndex = (y * width + x) * 3
                let red = Float(rgba[rgbaIndex]) / 255.0
                let green = Float(rgba[rgbaIndex + 1]) / 255.0
                let blue = Float(rgba[rgbaIndex + 2]) / 255.0
                rgb[pixelIndex] = (red - mean[0]) / std[0]
                rgb[pixelIndex + 1] = (green - mean[1]) / std[1]
                rgb[pixelIndex + 2] = (blue - mean[2]) / std[2]
            }
        }
        return rgb
    }

    private func makeFlattenedPatches(
        normalizedPixels: [Float],
        width: Int,
        height: Int,
        patchSize: Int
    ) -> [Float] {
        let gridH = height / patchSize
        let gridW = width / patchSize
        let flattenedPatchSize = 3 * patchSize * patchSize
        var flattened = [Float]()
        flattened.reserveCapacity(gridH * gridW * flattenedPatchSize)

        for patchYIndex in 0..<gridH {
            for patchXIndex in 0..<gridW {
                let patchOriginY = patchYIndex * patchSize
                let patchOriginX = patchXIndex * patchSize
                for channel in 0..<3 {
                    for patchY in 0..<patchSize {
                        for patchX in 0..<patchSize {
                            let sourceY = patchOriginY + patchY
                            let sourceX = patchOriginX + patchX
                            let sourceIndex = ((sourceY * width + sourceX) * 3) + channel
                            flattened.append(normalizedPixels[sourceIndex])
                        }
                    }
                }
            }
        }
        return flattened
    }

    private func aspectRatioPreservingResize(
        height: Int,
        width: Int,
        patchSize: Int,
        maxPatches: Int,
        poolingKernelSize: Int
    ) throws -> (height: Int, width: Int) {
        let totalPixels = height * width
        let targetPixels = maxPatches * patchSize * patchSize
        let factor = sqrt(Double(targetPixels) / Double(max(totalPixels, 1)))
        let idealHeight = factor * Double(height)
        let idealWidth = factor * Double(width)
        let sideMultiple = poolingKernelSize * patchSize

        var targetHeight = Int(floor(idealHeight / Double(sideMultiple))) * sideMultiple
        var targetWidth = Int(floor(idealWidth / Double(sideMultiple))) * sideMultiple

        if targetHeight == 0 && targetWidth == 0 {
            throw ModelContainerError.multimodalInputNotSupported(
                "Gemma4 image resize collapsed to zero dimensions."
            )
        }

        let maxSideLength = (maxPatches / (poolingKernelSize * poolingKernelSize)) * sideMultiple
        if targetHeight == 0 {
            targetHeight = sideMultiple
            targetWidth = min(maxSideLength, max(sideMultiple, Int(floor(Double(width) / Double(height))) * sideMultiple))
        } else if targetWidth == 0 {
            targetWidth = sideMultiple
            targetHeight = min(maxSideLength, max(sideMultiple, Int(floor(Double(height) / Double(width))) * sideMultiple))
        }

        return (height: targetHeight, width: targetWidth)
    }
}
