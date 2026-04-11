import CoreGraphics
import Foundation
import ImageIO

struct QwenVisionImagePreprocessor {
    private let configuration: ModelVisionConfiguration

    init(configuration: ModelVisionConfiguration) {
        self.configuration = configuration
    }

    func prepare(_ image: InputImage) throws -> PreparedPrompt.Multimodal.Image {
        let decodedImage = try decodeCGImage(image)
        let patchSize = configuration.patchSize ?? 16
        let temporalPatchSize = configuration.temporalPatchSize ?? 2
        let mergeSize = configuration.mergeSize ?? configuration.spatialMergeSize ?? 2
        let minPixels = configuration.minimumPixelCount ?? 56 * 56
        let maxPixels = configuration.maximumPixelCount ?? 28 * 28 * 1280
        let factor = patchSize * mergeSize
        let resizedSize = smartResize(
            height: decodedImage.height,
            width: decodedImage.width,
            factor: factor,
            minPixels: minPixels,
            maxPixels: maxPixels
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
            patchSize: patchSize,
            temporalPatchSize: temporalPatchSize,
            mergeSize: mergeSize
        )
        let placeholderTokenCount = (gridH * gridW) / (mergeSize * mergeSize)

        return PreparedPrompt.Multimodal.Image(
            gridTHW: [1, gridH, gridW],
            placeholderTokenCount: placeholderTokenCount,
            pixelValuesShape: [
                gridH * gridW,
                3 * temporalPatchSize * patchSize * patchSize,
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
        return [0.5, 0.5, 0.5]
    }

    private var resolvedImageStd: [Float] {
        let std = configuration.imageStd
        if std.count == 3 {
            return std.map(Float.init)
        }
        return [0.5, 0.5, 0.5]
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
            throw LanguageModelContextError.unsupportedInputForModel(
                "Could not decode image data for Qwen vision preprocessing."
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
            throw LanguageModelContextError.unsupportedInputForModel(
                "Could not create image resize context for Qwen vision preprocessing."
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
        patchSize: Int,
        temporalPatchSize: Int,
        mergeSize: Int
    ) -> [Float] {
        let gridH = height / patchSize
        let gridW = width / patchSize
        let blockH = gridH / mergeSize
        let blockW = gridW / mergeSize
        let flattenedPatchSize = 3 * temporalPatchSize * patchSize * patchSize
        var flattened = [Float]()
        flattened.reserveCapacity(gridH * gridW * flattenedPatchSize)

        for blockY in 0..<blockH {
            for blockX in 0..<blockW {
                for mergeY in 0..<mergeSize {
                    for mergeX in 0..<mergeSize {
                        let patchOriginY = (blockY * mergeSize + mergeY) * patchSize
                        let patchOriginX = (blockX * mergeSize + mergeX) * patchSize
                        for channel in 0..<3 {
                            for temporal in 0..<temporalPatchSize {
                                for patchY in 0..<patchSize {
                                    for patchX in 0..<patchSize {
                                        let sourceY = patchOriginY + patchY
                                        let sourceX = patchOriginX + patchX
                                        let sourceIndex = ((sourceY * width + sourceX) * 3) + channel
                                        let value = normalizedPixels[sourceIndex]
                                        flattened.append(value)
                                        if temporal + 1 < temporalPatchSize {
                                            continue
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        return flattened
    }

    private func smartResize(
        height: Int,
        width: Int,
        factor: Int,
        minPixels: Int,
        maxPixels: Int
    ) -> (height: Int, width: Int) {
        let aspectRatio = Double(max(height, width)) / Double(min(height, width))
        if aspectRatio > 200 {
            return (height: factor, width: factor)
        }

        var resizedHeight = roundedToFactor(height, factor: factor)
        var resizedWidth = roundedToFactor(width, factor: factor)
        let originalPixels = Double(height * width)
        let resizedPixels = resizedHeight * resizedWidth

        if resizedPixels > maxPixels {
            let beta = sqrt(originalPixels / Double(maxPixels))
            resizedHeight = max(factor, Int(floor(Double(height) / beta / Double(factor))) * factor)
            resizedWidth = max(factor, Int(floor(Double(width) / beta / Double(factor))) * factor)
        } else if resizedPixels < minPixels {
            let beta = sqrt(Double(minPixels) / originalPixels)
            resizedHeight = Int(ceil(Double(height) * beta / Double(factor))) * factor
            resizedWidth = Int(ceil(Double(width) * beta / Double(factor))) * factor
        }

        return (height: resizedHeight, width: resizedWidth)
    }

    private func roundedToFactor(_ value: Int, factor: Int) -> Int {
        max(factor, Int((Double(value) / Double(factor)).rounded()) * factor)
    }
}
