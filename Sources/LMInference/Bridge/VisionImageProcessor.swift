import CoreImage
import Foundation
import MLX

/// Unified image preprocessor for vision-language models.
///
/// Performs smart resizing (dimensions divisible by imageFactor, aspect ratio preserved),
/// normalization, and pixel array construction. All parameters come from ``VisionConfig``.
///
/// For Conv3d-based encoders (`temporalPatchSize > 1`), the single frame is
/// tiled along the temporal axis. For Conv2d-based encoders (`temporalPatchSize == 1`),
/// the output is `[1, H, W, C]`.
struct VisionImageProcessor: Sendable {

    let imageFactor: Int
    let patchSize: Int
    let spatialMergeSize: Int
    let minPixels: Int
    let maxPixels: Int
    let imageMean: [Float]
    let imageStd: [Float]
    let temporalPatchSize: Int

    init(config: VisionConfig) {
        self.imageFactor = config.imageFactor
        self.patchSize = config.patchSize
        self.spatialMergeSize = config.spatialMergeSize
        self.minPixels = config.minPixels
        self.maxPixels = config.maxPixels
        self.imageMean = config.imageMean
        self.imageStd = config.imageStd
        self.temporalPatchSize = config.temporalPatchSize
    }

    /// Preprocess a single image for the vision encoder.
    ///
    /// - Parameter image: Input image in any format.
    /// - Returns: Pixel tensor and grid dimensions for the vision encoder.
    func preprocess(image: CIImage) throws -> (MLXArray, LMInput.THW) {
        let (resizedWidth, resizedHeight) = smartResize(
            width: Int(image.extent.width),
            height: Int(image.extent.height)
        )

        let scaleX = CGFloat(resizedWidth) / image.extent.width
        let scaleY = CGFloat(resizedHeight) / image.extent.height
        let resized = image.transformed(by: CGAffineTransform(scaleX: scaleX, y: scaleY))

        let context = CIContext()
        let colorSpace = CGColorSpace(name: CGColorSpace.sRGB)!
        let width = resizedWidth
        let height = resizedHeight
        let bytesPerRow = width * 4

        var pixelData = [UInt8](repeating: 0, count: height * bytesPerRow)
        context.render(
            resized,
            toBitmap: &pixelData,
            rowBytes: bytesPerRow,
            bounds: CGRect(x: 0, y: 0, width: width, height: height),
            format: .RGBA8,
            colorSpace: colorSpace
        )

        // Convert to float and normalize: [H, W, 3]
        var floatPixels = [Float](repeating: 0, count: height * width * 3)
        for y in 0..<height {
            for x in 0..<width {
                let srcIdx = y * bytesPerRow + x * 4
                let dstIdx = y * width * 3 + x * 3
                floatPixels[dstIdx + 0] = (Float(pixelData[srcIdx + 0]) / 255.0 - imageMean[0]) / imageStd[0]
                floatPixels[dstIdx + 1] = (Float(pixelData[srcIdx + 1]) / 255.0 - imageMean[1]) / imageStd[1]
                floatPixels[dstIdx + 2] = (Float(pixelData[srcIdx + 2]) / 255.0 - imageMean[2]) / imageStd[2]
            }
        }

        let pixels: MLXArray
        if temporalPatchSize > 1 {
            // Conv3d: [1, T, H, W, C] — tile single frame along temporal axis
            let frame = MLXArray(floatPixels, [1, height, width, 3])
            pixels = tiled(
                frame.expandedDimensions(axis: 1),
                repetitions: [1, temporalPatchSize, 1, 1, 1]
            )
        } else {
            // Conv2d: [1, H, W, C]
            pixels = MLXArray(floatPixels, [1, height, width, 3])
        }

        let gridH = height / patchSize
        let gridW = width / patchSize
        let gridT = 1

        let thw = LMInput.THW(t: gridT, h: gridH, w: gridW)
        return (pixels, thw)
    }

    /// Preprocess multiple images.
    func preprocess(images: [CIImage]) throws -> (MLXArray, [LMInput.THW]) {
        var allPixels = [MLXArray]()
        var allTHW = [LMInput.THW]()

        for image in images {
            let (p, thw) = try preprocess(image: image)
            allPixels.append(p)
            allTHW.append(thw)
        }

        let combined = concatenated(allPixels, axis: 0)
        return (combined, allTHW)
    }

    // MARK: - Smart Resize

    /// Compute target dimensions preserving aspect ratio with both sides divisible by imageFactor.
    ///
    /// Constrains total pixel count between minPixels and maxPixels.
    func smartResize(width: Int, height: Int) -> (width: Int, height: Int) {
        let factor = imageFactor
        var h = height
        var w = width

        h = max(factor, ((h + factor / 2) / factor) * factor)
        w = max(factor, ((w + factor / 2) / factor) * factor)

        let totalPixels = h * w

        if totalPixels > maxPixels {
            let scale = sqrt(Double(maxPixels) / Double(totalPixels))
            h = Int(Double(h) * scale)
            w = Int(Double(w) * scale)
            h = max(factor, (h / factor) * factor)
            w = max(factor, (w / factor) * factor)
        }

        if h * w < minPixels {
            let scale = sqrt(Double(minPixels) / Double(h * w))
            h = Int(Double(h) * scale)
            w = Int(Double(w) * scale)
            h = max(factor, ((h + factor - 1) / factor) * factor)
            w = max(factor, ((w + factor - 1) / factor) * factor)
        }

        return (w, h)
    }
}
