import CoreImage
import Foundation
import MLX

/// Image preprocessor for Qwen2.5-VL.
///
/// Performs smart resizing (dimensions divisible by 28, aspect ratio preserved),
/// normalization, and pixel array construction for the vision encoder.
struct Qwen25VLImageProcessor: Sendable {

    let imageFactor: Int
    let minPixels: Int
    let maxPixels: Int
    let imageMean: (Float, Float, Float)
    let imageStd: (Float, Float, Float)
    let temporalPatchSize: Int

    init(
        config: Qwen25VLConfiguration.VisionConfiguration = .init(),
        minPixels: Int = 3136,
        maxPixels: Int = 12_845_056,
        imageMean: (Float, Float, Float) = (0.5, 0.5, 0.5),
        imageStd: (Float, Float, Float) = (0.5, 0.5, 0.5)
    ) {
        self.imageFactor = config.imageFactor
        self.minPixels = minPixels
        self.maxPixels = maxPixels
        self.imageMean = imageMean
        self.imageStd = imageStd
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

        // Resize the image
        let scaleX = CGFloat(resizedWidth) / image.extent.width
        let scaleY = CGFloat(resizedHeight) / image.extent.height
        let resized = image.transformed(by: CGAffineTransform(scaleX: scaleX, y: scaleY))

        // Render to pixel buffer
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
                // Rescale [0,255] → [0,1], then normalize
                floatPixels[dstIdx + 0] = (Float(pixelData[srcIdx + 0]) / 255.0 - imageMean.0) / imageStd.0
                floatPixels[dstIdx + 1] = (Float(pixelData[srcIdx + 1]) / 255.0 - imageMean.1) / imageStd.1
                floatPixels[dstIdx + 2] = (Float(pixelData[srcIdx + 2]) / 255.0 - imageMean.2) / imageStd.2
            }
        }

        // Create MLXArray: [1, T=temporalPatchSize, H, W, C=3] (NDHWC for Conv3d)
        // For single image, duplicate frame to match temporal_patch_size
        let pixels = MLXArray(floatPixels, [1, height, width, 3])
        let temporal = tiled(pixels.expandedDimensions(axis: 1), repetitions: [1, temporalPatchSize, 1, 1, 1])

        // Grid dimensions (in patch units, before spatial merge)
        let patchSize = imageFactor / (imageFactor / 14)  // = 14
        let gridH = height / patchSize
        let gridW = width / patchSize
        let gridT = temporalPatchSize / temporalPatchSize  // = 1

        let thw = LMInput.THW(t: gridT, h: gridH, w: gridW)
        return (temporal, thw)
    }

    /// Preprocess multiple images.
    func preprocess(images: [CIImage]) throws -> (MLXArray, [LMInput.THW]) {
        var allPixels = [MLXArray]()
        var allTHW = [LMInput.THW]()

        for image in images {
            let (pixels, thw) = try preprocess(image: image)
            allPixels.append(pixels)
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

        // Round to nearest multiple of factor
        h = max(factor, ((h + factor / 2) / factor) * factor)
        w = max(factor, ((w + factor / 2) / factor) * factor)

        let totalPixels = h * w

        // Scale down if too many pixels
        if totalPixels > maxPixels {
            let scale = sqrt(Double(maxPixels) / Double(totalPixels))
            h = Int(Double(h) * scale)
            w = Int(Double(w) * scale)
            h = max(factor, (h / factor) * factor)
            w = max(factor, (w / factor) * factor)
        }

        // Scale up if too few pixels
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
