import AVFoundation
import CoreGraphics
import Foundation

struct QwenVisionVideoPreprocessor {
    private let configuration: ModelVisionConfiguration

    init(configuration: ModelVisionConfiguration) {
        self.configuration = configuration
    }

    func prepare(_ video: InputVideo) async throws -> PreparedPrompt.Multimodal.Video {
        let decoded = try await decodeFrames(from: video)
        let patchSize = configuration.patchSize ?? 16
        let temporalPatchSize = configuration.temporalPatchSize ?? 2
        let mergeSize = configuration.mergeSize ?? configuration.spatialMergeSize ?? 2
        let minPixels = configuration.minimumPixelCount ?? 128 * 128
        let maxPixels = configuration.maximumPixelCount ?? 16 * 16 * 2 * 2 * 2 * 6144
        let factor = patchSize * mergeSize
        let resizedSize = smartResize(
            frameCount: decoded.frames.count,
            height: decoded.height,
            width: decoded.width,
            temporalFactor: temporalPatchSize,
            factor: factor,
            minPixels: minPixels,
            maxPixels: maxPixels
        )

        let normalizedFrames = try decoded.frames.map {
            try resizeAndNormalize(
                $0,
                width: resizedSize.width,
                height: resizedSize.height,
                mean: resolvedImageMean,
                std: resolvedImageStd
            )
        }
        let gridH = resizedSize.height / patchSize
        let gridW = resizedSize.width / patchSize
        let flattenedPatches = makeFlattenedPatches(
            normalizedFrames: normalizedFrames,
            width: resizedSize.width,
            height: resizedSize.height,
            patchSize: patchSize,
            temporalPatchSize: temporalPatchSize,
            mergeSize: mergeSize
        )
        let paddedFrameCount = paddedFrameIndices(
            decoded.frameIndices,
            temporalPatchSize: temporalPatchSize
        ).count
        let gridT = paddedFrameCount / temporalPatchSize
        let placeholderTokenCount = (gridT * gridH * gridW) / (mergeSize * mergeSize)
        let timestamps = averagedTimestamps(
            frameIndices: decoded.frameIndices,
            framesPerSecond: decoded.sourceFramesPerSecond,
            temporalPatchSize: temporalPatchSize
        )

        return PreparedPrompt.Multimodal.Video(
            gridTHW: [gridT, gridH, gridW],
            placeholderTokenCount: placeholderTokenCount,
            pixelValuesShape: [
                gridT * gridH * gridW,
                3 * temporalPatchSize * patchSize * patchSize,
            ],
            pixelValues: flattenedPatches,
            frameTimestamps: timestamps,
            sampledFrameCount: decoded.frameIndices.count,
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

    private func decodeFrames(from video: InputVideo) async throws -> DecodedVideo {
        let targetFPS = configuration.videoFramesPerSecond ?? 2
        let minFrames = configuration.minimumFrameCount ?? 4
        let maxFrames = configuration.maximumFrameCount ?? 768

        switch video.source {
        case .fileURL(let url):
            return try await decodeFrames(
                from: AVURLAsset(url: url),
                targetFPS: targetFPS,
                minFrames: minFrames,
                maxFrames: maxFrames
            )
        case .data(let data, let mimeType):
            let temporaryURL = FileManager.default.temporaryDirectory
                .appendingPathComponent(UUID().uuidString)
                .appendingPathExtension(fileExtension(for: mimeType))
            try data.write(to: temporaryURL, options: .atomic)
            defer {
                do {
                    try FileManager.default.removeItem(at: temporaryURL)
                } catch {
                    print("[QwenVisionVideoPreprocessor] failed to remove temp file: \(error)")
                }
            }
            return try await decodeFrames(
                from: AVURLAsset(url: temporaryURL),
                targetFPS: targetFPS,
                minFrames: minFrames,
                maxFrames: maxFrames
            )
        }
    }

    private func decodeFrames(
        from asset: AVAsset,
        targetFPS: Double,
        minFrames: Int,
        maxFrames: Int
    ) async throws -> DecodedVideo {
        do {
            guard let track = try await asset.loadTracks(withMediaType: .video).first else {
                throw InferenceSessionError.unsupportedInputForModel(
                    "Could not find a video track for Qwen vision preprocessing."
                )
            }
            let duration = try await asset.load(.duration)
            let durationSeconds = CMTimeGetSeconds(duration)
            guard durationSeconds.isFinite, durationSeconds > 0 else {
                throw InferenceSessionError.unsupportedInputForModel(
                    "Video duration is invalid for Qwen vision preprocessing."
                )
            }

            let nominalFrameRate = try await track.load(.nominalFrameRate)
            let sourceFramesPerSecond = nominalFrameRate > 0
                ? Double(nominalFrameRate)
                : 24.0
            let totalFrameCount = max(1, Int((durationSeconds * sourceFramesPerSecond).rounded()))
            let desiredFrameCount = min(
                max(Int((Double(totalFrameCount) / sourceFramesPerSecond * targetFPS).rounded(.down)), minFrames),
                maxFrames,
                totalFrameCount
            )
            let frameIndices = sampledFrameIndices(
                totalFrameCount: totalFrameCount,
                desiredFrameCount: desiredFrameCount
            )

            let generator = AVAssetImageGenerator(asset: asset)
            generator.appliesPreferredTrackTransform = true
            generator.requestedTimeToleranceBefore = .zero
            generator.requestedTimeToleranceAfter = .zero

            let times = frameIndices.map {
                CMTime(seconds: Double($0) / sourceFramesPerSecond, preferredTimescale: 600)
            }
            var frames: [CGImage] = []
            frames.reserveCapacity(frameIndices.count)
            for time in times {
                let image = try await generator.image(at: time).image
                frames.append(image)
            }
            guard let firstFrame = frames.first else {
                throw InferenceSessionError.unsupportedInputForModel(
                    "Could not decode video frames for Qwen vision preprocessing."
                )
            }

            return DecodedVideo(
                frames: frames,
                frameIndices: frameIndices,
                sourceFramesPerSecond: sourceFramesPerSecond,
                width: firstFrame.width,
                height: firstFrame.height
            )
        } catch let error as InferenceSessionError {
            throw error
        } catch {
            throw InferenceSessionError.unsupportedInputForModel(
                "Could not decode video frames for Qwen vision preprocessing: \(error.localizedDescription)"
            )
        }
    }

    private func sampledFrameIndices(
        totalFrameCount: Int,
        desiredFrameCount: Int
    ) -> [Int] {
        guard desiredFrameCount > 1 else {
            return [0]
        }
        let denominator = Double(desiredFrameCount - 1)
        return (0..<desiredFrameCount).map { sampleIndex in
            Int((Double(sampleIndex) * Double(totalFrameCount - 1) / denominator).rounded())
        }
    }

    private func paddedFrameIndices(
        _ frameIndices: [Int],
        temporalPatchSize: Int
    ) -> [Int] {
        var padded = frameIndices
        let remainder = padded.count % temporalPatchSize
        if remainder != 0, let last = padded.last {
            padded.append(contentsOf: Array(repeating: last, count: temporalPatchSize - remainder))
        }
        return padded
    }

    private func averagedTimestamps(
        frameIndices: [Int],
        framesPerSecond: Double,
        temporalPatchSize: Int
    ) -> [Double] {
        let padded = paddedFrameIndices(frameIndices, temporalPatchSize: temporalPatchSize)
        let timestamps = padded.map { Double($0) / framesPerSecond }
        return stride(from: 0, to: timestamps.count, by: temporalPatchSize).map { start in
            let end = start + temporalPatchSize - 1
            return (timestamps[start] + timestamps[end]) / 2
        }
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
            throw InferenceSessionError.unsupportedInputForModel(
                "Could not create video resize context for Qwen vision preprocessing."
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
        normalizedFrames: [[Float]],
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

        var paddedFrames = normalizedFrames
        if let lastFrame = paddedFrames.last {
            let remainder = paddedFrames.count % temporalPatchSize
            if remainder != 0 {
                paddedFrames.append(
                    contentsOf: Array(
                        repeating: lastFrame,
                        count: temporalPatchSize - remainder
                    )
                )
            }
        }

        let gridT = paddedFrames.count / temporalPatchSize
        var flattened: [Float] = []
        flattened.reserveCapacity(gridT * gridH * gridW * flattenedPatchSize)

        for temporalBlock in 0..<gridT {
            for blockY in 0..<blockH {
                for blockX in 0..<blockW {
                    for mergeY in 0..<mergeSize {
                        for mergeX in 0..<mergeSize {
                            let patchOriginY = (blockY * mergeSize + mergeY) * patchSize
                            let patchOriginX = (blockX * mergeSize + mergeX) * patchSize
                            for channel in 0..<3 {
                                for temporalOffset in 0..<temporalPatchSize {
                                    let frame = paddedFrames[temporalBlock * temporalPatchSize + temporalOffset]
                                    for patchY in 0..<patchSize {
                                        for patchX in 0..<patchSize {
                                            let sourceY = patchOriginY + patchY
                                            let sourceX = patchOriginX + patchX
                                            let sourceIndex = ((sourceY * width + sourceX) * 3) + channel
                                            flattened.append(frame[sourceIndex])
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
        frameCount: Int,
        height: Int,
        width: Int,
        temporalFactor: Int,
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
        let resizedFrameCount = Int(ceil(Double(frameCount) / Double(temporalFactor))) * temporalFactor
        let resizedPixels = resizedFrameCount * resizedHeight * resizedWidth
        let originalPixels = Double(frameCount * height * width)

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

    private func fileExtension(for mimeType: String?) -> String {
        switch mimeType {
        case "video/quicktime":
            return "mov"
        case "video/webm":
            return "webm"
        default:
            return "mp4"
        }
    }
}

private struct DecodedVideo {
    let frames: [CGImage]
    let frameIndices: [Int]
    let sourceFramesPerSecond: Double
    let width: Int
    let height: Int
}
