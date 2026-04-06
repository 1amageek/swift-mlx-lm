import AVFoundation
import CoreGraphics
import CoreVideo
import Foundation

enum TestVideoFixtures {
    static func makeMP4Data(
        frameCount: Int = 4,
        width: Int = 32,
        height: Int = 32,
        framesPerSecond: Int32 = 2
    ) async throws -> Data {
        let outputURL = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
            .appendingPathExtension("mp4")
        defer {
            do {
                try FileManager.default.removeItem(at: outputURL)
            } catch {
                print("[TestVideoFixtures] failed to remove temp file: \(error)")
            }
        }

        let writer = try AVAssetWriter(outputURL: outputURL, fileType: .mp4)
        let settings: [String: Any] = [
            AVVideoCodecKey: AVVideoCodecType.h264,
            AVVideoWidthKey: width,
            AVVideoHeightKey: height,
        ]
        let input = AVAssetWriterInput(mediaType: .video, outputSettings: settings)
        input.expectsMediaDataInRealTime = false
        let adaptor = AVAssetWriterInputPixelBufferAdaptor(
            assetWriterInput: input,
            sourcePixelBufferAttributes: [
                kCVPixelBufferPixelFormatTypeKey as String: Int(kCVPixelFormatType_32ARGB),
                kCVPixelBufferWidthKey as String: width,
                kCVPixelBufferHeightKey as String: height,
            ]
        )

        guard writer.canAdd(input) else {
            throw CocoaError(.fileWriteUnknown)
        }
        writer.add(input)
        writer.startWriting()
        writer.startSession(atSourceTime: .zero)

        for frameIndex in 0..<frameCount {
            while !input.isReadyForMoreMediaData {
                try await Task.sleep(for: .milliseconds(10))
            }
            let pixelBuffer = try makePixelBuffer(
                width: width,
                height: height,
                frameIndex: frameIndex
            )
            let presentationTime = CMTime(value: Int64(frameIndex), timescale: framesPerSecond)
            guard adaptor.append(pixelBuffer, withPresentationTime: presentationTime) else {
                throw CocoaError(.fileWriteUnknown)
            }
        }

        input.markAsFinished()
        try await finishWriting(writer)
        return try Data(contentsOf: outputURL)
    }

    static func writeTemporaryMP4(
        frameCount: Int = 4,
        width: Int = 32,
        height: Int = 32,
        framesPerSecond: Int32 = 2
    ) async throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
            .appendingPathExtension("mp4")
        let data = try await makeMP4Data(
            frameCount: frameCount,
            width: width,
            height: height,
            framesPerSecond: framesPerSecond
        )
        try data.write(to: url, options: .atomic)
        return url
    }

    private static func finishWriting(_ writer: AVAssetWriter) async throws {
        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            writer.finishWriting {
                if let error = writer.error {
                    continuation.resume(throwing: error)
                } else {
                    continuation.resume()
                }
            }
        }
    }

    private static func makePixelBuffer(
        width: Int,
        height: Int,
        frameIndex: Int
    ) throws -> CVPixelBuffer {
        var pixelBuffer: CVPixelBuffer?
        let status = CVPixelBufferCreate(
            nil,
            width,
            height,
            kCVPixelFormatType_32ARGB,
            [
                kCVPixelBufferCGImageCompatibilityKey: true,
                kCVPixelBufferCGBitmapContextCompatibilityKey: true,
            ] as CFDictionary,
            &pixelBuffer
        )
        guard status == kCVReturnSuccess, let pixelBuffer else {
            throw CocoaError(.coderInvalidValue)
        }

        CVPixelBufferLockBaseAddress(pixelBuffer, [])
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, []) }

        let colorSpace = CGColorSpace(name: CGColorSpace.sRGB) ?? CGColorSpaceCreateDeviceRGB()
        guard let context = CGContext(
            data: CVPixelBufferGetBaseAddress(pixelBuffer),
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: CVPixelBufferGetBytesPerRow(pixelBuffer),
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue | CGBitmapInfo.byteOrder32Big.rawValue
        ) else {
            throw CocoaError(.coderInvalidValue)
        }

        let hue = CGFloat(frameIndex % 8) / 8.0
        context.setFillColor(
            CGColor(
                colorSpace: colorSpace,
                components: [hue, 1.0 - hue, 0.5, 1.0]
            ) ?? CGColor(gray: 0.5, alpha: 1.0)
        )
        context.fill(CGRect(x: 0, y: 0, width: width, height: height))
        return pixelBuffer
    }
}
