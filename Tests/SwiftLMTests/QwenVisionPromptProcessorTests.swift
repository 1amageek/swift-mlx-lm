import Foundation
import Testing
@testable import SwiftLM

@Suite("Qwen Vision Prompt Preparation")
struct QwenVisionPromptProcessorTests {
    @Test("Expand Qwen image placeholder count from processor settings")
    func expandImagePlaceholders() async throws {
        let decoder = HFConfigDecoder()
        let configJSON = QwenVisionTestSupport.completeConfigJSON(includeVideo: false)
        let preprocessorJSON = QwenVisionTestSupport.preprocessorJSON(videoProcessorType: nil)

        let vision = try #require(
            try decoder.visionConfiguration(
                from: Data(configJSON.utf8),
                preprocessorConfigData: Data(preprocessorJSON.utf8)
            )
        )
        let configuration = ModelConfiguration(
            name: "Qwen3-VL",
            eosTokenIds: [],
            inputCapabilities: .init(supportsText: true, supportsImages: true, supportsVideo: false),
            vision: vision
        )
        let processor = QwenVisionPromptProcessor(configuration: configuration)
        let imageData = try TestImageFixtures.makeOnePixelPNGData()

        let prepared = try await processor.prepare(
            renderedText: "<|vision_start|><|image_pad|><|vision_end|>\nDescribe the image.",
            messages: [
                .user([
                    .image(InputImage(data: imageData, mimeType: "image/png")),
                    .text("Describe the image."),
                ])
            ]
        )

        let multimodal = try #require(prepared.multimodal)
        let image = try #require(multimodal.images.first)
        #expect(image.gridTHW == [1, 16, 16])
        #expect(image.placeholderTokenCount == 64)
        #expect(image.pixelValuesShape == [256, 1536])
        #expect(image.pixelValues.count == 256 * 1536)
        #expect(image.resizedSize == [256, 256])
        #expect(prepared.text.components(separatedBy: "<|image_pad|>").count - 1 == 64)
    }

    @Test("Expand Qwen video placeholder count and timestamps from processor settings")
    func expandVideoPlaceholders() async throws {
        let decoder = HFConfigDecoder()
        let configJSON = QwenVisionTestSupport.completeConfigJSON(includeImage: false)
        let preprocessorJSON = QwenVisionTestSupport.preprocessorJSON(imageProcessorType: nil)

        let vision = try #require(
            try decoder.visionConfiguration(
                from: Data(configJSON.utf8),
                preprocessorConfigData: Data(preprocessorJSON.utf8)
            )
        )
        let configuration = ModelConfiguration(
            name: "Qwen3-VL",
            eosTokenIds: [],
            inputCapabilities: .init(supportsText: true, supportsImages: false, supportsVideo: true),
            executionCapabilities: .init(supportsVideoPromptPreparation: true),
            vision: vision
        )
        let processor = QwenVisionPromptProcessor(configuration: configuration)
        let videoData = try await TestVideoFixtures.makeMP4Data()

        let prepared = try await processor.prepare(
            renderedText: "<|vision_start|><|video_pad|><|vision_end|>\nDescribe the video.",
            messages: [
                .user([
                    .video(InputVideo(data: videoData, mimeType: "video/mp4")),
                    .text("Describe the video."),
                ])
            ]
        )

        let multimodal = try #require(prepared.multimodal)
        let video = try #require(multimodal.videos.first)
        #expect(video.gridTHW == [2, 8, 8])
        #expect(video.placeholderTokenCount == 32)
        #expect(video.pixelValuesShape == [128, 1536])
        #expect(video.pixelValues.count == 128 * 1536)
        #expect(video.frameTimestamps.count == 2)
        #expect(prepared.text.contains("<0.2 seconds>"))
        #expect(prepared.text.contains("<1.2 seconds>"))
        #expect(prepared.text.components(separatedBy: "<|video_pad|>").count - 1 == 32)
    }

    @Test("Decode Qwen3-VL processor sizing metadata")
    func decodeProcessorSizingMetadata() throws {
        let decoder = HFConfigDecoder()
        let configJSON = QwenVisionTestSupport.completeConfigJSON(includeVideo: false)
        let preprocessorJSON = QwenVisionTestSupport.preprocessorJSON(videoProcessorType: nil)

        let vision = try #require(
            try decoder.visionConfiguration(
                from: Data(configJSON.utf8),
                preprocessorConfigData: Data(preprocessorJSON.utf8)
            )
        )

        #expect(vision.processorClass == "Qwen3VLProcessor")
        #expect(vision.imageProcessorType == "Qwen2VLImageProcessorFast")
        #expect(vision.patchSize == 16)
        #expect(vision.temporalPatchSize == 2)
        #expect(vision.mergeSize == 2)
        #expect(vision.minimumPixelCount == 65536)
        #expect(vision.maximumPixelCount == 16777216)
        #expect(vision.imageMean == [0.5, 0.5, 0.5])
        #expect(vision.imageStd == [0.5, 0.5, 0.5])
    }

    @Test("Image preprocessor supports file URL and data inputs")
    func imagePreprocessorSupportsFileAndData() throws {
        let vision = QwenVisionTestSupport.visionConfiguration(
            outHiddenSize: 64,
            supportsImages: true,
            supportsVideo: false
        )
        let preprocessor = QwenVisionImagePreprocessor(configuration: vision)
        let imageData = try TestImageFixtures.makeOnePixelPNGData()
        let imageURL = try TestImageFixtures.writeTemporaryPNG()
        defer {
            do {
                try FileManager.default.removeItem(at: imageURL)
            } catch {
                print("[QwenVisionPromptProcessorTests] failed to remove temporary image: \(error)")
            }
        }

        let fromData = try preprocessor.prepare(InputImage(data: imageData, mimeType: "image/png"))
        let fromFile = try preprocessor.prepare(InputImage(fileURL: imageURL))

        #expect(fromData.gridTHW == fromFile.gridTHW)
        #expect(fromData.placeholderTokenCount == fromFile.placeholderTokenCount)
        #expect(fromData.pixelValuesShape == fromFile.pixelValuesShape)
        #expect(fromFile.resizedSize == [4, 4])
    }

    @Test("Image preprocessor clamps and aligns resized dimensions")
    func imagePreprocessorClampsAndAlignsSize() throws {
        let vision = ModelVisionConfiguration(
            patchSize: 16,
            temporalPatchSize: 2,
            mergeSize: 2,
            minimumPixelCount: 16 * 16 * 4,
            maximumPixelCount: 16 * 16 * 8,
            imageMean: [0.5, 0.5, 0.5],
            imageStd: [0.5, 0.5, 0.5]
        )
        let preprocessor = QwenVisionImagePreprocessor(configuration: vision)
        let imageData = try TestImageFixtures.makePNGData(width: 128, height: 32)
        let prepared = try preprocessor.prepare(InputImage(data: imageData, mimeType: "image/png"))

        #expect(prepared.resizedSize[0] % 32 == 0)
        #expect(prepared.resizedSize[1] % 32 == 0)
        let pixelCount = prepared.resizedSize[0] * prepared.resizedSize[1]
        #expect(pixelCount <= 16 * 16 * 8)
        #expect(pixelCount >= 16 * 16 * 4)
    }

    @Test("Image preprocessor rejects invalid image data")
    func rejectInvalidImageData() {
        let preprocessor = QwenVisionImagePreprocessor(
            configuration: QwenVisionTestSupport.visionConfiguration(
                outHiddenSize: 64,
                supportsImages: true,
                supportsVideo: false
            )
        )

        #expect(throws: InferenceSessionError.self) {
            _ = try preprocessor.prepare(InputImage(data: Data("not an image".utf8), mimeType: "image/png"))
        }
    }

    @Test("Video preprocessor supports file URL and data inputs")
    func videoPreprocessorSupportsFileAndData() async throws {
        let vision = QwenVisionTestSupport.visionConfiguration(
            outHiddenSize: 64,
            supportsImages: false,
            supportsVideo: true
        )
        let preprocessor = QwenVisionVideoPreprocessor(configuration: vision)
        let videoData = try await TestVideoFixtures.makeMP4Data()
        let videoURL = try await TestVideoFixtures.writeTemporaryMP4()
        defer {
            do {
                try FileManager.default.removeItem(at: videoURL)
            } catch {
                print("[QwenVisionPromptProcessorTests] failed to remove temporary video: \(error)")
            }
        }

        let fromData = try await preprocessor.prepare(InputVideo(data: videoData, mimeType: "video/mp4"))
        let fromFile = try await preprocessor.prepare(InputVideo(fileURL: videoURL))

        #expect(fromData.gridTHW == fromFile.gridTHW)
        #expect(fromData.placeholderTokenCount == fromFile.placeholderTokenCount)
        #expect(fromData.frameTimestamps == fromFile.frameTimestamps)
    }

    @Test("Video preprocessor honors fps min max and temporal padding")
    func videoPreprocessorHonorsSamplingControls() async throws {
        let lowFPSVision = QwenVisionTestSupport.visionConfiguration(
            outHiddenSize: 64,
            supportsImages: false,
            supportsVideo: true,
            videoFramesPerSecond: 1,
            minimumFrameCount: 4,
            maximumFrameCount: 8
        )
        let highFPSVision = QwenVisionTestSupport.visionConfiguration(
            outHiddenSize: 64,
            supportsImages: false,
            supportsVideo: true,
            videoFramesPerSecond: 4,
            minimumFrameCount: 1,
            maximumFrameCount: 2
        )
        let videoData = try await TestVideoFixtures.makeMP4Data(frameCount: 6, framesPerSecond: 2)

        let lowFPSPrepared = try await QwenVisionVideoPreprocessor(configuration: lowFPSVision)
            .prepare(InputVideo(data: videoData, mimeType: "video/mp4"))
        let highFPSPrepared = try await QwenVisionVideoPreprocessor(configuration: highFPSVision)
            .prepare(InputVideo(data: videoData, mimeType: "video/mp4"))

        #expect(lowFPSPrepared.sampledFrameCount == 4)
        #expect(highFPSPrepared.sampledFrameCount == 2)
        #expect(lowFPSPrepared.gridTHW[0] % 1 == 0)
        #expect(highFPSPrepared.gridTHW[0] == 2)
        #expect(highFPSPrepared.frameTimestamps.count == 2)
    }

    @Test("Video preprocessor rejects invalid video data")
    func rejectInvalidVideoData() async {
        let preprocessor = QwenVisionVideoPreprocessor(
            configuration: QwenVisionTestSupport.visionConfiguration(
                outHiddenSize: 64,
                supportsImages: false,
                supportsVideo: true
            )
        )

        await #expect(throws: InferenceSessionError.self) {
            _ = try await preprocessor.prepare(
                InputVideo(data: Data("not a video".utf8), mimeType: "video/mp4")
            )
        }
    }

    @Test("Reject extra image placeholders")
    func rejectExtraImagePlaceholders() async throws {
        let configuration = ModelConfiguration(
            name: "Qwen3-VL",
            inputCapabilities: .init(supportsText: true, supportsImages: true, supportsVideo: false),
            executionCapabilities: .init(supportsImagePromptPreparation: true),
            vision: ModelVisionConfiguration(
                processorClass: "Qwen3VLProcessor",
                imageTokenID: 151655,
                imageProcessorType: "Qwen2VLImageProcessorFast",
                patchSize: 16,
                temporalPatchSize: 2,
                mergeSize: 2,
                minimumPixelCount: 65536,
                maximumPixelCount: 16777216
            )
        )
        let processor = QwenVisionPromptProcessor(configuration: configuration)
        let imageData = try TestImageFixtures.makeOnePixelPNGData()

        do {
            _ = try await processor.prepare(
                renderedText: "<|vision_start|><|image_pad|><|vision_end|><|vision_start|><|image_pad|><|vision_end|>",
                messages: [
                    .user([
                        .image(InputImage(data: imageData, mimeType: "image/png")),
                    ])
                ]
            )
            Issue.record("Expected extra image placeholders to be rejected")
        } catch is InferenceSessionError {
            // Expected.
        }
    }

    @Test("Reject missing image placeholders")
    func rejectMissingImagePlaceholders() async throws {
        let configuration = ModelConfiguration(
            name: "Qwen3-VL",
            inputCapabilities: .init(supportsText: true, supportsImages: true, supportsVideo: false),
            executionCapabilities: .init(supportsImagePromptPreparation: true),
            vision: ModelVisionConfiguration(
                processorClass: "Qwen3VLProcessor",
                imageTokenID: 151655,
                imageProcessorType: "Qwen2VLImageProcessorFast",
                patchSize: 16,
                temporalPatchSize: 2,
                mergeSize: 2,
                minimumPixelCount: 65536,
                maximumPixelCount: 16777216
            )
        )
        let processor = QwenVisionPromptProcessor(configuration: configuration)
        let imageData = try TestImageFixtures.makeOnePixelPNGData()

        do {
            _ = try await processor.prepare(
                renderedText: "Describe the image.",
                messages: [
                    .user([
                        .image(InputImage(data: imageData, mimeType: "image/png")),
                        .text("Describe the image."),
                    ])
                ]
            )
            Issue.record("Expected missing image placeholders to be rejected")
        } catch is InferenceSessionError {
            // Expected.
        }
    }

    @Test("Reject extra video placeholders")
    func rejectExtraVideoPlaceholders() async throws {
        let configuration = ModelConfiguration(
            name: "Qwen3-VL",
            inputCapabilities: .init(supportsText: true, supportsImages: false, supportsVideo: true),
            executionCapabilities: .init(supportsVideoPromptPreparation: true),
            vision: QwenVisionTestSupport.visionConfiguration(
                outHiddenSize: 64,
                supportsImages: false,
                supportsVideo: true
            )
        )
        let processor = QwenVisionPromptProcessor(configuration: configuration)
        let videoData = try await TestVideoFixtures.makeMP4Data()

        await #expect(throws: InferenceSessionError.self) {
            _ = try await processor.prepare(
                renderedText: "<|vision_start|><|video_pad|><|vision_end|><|vision_start|><|video_pad|><|vision_end|>",
                messages: [
                    .user([
                        .video(InputVideo(data: videoData, mimeType: "video/mp4")),
                    ])
                ]
            )
        }
    }

    @Test("Reject missing video placeholders")
    func rejectMissingVideoPlaceholders() async throws {
        let configuration = ModelConfiguration(
            name: "Qwen3-VL",
            inputCapabilities: .init(supportsText: true, supportsImages: false, supportsVideo: true),
            executionCapabilities: .init(supportsVideoPromptPreparation: true),
            vision: QwenVisionTestSupport.visionConfiguration(
                outHiddenSize: 64,
                supportsImages: false,
                supportsVideo: true
            )
        )
        let processor = QwenVisionPromptProcessor(configuration: configuration)
        let videoData = try await TestVideoFixtures.makeMP4Data()

        await #expect(throws: InferenceSessionError.self) {
            _ = try await processor.prepare(
                renderedText: "Describe the video.",
                messages: [
                    .user([
                        .video(InputVideo(data: videoData, mimeType: "video/mp4")),
                        .text("Describe the video."),
                    ])
                ]
            )
        }
    }

    @Test("Expand mixed image and video prompts and derive multimodal token types")
    func expandMixedImageAndVideoPrompt() async throws {
        let configuration = ModelConfiguration(
            name: "Qwen3-VL",
            inputCapabilities: .init(supportsText: true, supportsImages: true, supportsVideo: true),
            executionCapabilities: .init(
                supportsImagePromptPreparation: true,
                supportsVideoPromptPreparation: true
            ),
            vision: QwenVisionTestSupport.visionConfiguration(outHiddenSize: 64)
        )
        let processor = QwenVisionPromptProcessor(configuration: configuration)
        let tokenizer = QwenVisionTestTokenizer()
        let imageData = try TestImageFixtures.makeOnePixelPNGData()
        let videoData = try await TestVideoFixtures.makeMP4Data()

        let prepared = try await processor.prepare(
            renderedText: """
            system
            <|vision_start|><|image_pad|><|vision_end|>
            bridge
            <|vision_start|><|video_pad|><|vision_end|>
            done
            """,
            messages: [
                .user([
                    .text("system"),
                    .image(InputImage(data: imageData, mimeType: "image/png")),
                    .text("bridge"),
                    .video(InputVideo(data: videoData, mimeType: "video/mp4")),
                    .text("done"),
                ])
            ]
        )

        let multimodal = try #require(prepared.multimodal)
        let tokenIDs = tokenizer.encode(text: prepared.text)
        let mmTokenTypeIDs = processor.multimodalTokenTypes(for: tokenIDs)

        #expect(multimodal.images.count == 1)
        #expect(multimodal.videos.count == 1)
        #expect(mmTokenTypeIDs.contains(1))
        #expect(mmTokenTypeIDs.contains(2))
        #expect(mmTokenTypeIDs.count == tokenIDs.count)
    }

    @Test("Parity fixture matches official-style prompt preparation counts")
    func promptPreparationParityFixture() async throws {
        let parity = try QwenVisionTestSupport.parityFixture()
        let decoder = HFConfigDecoder()
        let vision = try #require(
            try decoder.visionConfiguration(
                from: Data(QwenVisionTestSupport.completeConfigJSON().utf8),
                preprocessorConfigData: Data(QwenVisionTestSupport.preprocessorJSON().utf8)
            )
        )
        let configuration = ModelConfiguration(
            name: "Qwen3-VL",
            inputCapabilities: .init(supportsText: true, supportsImages: true, supportsVideo: true),
            executionCapabilities: .init(
                supportsImagePromptPreparation: true,
                supportsVideoPromptPreparation: true
            ),
            vision: vision
        )
        let processor = QwenVisionPromptProcessor(configuration: configuration)
        let imageData = try TestImageFixtures.makeOnePixelPNGData()
        let videoData = try await TestVideoFixtures.makeMP4Data()

        let imagePrepared = try await processor.prepare(
            renderedText: "<|vision_start|><|image_pad|><|vision_end|>",
            messages: [.user([.image(InputImage(data: imageData, mimeType: "image/png"))])]
        )
        let videoPrepared = try await processor.prepare(
            renderedText: "<|vision_start|><|video_pad|><|vision_end|>",
            messages: [.user([.video(InputVideo(data: videoData, mimeType: "video/mp4"))])]
        )

        let image = try #require(imagePrepared.multimodal?.images.first)
        let video = try #require(videoPrepared.multimodal?.videos.first)
        #expect(image.gridTHW == parity.imagePrompt.gridTHW)
        #expect(image.placeholderTokenCount == parity.imagePrompt.placeholderTokenCount)
        #expect(image.pixelValuesShape == parity.imagePrompt.pixelValuesShape)
        #expect(video.gridTHW == parity.videoPrompt.gridTHW)
        #expect(video.placeholderTokenCount == parity.videoPrompt.placeholderTokenCount)
        #expect(video.pixelValuesShape == parity.videoPrompt.pixelValuesShape)
        #expect(video.frameTimestamps.count == parity.videoPrompt.frameTimestampCount)
    }
}
