import Testing
@testable import SwiftLM

@Suite("Qwen Vision Encoder")
struct QwenVisionEncoderTests {
    @Test("Encode synthetic image features and deepstack outputs")
    func encodeSyntheticImage() throws {
        let configuration = QwenVisionTestSupport.visionConfiguration(outHiddenSize: 8)
        let weights = QwenVisionTestSupport.syntheticVisionWeights(outHiddenSize: 8)
        let encoder = try QwenVisionEncoder(configuration: configuration, weights: weights)
        let image = PreparedPrompt.Multimodal.Image(
            gridTHW: [1, 2, 2],
            placeholderTokenCount: 1,
            pixelValuesShape: [4, 12],
            pixelValues: (0..<48).map { Float($0) / 48.0 },
            resizedSize: [4, 4]
        )

        let outputs = try encoder.encode(images: [image])

        #expect(outputs.visualTokenEmbeddings.count == 1)
        #expect(outputs.visualTokenEmbeddings[0].count == 8)
        #expect(outputs.deepstackFeaturesByLayer[0]?.count == 1)
        #expect(outputs.deepstackFeaturesByLayer[0]?[0].count == 8)
        #expect(outputs.visualTokenEmbeddings[0].allSatisfy { $0 == 0 })
        #expect(outputs.deepstackFeaturesByLayer[0]?[0].allSatisfy { $0 == 0 } == true)
    }

    @Test("Encode synthetic video features and deepstack outputs")
    func encodeSyntheticVideo() throws {
        let configuration = QwenVisionTestSupport.visionConfiguration(
            outHiddenSize: 8,
            supportsImages: false,
            supportsVideo: true
        )
        let encoder = try QwenVisionEncoder(
            configuration: configuration,
            weights: QwenVisionTestSupport.syntheticVisionWeights(outHiddenSize: 8)
        )
        let video = PreparedPrompt.Multimodal.Video(
            gridTHW: [2, 2, 2],
            placeholderTokenCount: 2,
            pixelValuesShape: [8, 12],
            pixelValues: (0..<96).map { Float($0) / 96.0 },
            frameTimestamps: [0.0, 1.0],
            sampledFrameCount: 2,
            resizedSize: [4, 4]
        )

        let outputs = try encoder.encode(videos: [video])

        #expect(outputs.visualTokenEmbeddings.count == 2)
        #expect(outputs.visualTokenEmbeddings[0].count == 8)
        #expect(outputs.deepstackFeaturesByLayer[0]?.count == 2)
        #expect(outputs.deepstackFeaturesByLayer[0]?[0].count == 8)
    }

    @Test("Missing vision weights fail with typed error")
    func missingVisionWeightsFail() throws {
        let configuration = QwenVisionTestSupport.visionConfiguration(outHiddenSize: 8)
        let weights = QwenVisionWeightStore(denseTensors: [:])
        let encoder = try QwenVisionEncoder(configuration: configuration, weights: weights)
        let image = PreparedPrompt.Multimodal.Image(
            gridTHW: [1, 2, 2],
            placeholderTokenCount: 1,
            pixelValuesShape: [4, 12],
            pixelValues: Array(repeating: 0, count: 48),
            resizedSize: [4, 4]
        )

        #expect(throws: ModelBundleLoaderError.self) {
            _ = try encoder.encode(images: [image])
        }
    }
}
