import Foundation
import Testing
@testable import SwiftLM
@testable import LMArchitecture

@Suite("Qwen Vision Capability")
struct QwenVisionCapabilityTests {
    @Test("qwen3_vl declares image and video support")
    func detectQwen3VLCapabilities() throws {
        let decoder = HFConfigDecoder()
        let capabilities = try decoder.inputCapabilities(
            from: Data(QwenVisionTestSupport.completeConfigJSON().utf8),
            preprocessorConfigData: Data(QwenVisionTestSupport.preprocessorJSON().utf8)
        )

        #expect(capabilities.supportsText)
        #expect(capabilities.supportsImages)
        #expect(capabilities.supportsVideo)
    }

    @Test("processor class alone does not enable image or video support")
    func processorClassAloneDoesNotEnableVision() throws {
        let decoder = HFConfigDecoder()
        let configJSON = """
        {
          "model_type": "qwen3",
          "hidden_size": 4096,
          "num_hidden_layers": 36,
          "intermediate_size": 12288,
          "vocab_size": 151936,
          "num_attention_heads": 32,
          "num_key_value_heads": 8,
          "head_dim": 128
        }
        """
        let preprocessorJSON = """
        {
          "processor_class": "AutoProcessor"
        }
        """

        let vision = try decoder.visionConfiguration(
            from: Data(configJSON.utf8),
            preprocessorConfigData: Data(preprocessorJSON.utf8)
        )
        #expect(vision == nil)

        let capabilities = try decoder.inputCapabilities(
            from: Data(configJSON.utf8),
            preprocessorConfigData: Data(preprocessorJSON.utf8)
        )
        #expect(!capabilities.supportsImages)
        #expect(!capabilities.supportsVideo)
    }

    @Test("vision config plus image token enables image support")
    func imageSupportRequiresExplicitVisionMetadata() throws {
        let decoder = HFConfigDecoder()
        let capabilities = try decoder.inputCapabilities(
            from: Data(QwenVisionTestSupport.completeConfigJSON(includeVideo: false).utf8),
            preprocessorConfigData: Data(
                QwenVisionTestSupport.preprocessorJSON(videoProcessorType: nil).utf8
            )
        )

        #expect(capabilities.supportsImages)
        #expect(!capabilities.supportsVideo)
    }

    @Test("video token and known video processor metadata enable video support")
    func videoSupportRequiresExplicitVisionMetadata() throws {
        let decoder = HFConfigDecoder()
        let capabilities = try decoder.inputCapabilities(
            from: Data(QwenVisionTestSupport.completeConfigJSON(includeImage: false).utf8),
            preprocessorConfigData: Data(
                QwenVisionTestSupport.preprocessorJSON(imageProcessorType: nil).utf8
            )
        )

        #expect(!capabilities.supportsImages)
        #expect(capabilities.supportsVideo)
    }

    @Test("decode Qwen3.5 vision metadata from official-style config")
    func decodeQwen35VisionMetadata() throws {
        let decoder = HFConfigDecoder()
        let vision = try #require(
            try decoder.visionConfiguration(
                from: Data(QwenVisionTestSupport.completeConfigJSON(modelType: "qwen3_5").utf8),
                preprocessorConfigData: Data(QwenVisionTestSupport.preprocessorJSON().utf8)
            )
        )
        let capabilities = try decoder.inputCapabilities(
            from: Data(QwenVisionTestSupport.completeConfigJSON(modelType: "qwen3_5").utf8),
            preprocessorConfigData: Data(QwenVisionTestSupport.preprocessorJSON().utf8),
            visionConfiguration: vision
        )

        #expect(vision.imageTokenID == 151655)
        #expect(vision.videoTokenID == 151656)
        #expect(vision.visionStartTokenID == 151652)
        #expect(vision.visionEndTokenID == 151653)
        #expect(vision.processorClass == "Qwen3VLProcessor")
        #expect(vision.imageProcessorType == "Qwen2VLImageProcessorFast")
        #expect(vision.videoProcessorType == "Qwen2VLVideoProcessor")
        #expect(capabilities.supportsImages)
        #expect(capabilities.supportsVideo)
    }

    @Test("resolve qwen3_vl text backbone")
    func resolveQwen3VLTextBackbone() throws {
        let decoder = HFConfigDecoder()
        let config = try decoder.decode(from: Data(QwenVisionTestSupport.completeConfigJSON().utf8))
        let graph = try ModelGraphResolver().resolveModelGraph(modelType: "qwen3_vl", config: config)

        #expect(countOperations(graph.rootRegion) > 0)
    }

    @Test("reject incomplete qwen3_vl text backbone metadata")
    func rejectIncompleteQwen3VLTextBackbone() throws {
        let decoder = HFConfigDecoder()
        let config = try decoder.decode(
            from: Data(QwenVisionTestSupport.incompleteBackboneConfigJSON().utf8)
        )

        #expect(throws: ModelBundleLoaderError.self) {
            _ = try ModelGraphResolver().resolveModelGraph(modelType: "qwen3_vl", config: config)
        }
    }

    @Test("qwen2 vision model types stay permissive even though they are not explicitly documented")
    func qwen2VisionTypesRemainPermissive() throws {
        let decoder = HFConfigDecoder()
        for modelType in ["qwen2_vl", "qwen2_5_vl"] {
            let config = try decoder.decode(
                from: Data(QwenVisionTestSupport.completeConfigJSON(modelType: modelType).utf8)
            )
            let capabilities = try decoder.inputCapabilities(
                from: Data(QwenVisionTestSupport.completeConfigJSON(modelType: modelType).utf8),
                preprocessorConfigData: Data(QwenVisionTestSupport.preprocessorJSON().utf8)
            )
            let graph = try ModelGraphResolver().resolveModelGraph(modelType: modelType, config: config)

            #expect(capabilities.supportsImages)
            #expect(capabilities.supportsVideo)
            #expect(countOperations(graph.rootRegion) > 0)
        }
    }

    @Test("Qwen vision support detects image and video prompt preparation")
    func qwen3VLVisionSupportDetection() {
        let vision = ModelVisionConfiguration(
            processorClass: "Qwen3VLProcessor",
            imageTokenID: 151655,
            videoTokenID: 151656,
            imageProcessorType: "Qwen2VLImageProcessorFast",
            videoProcessorType: "Qwen2VLVideoProcessor"
        )

        #expect(QwenVisionSupport.supportsImagePromptPreparation(vision: vision))
        #expect(QwenVisionSupport.supportsVideoPromptPreparation(vision: vision))
    }

    private func countOperations(_ region: Region) -> Int {
        var count = 0
        for op in region.operations {
            count += 1
            switch op.kind {
            case .residual(_, let body): count += countOperations(body)
            case .repeating(_, let body): count += countOperations(body)
            case .conditional(_, let t, let e): count += countOperations(t) + countOperations(e)
            case .parallel(_, let branches): for b in branches { count += countOperations(b) }
            case .primitive: break
            }
        }
        return count
    }
}
