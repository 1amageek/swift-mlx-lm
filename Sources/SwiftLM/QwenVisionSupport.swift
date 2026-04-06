struct QwenVisionSupport {
    static func supportsImageProcessorClass(_ processorClass: String?) -> Bool {
        switch processorClass {
        case "Qwen3VLProcessor", "Qwen2VLProcessor":
            return true
        default:
            return false
        }
    }

    static func supportsImageProcessorType(_ imageProcessorType: String?) -> Bool {
        guard let imageProcessorType else { return false }
        return imageProcessorType.contains("Qwen2VLImageProcessor")
            || imageProcessorType == "QwenImageProcessor"
    }

    static func supportsVideoProcessorType(_ videoProcessorType: String?) -> Bool {
        guard let videoProcessorType else { return false }
        return videoProcessorType.contains("Qwen2VLVideoProcessor")
            || videoProcessorType == "QwenVideoProcessor"
    }

    static func supportsVideoProcessorClass(_ processorClass: String?) -> Bool {
        supportsImageProcessorClass(processorClass)
    }

    static func supportsImagePromptPreparation(vision: ModelVisionConfiguration?) -> Bool {
        guard let vision else { return false }
        return vision.imageTokenID != nil && (
            supportsImageProcessorClass(vision.processorClass)
            || supportsImageProcessorType(vision.imageProcessorType)
        )
    }

    static func supportsVideoPromptPreparation(vision: ModelVisionConfiguration?) -> Bool {
        guard let vision else { return false }
        return vision.videoTokenID != nil && (
            supportsVideoProcessorClass(vision.processorClass)
            || supportsVideoProcessorType(vision.videoProcessorType)
        )
    }
}
