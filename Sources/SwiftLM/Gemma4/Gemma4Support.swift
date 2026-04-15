struct Gemma4Support {
    static func supportsImageProcessorClass(_ processorClass: String?) -> Bool {
        processorClass == "Gemma4Processor"
    }

    static func supportsImagePromptPreparation(vision: ModelVisionConfiguration?) -> Bool {
        guard let vision else { return false }
        return vision.imageTokenID != nil
            && (
                supportsImageProcessorClass(vision.processorClass)
                || vision.processorClass == nil
            )
            && vision.patchSize != nil
            && vision.poolingKernelSize != nil
            && vision.hiddenSize != nil
    }
}
