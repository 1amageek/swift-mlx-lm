import Foundation
import Darwin
import Metal
import OrderedCollections
import MetalCompiler
import Tokenizers
import Jinja

/// Thread-safe container for a compiled inference model.
///
/// Wraps MetalInferenceModel + Tokenizer. Provides the public API
/// consumed by AnyFoundationModels and Jardis.
///
/// ```swift
/// let container = try await ModelBundleLoader().load(repo: "mlx-community/Qwen2.5-0.5B-Instruct")
/// let stream = try container.generate(
///     prompt: ExecutablePrompt(tokenIDs: tokenizer.encode(text: "Hello")),
///     parameters: GenerateParameters(maxTokens: 100)
/// )
/// for await generation in stream {
///     if let text = generation.chunk { print(text, terminator: "") }
/// }
/// ```
public final class ModelContainer: @unchecked Sendable {

    private var inferenceModel: MetalInferenceModel
    private let modelTokenizer: any Tokenizer
    private let modelConfiguration: ModelConfiguration
    /// Jinja chat template loaded from model bundle. nil if not available.
    private let chatTemplate: Template?
    private let visionRuntime: QwenVisionRuntime?
    private let gemma4Runtime: Gemma4Runtime?

    public convenience init(
        inferenceModel: MetalInferenceModel,
        tokenizer: any Tokenizer,
        configuration: ModelConfiguration,
        chatTemplate: Template? = nil
    ) {
        self.init(
            inferenceModel: inferenceModel,
            tokenizer: tokenizer,
            configuration: configuration,
            chatTemplate: chatTemplate,
            visionRuntime: nil,
            gemma4Runtime: nil
        )
    }

    init(
        inferenceModel: MetalInferenceModel,
        tokenizer: any Tokenizer,
        configuration: ModelConfiguration,
        chatTemplate: Template? = nil,
        visionRuntime: QwenVisionRuntime? = nil,
        gemma4Runtime: Gemma4Runtime? = nil
    ) {
        self.inferenceModel = inferenceModel
        self.modelTokenizer = tokenizer
        self.modelConfiguration = configuration
        self.chatTemplate = chatTemplate
        self.visionRuntime = visionRuntime
        self.gemma4Runtime = gemma4Runtime
    }

    /// Model configuration (name, EOS tokens).
    public var configuration: ModelConfiguration {
        modelConfiguration
    }

    /// The tokenizer used by this model.
    public var tokenizer: any Tokenizer {
        modelTokenizer
    }

    /// Prepare user-facing input into rendered text, tokens, and prompt metadata.
    ///
    /// For chat messages, applies the Jinja chat template from the model bundle
    /// (chat_template.jinja or tokenizer_config.json). Falls back to simple
    /// role-prefixed formatting if no template is available.
    public func prepare(input: ModelInput) async throws -> PreparedInput {
        let preparedPrompt: PreparedPrompt
        switch input.prompt {
        case .text(let prompt):
            preparedPrompt = PreparedPrompt(text: prompt, multimodal: nil)
        case .chat(let messages):
            preparedPrompt = try await applyChatTemplate(messages: messages)
        }
        let tokens = modelTokenizer.encode(text: preparedPrompt.text)
        var multimodal = preparedPrompt.multimodal
        if multimodal != nil {
            if gemma4Runtime != nil {
                let processor = Gemma4PromptProcessor(configuration: configuration)
                multimodal?.mmTokenTypeIDs = processor.multimodalTokenTypes(for: tokens)
            } else if visionRuntime != nil {
                let processor = QwenVisionPromptProcessor(configuration: configuration)
                multimodal?.mmTokenTypeIDs = processor.multimodalTokenTypes(for: tokens)
            } else {
                throw ModelContainerError.multimodalInputNotSupported(
                    "No vision runtime available for multimodal token type assignment."
                )
            }
        }
        return PreparedInput(
            renderedText: preparedPrompt.text,
            tokenIDs: tokens,
            multimodalMetadata: multimodal
        )
    }

    /// Apply the Jinja chat template to format chat messages into a prompt string.
    private func applyChatTemplate(messages: [InputMessage]) async throws -> PreparedPrompt {
        let containsImages = messages.contains(where: \.containsImageContent)
        let containsVideos = messages.contains(where: \.containsVideoContent)
        if containsImages && !configuration.inputCapabilities.supportsImages {
            throw ModelContainerError.unsupportedInputForModel(
                "This model bundle does not declare image input support."
            )
        }
        if containsVideos && !configuration.inputCapabilities.supportsVideo {
            throw ModelContainerError.unsupportedInputForModel(
                "This model bundle does not declare video input support."
            )
        }

        if let template = chatTemplate {
            let context: [String: Value] = [
                "messages": .array(try messages.map(makeJinjaMessageValue)),
                "add_generation_prompt": .boolean(true),
                "add_vision_id": .boolean(false),
                "bos_token": .string(modelTokenizer.bosToken ?? ""),
                "eos_token": .string(modelTokenizer.eosToken ?? ""),
            ]
            let rendered = try template.render(context)
            return try await prepareRenderedPrompt(rendered, messages: messages)
        }

        // Fallback: simple role-prefixed format
        let rendered = messages.map { message in
            let content = message.content.map { item in
                switch item {
                case .text(let text):
                    return text
                case .image:
                    if gemma4Runtime != nil {
                        return "<|image|>"
                    }
                    return "<|vision_start|><|image_pad|><|vision_end|>"
                case .video:
                    if gemma4Runtime != nil {
                        return "<|video|>"
                    }
                    return "<|vision_start|><|video_pad|><|vision_end|>"
                }
            }
            .joined()
            return "\(message.role.rawValue): \(content)"
        }
        .joined(separator: "\n")
        return try await prepareRenderedPrompt(rendered, messages: messages)
    }

    private func prepareRenderedPrompt(
        _ rendered: String,
        messages: [InputMessage]
    ) async throws -> PreparedPrompt {
        guard messages.contains(where: \.containsVisualContent) else {
            return PreparedPrompt(text: rendered, multimodal: nil)
        }
        if gemma4Runtime != nil {
            let processor = Gemma4PromptProcessor(configuration: configuration)
            return try await processor.prepare(renderedText: rendered, messages: messages)
        }
        if visionRuntime != nil {
            let processor = QwenVisionPromptProcessor(configuration: configuration)
            return try await processor.prepare(renderedText: rendered, messages: messages)
        }
        throw ModelContainerError.multimodalInputNotSupported(
            "No vision runtime available for multimodal prompt preparation."
        )
    }

    private func makeJinjaMessageValue(message: InputMessage) throws -> Value {
        var object = OrderedDictionary<String, Value>()
        object["role"] = .string(message.role.rawValue)
        object["content"] = .array(try message.content.map(makeJinjaContentValue))
        return .object(object)
    }

    private func makeJinjaContentValue(item: InputMessage.Content) throws -> Value {
        var object = OrderedDictionary<String, Value>()
        switch item {
        case .text(let text):
            object["type"] = .string("text")
            object["text"] = .string(text)
        case .image(let image):
            object["type"] = .string("image")
            object["image"] = .string(try jinjaLocationString(for: image.source))
        case .video(let video):
            object["type"] = .string("video")
            object["video"] = .string(try jinjaLocationString(for: video.source))
        }
        return .object(object)
    }

    private func jinjaLocationString(for source: InputImage.Source) throws -> String {
        switch source {
        case .fileURL(let url):
            return url.absoluteString
        case .data:
            return "inline-image"
        }
    }

    private func jinjaLocationString(for source: InputVideo.Source) throws -> String {
        switch source {
        case .fileURL(let url):
            return url.absoluteString
        case .data:
            return "inline-video"
        }
    }

    private func streamGeneration(
        firstToken: Int32,
        promptTokenCount: Int,
        preparationTime: Double,
        requestStartTime: Double,
        ropePositionOffset: Int,
        parameters: GenerateParameters,
        continuation: AsyncStream<Generation>.Continuation
    ) {
        var tokenCount = 0
        let maxTokens = parameters.maxTokens ?? 1024
        let chunkTokenCount = max(1, parameters.streamChunkTokenCount)
        var bufferedTokenIDs: [Int] = []

        func emitBufferedChunkIfNeeded(force: Bool = false) {
            guard !bufferedTokenIDs.isEmpty else { return }
            guard force || bufferedTokenIDs.count >= chunkTokenCount else { return }
            let text = self.modelTokenizer.decode(tokens: bufferedTokenIDs)
            bufferedTokenIDs.removeAll(keepingCapacity: true)
            continuation.yield(.chunk(text))
        }

        guard firstToken >= 0 else {
            continuation.finish()
            return
        }
        if self.modelConfiguration.eosTokenIds.contains(Int(firstToken)) {
            continuation.finish()
            return
        }

        bufferedTokenIDs.append(Int(firstToken))
        tokenCount += 1

        if tokenCount < maxTokens {
            do {
                try self.scheduleDecodeStep(
                    tokenID: firstToken,
                    ropePositionOffset: ropePositionOffset
                )
            } catch {
                print("[ModelContainer] Failed to schedule decode: \(error)")
                continuation.finish()
                return
            }
        }

        emitBufferedChunkIfNeeded()

        while tokenCount < maxTokens {
            let outputToken = self.inferenceModel.flush()

            if outputToken < 0 { break }

            let isEOS = self.modelConfiguration.eosTokenIds.contains(Int(outputToken))
            if !isEOS, tokenCount + 1 < maxTokens {
                do {
                    try self.scheduleDecodeStep(
                        tokenID: outputToken,
                        ropePositionOffset: ropePositionOffset
                    )
                } catch {
                    print("[ModelContainer] Failed to schedule decode: \(error)")
                    continuation.finish()
                    return
                }
            }
            if isEOS {
                emitBufferedChunkIfNeeded(force: true)
                break
            }

            bufferedTokenIDs.append(Int(outputToken))
            tokenCount += 1
            emitBufferedChunkIfNeeded()
        }

        emitBufferedChunkIfNeeded(force: true)

        let totalTime = CFAbsoluteTimeGetCurrent() - requestStartTime
        let tokensPerSecond = totalTime > 0 ? Double(tokenCount) / totalTime : 0
        let preparationTokPerSec = preparationTime > 0 ? Double(promptTokenCount) / preparationTime : 0
        print("[ModelContainer] \(tokenCount) tokens (\(String(format: "%.0f", preparationTokPerSec)) prefill, \(String(format: "%.1f", tokensPerSecond)) decode tok/s) [\(String(format: "%.1f", totalTime))s]")
        continuation.yield(.info(CompletionInfo(
            tokenCount: tokenCount,
            tokensPerSecond: tokensPerSecond,
            totalTime: totalTime
        )))
        continuation.finish()
    }

    private func generationRoPEAxes(offset: Int) -> (UInt32, UInt32, UInt32)? {
        guard offset != 0 else { return nil }
        let ropePosition = inferenceModel.position + offset
        guard ropePosition >= 0 else { return nil }
        let value = UInt32(ropePosition)
        return (value, value, value)
    }

    private func scheduleDecodeStep(
        tokenID: Int32,
        ropePositionOffset: Int
    ) throws {
        if let gemma4Runtime {
            let perLayerInputs = try gemma4Runtime.buildDecodePerLayerInputs(tokenID: Int(tokenID))
            try inferenceModel.writeDecodePerLayerInputs(perLayerInputs)
        }
        _ = inferenceModel.decode(
            tokenID: tokenID,
            ropePositionAxes: generationRoPEAxes(offset: ropePositionOffset)
        )
    }

    private func prefill(prompt: ExecutablePrompt) throws -> (firstToken: Int32, ropePositionOffset: Int) {
        inferenceModel.resetCaches()
        if let gemma4PromptContext = prompt.gemma4PromptContext {
            let ropeAxes = (0..<gemma4PromptContext.promptEmbeddings.count).map { index -> (UInt32, UInt32, UInt32) in
                let value = UInt32(index)
                return (value, value, value)
            }
            try inferenceModel.writePrefillPerLayerInputs(gemma4PromptContext.perLayerInputs)
            let firstToken = try inferenceModel.prefillEmbeddings(
                gemma4PromptContext.promptEmbeddings,
                ropePositionAxes: ropeAxes
            )
            guard firstToken >= 0 else {
                throw ModelContainerError.invalidPrefillResult
            }
            return (firstToken: firstToken, ropePositionOffset: 0)
        }
        guard let visualContext = prompt.visualContext else {
            let promptTokens = prompt.tokenIDs.map(Int32.init)
            let firstToken = inferenceModel.prefill(tokens: promptTokens)
            guard firstToken >= 0 else {
                throw ModelContainerError.invalidPrefillResult
            }
            return (firstToken: firstToken, ropePositionOffset: 0)
        }

        let layout = visualContext.layout
        guard layout.tokenTypeIDs.count == prompt.tokenIDs.count else {
            throw ModelContainerError.multimodalInputNotSupported(
                "Executable multimodal prompt layout does not match token count."
            )
        }

        var imageTokenIndex = 0
        var videoTokenIndex = 0
        var firstToken: Int32 = -1
        for segment in layout.segments {
            let tokenType = segment.modality
            let tokenIndex = segment.tokenRange.lowerBound
            let endIndex = segment.tokenRange.upperBound
            let chunkTokens = prompt.tokenIDs[segment.tokenRange].map(Int32.init)
            switch tokenType {
            case 0:
                firstToken = inferenceModel.prefill(tokens: chunkTokens)
            case 1:
                let localTokenCount = endIndex - tokenIndex
                let endImageIndex = imageTokenIndex + localTokenCount
                guard endImageIndex <= visualContext.imageTokenEmbeddings.count else {
                    throw ModelContainerError.multimodalInputNotSupported(
                        "Vision encoder output is shorter than the image placeholder sequence."
                    )
                }
                let ropePositionAxes = segment.tokenRange.map { absoluteIndex in
                    layout.ropePositionIDs.axes(at: absoluteIndex)
                }
                let hiddenStates = Array(
                    visualContext.imageTokenEmbeddings[imageTokenIndex..<endImageIndex]
                )
                var deepstackFeaturesByLayer: [Int: [[Float]]] = [:]
                for (layerIndex, features) in visualContext.imageDeepstackFeaturesByLayer {
                    guard endImageIndex <= features.count else {
                        throw ModelContainerError.multimodalInputNotSupported(
                            "Deepstack visual feature count mismatch at layer \(layerIndex)."
                        )
                    }
                    deepstackFeaturesByLayer[layerIndex] = Array(features[imageTokenIndex..<endImageIndex])
                }
                firstToken = try inferenceModel.prefillEmbeddings(
                    hiddenStates,
                    ropePositionAxes: ropePositionAxes,
                    deepstackFeaturesByLayer: deepstackFeaturesByLayer
                )
                imageTokenIndex = endImageIndex
            case 2:
                let localTokenCount = endIndex - tokenIndex
                let endVideoIndex = videoTokenIndex + localTokenCount
                guard endVideoIndex <= visualContext.videoTokenEmbeddings.count else {
                    throw ModelContainerError.multimodalInputNotSupported(
                        "Vision encoder output is shorter than the video placeholder sequence."
                    )
                }
                let ropePositionAxes = segment.tokenRange.map { absoluteIndex in
                    layout.ropePositionIDs.axes(at: absoluteIndex)
                }
                let hiddenStates = Array(
                    visualContext.videoTokenEmbeddings[videoTokenIndex..<endVideoIndex]
                )
                var deepstackFeaturesByLayer: [Int: [[Float]]] = [:]
                for (layerIndex, features) in visualContext.videoDeepstackFeaturesByLayer {
                    guard endVideoIndex <= features.count else {
                        throw ModelContainerError.multimodalInputNotSupported(
                            "Deepstack video feature count mismatch at layer \(layerIndex)."
                        )
                    }
                    deepstackFeaturesByLayer[layerIndex] = Array(features[videoTokenIndex..<endVideoIndex])
                }
                firstToken = try inferenceModel.prefillEmbeddings(
                    hiddenStates,
                    ropePositionAxes: ropePositionAxes,
                    deepstackFeaturesByLayer: deepstackFeaturesByLayer
                )
                videoTokenIndex = endVideoIndex
            default:
                throw ModelContainerError.multimodalInputNotSupported(
                    "Unsupported multimodal token type ID: \(tokenType)"
                )
            }
        }

        guard firstToken >= 0 else {
            throw ModelContainerError.invalidPrefillResult
        }
        return (
            firstToken: firstToken,
            ropePositionOffset: layout.ropePositionDelta
        )
    }

    /// Convert prepared prompt data into runtime-executable prompt state.
    public func makeExecutablePrompt(from prepared: PreparedInput) throws -> ExecutablePrompt {
        if let gemma4Runtime {
            if let multimodal = prepared.multimodalMetadata, !multimodal.videos.isEmpty {
                throw ModelContainerError.multimodalInputNotSupported(
                    "Gemma4 video execution is not implemented yet."
                )
            }
            return ExecutablePrompt(
                tokenIDs: prepared.tokenIDs,
                attentionMask: prepared.attentionMask,
                gemma4PromptContext: try gemma4Runtime.makePromptContext(from: prepared)
            )
        }
        guard let multimodal = prepared.multimodalMetadata else {
            return ExecutablePrompt(tokenIDs: prepared.tokenIDs, attentionMask: prepared.attentionMask)
        }
        if !multimodal.videos.isEmpty && !configuration.executionCapabilities.supportsVideoExecution {
            throw ModelContainerError.multimodalInputNotSupported(
                "This runtime can prepare Qwen3.5/Qwen3-VL prompts, but video execution is unavailable for the loaded bundle."
            )
        }
        guard multimodal.images.isEmpty || configuration.executionCapabilities.supportsImageExecution else {
            throw ModelContainerError.multimodalInputNotSupported(
                "This runtime can prepare Qwen3.5/Qwen3-VL prompts, but image execution is unavailable for the loaded bundle."
            )
        }
        guard let visionRuntime else {
            throw ModelContainerError.multimodalInputNotSupported(
                "The loaded bundle does not have an active Qwen vision runtime."
            )
        }
        return ExecutablePrompt(
            tokenIDs: prepared.tokenIDs,
            attentionMask: prepared.attentionMask,
            visualContext: try visionRuntime.makeVisualContext(from: prepared)
        )
    }

    /// Build a reusable prompt state from an executable prompt.
    ///
    /// This runs prefill once, snapshots the decode state, and stores the
    /// first predicted token so the same prompt prefix can be reused later.
    public func makePromptState(prompt: ExecutablePrompt) throws -> PromptState {
        let prefillResult = try prefill(prompt: prompt)
        let metalState = try inferenceModel.makePromptState(firstToken: prefillResult.firstToken)
        return PromptState(
            metalState: metalState,
            promptTokenCount: prompt.tokenIDs.count,
            ropePositionOffset: prefillResult.ropePositionOffset
        )
    }

    /// Build a reusable prompt state from prepared prompt data.
    public func makePromptState(input: PreparedInput) throws -> PromptState {
        let prompt = try makeExecutablePrompt(from: input)
        return try makePromptState(prompt: prompt)
    }

    /// Build a reusable prompt state from user input.
    public func makePromptState(input: ModelInput) async throws -> PromptState {
        let prepared = try await prepare(input: input)
        return try makePromptState(input: prepared)
    }

    /// Generate text from an executable prompt.
    ///
    /// Returns an AsyncStream of Generation values (text chunks + completion info).
    /// Each `.chunk` may contain one or more decoded tokens.
    public func generate(
        prompt: ExecutablePrompt,
        parameters: GenerateParameters = GenerateParameters()
    ) throws -> AsyncStream<Generation> {
        let startTime = CFAbsoluteTimeGetCurrent()
        let prefillStart = CFAbsoluteTimeGetCurrent()
        let prefillResult = try prefill(prompt: prompt)
        let prefillTime = CFAbsoluteTimeGetCurrent() - prefillStart
        return AsyncStream { continuation in
            Task {
                self.streamGeneration(
                    firstToken: prefillResult.firstToken,
                    promptTokenCount: prompt.tokenIDs.count,
                    preparationTime: prefillTime,
                    requestStartTime: startTime,
                    ropePositionOffset: prefillResult.ropePositionOffset,
                    parameters: parameters,
                    continuation: continuation
                )
            }
        }
    }

    /// Generate text by restoring a reusable prompt state instead of re-running prefill.
    public func generate(
        from promptState: PromptState,
        parameters: GenerateParameters = GenerateParameters()
    ) throws -> AsyncStream<Generation> {
        let startTime = CFAbsoluteTimeGetCurrent()
        let restoreStart = CFAbsoluteTimeGetCurrent()
        do {
            try self.inferenceModel.restore(promptState: promptState.metalState)
        } catch {
            throw ModelContainerError.promptStateRestoreFailed(String(describing: error))
        }
        let restoreTime = CFAbsoluteTimeGetCurrent() - restoreStart
        return AsyncStream { continuation in
            Task {
                self.streamGeneration(
                    firstToken: promptState.metalState.firstToken,
                    promptTokenCount: promptState.promptTokenCount,
                    preparationTime: restoreTime,
                    requestStartTime: startTime,
                    ropePositionOffset: promptState.ropePositionOffset,
                    parameters: parameters,
                    continuation: continuation
                )
            }
        }
    }

    /// Prepare, validate, and generate from a public prompt shape in one step.
    public func generate(
        input: ModelInput,
        parameters: GenerateParameters = GenerateParameters()
    ) async throws -> AsyncStream<Generation> {
        let prepared = try await prepare(input: input)
        let prompt = try makeExecutablePrompt(from: prepared)
        return try generate(prompt: prompt, parameters: parameters)
    }

    /// Decode token IDs to text.
    public func decode(tokens: [Int]) -> String {
        modelTokenizer.decode(tokens: tokens)
    }

    /// Encode text to token IDs.
    public func encode(_ text: String) -> [Int] {
        modelTokenizer.encode(text: text)
    }

    /// Reset KV cache (call between independent conversations).
    public func resetCaches() {
        inferenceModel.resetCaches()
    }
}
