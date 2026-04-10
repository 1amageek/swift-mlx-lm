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
    private static let promptStateSamplingTailLimit = 256

    private var inferenceModel: MetalInferenceModel
    private let modelTokenizer: any Tokenizer
    private let modelConfiguration: ModelConfiguration
    private let vocabularySize: Int?
    private let finalLogitSoftcapping: Float?
    /// Jinja chat template loaded from model bundle. nil if not available.
    private let chatTemplate: Template?
    private let chatTemplateSource: String?
    private let thinkingTagPolicy: ThinkingTagPolicy?
    private let visionRuntime: QwenVisionRuntime?
    private let gemma4Runtime: Gemma4Runtime?

    public convenience init(
        inferenceModel: MetalInferenceModel,
        tokenizer: any Tokenizer,
        configuration: ModelConfiguration,
        chatTemplate: Template? = nil,
        chatTemplateSource: String? = nil,
        vocabularySize: Int? = nil,
        finalLogitSoftcapping: Float? = nil
    ) {
        self.init(
            inferenceModel: inferenceModel,
            tokenizer: tokenizer,
            configuration: configuration,
            chatTemplate: chatTemplate,
            chatTemplateSource: chatTemplateSource,
            vocabularySize: vocabularySize,
            finalLogitSoftcapping: finalLogitSoftcapping,
            visionRuntime: nil,
            gemma4Runtime: nil
        )
    }

    init(
        inferenceModel: MetalInferenceModel,
        tokenizer: any Tokenizer,
        configuration: ModelConfiguration,
        chatTemplate: Template? = nil,
        chatTemplateSource: String? = nil,
        vocabularySize: Int? = nil,
        finalLogitSoftcapping: Float? = nil,
        visionRuntime: QwenVisionRuntime? = nil,
        gemma4Runtime: Gemma4Runtime? = nil
    ) {
        self.inferenceModel = inferenceModel
        self.modelTokenizer = tokenizer
        self.modelConfiguration = configuration
        self.vocabularySize = vocabularySize
        self.finalLogitSoftcapping = finalLogitSoftcapping
        self.chatTemplate = chatTemplate
        self.chatTemplateSource = chatTemplateSource
        self.thinkingTagPolicy = Self.makeThinkingTagPolicy(
            tokenizer: tokenizer,
            chatTemplateSource: chatTemplateSource
        )
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
            if modelTokenizer.hasChatTemplate || chatTemplate != nil {
                preparedPrompt = try await applyChatTemplate(messages: [
                    .user([.text(prompt)])
                ])
            } else {
                preparedPrompt = PreparedPrompt(text: prompt, multimodal: nil)
            }
        case .chat(let messages):
            preparedPrompt = try await applyChatTemplate(messages: messages)
        }
        let tokens = preparedPrompt.tokenIDs ?? modelTokenizer.encode(text: preparedPrompt.text)
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
        let renderedMessages = renderedMessagesWithThinkingControl(from: messages)
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
                "messages": .array(try renderedMessages.map(makeJinjaMessageValue)),
                "add_generation_prompt": .boolean(true),
                "add_vision_id": .boolean(false),
                "enable_thinking": .boolean(false),
                "bos_token": .string(modelTokenizer.bosToken ?? ""),
                "eos_token": .string(modelTokenizer.eosToken ?? ""),
            ]
            let rendered = try template.render(context)
            return try await prepareRenderedPrompt(rendered, messages: messages)
        }

        if modelTokenizer.hasChatTemplate {
            let renderedTokenIDs = try modelTokenizer.applyChatTemplate(
                messages: try renderedMessages.map(makeTokenizerMessage),
                tools: nil,
                additionalContext: [
                    "add_vision_id": false,
                    "enable_thinking": false,
                ]
            )
            let rendered = modelTokenizer.decode(tokens: renderedTokenIDs, skipSpecialTokens: false)
            let prepared = try await prepareRenderedPrompt(rendered, messages: messages)
            let preparedTokenIDs: [Int]?
            if prepared.multimodal == nil {
                preparedTokenIDs = renderedTokenIDs
            } else {
                preparedTokenIDs = nil
            }
            return PreparedPrompt(
                text: prepared.text,
                tokenIDs: preparedTokenIDs,
                multimodal: prepared.multimodal
            )
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

    private func renderedMessagesWithThinkingControl(
        from messages: [InputMessage]
    ) -> [InputMessage] {
        guard shouldInjectDirectAnswerSystemPrompt,
              messages.first?.role != .system else {
            return messages
        }

        return [
            .system([
                .text(
                    "Respond with the final answer only. Do not output internal reasoning, chain-of-thought, or <think> tags."
                )
            ])
        ] + messages
    }

    private var shouldInjectDirectAnswerSystemPrompt: Bool {
        guard let chatTemplateSource else { return false }
        return chatTemplateSource.contains("keep_past_thinking")
            && !chatTemplateSource.contains("enable_thinking")
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
        if message.content.count == 1, case .text(let text) = message.content[0] {
            object["content"] = .string(text)
        } else {
            object["content"] = .array(try message.content.map(makeJinjaContentValue))
        }
        return .object(object)
    }

    private func makeTokenizerMessage(message: InputMessage) throws -> Message {
        var payload: Message = [
            "role": message.role.rawValue
        ]
        if message.content.count == 1, case .text(let text) = message.content[0] {
            payload["content"] = text
        } else {
            payload["content"] = try message.content.map(makeTokenizerContentValue)
        }
        return payload
    }

    private func makeTokenizerContentValue(item: InputMessage.Content) throws -> [String: any Sendable] {
        switch item {
        case .text(let text):
            return [
                "type": "text",
                "text": text,
            ]
        case .image(let image):
            return [
                "type": "image",
                "image": try jinjaLocationString(for: image.source),
            ]
        case .video(let video):
            return [
                "type": "video",
                "video": try jinjaLocationString(for: video.source),
            ]
        }
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
        samplingState initialSamplingState: GenerationSamplingState,
        promptTokenCount: Int,
        preparationTime: Double,
        requestStartTime: Double,
        ropePositionOffset: Int,
        parameters: GenerateParameters,
        continuation: AsyncStream<Generation>.Continuation
    ) {
        var samplingState = initialSamplingState
        var visibleText = ""
        var rawTokenCount = 0
        let maxVisibleTokens = parameters.maxTokens ?? 1024
        let maxRawTokens = maxRawTokenCount(forVisibleLimit: maxVisibleTokens)
        let chunkTokenCount = max(1, parameters.streamChunkTokenCount)
        var bufferedRawTokenIDs: [Int] = []
        var fallbackTokenIDs: [Int] = []
        var visibilityState = GenerationVisibilityState(policy: thinkingTagPolicy)

        func emitBufferedChunkIfNeeded(force: Bool = false) {
            guard !bufferedRawTokenIDs.isEmpty else { return }
            guard force || bufferedRawTokenIDs.count >= chunkTokenCount else { return }
            let decodedText = self.modelTokenizer.decode(
                tokens: bufferedRawTokenIDs,
                skipSpecialTokens: false
            )
            bufferedRawTokenIDs.removeAll(keepingCapacity: true)
            let visibleChunk = visibilityState.append(decodedText: decodedText)
            guard !visibleChunk.isEmpty else { return }
            visibleText += visibleChunk
            continuation.yield(.chunk(visibleChunk))
        }

        func recordGeneratedToken(_ tokenID: Int32) {
            rawTokenCount += 1
            if fallbackTokenIDs.count < maxVisibleTokens {
                fallbackTokenIDs.append(Int(tokenID))
            }
            bufferedRawTokenIDs.append(Int(tokenID))
            emitBufferedChunkIfNeeded()
        }

        guard firstToken >= 0 else {
            continuation.finish()
            return
        }
        if self.modelConfiguration.eosTokenIds.contains(Int(firstToken)) {
            continuation.finish()
            return
        }

        recordGeneratedToken(firstToken)

        var nextTokenID = firstToken
        while rawTokenCount < maxRawTokens {
            let argmaxToken: Int32
            do {
                argmaxToken = try self.executeDecodeStep(
                    tokenID: nextTokenID,
                    ropePositionOffset: ropePositionOffset
                )
            } catch {
                print("[ModelContainer] Failed to decode: \(error)")
                break
            }

            let outputToken = self.resolveSampledDecodeToken(
                fallbackToken: argmaxToken,
                parameters: parameters,
                samplingState: &samplingState
            )
            if outputToken < 0 { break }

            let isEOS = self.modelConfiguration.eosTokenIds.contains(Int(outputToken))
            if isEOS {
                emitBufferedChunkIfNeeded(force: true)
                break
            }

            samplingState.record(Int(outputToken), maxContextSize: parameters.repetitionContextSize)
            recordGeneratedToken(outputToken)
            nextTokenID = outputToken
            let visibleTokenCount = visibleText.isEmpty
                ? 0
                : self.modelTokenizer.encode(text: visibleText, addSpecialTokens: false).count
            if visibleTokenCount >= maxVisibleTokens {
                break
            }
        }

        emitBufferedChunkIfNeeded(force: true)
        let trailingVisibleText = visibilityState.finalize()
        if !trailingVisibleText.isEmpty {
            visibleText += trailingVisibleText
            continuation.yield(.chunk(trailingVisibleText))
        }

        if visibleText.isEmpty,
           !fallbackTokenIDs.isEmpty,
           !visibilityState.didSuppressReasoning {
            let fallbackText = self.modelTokenizer.decode(
                tokens: fallbackTokenIDs,
                skipSpecialTokens: true
            )
            if !fallbackText.isEmpty {
                visibleText = fallbackText
                continuation.yield(.chunk(fallbackText))
            }
        }

        let visibleTokenCount = visibleText.isEmpty
            ? 0
            : self.modelTokenizer.encode(text: visibleText, addSpecialTokens: false).count

        let totalTime = CFAbsoluteTimeGetCurrent() - requestStartTime
        let tokensPerSecond = totalTime > 0 ? Double(visibleTokenCount) / totalTime : 0
        let preparationTokPerSec = preparationTime > 0 ? Double(promptTokenCount) / preparationTime : 0
        print("[ModelContainer] \(visibleTokenCount) tokens (\(String(format: "%.0f", preparationTokPerSec)) prefill, \(String(format: "%.1f", tokensPerSecond)) decode tok/s) [\(String(format: "%.1f", totalTime))s]")
        continuation.yield(.info(CompletionInfo(
            tokenCount: visibleTokenCount,
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

    private func executeDecodeStep(
        tokenID: Int32,
        ropePositionOffset: Int
    ) throws -> Int32 {
        if let gemma4Runtime {
            let perLayerInputs = try gemma4Runtime.buildDecodePerLayerInputs(tokenID: Int(tokenID))
            try inferenceModel.writeDecodePerLayerInputs(perLayerInputs)
        }
        return inferenceModel.decode(
            tokenID: tokenID,
            ropePositionAxes: generationRoPEAxes(offset: ropePositionOffset)
        )
    }

    private func prefill(prompt: ExecutablePrompt) throws -> (firstToken: Int32, ropePositionOffset: Int) {
        inferenceModel.resetCaches()
        if let gemma4PromptContext = prompt.gemma4PromptContext {
            try inferenceModel.writePrefillPerLayerInputs(gemma4PromptContext.perLayerInputs)
            let firstToken: Int32
            if gemma4PromptContext.usesEmbeddingOverrides {
                let ropeAxes = (0..<gemma4PromptContext.promptEmbeddings.count).map { index -> (UInt32, UInt32, UInt32) in
                    let value = UInt32(index)
                    return (value, value, value)
                }
                firstToken = try inferenceModel.prefillEmbeddings(
                    gemma4PromptContext.promptEmbeddings,
                    ropePositionAxes: ropeAxes
                )
            } else {
                firstToken = inferenceModel.prefill(tokens: prompt.tokenIDs.map(Int32.init))
            }
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
            ropePositionOffset: prefillResult.ropePositionOffset,
            samplingSeed: samplingSeed(for: prompt.tokenIDs),
            promptTokenTail: Array(prompt.tokenIDs.suffix(Self.promptStateSamplingTailLimit))
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
        let resolvedParameters = resolvedGenerateParameters(parameters)
        let startTime = CFAbsoluteTimeGetCurrent()
        let prefillStart = CFAbsoluteTimeGetCurrent()
        let prefillResult = try prefill(prompt: prompt)
        let prefillTime = CFAbsoluteTimeGetCurrent() - prefillStart
        var initialSamplingState = makeInitialSamplingState(
            promptTokenIDs: prompt.tokenIDs,
            parameters: resolvedParameters
        )
        let firstToken = resolveSampledPrefillToken(
            fallbackToken: prefillResult.firstToken,
            parameters: resolvedParameters,
            samplingState: &initialSamplingState
        )
        let streamingSamplingState = initialSamplingState.withRecordedFirstToken(
            firstToken,
            maxContextSize: resolvedParameters.repetitionContextSize
        )
        return AsyncStream { continuation in
            Task {
                self.streamGeneration(
                    firstToken: firstToken,
                    samplingState: streamingSamplingState,
                    promptTokenCount: prompt.tokenIDs.count,
                    preparationTime: prefillTime,
                    requestStartTime: startTime,
                    ropePositionOffset: prefillResult.ropePositionOffset,
                    parameters: resolvedParameters,
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
        let resolvedParameters = resolvedGenerateParameters(parameters)
        let startTime = CFAbsoluteTimeGetCurrent()
        let restoreStart = CFAbsoluteTimeGetCurrent()
        do {
            try self.inferenceModel.restore(promptState: promptState.metalState)
        } catch {
            throw ModelContainerError.promptStateRestoreFailed(String(describing: error))
        }
        let restoreTime = CFAbsoluteTimeGetCurrent() - restoreStart
        var initialSamplingState = GenerationSamplingState(
            rngState: promptState.samplingSeed,
            recentTokenIDs: Array(promptState.promptTokenTail.suffix(resolvedParameters.repetitionContextSize))
        )
        let firstToken = resolveSampledPromptStateToken(
            fallbackToken: promptState.metalState.firstToken,
            logitsBuffer: inferenceModel.decodePlan.buffers.logits,
            parameters: resolvedParameters,
            samplingState: &initialSamplingState
        )
        let streamingSamplingState = initialSamplingState.withRecordedFirstToken(
            firstToken,
            maxContextSize: resolvedParameters.repetitionContextSize
        )
        return AsyncStream { continuation in
            Task {
                self.streamGeneration(
                    firstToken: firstToken,
                    samplingState: streamingSamplingState,
                    promptTokenCount: promptState.promptTokenCount,
                    preparationTime: restoreTime,
                    requestStartTime: startTime,
                    ropePositionOffset: promptState.ropePositionOffset,
                    parameters: resolvedParameters,
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
        modelTokenizer.decode(tokens: tokens, skipSpecialTokens: true)
    }

    /// Encode text to token IDs.
    public func encode(_ text: String) -> [Int] {
        modelTokenizer.encode(text: text)
    }

    /// Reset KV cache (call between independent conversations).
    public func resetCaches() {
        inferenceModel.resetCaches()
    }

    internal func debugPrefillTopLogits(
        prompt: ExecutablePrompt,
        topK: Int = 20
    ) throws -> [(tokenID: Int, logit: Float, decoded: String)] {
        _ = try prefill(prompt: prompt)
        let decodePlan = inferenceModel.decodePlan
        let maximumBufferVocabularySize = decodePlan.buffers.logits.length / decodePlan.buffers.bufferPrecision.byteSize
        let resolvedVocabularySize = min(vocabularySize ?? maximumBufferVocabularySize, maximumBufferVocabularySize)
        guard resolvedVocabularySize > 0 else {
            return []
        }

        let logits = readModelLogits(
            from: decodePlan.buffers.logits,
            vocabularySize: resolvedVocabularySize,
            precision: decodePlan.buffers.bufferPrecision
        )
        let ranked = logits.enumerated()
            .sorted { lhs, rhs in
                if lhs.element == rhs.element {
                    return lhs.offset < rhs.offset
                }
                return lhs.element > rhs.element
            }
            .prefix(max(0, topK))

        return ranked.map { entry in
            let tokenID = entry.offset
            return (
                tokenID: tokenID,
                logit: entry.element,
                decoded: modelTokenizer.decode(tokens: [tokenID], skipSpecialTokens: false)
            )
        }
    }

    internal func debugPrefillOutputHeadDiagnostics(
        prompt: ExecutablePrompt,
        topK: Int = 20
    ) throws -> (
        topLogits: [(tokenID: Int, logit: Float, decoded: String)],
        inputLayout: (offset: Int, bufferLength: Int, hiddenCount: Int, readableCount: Int, bufferKind: String),
        transferLayout: (
            sourceOffset: Int,
            sourceBufferLength: Int,
            sourceReadableCount: Int,
            sourceBufferKind: String,
            destinationOffset: Int,
            destinationBufferLength: Int,
            destinationReadableCount: Int,
            destinationBufferKind: String,
            transferCount: Int
        ),
        transferSource: [Float],
        transferDestination: [Float]
    ) {
        _ = try prefill(prompt: prompt)
        guard let prefillPlan = inferenceModel.prefillPlan else {
            return (
                topLogits: [],
                inputLayout: (0, 0, 0, 0, "missing"),
                transferLayout: (0, 0, 0, "missing", 0, 0, 0, "missing", 0),
                transferSource: [],
                transferDestination: []
            )
        }

        let decodePlan = inferenceModel.decodePlan
        let inputBinding = decodePlan.outputHeadInputBinding()
        let decodeElementSize = max(decodePlan.buffers.bufferPrecision.byteSize, 1)
        let hiddenCount = decodePlan.buffers.hidden.length / decodeElementSize
        let readableCount = max((inputBinding.buffer.length - inputBinding.offset) / decodeElementSize, 0)

        let source = prefillPlan.finalHiddenSource(sequenceLength: prompt.tokenIDs.count)
        let prefillElementSize = max(prefillPlan.buffers.bufferPrecision.byteSize, 1)
        let sourceReadableCount = max((source.buffer.length - source.offset) / prefillElementSize, 0)
        let destinationReadableCount = max((inputBinding.buffer.length - inputBinding.offset) / decodeElementSize, 0)
        let transferCount = min(
            prefillPlan.buffers.hidden.length
                / max(prefillPlan.maximumSequenceLength, 1)
                / prefillElementSize,
            decodePlan.buffers.hidden.length / decodeElementSize
        )
        let safeTransferCount = max(
            min(transferCount, min(sourceReadableCount, destinationReadableCount)),
            0
        )

        let maximumBufferVocabularySize = decodePlan.buffers.logits.length / decodePlan.buffers.bufferPrecision.byteSize
        let resolvedVocabularySize = min(vocabularySize ?? maximumBufferVocabularySize, maximumBufferVocabularySize)
        let topLogits: [(tokenID: Int, logit: Float, decoded: String)]
        if resolvedVocabularySize > 0 {
            let logits = readModelLogits(
                from: decodePlan.buffers.logits,
                vocabularySize: resolvedVocabularySize,
                precision: decodePlan.buffers.bufferPrecision
            )
            let ranked = logits.enumerated()
                .sorted { lhs, rhs in
                    if lhs.element == rhs.element {
                        return lhs.offset < rhs.offset
                    }
                    return lhs.element > rhs.element
                }
                .prefix(max(0, topK))
            topLogits = ranked.map { entry in
                let tokenID = entry.offset
                return (
                    tokenID: tokenID,
                    logit: entry.element,
                    decoded: modelTokenizer.decode(tokens: [tokenID], skipSpecialTokens: false)
                )
            }
        } else {
            topLogits = []
        }

        return (
            topLogits: topLogits,
            inputLayout: (
                offset: inputBinding.offset,
                bufferLength: inputBinding.buffer.length,
                hiddenCount: hiddenCount,
                readableCount: readableCount,
                bufferKind: describeBufferKind(
                    buffer: inputBinding.buffer,
                    hidden: decodePlan.buffers.hidden,
                    residual: decodePlan.buffers.residual,
                    scratch: decodePlan.buffers.scratch
                )
            ),
            transferLayout: (
                sourceOffset: source.offset,
                sourceBufferLength: source.buffer.length,
                sourceReadableCount: sourceReadableCount,
                sourceBufferKind: describeBufferKind(
                    buffer: source.buffer,
                    hidden: prefillPlan.buffers.hidden,
                    residual: prefillPlan.buffers.residual,
                    scratch: prefillPlan.buffers.scratch
                ),
                destinationOffset: inputBinding.offset,
                destinationBufferLength: inputBinding.buffer.length,
                destinationReadableCount: destinationReadableCount,
                destinationBufferKind: describeBufferKind(
                    buffer: inputBinding.buffer,
                    hidden: decodePlan.buffers.hidden,
                    residual: decodePlan.buffers.residual,
                    scratch: decodePlan.buffers.scratch
                ),
                transferCount: transferCount
            ),
            transferSource: readVector(
                buffer: source.buffer,
                offset: source.offset,
                precision: prefillPlan.buffers.bufferPrecision,
                count: safeTransferCount
            ),
            transferDestination: readVector(
                buffer: inputBinding.buffer,
                offset: inputBinding.offset,
                precision: decodePlan.buffers.bufferPrecision,
                count: safeTransferCount
            )
        )
    }

    internal func debugContinuationLogitComparison(
        prompt: ExecutablePrompt,
        appendedTokenID: Int,
        topK: Int = 10
    ) throws -> (
        prefillTopLogits: [(tokenID: Int, logit: Float, decoded: String)],
        decodeTopLogits: [(tokenID: Int, logit: Float, decoded: String)],
        prefillLogitsFingerprint: UInt64,
        decodeLogitsFingerprint: UInt64,
        maxAbsDiff: Float,
        differingCount: Int
    ) {
        if prompt.visualContext != nil || prompt.gemma4PromptContext != nil {
            throw ModelContainerError.multimodalInputNotSupported(
                "Continuation logit comparison currently supports text-only prompts."
            )
        }

        let extendedPrompt = ExecutablePrompt(
            tokenIDs: prompt.tokenIDs + [appendedTokenID],
            attentionMask: prompt.attentionMask.map { $0 + [1] }
        )

        _ = try prefill(prompt: extendedPrompt)
        let prefillLogits = readModelLogits(
            from: inferenceModel.decodePlan.buffers.logits,
            vocabularySize: resolvedLogitVocabularySize(),
            precision: inferenceModel.decodePlan.buffers.bufferPrecision
        )
        let prefillTopLogits = rankedLogits(prefillLogits, topK: topK)

        resetCaches()
        let prefillResult = try prefill(prompt: prompt)
        _ = try executeDecodeStep(
            tokenID: Int32(appendedTokenID),
            ropePositionOffset: prefillResult.ropePositionOffset
        )
        let decodeLogits = readModelLogits(
            from: inferenceModel.decodePlan.buffers.logits,
            vocabularySize: resolvedLogitVocabularySize(),
            precision: inferenceModel.decodePlan.buffers.bufferPrecision
        )
        let decodeTopLogits = rankedLogits(decodeLogits, topK: topK)

        return (
            prefillTopLogits: prefillTopLogits,
            decodeTopLogits: decodeTopLogits,
            prefillLogitsFingerprint: fingerprint(for: prefillLogits),
            decodeLogitsFingerprint: fingerprint(for: decodeLogits),
            maxAbsDiff: maximumAbsoluteDifference(prefillLogits, decodeLogits),
            differingCount: differingElementCount(prefillLogits, decodeLogits)
        )
    }

    internal func debugPrefillFinalHidden(
        prompt: ExecutablePrompt
    ) throws -> [Float] {
        _ = try prefill(prompt: prompt)
        let decodePlan = inferenceModel.decodePlan
        let inputBinding = decodePlan.outputHeadInputBinding()
        let elementSize = max(decodePlan.buffers.bufferPrecision.byteSize, 1)
        let hiddenSize = decodePlan.buffers.hidden.length / elementSize
        let remainingCount = max((inputBinding.buffer.length - inputBinding.offset) / elementSize, 0)
        let readableCount = min(hiddenSize, remainingCount)
        guard readableCount > 0 else {
            return []
        }

        switch decodePlan.buffers.bufferPrecision {
        case .float32:
            let pointer = inputBinding.buffer.contents().bindMemory(
                to: Float.self,
                capacity: inputBinding.buffer.length / MemoryLayout<Float>.stride
            )
            let elementOffset = inputBinding.offset / MemoryLayout<Float>.stride
            return Array(UnsafeBufferPointer(start: pointer + elementOffset, count: readableCount))
        case .float16:
            let pointer = inputBinding.buffer.contents().bindMemory(
                to: Float16.self,
                capacity: inputBinding.buffer.length / MemoryLayout<Float16>.stride
            )
            let elementOffset = inputBinding.offset / MemoryLayout<Float16>.stride
            return (0..<readableCount).map { Float(pointer[elementOffset + $0]) }
        case .bfloat16:
            let pointer = inputBinding.buffer.contents().bindMemory(
                to: UInt16.self,
                capacity: inputBinding.buffer.length / MemoryLayout<UInt16>.stride
            )
            let elementOffset = inputBinding.offset / MemoryLayout<UInt16>.stride
            return (0..<readableCount).map { index in
                let bits = UInt32(pointer[elementOffset + index]) << 16
                return Float(bitPattern: bits)
            }
        }
    }

    internal func debugClone(
        compiledModel: MetalCompiledModel
    ) throws -> ModelContainer {
        let clonedInferenceModel = try MetalInferenceModel(
            compiledModel: compiledModel,
            device: inferenceModel.device
        )
        return ModelContainer(
            inferenceModel: clonedInferenceModel,
            tokenizer: modelTokenizer,
            configuration: modelConfiguration,
            vocabularySize: vocabularySize,
            finalLogitSoftcapping: finalLogitSoftcapping,
            visionRuntime: visionRuntime,
            gemma4Runtime: gemma4Runtime
        )
    }

    internal var debugCompiledModel: MetalCompiledModel {
        inferenceModel.compiledModel
    }

    internal func debugRepeatedPrefillFinalHiddenDiagnostics(
        prompt: ExecutablePrompt
    ) throws -> (
        firstFingerprint: UInt64,
        secondFingerprint: UInt64,
        firstNaNCount: Int,
        secondNaNCount: Int,
        differingCount: Int
    ) {
        let first = try debugPrefillFinalHidden(prompt: prompt)
        resetCaches()
        let second = try debugPrefillFinalHidden(prompt: prompt)
        return (
            firstFingerprint: fingerprint(for: first),
            secondFingerprint: fingerprint(for: second),
            firstNaNCount: first.filter(\.isNaN).count,
            secondNaNCount: second.filter(\.isNaN).count,
            differingCount: differingElementCount(first, second)
        )
    }

    internal func debugPrefillOutputHeadInputLayout(
        prompt: ExecutablePrompt
    ) throws -> (offset: Int, bufferLength: Int, hiddenCount: Int, readableCount: Int, bufferKind: String) {
        _ = try prefill(prompt: prompt)
        let decodePlan = inferenceModel.decodePlan
        let inputBinding = decodePlan.outputHeadInputBinding()
        let elementSize = max(decodePlan.buffers.bufferPrecision.byteSize, 1)
        let hiddenCount = decodePlan.buffers.hidden.length / elementSize
        let readableCount = max((inputBinding.buffer.length - inputBinding.offset) / elementSize, 0)
        let bufferKind: String
        if inputBinding.buffer === decodePlan.buffers.hidden {
            bufferKind = "hidden"
        } else if inputBinding.buffer === decodePlan.buffers.residual {
            bufferKind = "residual"
        } else if inputBinding.buffer === decodePlan.buffers.scratch {
            bufferKind = "scratch"
        } else {
            bufferKind = "other"
        }
        return (
            offset: inputBinding.offset,
            bufferLength: inputBinding.buffer.length,
            hiddenCount: hiddenCount,
            readableCount: readableCount,
            bufferKind: bufferKind
        )
    }

    internal func debugCurrentDecodeTopLogits(
        topK: Int = 20
    ) -> [(tokenID: Int, logit: Float, decoded: String)] {
        let decodePlan = inferenceModel.decodePlan
        let maximumBufferVocabularySize = decodePlan.buffers.logits.length / decodePlan.buffers.bufferPrecision.byteSize
        let resolvedVocabularySize = min(vocabularySize ?? maximumBufferVocabularySize, maximumBufferVocabularySize)
        guard resolvedVocabularySize > 0 else {
            return []
        }

        let logits = readModelLogits(
            from: decodePlan.buffers.logits,
            vocabularySize: resolvedVocabularySize,
            precision: decodePlan.buffers.bufferPrecision
        )
        let ranked = logits.enumerated()
            .sorted { lhs, rhs in
                if lhs.element == rhs.element {
                    return lhs.offset < rhs.offset
                }
                return lhs.element > rhs.element
            }
            .prefix(max(0, topK))

        return ranked.map { entry in
            let tokenID = entry.offset
            return (
                tokenID: tokenID,
                logit: entry.element,
                decoded: modelTokenizer.decode(tokens: [tokenID], skipSpecialTokens: false)
            )
        }
    }

    private func resolvedLogitVocabularySize() -> Int {
        let decodePlan = inferenceModel.decodePlan
        let maximumBufferVocabularySize = decodePlan.buffers.logits.length / decodePlan.buffers.bufferPrecision.byteSize
        return min(vocabularySize ?? maximumBufferVocabularySize, maximumBufferVocabularySize)
    }

    internal func debugCurrentDecodeTokenOut() -> Int32 {
        inferenceModel.decodePlan.buffers.tokenOut.contents()
            .bindMemory(to: Int32.self, capacity: 1)
            .pointee
    }

    internal func debugGeneratedTokenIDs(
        prompt: ExecutablePrompt,
        parameters: GenerateParameters
    ) throws -> [Int] {
        let resolvedParameters = resolvedGenerateParameters(parameters)
        let prefillResult = try prefill(prompt: prompt)
        var samplingState = makeInitialSamplingState(
            promptTokenIDs: prompt.tokenIDs,
            parameters: resolvedParameters
        )
        let firstToken = resolveSampledPrefillToken(
            fallbackToken: prefillResult.firstToken,
            parameters: resolvedParameters,
            samplingState: &samplingState
        )
        let streamingSamplingState = samplingState.withRecordedFirstToken(
            firstToken,
            maxContextSize: resolvedParameters.repetitionContextSize
        )
        return try collectGeneratedTokenIDs(
            firstToken: firstToken,
            samplingState: streamingSamplingState,
            ropePositionOffset: prefillResult.ropePositionOffset,
            parameters: resolvedParameters,
            collectionMode: .visibleOnly
        )
    }

    internal func debugPromptStateGeneratedTokenIDs(
        promptState: PromptState,
        parameters: GenerateParameters
    ) throws -> [Int] {
        let resolvedParameters = resolvedGenerateParameters(parameters)
        try inferenceModel.restore(promptState: promptState.metalState)
        var samplingState = GenerationSamplingState(
            rngState: promptState.samplingSeed,
            recentTokenIDs: Array(promptState.promptTokenTail.suffix(resolvedParameters.repetitionContextSize))
        )
        let firstToken = resolveSampledPromptStateToken(
            fallbackToken: promptState.metalState.firstToken,
            logitsBuffer: inferenceModel.decodePlan.buffers.logits,
            parameters: resolvedParameters,
            samplingState: &samplingState
        )
        let streamingSamplingState = samplingState.withRecordedFirstToken(
            firstToken,
            maxContextSize: resolvedParameters.repetitionContextSize
        )
        return try collectGeneratedTokenIDs(
            firstToken: firstToken,
            samplingState: streamingSamplingState,
            ropePositionOffset: promptState.ropePositionOffset,
            parameters: resolvedParameters,
            collectionMode: .visibleOnly
        )
    }

    internal func debugRawGeneratedTokenIDs(
        prompt: ExecutablePrompt,
        parameters: GenerateParameters
    ) throws -> [Int] {
        let resolvedParameters = resolvedGenerateParameters(parameters)
        let prefillResult = try prefill(prompt: prompt)
        var samplingState = makeInitialSamplingState(
            promptTokenIDs: prompt.tokenIDs,
            parameters: resolvedParameters
        )
        let firstToken = resolveSampledPrefillToken(
            fallbackToken: prefillResult.firstToken,
            parameters: resolvedParameters,
            samplingState: &samplingState
        )
        let streamingSamplingState = samplingState.withRecordedFirstToken(
            firstToken,
            maxContextSize: resolvedParameters.repetitionContextSize
        )
        return try collectGeneratedTokenIDs(
            firstToken: firstToken,
            samplingState: streamingSamplingState,
            ropePositionOffset: prefillResult.ropePositionOffset,
            parameters: resolvedParameters,
            collectionMode: .raw
        )
    }

    internal func debugPromptStateGenerationTrace(
        prompt: ExecutablePrompt,
        parameters: GenerateParameters,
        topK: Int = 10
    ) throws -> (
        directBoundary: DebugGenerationBoundaryState,
        restoredBoundary: DebugGenerationBoundaryState,
        directSteps: [DebugGenerationStepTrace],
        restoredSteps: [DebugGenerationStepTrace]
    ) {
        let resolvedParameters = resolvedGenerateParameters(parameters)

        let prefillResult = try prefill(prompt: prompt)
        var directInitialSamplingState = makeInitialSamplingState(
            promptTokenIDs: prompt.tokenIDs,
            parameters: resolvedParameters
        )
        let directFirstToken = resolveSampledPrefillToken(
            fallbackToken: prefillResult.firstToken,
            parameters: resolvedParameters,
            samplingState: &directInitialSamplingState
        )
        let directStreamingSamplingState = directInitialSamplingState.withRecordedFirstToken(
            directFirstToken,
            maxContextSize: resolvedParameters.repetitionContextSize
        )
        let directBoundary = debugCurrentDecodeBoundaryState(
            firstToken: directFirstToken,
            parameters: resolvedParameters,
            recentTokenIDs: directStreamingSamplingState.recentTokenIDs,
            topK: topK
        )
        let directSteps = try debugDecodeTrace(
            firstToken: directFirstToken,
            samplingState: directStreamingSamplingState,
            ropePositionOffset: prefillResult.ropePositionOffset,
            parameters: resolvedParameters,
            topK: topK
        )

        resetCaches()
        let promptState = try makePromptState(prompt: prompt)
        try inferenceModel.restore(promptState: promptState.metalState)
        var restoredInitialSamplingState = GenerationSamplingState(
            rngState: promptState.samplingSeed,
            recentTokenIDs: Array(promptState.promptTokenTail.suffix(resolvedParameters.repetitionContextSize))
        )
        let restoredFirstToken = resolveSampledPromptStateToken(
            fallbackToken: promptState.metalState.firstToken,
            logitsBuffer: inferenceModel.decodePlan.buffers.logits,
            parameters: resolvedParameters,
            samplingState: &restoredInitialSamplingState
        )
        let restoredStreamingSamplingState = restoredInitialSamplingState.withRecordedFirstToken(
            restoredFirstToken,
            maxContextSize: resolvedParameters.repetitionContextSize
        )
        let restoredBoundary = debugCurrentDecodeBoundaryState(
            firstToken: restoredFirstToken,
            parameters: resolvedParameters,
            recentTokenIDs: restoredStreamingSamplingState.recentTokenIDs,
            topK: topK
        )
        let restoredSteps = try debugDecodeTrace(
            firstToken: restoredFirstToken,
            samplingState: restoredStreamingSamplingState,
            ropePositionOffset: promptState.ropePositionOffset,
            parameters: resolvedParameters,
            topK: topK
        )

        return (
            directBoundary: directBoundary,
            restoredBoundary: restoredBoundary,
            directSteps: directSteps,
            restoredSteps: restoredSteps
        )
    }

    internal func debugPromptStateRestoreDiagnostics(
        prompt: ExecutablePrompt,
        topK: Int = 10
    ) throws -> (
        directTopLogits: [(tokenID: Int, logit: Float, decoded: String)],
        directTokenOut: Int32,
        promptStateFirstToken: Int32,
        restoredTopLogits: [(tokenID: Int, logit: Float, decoded: String)],
        restoredTokenOut: Int32
    ) {
        let directTopLogits = try debugPrefillTopLogits(prompt: prompt, topK: topK)
        let directTokenOut = debugCurrentDecodeTokenOut()

        resetCaches()
        let promptState = try makePromptState(prompt: prompt)
        try inferenceModel.restore(promptState: promptState.metalState)

        return (
            directTopLogits: directTopLogits,
            directTokenOut: directTokenOut,
            promptStateFirstToken: promptState.metalState.firstToken,
            restoredTopLogits: debugCurrentDecodeTopLogits(topK: topK),
            restoredTokenOut: debugCurrentDecodeTokenOut()
        )
    }

    internal func debugPromptStateSampledFirstTokens(
        prompt: ExecutablePrompt,
        parameters: GenerateParameters
    ) throws -> (
        direct: Int32,
        restored: Int32,
        directRecentTokenIDs: [Int],
        restoredRecentTokenIDs: [Int],
        directTopLogits: [(tokenID: Int, logit: Float, decoded: String)],
        restoredTopLogits: [(tokenID: Int, logit: Float, decoded: String)]
    ) {
        let resolvedParameters = resolvedGenerateParameters(parameters)
        let prefillResult = try prefill(prompt: prompt)
        var directSamplingState = makeInitialSamplingState(
            promptTokenIDs: prompt.tokenIDs,
            parameters: resolvedParameters
        )
        let directRecentTokenIDs = directSamplingState.recentTokenIDs
        let directTopLogits = debugCurrentSamplingTopLogits(
            parameters: resolvedParameters,
            recentTokenIDs: directRecentTokenIDs,
            topK: 10
        )
        let direct = resolveSampledPrefillToken(
            fallbackToken: prefillResult.firstToken,
            parameters: resolvedParameters,
            samplingState: &directSamplingState
        )

        resetCaches()
        let promptState = try makePromptState(prompt: prompt)
        try inferenceModel.restore(promptState: promptState.metalState)
        var restoredSamplingState = GenerationSamplingState(
            rngState: promptState.samplingSeed,
            recentTokenIDs: Array(promptState.promptTokenTail.suffix(resolvedParameters.repetitionContextSize))
        )
        let restoredRecentTokenIDs = restoredSamplingState.recentTokenIDs
        let restoredTopLogits = debugCurrentSamplingTopLogits(
            parameters: resolvedParameters,
            recentTokenIDs: restoredRecentTokenIDs,
            topK: 10
        )
        let restored = resolveSampledPromptStateToken(
            fallbackToken: promptState.metalState.firstToken,
            logitsBuffer: inferenceModel.decodePlan.buffers.logits,
            parameters: resolvedParameters,
            samplingState: &restoredSamplingState
        )

        return (
            direct: direct,
            restored: restored,
            directRecentTokenIDs: directRecentTokenIDs,
            restoredRecentTokenIDs: restoredRecentTokenIDs,
            directTopLogits: directTopLogits,
            restoredTopLogits: restoredTopLogits
        )
    }

    internal func debugRepeatedPrefillSampledFirstTokens(
        prompt: ExecutablePrompt,
        parameters: GenerateParameters
    ) throws -> (
        first: Int32,
        second: Int32,
        firstTopLogits: [(tokenID: Int, logit: Float, decoded: String)],
        secondTopLogits: [(tokenID: Int, logit: Float, decoded: String)],
        firstLogitFingerprint: UInt64,
        secondLogitFingerprint: UInt64,
        maxAbsDiff: Float,
        differingCount: Int,
        firstNaNCount: Int,
        secondNaNCount: Int,
        firstDifferingEntries: [(tokenID: Int, first: Float, second: Float)]
    ) {
        let resolvedParameters = resolvedGenerateParameters(parameters)
        let first = try debugSinglePrefillSampledFirstToken(
            prompt: prompt,
            parameters: resolvedParameters
        )

        resetCaches()

        let second = try debugSinglePrefillSampledFirstToken(
            prompt: prompt,
            parameters: resolvedParameters
        )

        return (
            first: first.sampledToken,
            second: second.sampledToken,
            firstTopLogits: first.topLogits,
            secondTopLogits: second.topLogits,
            firstLogitFingerprint: fingerprint(for: first.processedLogits),
            secondLogitFingerprint: fingerprint(for: second.processedLogits),
            maxAbsDiff: maximumAbsoluteDifference(first.processedLogits, second.processedLogits),
            differingCount: differingElementCount(first.processedLogits, second.processedLogits),
            firstNaNCount: first.processedLogits.filter(\.isNaN).count,
            secondNaNCount: second.processedLogits.filter(\.isNaN).count,
            firstDifferingEntries: firstDifferingEntries(first.processedLogits, second.processedLogits)
        )
    }

    internal func debugPrefillTransferVectors(
        prompt: ExecutablePrompt
    ) throws -> (source: [Float], destination: [Float]) {
        _ = try prefill(prompt: prompt)
        guard let prefillPlan = inferenceModel.prefillPlan else {
            return ([], [])
        }

        let source = prefillPlan.finalHiddenSource(sequenceLength: prompt.tokenIDs.count)
        let decodePlan = inferenceModel.decodePlan
        let destination = decodePlan.outputHeadInputBinding()

        let prefillElementSize = max(prefillPlan.buffers.bufferPrecision.byteSize, 1)
        let decodeElementSize = max(decodePlan.buffers.bufferPrecision.byteSize, 1)
        let prefillHiddenCount = max(
            min(
                prefillPlan.buffers.hidden.length
                    / max(prefillPlan.maximumSequenceLength, 1)
                    / prefillElementSize,
                max((source.buffer.length - source.offset) / prefillElementSize, 0)
            ),
            0
        )
        let decodeHiddenCount = max(
            min(
                decodePlan.buffers.hidden.length / decodeElementSize,
                max((destination.buffer.length - destination.offset) / decodeElementSize, 0)
            ),
            0
        )
        let count = min(prefillHiddenCount, decodeHiddenCount)
        guard count > 0 else {
            return ([], [])
        }

        return (
            readVector(
                buffer: source.buffer,
                offset: source.offset,
                precision: prefillPlan.buffers.bufferPrecision,
                count: count
            ),
            readVector(
                buffer: destination.buffer,
                offset: destination.offset,
                precision: decodePlan.buffers.bufferPrecision,
                count: count
            )
        )
    }

    internal func debugPrefillTransferLayout(
        prompt: ExecutablePrompt
    ) throws -> (
        sourceOffset: Int,
        sourceBufferLength: Int,
        sourceReadableCount: Int,
        sourceBufferKind: String,
        destinationOffset: Int,
        destinationBufferLength: Int,
        destinationReadableCount: Int,
        destinationBufferKind: String,
        transferCount: Int
    ) {
        _ = try prefill(prompt: prompt)
        guard let prefillPlan = inferenceModel.prefillPlan else {
            return (0, 0, 0, "missing", 0, 0, 0, "missing", 0)
        }

        let source = prefillPlan.finalHiddenSource(sequenceLength: prompt.tokenIDs.count)
        let decodePlan = inferenceModel.decodePlan
        let destination = decodePlan.outputHeadInputBinding()
        let prefillElementSize = max(prefillPlan.buffers.bufferPrecision.byteSize, 1)
        let decodeElementSize = max(decodePlan.buffers.bufferPrecision.byteSize, 1)
        let sourceReadableCount = max((source.buffer.length - source.offset) / prefillElementSize, 0)
        let destinationReadableCount = max((destination.buffer.length - destination.offset) / decodeElementSize, 0)
        let transferCount = min(
            prefillPlan.buffers.hidden.length
                / max(prefillPlan.maximumSequenceLength, 1)
                / prefillElementSize,
            decodePlan.buffers.hidden.length / decodeElementSize
        )

        return (
            sourceOffset: source.offset,
            sourceBufferLength: source.buffer.length,
            sourceReadableCount: sourceReadableCount,
            sourceBufferKind: describeBufferKind(
                buffer: source.buffer,
                hidden: prefillPlan.buffers.hidden,
                residual: prefillPlan.buffers.residual,
                scratch: prefillPlan.buffers.scratch
            ),
            destinationOffset: destination.offset,
            destinationBufferLength: destination.buffer.length,
            destinationReadableCount: destinationReadableCount,
            destinationBufferKind: describeBufferKind(
                buffer: destination.buffer,
                hidden: decodePlan.buffers.hidden,
                residual: decodePlan.buffers.residual,
                scratch: decodePlan.buffers.scratch
            ),
            transferCount: transferCount
        )
    }

    internal func debugPrefillLastTokenHiddenSnapshots(
        prompt: ExecutablePrompt,
        stepIndices: Set<Int>
    ) throws -> [Int: [Float]] {
        if let gemma4PromptContext = prompt.gemma4PromptContext {
            if gemma4PromptContext.usesEmbeddingOverrides == false {
                return try inferenceModel.debugPrefillLastTokenHiddenSnapshots(
                    tokens: prompt.tokenIDs.map(Int32.init),
                    stepIndices: stepIndices,
                    prefillPerLayerInputs: gemma4PromptContext.perLayerInputs
                )
            }
            let ropeAxes = (0..<gemma4PromptContext.promptEmbeddings.count).map { index -> (UInt32, UInt32, UInt32) in
                let value = UInt32(index)
                return (value, value, value)
            }
            return try inferenceModel.debugPrefillLastTokenHiddenSnapshots(
                hiddenStates: gemma4PromptContext.promptEmbeddings,
                ropePositionAxes: ropeAxes,
                stepIndices: stepIndices,
                prefillPerLayerInputs: gemma4PromptContext.perLayerInputs
            )
        }
        return try inferenceModel.debugPrefillLastTokenHiddenSnapshots(
            tokens: prompt.tokenIDs.map(Int32.init),
            stepIndices: stepIndices
        )
    }

    internal func debugPrefillLastTokenResidualSnapshots(
        prompt: ExecutablePrompt,
        stepIndices: Set<Int>
    ) throws -> [Int: [Float]] {
        if let gemma4PromptContext = prompt.gemma4PromptContext {
            if gemma4PromptContext.usesEmbeddingOverrides == false {
                return try inferenceModel.debugPrefillLastTokenResidualSnapshots(
                    tokens: prompt.tokenIDs.map(Int32.init),
                    stepIndices: stepIndices
                )
            }
            let ropeAxes = (0..<gemma4PromptContext.promptEmbeddings.count).map { index -> (UInt32, UInt32, UInt32) in
                let value = UInt32(index)
                return (value, value, value)
            }
            return try inferenceModel.debugPrefillLastTokenResidualSnapshots(
                hiddenStates: gemma4PromptContext.promptEmbeddings,
                ropePositionAxes: ropeAxes,
                stepIndices: stepIndices,
                prefillPerLayerInputs: gemma4PromptContext.perLayerInputs
            )
        }
        return try inferenceModel.debugPrefillLastTokenResidualSnapshots(
            tokens: prompt.tokenIDs.map(Int32.init),
            stepIndices: stepIndices
        )
    }

    internal func debugPrefillLastTokenScratchSnapshots(
        prompt: ExecutablePrompt,
        stepIndices: Set<Int>,
        slotIndex: Int,
        rowStride: Int,
        count: Int
    ) throws -> [Int: [Float]] {
        if let gemma4PromptContext = prompt.gemma4PromptContext {
            if gemma4PromptContext.usesEmbeddingOverrides == false {
                return try inferenceModel.debugPrefillLastTokenScratchSnapshots(
                    tokens: prompt.tokenIDs.map(Int32.init),
                    stepIndices: stepIndices,
                    slotIndex: slotIndex,
                    rowStride: rowStride,
                    count: count,
                    prefillPerLayerInputs: gemma4PromptContext.perLayerInputs
                )
            }
            let ropeAxes = (0..<gemma4PromptContext.promptEmbeddings.count).map { index -> (UInt32, UInt32, UInt32) in
                let value = UInt32(index)
                return (value, value, value)
            }
            return try inferenceModel.debugPrefillLastTokenScratchSnapshots(
                hiddenStates: gemma4PromptContext.promptEmbeddings,
                ropePositionAxes: ropeAxes,
                stepIndices: stepIndices,
                slotIndex: slotIndex,
                rowStride: rowStride,
                count: count,
                prefillPerLayerInputs: gemma4PromptContext.perLayerInputs
            )
        }
        return try inferenceModel.debugPrefillLastTokenScratchSnapshots(
            tokens: prompt.tokenIDs.map(Int32.init),
            stepIndices: stepIndices,
            slotIndex: slotIndex,
            rowStride: rowStride,
            count: count
        )
    }

    internal func debugPrefillStepSummaries() -> [(index: Int, kernelName: String, layerIndex: Int?)] {
        guard let prefillPlan = inferenceModel.prefillPlan else {
            return []
        }
        return prefillPlan.steps.enumerated().map { index, step in
            (
                index: index,
                kernelName: step.metadata.kernelName ?? step.pipeline.label ?? "(unlabeled)",
                layerIndex: step.metadata.layerIndex
            )
        }
    }

    internal func debugPrefillSteps() -> [MetalPrefillStep] {
        inferenceModel.prefillPlan?.steps ?? []
    }

    internal func debugDescribePrefillSteps(
        indices: Set<Int>
    ) -> [Int: String] {
        guard let prefillPlan = inferenceModel.prefillPlan else {
            return [:]
        }
        var descriptions: [Int: String] = [:]
        for index in indices.sorted() where index >= 0 && index < prefillPlan.steps.count {
            let step = prefillPlan.steps[index]
            let buffers = step.bufferBindings.map { binding in
                let kind = describeBufferKind(
                    buffer: binding.buffer,
                    hidden: prefillPlan.buffers.hidden,
                    residual: prefillPlan.buffers.residual,
                    scratch: prefillPlan.buffers.scratch
                )
                return "\(binding.index):\(kind)@\(binding.offset)"
            }.joined(separator: ", ")
            let bytes = step.bytesBindings.map { binding in
                "\(binding.index):\(binding.value.count)b"
            }.joined(separator: ", ")
            let barrier = String(describing: step.barrierPolicy)
            let entryIndex = step.metadata.entryIndex.map(String.init) ?? "-"
            descriptions[index] = "entry=\(entryIndex) barrier=\(barrier) buffers=[\(buffers)] bytes=[\(bytes)]"
        }
        return descriptions
    }

#if ENABLE_METAL_PROBES
    internal func debugDecodeStepSummaries() -> [(index: Int, kernelName: String, layerIndex: Int?)] {
        inferenceModel.decodePlan.steps.enumerated().map { index, step in
            (
                index: index,
                kernelName: step.metadata.kernelName ?? step.pipeline.label ?? "(unlabeled)",
                layerIndex: step.metadata.layerIndex
            )
        }
    }

    internal func debugDecodeBindingProbes(
        prompt: ExecutablePrompt,
        tokenID: Int,
        probes: [MetalInferenceModel.DebugDecodeBindingProbe],
        visibilityOptions: MTL4VisibilityOptions = []
    ) throws -> [String: [Float]] {
        guard prompt.visualContext == nil, prompt.gemma4PromptContext == nil else {
            throw ModelContainerError.multimodalInputNotSupported(
                "Decode binding probes currently support text-only prompts."
            )
        }
        return try inferenceModel.debugDecodeBindingProbes(
            promptTokens: prompt.tokenIDs.map(Int32.init),
            tokenID: Int32(tokenID),
            ropePositionAxes: generationRoPEAxes(offset: 0),
            probes: probes,
            visibilityOptions: visibilityOptions
        )
    }

    internal func debugPrefillBindingProbes(
        prompt: ExecutablePrompt,
        stepIndex: Int,
        probes: [MetalInferenceModel.DebugPrefillBindingProbe],
        isolatedSubmission: Bool = true,
        visibilityOptions: MTL4VisibilityOptions = [],
        stepVisibilityOptions: MTL4VisibilityOptions? = nil,
        probeVisibilityOptions: MTL4VisibilityOptions? = nil
    ) throws -> [String: [Float]] {
        if let gemma4PromptContext = prompt.gemma4PromptContext {
            let ropeAxes = (0..<gemma4PromptContext.promptEmbeddings.count).map { index -> (UInt32, UInt32, UInt32) in
                let value = UInt32(index)
                return (value, value, value)
            }
            return try inferenceModel.debugPrefillBindingProbes(
                hiddenStates: gemma4PromptContext.promptEmbeddings,
                ropePositionAxes: ropeAxes,
                stepIndex: stepIndex,
                probes: probes,
                prefillPerLayerInputs: gemma4PromptContext.perLayerInputs,
                isolatedSubmission: isolatedSubmission,
                visibilityOptions: visibilityOptions,
                stepVisibilityOptions: stepVisibilityOptions,
                probeVisibilityOptions: probeVisibilityOptions
            )
        }

        return try inferenceModel.debugPrefillBindingProbes(
            tokens: prompt.tokenIDs.map(Int32.init),
            stepIndex: stepIndex,
            probes: probes,
            isolatedSubmission: isolatedSubmission,
            visibilityOptions: visibilityOptions,
            stepVisibilityOptions: stepVisibilityOptions,
            probeVisibilityOptions: probeVisibilityOptions
        )
    }

    internal func debugActualPrefillBindingProbes(
        prompt: ExecutablePrompt,
        stepIndex: Int,
        probes: [MetalInferenceModel.DebugPrefillBindingProbe],
        visibilityOptions: MTL4VisibilityOptions = []
    ) throws -> [String: [Float]] {
        if let gemma4PromptContext = prompt.gemma4PromptContext {
            let ropeAxes = (0..<gemma4PromptContext.promptEmbeddings.count).map { index -> (UInt32, UInt32, UInt32) in
                let value = UInt32(index)
                return (value, value, value)
            }
            return try inferenceModel.debugActualPrefillBindingProbes(
                hiddenStates: gemma4PromptContext.promptEmbeddings,
                ropePositionAxes: ropeAxes,
                stepIndex: stepIndex,
                probes: probes,
                prefillPerLayerInputs: gemma4PromptContext.perLayerInputs,
                visibilityOptions: visibilityOptions
            )
        }

        return try inferenceModel.debugPrefillBindingProbes(
            tokens: prompt.tokenIDs.map(Int32.init),
            stepIndex: stepIndex,
            probes: probes,
            isolatedSubmission: true,
            visibilityOptions: visibilityOptions,
            stepVisibilityOptions: nil,
            probeVisibilityOptions: nil
        )
    }
#endif

    private func makeInitialSamplingState(
        promptTokenIDs: [Int],
        parameters: GenerateParameters
    ) -> GenerationSamplingState {
        GenerationSamplingState(
            rngState: samplingSeed(for: promptTokenIDs),
            recentTokenIDs: Array(promptTokenIDs.suffix(parameters.repetitionContextSize))
        )
    }

    private func resolvedGenerateParameters(_ parameters: GenerateParameters) -> GenerateParameters {
        var resolved = parameters
        let modelName = modelConfiguration.name.lowercased()

        if thinkingTagPolicy != nil,
           modelName.hasPrefix("lfm"),
           parameters.usesLibraryDefaults {
            resolved.temperature = 0.1
            resolved.topP = 0.1
            resolved.topK = 50
            resolved.repetitionPenalty = 1.05
            resolved.presencePenalty = nil
            return resolved
        }

        let usesQwenChatDefaults = modelName == "qwen3" || modelName.hasPrefix("qwen3_")
        guard usesQwenChatDefaults, parameters.usesLibraryDefaults else {
            return parameters
        }
        resolved.temperature = 0.7
        resolved.topP = 0.8
        resolved.topK = 20
        resolved.presencePenalty = 1.5
        return resolved
    }

    private func resolveSampledDecodeToken(
        fallbackToken: Int32,
        parameters: GenerateParameters,
        samplingState: inout GenerationSamplingState
    ) -> Int32 {
        resolveSampledToken(
            fallbackToken: fallbackToken,
            logitsBuffer: inferenceModel.decodePlan.buffers.logits,
            precision: inferenceModel.decodePlan.buffers.bufferPrecision,
            parameters: parameters,
            samplingState: &samplingState
        )
    }

    private func resolveSampledPrefillToken(
        fallbackToken: Int32,
        parameters: GenerateParameters,
        samplingState: inout GenerationSamplingState
    ) -> Int32 {
        guard inferenceModel.prefillPlan != nil else {
            return fallbackToken
        }
        return resolveSampledToken(
            fallbackToken: fallbackToken,
            logitsBuffer: inferenceModel.decodePlan.buffers.logits,
            precision: inferenceModel.decodePlan.buffers.bufferPrecision,
            parameters: parameters,
            samplingState: &samplingState
        )
    }

    private func readVector(
        buffer: MTLBuffer,
        offset: Int,
        precision: BufferPrecision,
        count: Int
    ) -> [Float] {
        guard count > 0 else { return [] }
        switch precision {
        case .float32:
            let pointer = buffer.contents().bindMemory(
                to: Float.self,
                capacity: buffer.length / MemoryLayout<Float>.stride
            )
            let elementOffset = offset / MemoryLayout<Float>.stride
            return Array(UnsafeBufferPointer(start: pointer + elementOffset, count: count))
        case .float16:
            let pointer = buffer.contents().bindMemory(
                to: Float16.self,
                capacity: buffer.length / MemoryLayout<Float16>.stride
            )
            let elementOffset = offset / MemoryLayout<Float16>.stride
            return (0..<count).map { Float(pointer[elementOffset + $0]) }
        case .bfloat16:
            let pointer = buffer.contents().bindMemory(
                to: UInt16.self,
                capacity: buffer.length / MemoryLayout<UInt16>.stride
            )
            let elementOffset = offset / MemoryLayout<UInt16>.stride
            return (0..<count).map { index in
                Float(bitPattern: UInt32(pointer[elementOffset + index]) << 16)
            }
        }
    }

    private func describeBufferKind(
        buffer: MTLBuffer,
        hidden: MTLBuffer,
        residual: MTLBuffer,
        scratch: MTLBuffer
    ) -> String {
        if buffer === hidden { return "hidden" }
        if buffer === residual { return "residual" }
        if buffer === scratch { return "scratch" }
        return "other"
    }

    private func resolveSampledPromptStateToken(
        fallbackToken: Int32,
        logitsBuffer: MTLBuffer,
        parameters: GenerateParameters,
        samplingState: inout GenerationSamplingState
    ) -> Int32 {
        resolveSampledToken(
            fallbackToken: fallbackToken,
            logitsBuffer: logitsBuffer,
            precision: inferenceModel.decodePlan.buffers.bufferPrecision,
            parameters: parameters,
            samplingState: &samplingState
        )
    }

    private func resolveSampledToken(
        fallbackToken: Int32,
        logitsBuffer: MTLBuffer,
        precision: BufferPrecision,
        parameters: GenerateParameters,
        samplingState: inout GenerationSamplingState
    ) -> Int32 {
        if ProcessInfo.processInfo.environment["SWIFTLM_DISABLE_HOST_SAMPLING"] == "1" {
            return fallbackToken
        }
        let suppressedTokenIDs = specialTokensToSuppress
        let needsGreedyRepair = suppressedTokenIDs.contains(Int(fallbackToken))
        let requiresHostLogitPostprocessing = finalLogitSoftcapping != nil
        let shouldSampleOnHost =
            parameters.temperature > 0
            || parameters.topP < 1
            || parameters.repetitionPenalty != nil
            || needsGreedyRepair
            || requiresHostLogitPostprocessing

        if ProcessInfo.processInfo.environment["SWIFTLM_TRACE_SAMPLING"] == "1" {
            print(
                "[ModelContainer] sampling trace: fallback=\(fallbackToken) temp=\(parameters.temperature) topP=\(parameters.topP) "
                    + "needsRepair=\(needsGreedyRepair) hostPost=\(requiresHostLogitPostprocessing) host=\(shouldSampleOnHost)"
            )
        }

        guard shouldSampleOnHost else {
            return fallbackToken
        }

        let maximumBufferVocabularySize = logitsBuffer.length / precision.byteSize
        let vocabularySize = min(vocabularySize ?? maximumBufferVocabularySize, maximumBufferVocabularySize)
        guard vocabularySize > 0 else {
            return fallbackToken
        }

        var logits = readModelLogits(
            from: logitsBuffer,
            vocabularySize: vocabularySize,
            precision: precision
        )
        applySuppressedTokenMask(suppressedTokenIDs, to: &logits)
        applyPresencePenalty(
            to: &logits,
            recentTokenIDs: samplingState.recentTokenIDs,
            penalty: parameters.presencePenalty
        )
        applyRepetitionPenalty(
            to: &logits,
            recentTokenIDs: samplingState.recentTokenIDs,
            penalty: parameters.repetitionPenalty
        )
        return Int32(sampleTokenID(
            from: logits,
            fallbackTokenID: Int(fallbackToken),
            temperature: parameters.temperature,
            topP: parameters.topP,
            topK: parameters.topK,
            minP: parameters.minP,
            random: samplingState.nextUnitFloat()
        ))
    }

    private func debugSinglePrefillSampledFirstToken(
        prompt: ExecutablePrompt,
        parameters: GenerateParameters
    ) throws -> (
        sampledToken: Int32,
        topLogits: [(tokenID: Int, logit: Float, decoded: String)],
        processedLogits: [Float]
    ) {
        let prefillResult = try prefill(prompt: prompt)
        var samplingState = makeInitialSamplingState(
            promptTokenIDs: prompt.tokenIDs,
            parameters: parameters
        )
        let processedLogits = preparedSamplingLogits(
            logitsBuffer: inferenceModel.decodePlan.buffers.logits,
            precision: inferenceModel.decodePlan.buffers.bufferPrecision,
            parameters: parameters,
            recentTokenIDs: samplingState.recentTokenIDs
        )
        let topLogits = rankedLogits(processedLogits, topK: 10)
        let sampledToken = resolveSampledPrefillToken(
            fallbackToken: prefillResult.firstToken,
            parameters: parameters,
            samplingState: &samplingState
        )
        return (
            sampledToken: sampledToken,
            topLogits: topLogits,
            processedLogits: processedLogits
        )
    }

    private func debugCurrentSamplingTopLogits(
        parameters: GenerateParameters,
        recentTokenIDs: [Int],
        topK: Int
    ) -> [(tokenID: Int, logit: Float, decoded: String)] {
        rankedLogits(
            preparedSamplingLogits(
                logitsBuffer: inferenceModel.decodePlan.buffers.logits,
                precision: inferenceModel.decodePlan.buffers.bufferPrecision,
                parameters: parameters,
                recentTokenIDs: recentTokenIDs
            ),
            topK: topK
        )
    }

    private func debugCurrentDecodeBoundaryState(
        firstToken: Int32,
        parameters: GenerateParameters,
        recentTokenIDs: [Int],
        topK: Int
    ) -> DebugGenerationBoundaryState {
        let buffers = inferenceModel.decodePlan.buffers
        let positionValue = buffers.position.contents().bindMemory(to: UInt32.self, capacity: 1).pointee
        let ropePointer = buffers.ropePositionAxes.contents().bindMemory(to: UInt32.self, capacity: 3)
        let tokenInValue = buffers.tokenIn.contents().bindMemory(to: Int32.self, capacity: 1).pointee
        let tokenOutValue = buffers.tokenOut.contents().bindMemory(to: Int32.self, capacity: 1).pointee
        let processedLogits = preparedSamplingLogits(
            logitsBuffer: buffers.logits,
            precision: buffers.bufferPrecision,
            parameters: parameters,
            recentTokenIDs: recentTokenIDs
        )
        return DebugGenerationBoundaryState(
            firstToken: Int(firstToken),
            position: inferenceModel.position,
            positionBufferValue: Int(positionValue),
            ropePositionAxes: [ropePointer[0], ropePointer[1], ropePointer[2]].map(Int.init),
            tokenIn: Int(tokenInValue),
            tokenOut: Int(tokenOutValue),
            processedLogitsFingerprint: fingerprint(for: processedLogits),
            topLogits: rankedLogits(processedLogits, topK: topK),
            recentTokenIDs: recentTokenIDs
        )
    }

    private func debugDecodeTrace(
        firstToken: Int32,
        samplingState initialSamplingState: GenerationSamplingState,
        ropePositionOffset: Int,
        parameters: GenerateParameters,
        topK: Int
    ) throws -> [DebugGenerationStepTrace] {
        var samplingState = initialSamplingState
        let maxTokens = parameters.maxTokens ?? 1024
        guard maxTokens > 0, firstToken >= 0 else {
            return []
        }
        guard !self.modelConfiguration.eosTokenIds.contains(Int(firstToken)) else {
            return []
        }

        var traces: [DebugGenerationStepTrace] = []
        var tokenCount = 1
        var nextTokenID = firstToken
        while tokenCount < maxTokens {
            let positionBefore = inferenceModel.position
            let tokenInBefore = Int(
                inferenceModel.decodePlan.buffers.tokenIn.contents()
                    .bindMemory(to: Int32.self, capacity: 1)
                    .pointee
            )
            let tokenOutBefore = Int(
                inferenceModel.decodePlan.buffers.tokenOut.contents()
                    .bindMemory(to: Int32.self, capacity: 1)
                    .pointee
            )
            let rngStateBefore = samplingState.rngState
            var probeRNG = DeterministicRNG(state: rngStateBefore)
            let randomValue = probeRNG.nextUnitFloat()

            let argmaxToken = try self.executeDecodeStep(
                tokenID: nextTokenID,
                ropePositionOffset: ropePositionOffset
            )

            let processedLogits = preparedSamplingLogits(
                logitsBuffer: inferenceModel.decodePlan.buffers.logits,
                precision: inferenceModel.decodePlan.buffers.bufferPrecision,
                parameters: parameters,
                recentTokenIDs: samplingState.recentTokenIDs
            )
            let outputToken = self.resolveSampledDecodeToken(
                fallbackToken: argmaxToken,
                parameters: parameters,
                samplingState: &samplingState
            )
            let tokenOutAfter = Int(
                inferenceModel.decodePlan.buffers.tokenOut.contents()
                    .bindMemory(to: Int32.self, capacity: 1)
                    .pointee
            )

            traces.append(
                DebugGenerationStepTrace(
                    stepIndex: traces.count,
                    inputTokenID: Int(nextTokenID),
                    positionBefore: positionBefore,
                    tokenInBefore: tokenInBefore,
                    tokenOutBefore: tokenOutBefore,
                    argmaxTokenID: Int(argmaxToken),
                    sampledTokenID: Int(outputToken),
                    tokenOutAfter: tokenOutAfter,
                    rngStateBefore: rngStateBefore,
                    randomValue: randomValue,
                    processedLogitsFingerprint: fingerprint(for: processedLogits),
                    topLogits: rankedLogits(processedLogits, topK: topK),
                    recentTokenIDs: samplingState.recentTokenIDs
                )
            )

            if outputToken < 0 || self.modelConfiguration.eosTokenIds.contains(Int(outputToken)) {
                break
            }

            tokenCount += 1
            samplingState.record(Int(outputToken), maxContextSize: parameters.repetitionContextSize)
            nextTokenID = outputToken
        }

        return traces
    }

    private func preparedSamplingLogits(
        logitsBuffer: MTLBuffer,
        precision: BufferPrecision,
        parameters: GenerateParameters,
        recentTokenIDs: [Int]
    ) -> [Float] {
        let maximumBufferVocabularySize = logitsBuffer.length / precision.byteSize
        let resolvedVocabularySize = min(vocabularySize ?? maximumBufferVocabularySize, maximumBufferVocabularySize)
        guard resolvedVocabularySize > 0 else {
            return []
        }

        var logits = readModelLogits(
            from: logitsBuffer,
            vocabularySize: resolvedVocabularySize,
            precision: precision
        )
        applySuppressedTokenMask(specialTokensToSuppress, to: &logits)
        applyPresencePenalty(
            to: &logits,
            recentTokenIDs: recentTokenIDs,
            penalty: parameters.presencePenalty
        )
        applyRepetitionPenalty(
            to: &logits,
            recentTokenIDs: recentTokenIDs,
            penalty: parameters.repetitionPenalty
        )
        return logits
    }

    private func rankedLogits(
        _ logits: [Float],
        topK: Int
    ) -> [(tokenID: Int, logit: Float, decoded: String)] {
        logits.enumerated()
            .sorted { lhs, rhs in
                if lhs.element == rhs.element {
                    return lhs.offset < rhs.offset
                }
                return lhs.element > rhs.element
            }
            .prefix(max(0, topK))
            .map { entry in
                let tokenID = entry.offset
                return (
                    tokenID: tokenID,
                    logit: entry.element,
                    decoded: modelTokenizer.decode(tokens: [tokenID], skipSpecialTokens: false)
                )
            }
    }

    private func fingerprint(for logits: [Float]) -> UInt64 {
        var hash: UInt64 = 0xcbf29ce484222325
        for value in logits {
            hash ^= UInt64(value.bitPattern)
            hash &*= 0x100000001b3
        }
        return hash
    }

    private func maximumAbsoluteDifference(_ lhs: [Float], _ rhs: [Float]) -> Float {
        let count = min(lhs.count, rhs.count)
        guard count > 0 else { return 0 }
        var maximum: Float = 0
        for index in 0..<count {
            maximum = max(maximum, abs(lhs[index] - rhs[index]))
        }
        return maximum
    }

    private func differingElementCount(_ lhs: [Float], _ rhs: [Float]) -> Int {
        let count = min(lhs.count, rhs.count)
        guard count > 0 else { return 0 }
        var differenceCount = 0
        for index in 0..<count where lhs[index].bitPattern != rhs[index].bitPattern {
            differenceCount += 1
        }
        return differenceCount
    }

    private func firstDifferingEntries(
        _ lhs: [Float],
        _ rhs: [Float],
        limit: Int = 10
    ) -> [(tokenID: Int, first: Float, second: Float)] {
        let count = min(lhs.count, rhs.count)
        guard count > 0 else { return [] }
        var entries: [(tokenID: Int, first: Float, second: Float)] = []
        entries.reserveCapacity(limit)
        for index in 0..<count where lhs[index].bitPattern != rhs[index].bitPattern {
            entries.append((tokenID: index, first: lhs[index], second: rhs[index]))
            if entries.count == limit {
                break
            }
        }
        return entries
    }

    private var specialTokensToSuppress: Set<Int> {
        var tokenIDs = Set<Int>()
        let specialTokens = ["<pad>", modelTokenizer.bosToken]
        for token in specialTokens {
            guard let token, let tokenID = modelTokenizer.convertTokenToId(token) else { continue }
            tokenIDs.insert(tokenID)
        }
        return tokenIDs.subtracting(modelConfiguration.eosTokenIds)
    }

    private func samplingSeed(for tokenIDs: [Int]) -> UInt64 {
        var hash: UInt64 = 0xcbf29ce484222325
        for tokenID in tokenIDs {
            hash ^= UInt64(bitPattern: Int64(tokenID))
            hash &*= 0x100000001b3
        }
        return hash
    }

    private func readLogits(
        from buffer: MTLBuffer,
        vocabularySize: Int,
        precision: BufferPrecision
    ) -> [Float] {
        switch precision {
        case .float32:
            let pointer = buffer.contents().bindMemory(to: Float.self, capacity: vocabularySize)
            return Array(UnsafeBufferPointer(start: pointer, count: vocabularySize))
        case .float16:
            let pointer = buffer.contents().bindMemory(to: Float16.self, capacity: vocabularySize)
            return (0..<vocabularySize).map { Float(pointer[$0]) }
        case .bfloat16:
            let pointer = buffer.contents().bindMemory(to: UInt16.self, capacity: vocabularySize)
            return (0..<vocabularySize).map { index in
                let bits = UInt32(pointer[index]) << 16
                return Float(bitPattern: bits)
            }
        }
    }

    private func readModelLogits(
        from buffer: MTLBuffer,
        vocabularySize: Int,
        precision: BufferPrecision
    ) -> [Float] {
        var logits = readLogits(
            from: buffer,
            vocabularySize: vocabularySize,
            precision: precision
        )
        applyModelLogitPostprocessing(to: &logits)
        return logits
    }

    private func applyModelLogitPostprocessing(to logits: inout [Float]) {
        guard let cap = finalLogitSoftcapping, cap > 0 else {
            return
        }
        for index in logits.indices {
            logits[index] = tanhf(logits[index] / cap) * cap
        }
    }

    private func applySuppressedTokenMask(_ tokenIDs: Set<Int>, to logits: inout [Float]) {
        guard !tokenIDs.isEmpty else { return }
        for tokenID in tokenIDs where tokenID >= 0 && tokenID < logits.count {
            logits[tokenID] = -Float.greatestFiniteMagnitude
        }
    }

    private func applyRepetitionPenalty(
        to logits: inout [Float],
        recentTokenIDs: [Int],
        penalty: Float?
    ) {
        guard let penalty, penalty > 1 else { return }
        for tokenID in Set(recentTokenIDs) where tokenID >= 0 && tokenID < logits.count {
            let value = logits[tokenID]
            logits[tokenID] = value >= 0 ? value / penalty : value * penalty
        }
    }

    private func applyPresencePenalty(
        to logits: inout [Float],
        recentTokenIDs: [Int],
        penalty: Float?
    ) {
        guard let penalty, penalty > 0 else { return }
        for tokenID in Set(recentTokenIDs) where tokenID >= 0 && tokenID < logits.count {
            logits[tokenID] -= penalty
        }
    }

    private func argmaxTokenID(from logits: [Float]) -> Int {
        var bestIndex = 0
        var bestValue = -Float.greatestFiniteMagnitude
        for (index, value) in logits.enumerated() where value > bestValue {
            bestValue = value
            bestIndex = index
        }
        return bestIndex
    }

    private func collectGeneratedTokenIDs(
        firstToken: Int32,
        samplingState initialSamplingState: GenerationSamplingState,
        ropePositionOffset: Int,
        parameters: GenerateParameters,
        collectionMode: GeneratedTokenCollectionMode
    ) throws -> [Int] {
        var samplingState = initialSamplingState
        let maxRequestedTokens = parameters.maxTokens ?? 1024
        guard maxRequestedTokens > 0, firstToken >= 0 else {
            return []
        }
        guard !self.modelConfiguration.eosTokenIds.contains(Int(firstToken)) else {
            return []
        }

        var rawTokenCount = 0
        let maxRawTokens = collectionMode == .raw
            ? maxRequestedTokens
            : maxRawTokenCount(forVisibleLimit: maxRequestedTokens)
        var rawTokenIDs: [Int] = []

        func appendGeneratedToken(_ tokenID: Int32) {
            rawTokenCount += 1
            rawTokenIDs.append(Int(tokenID))
        }

        appendGeneratedToken(firstToken)
        var nextTokenID = firstToken
        while rawTokenCount < maxRawTokens {
            if case .raw = collectionMode, rawTokenIDs.count >= maxRequestedTokens {
                break
            }
            let argmaxToken = try self.executeDecodeStep(
                tokenID: nextTokenID,
                ropePositionOffset: ropePositionOffset
            )
            let outputToken = self.resolveSampledDecodeToken(
                fallbackToken: argmaxToken,
                parameters: parameters,
                samplingState: &samplingState
            )
            if outputToken < 0 || self.modelConfiguration.eosTokenIds.contains(Int(outputToken)) {
                break
            }
            samplingState.record(Int(outputToken), maxContextSize: parameters.repetitionContextSize)
            appendGeneratedToken(outputToken)
            nextTokenID = outputToken
        }

        switch collectionMode {
        case .raw:
            return rawTokenIDs
        case .visibleOnly:
            var visibilityState = GenerationVisibilityState(policy: thinkingTagPolicy)
            let rawText = self.modelTokenizer.decode(tokens: rawTokenIDs, skipSpecialTokens: false)
            var visibleText = visibilityState.append(decodedText: rawText)
            visibleText += visibilityState.finalize()
            let visibleTokenIDs = self.modelTokenizer.encode(
                text: visibleText,
                addSpecialTokens: false
            )
            if visibleTokenIDs.isEmpty && !visibilityState.didSuppressReasoning {
                return Array(rawTokenIDs.prefix(maxRequestedTokens))
            }
            return Array(visibleTokenIDs.prefix(maxRequestedTokens))
        }
    }

    private func maxRawTokenCount(forVisibleLimit visibleLimit: Int) -> Int {
        guard thinkingTagPolicy != nil else { return visibleLimit }
        return max(visibleLimit * 256, visibleLimit + 1024)
    }

    private static func makeThinkingTagPolicy(
        tokenizer: any Tokenizer,
        chatTemplateSource: String?
    ) -> ThinkingTagPolicy? {
        guard let chatTemplateSource,
              chatTemplateSource.contains("keep_past_thinking") else {
            return nil
        }
        let openTag = "<think>"
        let closeTag = "</think>"
        return ThinkingTagPolicy(
            openTag: openTag,
            closeTag: closeTag,
            openTagTokenID: resolvedTagTokenID(tokenizer: tokenizer, text: openTag),
            closeTagTokenID: resolvedTagTokenID(tokenizer: tokenizer, text: closeTag)
        )
    }

    private static func resolvedTagTokenID(
        tokenizer: any Tokenizer,
        text: String
    ) -> Int? {
        guard let tokenID = tokenizer.convertTokenToId(text) else {
            return nil
        }
        guard tokenizer.decode(tokens: [tokenID], skipSpecialTokens: false) == text else {
            return nil
        }
        return tokenID
    }

    private func sampleTokenID(
        from logits: [Float],
        fallbackTokenID: Int,
        temperature: Float,
        topP: Float,
        topK: Int?,
        minP: Float,
        random: Float
    ) -> Int {
        let clippedTemperature = max(temperature, 0)
        if clippedTemperature == 0 {
            return argmaxTokenID(from: logits)
        }

        let finiteIndices = logits.indices.filter { logits[$0].isFinite }
        guard !finiteIndices.isEmpty else {
            return fallbackTokenID
        }
        let adjustedLogits = logits.map { $0 / clippedTemperature }
        guard let maxLogit = finiteIndices.map({ adjustedLogits[$0] }).max() else {
            return fallbackTokenID
        }

        var weighted: [(tokenID: Int, weight: Float)] = []
        weighted.reserveCapacity(finiteIndices.count)
        var totalWeight: Float = 0
        for tokenID in finiteIndices {
            let weight = expf(adjustedLogits[tokenID] - maxLogit)
            guard weight.isFinite, weight > 0 else { continue }
            weighted.append((tokenID: tokenID, weight: weight))
            totalWeight += weight
        }
        guard !weighted.isEmpty, totalWeight > 0 else {
            return fallbackTokenID
        }

        weighted.sort { lhs, rhs in
            if lhs.weight == rhs.weight {
                return lhs.tokenID < rhs.tokenID
            }
            return lhs.weight > rhs.weight
        }

        let clippedMinP = max(0, minP)
        let minPFiltered: [(tokenID: Int, weight: Float)]
        if clippedMinP > 0, let maxWeight = weighted.first?.weight, maxWeight > 0 {
            let threshold = maxWeight * clippedMinP
            let filtered = weighted.filter { $0.weight >= threshold }
            minPFiltered = filtered.isEmpty ? [weighted[0]] : filtered
        } else {
            minPFiltered = weighted
        }

        let topKFiltered: [(tokenID: Int, weight: Float)]
        if let topK, topK > 0 {
            topKFiltered = Array(minPFiltered.prefix(topK))
        } else {
            topKFiltered = minPFiltered
        }

        let clippedTopP = min(max(topP, 0), 1)
        let selected: [(tokenID: Int, weight: Float)]
        if clippedTopP > 0, clippedTopP < 1 {
            var cumulative: Float = 0
            var prefix: [(tokenID: Int, weight: Float)] = []
            let filteredTotalWeight = topKFiltered.reduce(into: Float(0)) { partial, entry in
                partial += entry.weight
            }
            guard filteredTotalWeight > 0 else {
                return topKFiltered.first?.tokenID ?? fallbackTokenID
            }
            prefix.reserveCapacity(topKFiltered.count)
            for entry in topKFiltered {
                prefix.append(entry)
                cumulative += entry.weight / filteredTotalWeight
                if cumulative >= clippedTopP {
                    break
                }
            }
            selected = prefix
        } else {
            selected = topKFiltered
        }

        let selectedWeight = selected.reduce(into: Float(0)) { partial, entry in
            partial += entry.weight
        }
        guard selectedWeight > 0 else {
            return selected.first?.tokenID ?? fallbackTokenID
        }

        let threshold = min(max(random, 0), 0.999_999_94) * selectedWeight
        var running: Float = 0
        for entry in selected {
            running += entry.weight
            if threshold < running {
                return entry.tokenID
            }
        }
        return selected.last?.tokenID ?? fallbackTokenID
    }

}

enum GeneratedTokenCollectionMode {
    case raw
    case visibleOnly
}

struct ThinkingTagPolicy {
    let openTag: String
    let closeTag: String
    let openTagTokenID: Int?
    let closeTagTokenID: Int?
}

struct GenerationVisibilityState {
    let policy: ThinkingTagPolicy?
    private(set) var suppressingReasoning = false
    private(set) var didSuppressReasoning = false
    private var pendingText = ""

    init(policy: ThinkingTagPolicy?) {
        self.policy = policy
    }

    mutating func append(decodedText: String) -> String {
        guard let policy else { return decodedText }
        pendingText += decodedText
        var visibleOutput = ""

        while !pendingText.isEmpty {
            if suppressingReasoning {
                if let closeRange = pendingText.range(of: policy.closeTag) {
                    didSuppressReasoning = true
                    pendingText.removeSubrange(pendingText.startIndex..<closeRange.upperBound)
                    suppressingReasoning = false
                    continue
                }

                let keepSuffix = pendingSuffixMatchingPrefix(of: policy.closeTag)
                if pendingText.count > keepSuffix.count {
                    didSuppressReasoning = true
                }
                pendingText = keepSuffix
                return visibleOutput
            }

            if let openRange = pendingText.range(of: policy.openTag) {
                visibleOutput += String(pendingText[..<openRange.lowerBound])
                pendingText.removeSubrange(pendingText.startIndex..<openRange.upperBound)
                suppressingReasoning = true
                didSuppressReasoning = true
                continue
            }

            let keepSuffix = pendingSuffixMatchingPrefix(of: policy.openTag)
            if pendingText.count > keepSuffix.count {
                let emitCount = pendingText.count - keepSuffix.count
                let emitEnd = pendingText.index(pendingText.startIndex, offsetBy: emitCount)
                visibleOutput += String(pendingText[..<emitEnd])
                pendingText = keepSuffix
            }
            return visibleOutput
        }

        return visibleOutput
    }

    mutating func finalize() -> String {
        guard policy != nil else {
            let text = pendingText
            pendingText = ""
            return text
        }
        guard !suppressingReasoning else {
            if !pendingText.isEmpty {
                didSuppressReasoning = true
            }
            pendingText = ""
            return ""
        }

        let text = pendingText
        pendingText = ""
        return text
    }

    private func pendingSuffixMatchingPrefix(of pattern: String) -> String {
        guard !pendingText.isEmpty, !pattern.isEmpty else { return "" }
        let candidateCount = min(pendingText.count, pattern.count - 1)
        guard candidateCount > 0 else { return "" }
        for suffixLength in stride(from: candidateCount, through: 1, by: -1) {
            let start = pendingText.index(pendingText.endIndex, offsetBy: -suffixLength)
            let suffix = String(pendingText[start...])
            if pattern.hasPrefix(suffix) {
                return suffix
            }
        }
        return ""
    }
}

private struct GenerationSamplingState {
    let rngState: UInt64
    let recentTokenIDs: [Int]

    func withRecordedFirstToken(_ tokenID: Int32, maxContextSize: Int) -> GenerationSamplingState {
        var copy = self
        copy.record(Int(tokenID), maxContextSize: maxContextSize)
        return copy
    }

    mutating func record(_ tokenID: Int, maxContextSize: Int) {
        let trimmedPrefix = recentTokenIDs.suffix(max(0, maxContextSize - 1))
        var updated = Array(trimmedPrefix)
        updated.append(tokenID)
        self = GenerationSamplingState(rngState: rngState, recentTokenIDs: updated)
    }

    mutating func nextUnitFloat() -> Float {
        var rng = DeterministicRNG(state: rngState)
        let value = rng.nextUnitFloat()
        self = GenerationSamplingState(rngState: rng.state, recentTokenIDs: recentTokenIDs)
        return value
    }
}

private extension GenerateParameters {
    var usesLibraryDefaults: Bool {
        temperature == 0.6
            && topP == 1.0
            && topK == nil
            && minP == 0
            && repetitionPenalty == nil
            && presencePenalty == nil
    }
}

internal struct DebugGenerationBoundaryState: Sendable {
    let firstToken: Int
    let position: Int
    let positionBufferValue: Int
    let ropePositionAxes: [Int]
    let tokenIn: Int
    let tokenOut: Int
    let processedLogitsFingerprint: UInt64
    let topLogits: [(tokenID: Int, logit: Float, decoded: String)]
    let recentTokenIDs: [Int]
}

internal struct DebugGenerationStepTrace: Sendable {
    let stepIndex: Int
    let inputTokenID: Int
    let positionBefore: Int
    let tokenInBefore: Int
    let tokenOutBefore: Int
    let argmaxTokenID: Int
    let sampledTokenID: Int
    let tokenOutAfter: Int
    let rngStateBefore: UInt64
    let randomValue: Float
    let processedLogitsFingerprint: UInt64
    let topLogits: [(tokenID: Int, logit: Float, decoded: String)]
    let recentTokenIDs: [Int]
}

private struct DeterministicRNG {
    var state: UInt64

    mutating func nextUnitFloat() -> Float {
        state = state &* 6364136223846793005 &+ 1
        let value = Double(state >> 11) / Double(1 << 53)
        return Float(value)
    }
}
