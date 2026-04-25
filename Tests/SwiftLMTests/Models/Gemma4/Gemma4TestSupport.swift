import Foundation
import Metal
import Testing
import Tokenizers
@testable import LMIR
@testable import MetalCompiler
@testable import SwiftLM
@testable import LMArchitecture
@testable import ModelDeclarations

enum Gemma4TestSupport {
    static func syntheticGemma4Container() async throws -> LanguageModelContext? {
        try await Gemma4SyntheticContainerCache.shared.container()
    }

    static func realGemma4Container() async throws -> LanguageModelContext? {
        try await Gemma4RealBundleCache.shared.container()
    }

    static func optionalRealGemma4Directory() throws -> URL? {
        let repositoryRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let directCandidates = [
            repositoryRoot.appendingPathComponent("TestData/gemma-4-E2B-it").path,
        ]
        for candidate in directCandidates {
            let directory = URL(fileURLWithPath: candidate)
            if try isUsableModelDirectory(directory) {
                return directory
            }
        }

        let envCandidates = [
            ProcessInfo.processInfo.environment["SWIFTLM_GEMMA4_DIR"],
            ProcessInfo.processInfo.environment["SWIFTLM_GEMMA4_E2B_DIR"],
        ].compactMap { $0 }
        for candidate in envCandidates {
            let url = URL(fileURLWithPath: NSString(string: candidate).expandingTildeInPath)
            if try isUsableModelDirectory(url) {
                return url
            }
        }

        let hubRoot = URL(
            fileURLWithPath: NSString(
                string: "~/.cache/huggingface/hub"
            ).expandingTildeInPath
        )
        guard FileManager.default.fileExists(atPath: hubRoot.path) else {
            return nil
        }
        let entries = try FileManager.default.contentsOfDirectory(
            at: hubRoot,
            includingPropertiesForKeys: nil
        )
        let gemmaDirectories = entries
            .filter {
                let name = $0.lastPathComponent.lowercased()
                return name.hasPrefix("models--google--gemma-4-e2b")
            }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
        for entry in gemmaDirectories {
            let snapshots = try snapshotDirectories(baseURL: entry)
            for snapshot in snapshots where try isUsableModelDirectory(snapshot) {
                return snapshot
            }
        }
        return nil
    }

    static func optionalRealGemma4RepoID() -> String? {
        let candidates = [
            ProcessInfo.processInfo.environment["SWIFTLM_GEMMA4_REPO"],
            ProcessInfo.processInfo.environment["SWIFTLM_GEMMA4_E2B_REPO"],
        ].compactMap { value -> String? in
            guard let value else { return nil }
            let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
            return trimmed.isEmpty ? nil : trimmed
        }
        return candidates.first
    }

    static func modelConfiguration(hiddenSize: Int) -> ModelConfiguration {
        let vision = ModelVisionConfiguration(
            hiddenSize: 32,
            depth: 2,
            intermediateSize: 64,
            outHiddenSize: hiddenSize,
            headCount: 4,
            processorClass: "Gemma4Processor",
            imageTokenID: Gemma4TestTokenizer.imageTokenID,
            videoTokenID: Gemma4TestTokenizer.videoTokenID,
            patchSize: 16,
            poolingKernelSize: 3,
            positionEmbeddingSize: 64,
            defaultOutputLength: 4,
            standardize: false,
            imageMean: [0, 0, 0],
            imageStd: [1, 1, 1]
        )
        return ModelConfiguration(
            name: "gemma4",
            eosTokenIds: [],
            inputCapabilities: .init(supportsText: true, supportsImages: true, supportsVideo: false),
            executionCapabilities: .init(
                supportsTextGeneration: true,
                supportsPromptStateReuse: true,
                supportsImagePromptPreparation: true,
                supportsImageExecution: true,
                supportsVideoPromptPreparation: false,
                supportsVideoExecution: false
            ),
            vision: vision
        )
    }

    static func syntheticConfig() -> ModelConfig {
        ModelConfig(
            hiddenSize: 64,
            layerCount: 2,
            intermediateSize: 128,
            vocabSize: 4096,
            attentionHeads: 4,
            kvHeads: 1,
            headDim: 16,
            attentionBias: false,
            mlpBias: false,
            normEps: 1e-6,
            normKind: .rmsNorm,
            ropeTheta: 10_000.0,
            ropeDimension: 16,
            ropeScaling: nil,
            tiedEmbeddings: true,
            expertCount: nil,
            expertsPerToken: nil,
            qkNorm: true,
            fullAttentionInterval: nil,
            ssmNumHeads: nil,
            ssmKeyHeadDim: nil,
            ssmValueHeadDim: nil,
            convKernelSize: nil,
            partialRotaryFactor: nil,
            slidingWindow: 32,
            layerTypes: ["sliding_attention", "full_attention"],
            hiddenSizePerLayerInput: 8,
            vocabSizePerLayerInput: 4096,
            globalHeadDim: 16,
            globalKVHeads: nil,
            numKVSharedLayers: 1,
            useDoubleWideMLP: false,
            attentionKEqualsV: true,
            fullAttentionRopeTheta: 1_000_000.0,
            fullAttentionPartialRotaryFactor: 0.25,
            fullAttentionRoPEScaling: RoPEScaling(kind: .custom("proportional"), factor: 1.0)
        )
    }

    static func syntheticRuntime(hiddenSize: Int) throws -> Gemma4Runtime {
        let config = syntheticConfig()
        let weightStore = Gemma4WeightStore(denseTensors: syntheticDenseTensors(hiddenSize: hiddenSize))
        let textRuntime = try Gemma4TextRuntime(config: config, weights: weightStore)
        let visionEncoder = try Gemma4VisionEncoder(
            configuration: modelConfiguration(hiddenSize: hiddenSize).vision ?? .init(),
            textHiddenSize: hiddenSize,
            weights: weightStore
        )
        return Gemma4Runtime(
            padTokenID: Gemma4TestTokenizer.padTokenID,
            textRuntime: textRuntime,
            visionEncoder: visionEncoder
        )
    }

    static func syntheticDenseTensors(hiddenSize: Int) -> [String: Gemma4WeightStore.DenseTensor] {
        let vocabSize = 4096
        let layerCount = 2
        let perLayerSize = 8
        let visionHidden = 32
        let visionIntermediate = 64
        let patchDimension = 3 * 16 * 16
        let positionEmbeddingSize = 64

        var tensors: [String: Gemma4WeightStore.DenseTensor] = [
            "model.language_model.embed_tokens.weight": .init(
                values: repeatingRamp(count: vocabSize * hiddenSize, scale: 0.0001),
                shape: [vocabSize, hiddenSize]
            ),
            "model.language_model.embed_tokens_per_layer.weight": .init(
                values: repeatingRamp(count: vocabSize * layerCount * perLayerSize, scale: 0.0002),
                shape: [vocabSize, layerCount * perLayerSize]
            ),
            "model.language_model.per_layer_model_projection.weight": .init(
                values: repeatingRamp(count: layerCount * perLayerSize * hiddenSize, scale: 0.0003),
                shape: [layerCount * perLayerSize, hiddenSize]
            ),
            "model.language_model.per_layer_projection_norm.weight": .init(
                values: Array(repeating: 1, count: perLayerSize),
                shape: [perLayerSize]
            ),
            "model.vision_tower.patch_embedder.input_proj.weight": .init(
                values: repeatingRamp(count: visionHidden * patchDimension, scale: 0.0004),
                shape: [visionHidden, patchDimension]
            ),
            "model.vision_tower.patch_embedder.position_embedding_table": .init(
                values: repeatingRamp(count: 2 * positionEmbeddingSize * visionHidden, scale: 0.0001),
                shape: [2, positionEmbeddingSize, visionHidden]
            ),
            "model.embed_vision.embedding_projection.weight": .init(
                values: repeatingRamp(count: hiddenSize * visionHidden, scale: 0.0002),
                shape: [hiddenSize, visionHidden]
            ),
        ]

        for layerIndex in 0..<layerCount {
            let prefix = "model.language_model.layers.\(layerIndex)"
            tensors["\(prefix).input_layernorm.weight"] = .init(
                values: Array(repeating: 1, count: hiddenSize),
                shape: [hiddenSize]
            )
            tensors["\(prefix).post_attention_layernorm.weight"] = .init(
                values: Array(repeating: 1, count: hiddenSize),
                shape: [hiddenSize]
            )
            tensors["\(prefix).pre_feedforward_layernorm.weight"] = .init(
                values: Array(repeating: 1, count: hiddenSize),
                shape: [hiddenSize]
            )
            tensors["\(prefix).post_feedforward_layernorm.weight"] = .init(
                values: Array(repeating: 1, count: hiddenSize),
                shape: [hiddenSize]
            )
            tensors["\(prefix).self_attn.q_norm.weight"] = .init(
                values: Array(repeating: 1, count: 16),
                shape: [16]
            )
            tensors["\(prefix).self_attn.k_norm.weight"] = .init(
                values: Array(repeating: 1, count: 16),
                shape: [16]
            )
            tensors["\(prefix).self_attn.v_norm.weight"] = .init(
                values: Array(repeating: Float(0), count: 16),
                shape: [16]
            )
            tensors["\(prefix).self_attn.q_proj.weight"] = .init(
                values: repeatingRamp(count: hiddenSize * hiddenSize, scale: 0.0002),
                shape: [hiddenSize, hiddenSize]
            )
            tensors["\(prefix).self_attn.k_proj.weight"] = .init(
                values: repeatingRamp(count: 16 * hiddenSize, scale: 0.0002),
                shape: [16, hiddenSize]
            )
            tensors["\(prefix).self_attn.v_proj.weight"] = .init(
                values: repeatingRamp(count: 16 * hiddenSize, scale: 0.00015),
                shape: [16, hiddenSize]
            )
            tensors["\(prefix).self_attn.o_proj.weight"] = .init(
                values: repeatingRamp(count: hiddenSize * hiddenSize, scale: 0.00018),
                shape: [hiddenSize, hiddenSize]
            )
            tensors["\(prefix).mlp.gate_proj.weight"] = .init(
                values: repeatingRamp(count: 128 * hiddenSize, scale: 0.00017),
                shape: [128, hiddenSize]
            )
            tensors["\(prefix).mlp.up_proj.weight"] = .init(
                values: repeatingRamp(count: 128 * hiddenSize, scale: 0.00013),
                shape: [128, hiddenSize]
            )
            tensors["\(prefix).mlp.down_proj.weight"] = .init(
                values: repeatingRamp(count: hiddenSize * 128, scale: 0.00019),
                shape: [hiddenSize, 128]
            )
            tensors["\(prefix).per_layer_input_gate.weight"] = .init(
                values: repeatingRamp(count: perLayerSize * hiddenSize, scale: 0.00011),
                shape: [perLayerSize, hiddenSize]
            )
            tensors["\(prefix).per_layer_projection.weight"] = .init(
                values: repeatingRamp(count: hiddenSize * perLayerSize, scale: 0.00009),
                shape: [hiddenSize, perLayerSize]
            )
            tensors["\(prefix).post_per_layer_input_norm.weight"] = .init(
                values: Array(repeating: 1, count: hiddenSize),
                shape: [hiddenSize]
            )
            tensors["\(prefix).layer_scalar"] = .init(
                values: [1],
                shape: [1]
            )

            let visionPrefix = "model.vision_tower.encoder.layers.\(layerIndex)"
            tensors["\(visionPrefix).input_layernorm.weight"] = .init(
                values: Array(repeating: 1, count: visionHidden),
                shape: [visionHidden]
            )
            tensors["\(visionPrefix).post_attention_layernorm.weight"] = .init(
                values: Array(repeating: 1, count: visionHidden),
                shape: [visionHidden]
            )
            tensors["\(visionPrefix).pre_feedforward_layernorm.weight"] = .init(
                values: Array(repeating: 1, count: visionHidden),
                shape: [visionHidden]
            )
            tensors["\(visionPrefix).post_feedforward_layernorm.weight"] = .init(
                values: Array(repeating: 1, count: visionHidden),
                shape: [visionHidden]
            )
            tensors["\(visionPrefix).self_attn.q_norm.weight"] = .init(
                values: Array(repeating: 1, count: visionHidden / 4),
                shape: [visionHidden / 4]
            )
            tensors["\(visionPrefix).self_attn.k_norm.weight"] = .init(
                values: Array(repeating: 1, count: visionHidden / 4),
                shape: [visionHidden / 4]
            )
            for name in [
                "self_attn.q_proj.linear.weight",
                "self_attn.k_proj.linear.weight",
                "self_attn.v_proj.linear.weight",
                "self_attn.o_proj.linear.weight",
            ] {
                tensors["\(visionPrefix).\(name)"] = .init(
                    values: repeatingRamp(count: visionHidden * visionHidden, scale: 0.0002),
                    shape: [visionHidden, visionHidden]
                )
            }
            tensors["\(visionPrefix).mlp.gate_proj.linear.weight"] = .init(
                values: repeatingRamp(count: visionIntermediate * visionHidden, scale: 0.0002),
                shape: [visionIntermediate, visionHidden]
            )
            tensors["\(visionPrefix).mlp.up_proj.linear.weight"] = .init(
                values: repeatingRamp(count: visionIntermediate * visionHidden, scale: 0.0001),
                shape: [visionIntermediate, visionHidden]
            )
            tensors["\(visionPrefix).mlp.down_proj.linear.weight"] = .init(
                values: repeatingRamp(count: visionHidden * visionIntermediate, scale: 0.0002),
                shape: [visionHidden, visionIntermediate]
            )
        }

        tensors["model.language_model.norm.weight"] = .init(
            values: Array(repeating: 1, count: hiddenSize),
            shape: [hiddenSize]
        )

        return tensors
    }

    private static func repeatingRamp(count: Int, scale: Float) -> [Float] {
        (0..<count).map { index in
            Float((index % 17) + 1) * scale
        }
    }

    static func syntheticSTAFWeightStore(
        graph: ModelGraph,
        hiddenSize: Int,
        device: MTLDevice
    ) throws -> STAFWeightStore {
        let denseTensors = syntheticDenseTensors(hiddenSize: hiddenSize)
        let maxElements = max(1, denseTensors.values.map { $0.shape.reduce(1, *) }.max() ?? 1)
        let payloadSize = maxElements * MemoryLayout<UInt16>.stride
        guard let buffer = device.makeBuffer(length: payloadSize, options: .storageModeShared) else {
            throw MetalCompilerError.deviceSetupFailed("Cannot allocate synthetic Gemma4 weight buffer")
        }
        memset(buffer.contents(), 0, buffer.length)

        var entries: [String: STAFTensorEntry] = [:]
        let allTensorNames = tensorNames(in: graph.rootRegion).union(denseTensors.keys)
        for tensorName in allTensorNames {
            let shape = denseTensors[tensorName]?.shape ?? [maxElements]
            entries[tensorName] = STAFTensorEntry(
                name: tensorName,
                payloadOffset: 0,
                payloadSize: payloadSize,
                schemeIdentifier: .fp16RowMajor,
                semanticRole: .unknown,
                shape: shape,
                blockSize: 0,
                groupSize: 0,
                bufferOffset: 0
            )
        }

        return STAFWeightStore(
            buffer: buffer,
            entries: entries,
            metadata: .empty,
            specializedBufferAccesses: [:]
        )
    }

    private static func tensorNames(in region: Region) -> Set<String> {
        var names = Set(region.operations.flatMap { operation in
            operation.parameterBindings.map(\.tensorName)
        })
        for operation in region.operations {
            switch operation.kind {
            case .primitive:
                break
            case .residual(_, let body):
                names.formUnion(tensorNames(in: body))
            case .parallel(_, let branches):
                for branch in branches {
                    names.formUnion(tensorNames(in: branch))
                }
            case .repeating(_, let body):
                names.formUnion(tensorNames(in: body))
            case .conditional(_, let thenRegion, let elseRegion):
                names.formUnion(tensorNames(in: thenRegion))
                names.formUnion(tensorNames(in: elseRegion))
            }
        }
        return names
    }

    private static func snapshotDirectories(baseURL: URL) throws -> [URL] {
        let snapshots = baseURL.appendingPathComponent("snapshots")
        guard FileManager.default.fileExists(atPath: snapshots.path) else {
            return []
        }
        return try FileManager.default.contentsOfDirectory(
            at: snapshots,
            includingPropertiesForKeys: nil
        )
    }

    private static func isUsableModelDirectory(_ url: URL) throws -> Bool {
        let requiredFiles = [
            "config.json",
            "tokenizer.json",
            "model.safetensors",
        ]
        return requiredFiles.allSatisfy { name in
            FileManager.default.fileExists(atPath: url.appendingPathComponent(name).path)
        }
    }
}

private actor Gemma4SyntheticContainerCache {
    static let shared = Gemma4SyntheticContainerCache()

    func container() async throws -> LanguageModelContext? {
        guard let device = MTLCreateSystemDefaultDevice() else {
            return nil
        }

        let config = Gemma4TestSupport.syntheticConfig()
        let graph = try ModelGraph(Gemma4(config: config))
        let resolvedGraph = ParameterResolver().resolve(graph: graph, convention: .gemma4Family)
        let stafWeightStore = try Gemma4TestSupport.syntheticSTAFWeightStore(
            graph: resolvedGraph,
            hiddenSize: config.hiddenSize,
            device: device
        )
        let compiler = MetalInferenceCompiler()
        var compiledModel = try compiler.compile(
            graph: resolvedGraph,
            hiddenSize: config.hiddenSize,
            intermediateSize: config.intermediateSize,
            vocabSize: config.vocabSize,
            stafWeightStore: stafWeightStore,
            device: device
        )
        let prefillPlan = try compiler.compilePrefill(
            graph: resolvedGraph,
            hiddenSize: config.hiddenSize,
            intermediateSize: config.intermediateSize,
            vocabSize: config.vocabSize,
            inferencePolicy: InferencePolicy(maximumSequenceLength: 256),
            stafWeightStore: stafWeightStore,
            sharedKVCache: compiledModel.buffers.kvCache,
            sharedConvState: compiledModel.buffers.convState,
            sharedConvStateDimension: compiledModel.buffers.convStateDimension,
            sharedConvStateKernelSize: compiledModel.buffers.convStateKernelSize,
            sharedRecurrentState: compiledModel.buffers.recurrentState,
            sharedRecurrentStateBytesPerLayer: compiledModel.buffers.recurrentStateBytesPerLayer,
            device: device
        )
        compiledModel = compiledModel.withPrefillPlan(prefillPlan)
        let inferenceModel = try MetalInferenceModel(plan: compiledModel, device: device)
        let hiddenSize = config.hiddenSize
        let chatTemplateSource = Gemma4DefaultChatTemplate.synthesizedSource()
        let chatTemplate = try Gemma4DefaultChatTemplate.template()
        let container = LanguageModelContext(
            inferenceModel: inferenceModel,
            tokenizer: Gemma4TestTokenizer(),
            configuration: Gemma4TestSupport.modelConfiguration(hiddenSize: hiddenSize),
            chatTemplate: chatTemplate,
            chatTemplateSource: chatTemplateSource,
            vocabularySize: config.vocabSize,
            gemma4Runtime: try Gemma4TestSupport.syntheticRuntime(hiddenSize: hiddenSize)
        )
        return container
    }
}

private actor Gemma4RealBundleCache {
    static let shared = Gemma4RealBundleCache()
    private var cachedContainer: LanguageModelContext?

    func container() async throws -> LanguageModelContext? {
        if let cachedContainer {
            cachedContainer.resetState()
            return cachedContainer
        }
        if let repo = Gemma4TestSupport.optionalRealGemma4RepoID() {
            let loaded = try await ModelBundleLoader().load(repo: repo)
            let context = try LanguageModelContext(loaded)
            cachedContainer = context
            return context
        }
        guard let directory = try Gemma4TestSupport.optionalRealGemma4Directory() else {
            return nil
        }
        let loaded = try await ModelBundleLoader().load(directory: directory)
        let context = try LanguageModelContext(loaded)
        cachedContainer = context
        return context
    }
}

final class Gemma4TestTokenizer: @unchecked Sendable, Tokenizer {
    static let padToken = "<pad>"
    static let imageToken = "<|image|>"
    static let boiToken = "<|image>"
    static let eoiToken = "<image|>"
    static let videoToken = "<|video|>"
    static let padTokenID = 0
    static let boiTokenID = 1
    static let imageTokenID = 2
    static let eoiTokenID = 3
    static let videoTokenID = 4

    var bosToken: String? { nil }
    var bosTokenId: Int? { nil }
    var eosToken: String? { nil }
    var eosTokenId: Int? { nil }
    var unknownToken: String? { nil }
    var unknownTokenId: Int? { nil }
    var hasChatTemplate: Bool { false }

    func tokenize(text: String) -> [String] {
        var tokens: [String] = []
        var index = text.startIndex
        while index < text.endIndex {
            if text[index].isWhitespace {
                index = text.index(after: index)
                continue
            }
            if text[index] == "<" {
                var end = text.index(after: index)
                while end < text.endIndex && text[end] != ">" {
                    end = text.index(after: end)
                }
                if end < text.endIndex {
                    tokens.append(String(text[index...end]))
                    index = text.index(after: end)
                    continue
                }
            }
            var end = index
            while end < text.endIndex && !text[end].isWhitespace && text[end] != "<" {
                end = text.index(after: end)
            }
            tokens.append(String(text[index..<end]))
            index = end
        }
        return tokens
    }

    func encode(text: String) -> [Int] {
        encode(text: text, addSpecialTokens: true)
    }

    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        tokenize(text: text).map { tokenID(for: $0) }
    }

    func callAsFunction(_ text: String, addSpecialTokens: Bool) -> [Int] {
        encode(text: text, addSpecialTokens: addSpecialTokens)
    }

    func decode(tokens: [Int]) -> String {
        decode(tokens: tokens, skipSpecialTokens: false)
    }

    func decode(tokens: [Int], skipSpecialTokens: Bool) -> String {
        tokens.map { convertIdToToken($0) ?? "<tok:\($0)>" }.joined(separator: " ")
    }

    func convertTokenToId(_ token: String) -> Int? {
        tokenID(for: token)
    }

    func convertTokensToIds(_ tokens: [String]) -> [Int?] {
        tokens.map(convertTokenToId)
    }

    func convertIdToToken(_ id: Int) -> String? {
        switch id {
        case Self.padTokenID:
            return Self.padToken
        case Self.imageTokenID:
            return Self.imageToken
        case Self.boiTokenID:
            return Self.boiToken
        case Self.eoiTokenID:
            return Self.eoiToken
        case Self.videoTokenID:
            return Self.videoToken
        default:
            return "tok\(id)"
        }
    }

    func convertIdsToTokens(_ ids: [Int]) -> [String?] {
        ids.map(convertIdToToken)
    }

    func applyChatTemplate(messages: [Message]) throws -> [Int] {
        throw CocoaError(.featureUnsupported)
    }

    func applyChatTemplate(messages: [Message], tools: [ToolSpec]?) throws -> [Int] {
        throw CocoaError(.featureUnsupported)
    }

    func applyChatTemplate(
        messages: [Message],
        tools: [ToolSpec]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        throw CocoaError(.featureUnsupported)
    }

    func applyChatTemplate(messages: [Message], chatTemplate: ChatTemplateArgument) throws -> [Int] {
        throw CocoaError(.featureUnsupported)
    }

    func applyChatTemplate(messages: [Message], chatTemplate: String) throws -> [Int] {
        throw CocoaError(.featureUnsupported)
    }

    func applyChatTemplate(
        messages: [Message],
        chatTemplate: ChatTemplateArgument?,
        addGenerationPrompt: Bool,
        truncation: Bool,
        maxLength: Int?,
        tools: [ToolSpec]?
    ) throws -> [Int] {
        throw CocoaError(.featureUnsupported)
    }

    func applyChatTemplate(
        messages: [Message],
        chatTemplate: ChatTemplateArgument?,
        addGenerationPrompt: Bool,
        truncation: Bool,
        maxLength: Int?,
        tools: [ToolSpec]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        throw CocoaError(.featureUnsupported)
    }

    private func tokenID(for token: String) -> Int {
        switch token {
        case Self.padToken:
            return Self.padTokenID
        case Self.imageToken:
            return Self.imageTokenID
        case Self.boiToken:
            return Self.boiTokenID
        case Self.eoiToken:
            return Self.eoiTokenID
        case Self.videoToken:
            return Self.videoTokenID
        default:
            let scalarSum = token.unicodeScalars.reduce(0) { $0 + Int($1.value) }
            return 32 + (scalarSum % 2048)
        }
    }
}
