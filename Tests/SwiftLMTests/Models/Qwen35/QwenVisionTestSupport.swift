import Foundation
import Jinja
import Metal
import Testing
import Tokenizers
@testable import MetalCompiler
@testable import SwiftLM
@testable import LMArchitecture
@testable import LMIR
@testable import ModelDeclarations

enum QwenVisionTestSupport {
    static func completeConfigJSON(
        modelType: String = "qwen3_vl",
        includeImage: Bool = true,
        includeVideo: Bool = true
    ) -> String {
        """
        {
          "model_type": "\(modelType)",
          \(includeImage ? "\"image_token_id\": 151655," : "")
          \(includeVideo ? "\"video_token_id\": 151656," : "")
          "vision_start_token_id": 151652,
          "vision_end_token_id": 151653,
          "vision_config": {
            "hidden_size": 8,
            "depth": 1,
            "intermediate_size": 16,
            "out_hidden_size": 64,
            "num_heads": 2,
            "num_position_embeddings": 4,
            "in_channels": 3,
            "hidden_act": "gelu",
            "patch_size": 2,
            "temporal_patch_size": 1,
            "spatial_merge_size": 2,
            "deepstack_visual_indexes": [0]
          },
          "text_config": {
            "hidden_size": 1024,
            "num_hidden_layers": 24,
            "intermediate_size": 3584,
            "vocab_size": 248320,
            "num_attention_heads": 8,
            "num_key_value_heads": 2,
            "head_dim": 256,
            "full_attention_interval": 4,
            "conv_kernel_size": 4,
            "linear_num_key_heads": 4,
            "linear_num_value_heads": 16,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "partial_rotary_factor": 0.25
          }
        }
        """
    }

    static func incompleteBackboneConfigJSON(modelType: String = "qwen3_vl") -> String {
        """
        {
          "model_type": "\(modelType)",
          "image_token_id": 151655,
          "vision_config": { "hidden_size": 8 },
          "text_config": {
            "hidden_size": 1024,
            "num_hidden_layers": 24,
            "intermediate_size": 3584,
            "vocab_size": 248320,
            "num_attention_heads": 8,
            "num_key_value_heads": 2,
            "head_dim": 256
          }
        }
        """
    }

    static func preprocessorJSON(
        processorClass: String = "Qwen3VLProcessor",
        imageProcessorType: String? = "Qwen2VLImageProcessorFast",
        videoProcessorType: String? = "Qwen2VLVideoProcessor",
        fps: Double? = 2,
        minFrames: Int? = 4,
        maxFrames: Int? = 768
    ) -> String {
        let imageLine = imageProcessorType.map { "\"image_processor_type\": \"\($0)\"," } ?? ""
        let videoLine = videoProcessorType.map { "\"video_processor_type\": \"\($0)\"," } ?? ""
        let fpsLine = fps.map { "\"fps\": \($0)," } ?? ""
        let minLine = minFrames.map { "\"min_frames\": \($0)," } ?? ""
        let maxLine = maxFrames.map { "\"max_frames\": \($0)," } ?? ""
        return """
        {
          "processor_class": "\(processorClass)",
          \(imageLine)
          \(videoLine)
          "patch_size": 16,
          "temporal_patch_size": 2,
          "merge_size": 2,
          \(fpsLine)
          \(minLine)
          \(maxLine)
          "image_mean": [0.5, 0.5, 0.5],
          "image_std": [0.5, 0.5, 0.5],
          "size": {
            "shortest_edge": 65536,
            "longest_edge": 16777216
          }
        }
        """
    }

    static func parityFixture() throws -> QwenVisionParityFixture {
        guard let url = Bundle.module.url(forResource: "QwenVisionParity", withExtension: "json") else {
            throw CocoaError(.fileNoSuchFile)
        }
        let data = try Data(contentsOf: url)
        return try JSONDecoder().decode(QwenVisionParityFixture.self, from: data)
    }

    static func localTextBundleDirectory() throws -> URL? {
        // Resolve from HF cache only — model bundles live under ~/.cache/huggingface/hub/.
        let cacheCandidates = [
            "~/.cache/huggingface/hub/models--LiquidAI--LFM2.5-1.2B-Thinking",
            "~/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B",
            "~/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct",
        ]
        for candidate in cacheCandidates {
            let snapshots = try snapshotDirectories(basePath: candidate)
            for snapshot in snapshots where try isUsableModelDirectory(snapshot) {
                return snapshot
            }
        }
        return nil
    }

    static func optionalRealQwen3VLDirectory() throws -> URL? {
        let envCandidates = [
            ProcessInfo.processInfo.environment["SWIFTLM_QWEN3_VL_DIR"],
            ProcessInfo.processInfo.environment["SWIFTLM_QWEN35_VL_DIR"],
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
        let qwenVisionDirectories = entries
            .filter {
                let name = $0.lastPathComponent.lowercased()
                guard name.hasPrefix("models--qwen--") else { return false }
                return name.contains("vl") || name.contains("qwen3.5")
            }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
        for entry in qwenVisionDirectories {
            let snapshots = try snapshotDirectories(baseURL: entry)
            for snapshot in snapshots where try isUsableModelDirectory(snapshot) {
                return snapshot
            }
        }
        return nil
    }

    static func optionalQwen35MLXBundleDirectory(name: String) throws -> URL? {
        let envVar = "SWIFTLM_QWEN35_MLX_\(name.replacingOccurrences(of: "-", with: "_").replacingOccurrences(of: ".", with: "_").uppercased())_DIR"
        if let envValue = ProcessInfo.processInfo.environment[envVar] {
            let url = URL(fileURLWithPath: NSString(string: envValue).expandingTildeInPath)
            if try isUsableModelDirectory(url) {
                return url
            }
        }

        let hubCandidate = NSString(
            string: "~/.cache/huggingface/hub/models--mlx-community--\(name)"
        ).expandingTildeInPath
        let hubSnapshots = try snapshotDirectories(
            baseURL: URL(fileURLWithPath: hubCandidate)
        )
        for snapshot in hubSnapshots where try isUsableModelDirectory(snapshot) {
            return snapshot
        }

        let flatCandidate = URL(
            fileURLWithPath: NSString(
                string: "~/Library/Caches/huggingface/models/mlx-community/\(name)"
            ).expandingTildeInPath
        )
        if try isUsableModelDirectory(flatCandidate) {
            return flatCandidate
        }

        return nil
    }

    static func optionalRealQwen35RepoID() -> String? {
        let candidates = [
            ProcessInfo.processInfo.environment["SWIFTLM_QWEN35_VL_REPO"],
            ProcessInfo.processInfo.environment["SWIFTLM_QWEN3_VL_REPO"],
        ].compactMap { value -> String? in
            guard let value else { return nil }
            let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
            return trimmed.isEmpty ? nil : trimmed
        }
        if let configured = candidates.first {
            return configured
        }
#if REAL_QWEN35_NETWORK_E2E
        return "Qwen/Qwen3.5-0.8B-Base"
#else
        return nil
#endif
    }

    static func syntheticMultimodalContainer() async throws -> LanguageModelContext? {
        try await QwenVisionSyntheticContainerCache.shared.container()
    }

    static func realQwen3VLContainer() async throws -> LanguageModelContext? {
        try await QwenVisionRealBundleCache.shared.container()
    }

    static func collectGeneration(
        from stream: AsyncStream<GenerationEvent>
    ) async -> (chunks: [String], completion: CompletionInfo?) {
        var chunks: [String] = []
        var completion: CompletionInfo?
        for await generation in stream {
            if let chunk = generation.text {
                chunks.append(chunk)
            }
            if let info = generation.completion {
                completion = info
            }
        }
        return (chunks, completion)
    }

    static func visionConfiguration(
        outHiddenSize: Int,
        supportsImages: Bool = true,
        supportsVideo: Bool = true,
        videoFramesPerSecond: Double = 2,
        minimumFrameCount: Int = 4,
        maximumFrameCount: Int = 768
    ) -> ModelVisionConfiguration {
        ModelVisionConfiguration(
            hiddenSize: 8,
            depth: 1,
            intermediateSize: 16,
            outHiddenSize: outHiddenSize,
            headCount: 2,
            numPositionEmbeddings: 4,
            inChannels: 3,
            hiddenAct: "gelu",
            deepstackVisualIndexes: [0],
            processorClass: "Qwen3VLProcessor",
            imageTokenID: supportsImages ? QwenVisionTestTokenizer.imageTokenID : nil,
            videoTokenID: supportsVideo ? QwenVisionTestTokenizer.videoTokenID : nil,
            visionStartTokenID: QwenVisionTestTokenizer.visionStartTokenID,
            visionEndTokenID: QwenVisionTestTokenizer.visionEndTokenID,
            imageProcessorType: supportsImages ? "Qwen2VLImageProcessorFast" : nil,
            videoProcessorType: supportsVideo ? "Qwen2VLVideoProcessor" : nil,
            patchSize: 2,
            temporalPatchSize: 1,
            mergeSize: 2,
            spatialMergeSize: 2,
            minimumPixelCount: 4,
            maximumPixelCount: 256,
            videoFramesPerSecond: videoFramesPerSecond,
            minimumFrameCount: minimumFrameCount,
            maximumFrameCount: maximumFrameCount,
            imageMean: [0.5, 0.5, 0.5],
            imageStd: [0.5, 0.5, 0.5]
        )
    }

    static func modelConfiguration(
        outHiddenSize: Int,
        supportsImages: Bool = true,
        supportsVideo: Bool = true,
        supportsExecution: Bool = true
    ) -> ModelConfiguration {
        let vision = visionConfiguration(
            outHiddenSize: outHiddenSize,
            supportsImages: supportsImages,
            supportsVideo: supportsVideo
        )
        return ModelConfiguration(
            name: "Synthetic Qwen3.5+ Vision",
            eosTokenIds: [],
            inputCapabilities: .init(
                supportsText: true,
                supportsImages: supportsImages,
                supportsVideo: supportsVideo
            ),
            executionCapabilities: .init(
                supportsTextGeneration: true,
                supportsPromptStateReuse: true,
                supportsImagePromptPreparation: supportsImages,
                supportsImageExecution: supportsExecution && supportsImages,
                supportsVideoPromptPreparation: supportsVideo,
                supportsVideoExecution: supportsExecution && supportsVideo
            ),
            vision: vision
        )
    }

    static func syntheticVisionRuntime(outHiddenSize: Int) throws -> QwenVisionRuntime {
        let configuration = visionConfiguration(outHiddenSize: outHiddenSize)
        let encoder = try QwenVisionEncoder(
            configuration: configuration,
            weights: syntheticVisionWeights(outHiddenSize: outHiddenSize)
        )
        return QwenVisionRuntime(encoder: encoder)
    }

    static func syntheticVisionWeights(outHiddenSize: Int) -> QwenVisionWeightStore {
        QwenVisionWeightStore(denseTensors: [
            "model.visual.patch_embed.proj.weight": .init(
                values: Array(repeating: 0.01, count: 8 * 12),
                shape: [8, 3, 1, 2, 2]
            ),
            "model.visual.patch_embed.proj.bias": .init(
                values: Array(repeating: 0, count: 8),
                shape: [8]
            ),
            "model.visual.pos_embed.weight": .init(
                values: Array(repeating: 0, count: 4 * 8),
                shape: [4, 8]
            ),
            "model.visual.blocks.0.norm1.weight": .init(values: Array(repeating: 1, count: 8), shape: [8]),
            "model.visual.blocks.0.norm1.bias": .init(values: Array(repeating: 0, count: 8), shape: [8]),
            "model.visual.blocks.0.norm2.weight": .init(values: Array(repeating: 1, count: 8), shape: [8]),
            "model.visual.blocks.0.norm2.bias": .init(values: Array(repeating: 0, count: 8), shape: [8]),
            "model.visual.blocks.0.attn.qkv.weight": .init(
                values: Array(repeating: 0, count: 24 * 8),
                shape: [24, 8]
            ),
            "model.visual.blocks.0.attn.qkv.bias": .init(
                values: Array(repeating: 0, count: 24),
                shape: [24]
            ),
            "model.visual.blocks.0.attn.proj.weight": .init(
                values: Array(repeating: 0, count: 8 * 8),
                shape: [8, 8]
            ),
            "model.visual.blocks.0.attn.proj.bias": .init(values: Array(repeating: 0, count: 8), shape: [8]),
            "model.visual.blocks.0.mlp.linear_fc1.weight": .init(
                values: Array(repeating: 0, count: 16 * 8),
                shape: [16, 8]
            ),
            "model.visual.blocks.0.mlp.linear_fc1.bias": .init(values: Array(repeating: 0, count: 16), shape: [16]),
            "model.visual.blocks.0.mlp.linear_fc2.weight": .init(
                values: Array(repeating: 0, count: 8 * 16),
                shape: [8, 16]
            ),
            "model.visual.blocks.0.mlp.linear_fc2.bias": .init(values: Array(repeating: 0, count: 8), shape: [8]),
            "model.visual.merger.norm.weight": .init(values: Array(repeating: 1, count: 8), shape: [8]),
            "model.visual.merger.norm.bias": .init(values: Array(repeating: 0, count: 8), shape: [8]),
            "model.visual.merger.linear_fc1.weight": .init(
                values: Array(repeating: 0, count: 32 * 32),
                shape: [32, 32]
            ),
            "model.visual.merger.linear_fc1.bias": .init(values: Array(repeating: 0, count: 32), shape: [32]),
            "model.visual.merger.linear_fc2.weight": .init(
                values: Array(repeating: 0, count: outHiddenSize * 32),
                shape: [outHiddenSize, 32]
            ),
            "model.visual.merger.linear_fc2.bias": .init(values: Array(repeating: 0, count: outHiddenSize), shape: [outHiddenSize]),
            "model.visual.deepstack_merger_list.0.norm.weight": .init(
                values: Array(repeating: 1, count: 32),
                shape: [32]
            ),
            "model.visual.deepstack_merger_list.0.norm.bias": .init(values: Array(repeating: 0, count: 32), shape: [32]),
            "model.visual.deepstack_merger_list.0.linear_fc1.weight": .init(
                values: Array(repeating: 0, count: 32 * 32),
                shape: [32, 32]
            ),
            "model.visual.deepstack_merger_list.0.linear_fc1.bias": .init(values: Array(repeating: 0, count: 32), shape: [32]),
            "model.visual.deepstack_merger_list.0.linear_fc2.weight": .init(
                values: Array(repeating: 0, count: outHiddenSize * 32),
                shape: [outHiddenSize, 32]
            ),
            "model.visual.deepstack_merger_list.0.linear_fc2.bias": .init(values: Array(repeating: 0, count: outHiddenSize), shape: [outHiddenSize]),
        ])
    }

    private static func snapshotDirectories(basePath: String) throws -> [URL] {
        try snapshotDirectories(
            baseURL: URL(fileURLWithPath: NSString(string: basePath).expandingTildeInPath)
        )
    }

    private static func snapshotDirectories(baseURL: URL) throws -> [URL] {
        let snapshotsURL = baseURL.appendingPathComponent("snapshots")
        guard FileManager.default.fileExists(atPath: snapshotsURL.path) else {
            return []
        }
        return try FileManager.default.contentsOfDirectory(
            at: snapshotsURL,
            includingPropertiesForKeys: nil
        ).sorted { $0.lastPathComponent < $1.lastPathComponent }
    }

    private static func isUsableModelDirectory(_ directory: URL) throws -> Bool {
        let configPath = directory.appendingPathComponent("config.json")
        let tokenizerPath = directory.appendingPathComponent("tokenizer.json")
        guard FileManager.default.fileExists(atPath: configPath.path),
              FileManager.default.fileExists(atPath: tokenizerPath.path) else {
            return false
        }
        let contents = try FileManager.default.contentsOfDirectory(
            at: directory,
            includingPropertiesForKeys: nil
        )
        return contents.contains { $0.pathExtension == "safetensors" }
    }
}

struct QwenVisionParityFixture: Decodable {
    struct Prompt: Decodable {
        let gridTHW: [Int]
        let placeholderTokenCount: Int
        let pixelValuesShape: [Int]
        let frameTimestampCount: Int?
    }

    struct Layout: Decodable {
        let mropePositionDelta: Int
        let axesByIndex: [String: [Int]]
    }

    let imagePrompt: Prompt
    let videoPrompt: Prompt
    let videoLayout: Layout
}

private actor QwenVisionSyntheticContainerCache {
    static let shared = QwenVisionSyntheticContainerCache()

    func container() async throws -> LanguageModelContext? {
        guard let device = MTLCreateSystemDefaultDevice() else {
            return nil
        }
        let config = ModelConfig(
            hiddenSize: 64,
            layerCount: 2,
            intermediateSize: 128,
            vocabSize: 248320,
            attentionHeads: 4,
            kvHeads: 2,
            headDim: 16,
            attentionBias: false,
            mlpBias: false,
            normEps: 1e-5,
            normKind: .rmsNorm,
            ropeTheta: 10_000,
            ropeDimension: 16,
            ropeScaling: nil,
            tiedEmbeddings: false,
            expertCount: nil,
            expertsPerToken: nil,
            qkNorm: false,
            fullAttentionInterval: nil,
            ssmNumHeads: nil,
            ssmKeyHeadDim: nil,
            ssmValueHeadDim: nil,
            convKernelSize: nil,
            partialRotaryFactor: nil,
            slidingWindow: nil
        )
        let graph = try ModelGraph(Transformer(config: config))
        let resolvedGraph = ParameterResolver().resolve(graph: graph, convention: .llamaFamily)
        let weightStore = try makeSyntheticWeightStore(config: config, device: device)
        let compiler = MetalInferenceCompiler()
        var compiledModel = try compiler.compile(
            graph: resolvedGraph,
            hiddenSize: config.hiddenSize,
            intermediateSize: config.intermediateSize,
            vocabSize: config.vocabSize,
            stafWeightStore: weightStore,
            device: device
        )
        let prefillPlan = try compiler.compilePrefill(
            graph: resolvedGraph,
            hiddenSize: config.hiddenSize,
            intermediateSize: config.intermediateSize,
            vocabSize: config.vocabSize,
            inferencePolicy: InferencePolicy(maximumSequenceLength: 512),
            stafWeightStore: weightStore,
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
        let configuration = QwenVisionTestSupport.modelConfiguration(
            outHiddenSize: config.hiddenSize,
            supportsImages: true,
            supportsVideo: true,
            supportsExecution: true
        )
        let runtime = try QwenVisionTestSupport.syntheticVisionRuntime(
            outHiddenSize: config.hiddenSize
        )
        let container = LanguageModelContext(
            inferenceModel: inferenceModel,
            tokenizer: QwenVisionTestTokenizer(),
            configuration: configuration,
            chatTemplate: try Template(Self.syntheticChatTemplateSource),
            chatTemplateSource: Self.syntheticChatTemplateSource,
            vocabularySize: config.vocabSize,
            visionRuntime: runtime
        )
        return container
    }

    private static let syntheticChatTemplateSource = """
        {%- for message in messages -%}
        {%- for item in message.content -%}
        {%- if item.type == "text" -%}{{ item.text }}{%- endif -%}
        {%- if item.type == "image" -%}<|vision_start|><|image_pad|><|vision_end|>{%- endif -%}
        {%- if item.type == "video" -%}<|vision_start|><|video_pad|><|vision_end|>{%- endif -%}
        {%- endfor -%}
        {%- endfor -%}
        {%- if add_generation_prompt -%}<|assistant|>{%- endif -%}
        """

    private func makeSyntheticWeightStore(config: ModelConfig, device: MTLDevice) throws -> STAFWeightStore {
        let maxElements = max(
            config.vocabSize * config.hiddenSize,
            config.hiddenSize * config.intermediateSize,
            config.hiddenSize * config.hiddenSize
        )
        let payloadSize = max(1, maxElements * MemoryLayout<UInt16>.stride)
        guard let buffer = device.makeBuffer(length: payloadSize, options: .storageModeShared) else {
            throw MetalCompilerError.deviceSetupFailed("Cannot allocate synthetic Qwen vision weight buffer")
        }
        buffer.contents().initializeMemory(as: UInt8.self, repeating: 0, count: payloadSize)

        let kvOutputSize = config.kvHeads * config.headDim
        var tensorShapes: [String: [Int]] = [
            "model.embed_tokens.weight": [config.vocabSize, config.hiddenSize],
            "model.norm.weight": [config.hiddenSize],
            "lm_head.weight": [config.vocabSize, config.hiddenSize]
        ]
        for layerIndex in 0..<config.layerCount {
            let prefix = "model.layers.\(layerIndex)"
            tensorShapes["\(prefix).input_layernorm.weight"] = [config.hiddenSize]
            tensorShapes["\(prefix).self_attn.q_proj.weight"] = [config.hiddenSize, config.hiddenSize]
            tensorShapes["\(prefix).self_attn.k_proj.weight"] = [kvOutputSize, config.hiddenSize]
            tensorShapes["\(prefix).self_attn.v_proj.weight"] = [kvOutputSize, config.hiddenSize]
            tensorShapes["\(prefix).self_attn.o_proj.weight"] = [config.hiddenSize, config.hiddenSize]
            tensorShapes["\(prefix).post_attention_layernorm.weight"] = [config.hiddenSize]
            tensorShapes["\(prefix).mlp.gate_proj.weight"] = [config.intermediateSize, config.hiddenSize]
            tensorShapes["\(prefix).mlp.up_proj.weight"] = [config.intermediateSize, config.hiddenSize]
            tensorShapes["\(prefix).mlp.down_proj.weight"] = [config.hiddenSize, config.intermediateSize]
        }

        var entries: [String: STAFTensorEntry] = [:]
        for (tensorName, shape) in tensorShapes {
            entries[tensorName] = STAFTensorEntry(
                name: tensorName,
                payloadOffset: 0,
                payloadSize: payloadSize,
                schemeIdentifier: .passthrough,
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
}

private actor QwenVisionRealBundleCache {
    static let shared = QwenVisionRealBundleCache()
    private var cachedContainer: LanguageModelContext?

    func container() async throws -> LanguageModelContext? {
        if let cachedContainer {
            cachedContainer.resetState()
            return cachedContainer
        }
        if let repo = QwenVisionTestSupport.optionalRealQwen35RepoID() {
            let loaded = try await ModelBundleLoader().load(repo: repo)
            let context = try LanguageModelContext(loaded)
            cachedContainer = context
            return context
        }
        guard let directory = try QwenVisionTestSupport.optionalRealQwen3VLDirectory() else {
            return nil
        }
        let loaded = try await ModelBundleLoader().load(directory: directory)
        let context = try LanguageModelContext(loaded)
        cachedContainer = context
        return context
    }
}

final class QwenVisionTestTokenizer: @unchecked Sendable, Tokenizer {
    static let visionStartToken = "<|vision_start|>"
    static let visionEndToken = "<|vision_end|>"
    static let imageToken = "<|image_pad|>"
    static let videoToken = "<|video_pad|>"
    static let visionStartTokenID = 151652
    static let visionEndTokenID = 151653
    static let imageTokenID = 151655
    static let videoTokenID = 151656

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
        case Self.visionStartTokenID:
            return Self.visionStartToken
        case Self.visionEndTokenID:
            return Self.visionEndToken
        case Self.imageTokenID:
            return Self.imageToken
        case Self.videoTokenID:
            return Self.videoToken
        default:
            return nil
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
        case Self.visionStartToken:
            return Self.visionStartTokenID
        case Self.visionEndToken:
            return Self.visionEndTokenID
        case Self.imageToken:
            return Self.imageTokenID
        case Self.videoToken:
            return Self.videoTokenID
        default:
            return 1000 + stableHash(token) % 50000
        }
    }

    private func stableHash(_ token: String) -> Int {
        var hash: UInt64 = 1_469_598_103_934_665_603
        for byte in token.utf8 {
            hash ^= UInt64(byte)
            hash &*= 1_099_511_628_211
        }
        return Int(hash % 1_000_000)
    }
}
