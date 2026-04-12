import Foundation
import Metal
import Testing
@testable import MetalCompiler
@testable import SwiftLM
@testable import LMArchitecture

@Suite("Text Embedding Runtime", .tags(.unit))
struct TextEmbeddingRuntimeTests {
    @Test("Text embedding snapshot patterns include nested module configs")
    func textEmbeddingSnapshotPatternsIncludeNestedModuleConfigs() {
        #expect(ModelBundleLoader.textEmbeddingSnapshotPatterns.contains("modules.json"))
        #expect(
            ModelBundleLoader.textEmbeddingSnapshotPatterns.contains("config_sentence_transformers.json")
        )
        #expect(ModelBundleLoader.textEmbeddingSnapshotPatterns.contains("**/config.json"))
    }

    @Test("HFConfigDecoder parses Gemma3Text embedding attention metadata")
    func parseGemma3TextEmbeddingConfig() throws {
        let configJSON = """
        {
          "model_type": "gemma3_text",
          "hidden_size": 768,
          "num_hidden_layers": 24,
          "intermediate_size": 1152,
          "vocab_size": 262144,
          "num_attention_heads": 3,
          "num_key_value_heads": 1,
          "head_dim": 256,
          "rms_norm_eps": 1.0e-6,
          "tie_word_embeddings": true,
          "attention_bias": false,
          "mlp_bias": false,
          "sliding_window": 512,
          "use_bidirectional_attention": true,
          "query_pre_attn_scalar": 256,
          "rope_local_base_freq": 10000.0,
          "rope_theta": 1000000.0,
          "layer_types": [
            "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention",
            "sliding_attention", "full_attention",
            "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention",
            "sliding_attention", "full_attention",
            "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention",
            "sliding_attention", "full_attention",
            "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention",
            "sliding_attention", "full_attention"
          ],
          "rope_parameters": {
            "sliding_attention": {
              "rope_theta": 10000.0
            },
            "full_attention": {
              "rope_theta": 1000000.0
            }
          }
        }
        """

        let data = try #require(configJSON.data(using: .utf8))
        let decoder = HFConfigDecoder()
        let config = try decoder.decode(from: data)
        let modelType = try decoder.modelType(from: data)

        #expect(modelType == "gemma3_text")
        #expect(config.useBidirectionalAttention)
        #expect(config.queryPreAttentionScalar != nil)
        #expect(abs((config.queryPreAttentionScalar ?? 0) - 256) < 0.0001)
        #expect(config.localAttentionRopeTheta != nil)
        #expect(abs((config.localAttentionRopeTheta ?? 0) - 10_000.0) < 0.0001)
        #expect(config.fullAttentionRopeTheta != nil)
        #expect(abs((config.fullAttentionRopeTheta ?? 0) - 1_000_000.0) < 0.0001)
        #expect(config.layerTypes?.count == 24)
    }

    @Test("SentenceTransformerMetadata loads prompts and module pipeline")
    func loadSentenceTransformerMetadata() throws {
        let directory = try makeTemporaryDirectory()
        defer {
            do {
                try FileManager.default.removeItem(at: directory)
            } catch {
                Issue.record("Failed to remove temporary directory: \(error)")
            }
        }

        try createDirectory(directory.appendingPathComponent("1_Pooling"))
        try createDirectory(directory.appendingPathComponent("2_Dense"))
        try createDirectory(directory.appendingPathComponent("3_Dense"))

        try writeJSON(
            """
            {
              "0": {
                "idx": 0,
                "name": "0",
                "path": "",
                "type": "sentence_transformers.models.Transformer"
              },
              "1": {
                "idx": 1,
                "name": "1",
                "path": "1_Pooling",
                "type": "sentence_transformers.models.Pooling"
              },
              "2": {
                "idx": 2,
                "name": "2",
                "path": "2_Dense",
                "type": "sentence_transformers.models.Dense"
              },
              "3": {
                "idx": 3,
                "name": "3",
                "path": "3_Dense",
                "type": "sentence_transformers.models.Dense"
              },
              "4": {
                "idx": 4,
                "name": "4",
                "path": "4_Normalize",
                "type": "sentence_transformers.models.Normalize"
              }
            }
            """,
            to: directory.appendingPathComponent("modules.json")
        )
        try writeJSON(
            """
            {
              "prompts": {
                "query": "task: search result | query: ",
                "document": "title: none | text: "
              },
              "default_prompt_name": "query",
              "similarity_fn_name": "cosine"
            }
            """,
            to: directory.appendingPathComponent("config_sentence_transformers.json")
        )
        try writeJSON(
            """
            {
              "pooling_mode_mean_tokens": true,
              "pooling_mode_cls_token": false,
              "pooling_mode_max_tokens": false,
              "pooling_mode_lasttoken": false,
              "include_prompt": true
            }
            """,
            to: directory.appendingPathComponent("1_Pooling/config.json")
        )
        try writeJSON(
            """
            {
              "in_features": 768,
              "out_features": 3072,
              "bias": false,
              "activation_function": "torch.nn.modules.linear.Identity"
            }
            """,
            to: directory.appendingPathComponent("2_Dense/config.json")
        )
        try writeJSON(
            """
            {
              "in_features": 3072,
              "out_features": 768,
              "bias": false,
              "activation_function": "torch.nn.Identity"
            }
            """,
            to: directory.appendingPathComponent("3_Dense/config.json")
        )

        let resources = ModelBundleResources(
            directory: directory,
            configData: Data(),
            config: dummyTextConfig(),
            modelType: "gemma3_text",
            safetensorsURLs: [],
            modulesData: try Data(contentsOf: directory.appendingPathComponent("modules.json")),
            sentenceTransformersConfigData: try Data(
                contentsOf: directory.appendingPathComponent("config_sentence_transformers.json")
            ),
            chatTemplate: nil,
            chatTemplateSource: nil,
            preprocessorConfigData: nil,
            inputCapabilities: .textOnly,
            visionConfiguration: nil
        )

        let metadata = try SentenceTransformerMetadata.load(from: resources)

        #expect(metadata.defaultPromptName == "query")
        #expect(metadata.availablePromptNames == ["document", "query"])
        #expect(metadata.pooling.strategy == .mean)
        #expect(metadata.pooling.includePrompt)
        #expect(metadata.denseLayers.count == 2)
        #expect(metadata.postprocessors == [.l2Normalize])
    }

    @Test("SentenceTransformerMetadata falls back to model-scoped defaults when module configs are absent")
    func loadSentenceTransformerMetadataWithModelScopedFallbacks() throws {
        let directory = try makeTemporaryDirectory()
        defer {
            do {
                try FileManager.default.removeItem(at: directory)
            } catch {
                Issue.record("Failed to remove temporary directory: \(error)")
            }
        }

        try writeJSON(
            """
            [
              {
                "idx": 0,
                "name": "0",
                "path": "",
                "type": "sentence_transformers.models.Transformer"
              },
              {
                "idx": 1,
                "name": "1",
                "path": "1_Pooling",
                "type": "sentence_transformers.models.Pooling"
              },
              {
                "idx": 2,
                "name": "2",
                "path": "2_Dense",
                "type": "sentence_transformers.models.Dense"
              },
              {
                "idx": 3,
                "name": "3",
                "path": "3_Dense",
                "type": "sentence_transformers.models.Dense"
              },
              {
                "idx": 4,
                "name": "4",
                "path": "4_Normalize",
                "type": "sentence_transformers.models.Normalize"
              }
            ]
            """,
            to: directory.appendingPathComponent("modules.json")
        )
        try writeJSON(
            """
            {
              "prompts": {
                "query": "task: search result | query: ",
                "document": "title: none | text: "
              }
            }
            """,
            to: directory.appendingPathComponent("config_sentence_transformers.json")
        )

        let resources = ModelBundleResources(
            directory: directory,
            configData: Data(),
            config: dummyTextConfig(),
            modelType: "gemma3_text",
            safetensorsURLs: [],
            modulesData: try Data(contentsOf: directory.appendingPathComponent("modules.json")),
            sentenceTransformersConfigData: try Data(
                contentsOf: directory.appendingPathComponent("config_sentence_transformers.json")
            ),
            chatTemplate: nil,
            chatTemplateSource: nil,
            preprocessorConfigData: nil,
            inputCapabilities: .textOnly,
            visionConfiguration: nil
        )

        let metadata = try SentenceTransformerMetadata.load(from: resources)

        #expect(metadata.pooling.strategy == .mean)
        #expect(metadata.pooling.includePrompt)
        #expect(metadata.denseLayers.count == 2)
        #expect(metadata.denseLayers.allSatisfy { $0.activation == .identity })
        #expect(metadata.postprocessors == [.l2Normalize])
    }

    @Test("SentenceTransformer metadata omits postprocessors when Normalize module is absent")
    func sentenceTransformerMetadataWithoutNormalizeHasNoPostprocessors() throws {
        let metadata = SentenceTransformerMetadata(
            prompts: [:],
            defaultPromptName: nil,
            similarityFunctionName: nil,
            pooling: .init(strategy: .mean, includePrompt: true),
            denseLayers: [],
            postprocessors: []
        )

        #expect(metadata.postprocessors.isEmpty)
        #expect(metadata.pooling.includePrompt)
    }

    @Test("TextEmbeddingInput keeps text and prompt selection together")
    func textEmbeddingInputCapturesRequestValue() {
        let implicit = TextEmbeddingInput("swift metal", promptName: "query")
        #expect(implicit.text == "swift metal")
        #expect(implicit.promptName == "query")

        let explicit = TextEmbeddingInput(text: "embedding", promptName: nil)
        #expect(explicit.text == "embedding")
        #expect(explicit.promptName == nil)
    }

    @Test("SentenceTransformer runtime applies pooling then dense then postprocessors")
    func sentenceTransformerRuntimePipeline() throws {
        let metadata = SentenceTransformerMetadata(
            prompts: ["query": "task: search | query: "],
            defaultPromptName: nil,
            similarityFunctionName: "cosine",
            pooling: .init(strategy: .mean, includePrompt: false),
            denseLayers: [
                .init(
                    weightName: "dense.0.weight",
                    biasName: "dense.0.bias",
                    inputDimension: 2,
                    outputDimension: 2,
                    activation: .identity
                )
            ],
            postprocessors: [.l2Normalize]
        )
        let weightStore = CPUWeightStore(denseTensors: [
            "dense.0.weight": .init(
                values: [
                    2, 0,
                    0, 1,
                ],
                shape: [2, 2]
            )
        ])
        let runtime = try SentenceTransformerTextEmbeddingRuntime(
            metadata: metadata,
            weightStore: weightStore
        )
        let tokenizer = QwenVisionTestTokenizer()
        let prepared = try runtime.prepare(
            text: "swift metal",
            promptName: "query",
            tokenizer: tokenizer
        )

        let hiddenStates = Array(repeating: [Float](repeating: 0, count: 2), count: prepared.promptTokenCount)
            + [[1, 2], [3, 4]]
        let embedding = try runtime.embed(
            hiddenStates: hiddenStates,
            promptTokenCount: prepared.promptTokenCount
        )

        #expect(prepared.renderedText == "task: search | query: swift metal")
        #expect(embedding.count == 2)
        #expect(abs(embedding[0] - 0.8) < 0.0001)
        #expect(abs(embedding[1] - 0.6) < 0.0001)
        #expect(prepared.promptTokenCount > 0)
    }

    @Test("SentenceTransformer runtime only normalizes when postprocessors request it")
    func sentenceTransformerRuntimeSkipsNormalizeWhenPostprocessorsAreEmpty() throws {
        let metadata = SentenceTransformerMetadata(
            prompts: ["query": "task: search | query: "],
            defaultPromptName: nil,
            similarityFunctionName: nil,
            pooling: .init(strategy: .mean, includePrompt: false),
            denseLayers: [
                .init(
                    weightName: "dense.0.weight",
                    biasName: "dense.0.bias",
                    inputDimension: 2,
                    outputDimension: 2,
                    activation: .identity
                )
            ],
            postprocessors: []
        )
        let weightStore = CPUWeightStore(denseTensors: [
            "dense.0.weight": .init(
                values: [
                    2, 0,
                    0, 1,
                ],
                shape: [2, 2]
            )
        ])
        let runtime = try SentenceTransformerTextEmbeddingRuntime(
            metadata: metadata,
            weightStore: weightStore
        )
        let tokenizer = QwenVisionTestTokenizer()
        let prepared = try runtime.prepare(
            text: "swift metal",
            promptName: "query",
            tokenizer: tokenizer
        )
        let hiddenStates = Array(
            repeating: [Float](repeating: 0, count: 2),
            count: prepared.promptTokenCount
        ) + [[1, 2], [3, 4]]

        let embedding = try runtime.embed(
            hiddenStates: hiddenStates,
            promptTokenCount: prepared.promptTokenCount
        )

        #expect(abs(embedding[0] - 4) < 0.0001)
        #expect(abs(embedding[1] - 3) < 0.0001)
        let norm = (embedding[0] * embedding[0] + embedding[1] * embedding[1]).squareRoot()
        #expect(abs(norm - 5) < 0.0001)
    }

    @Test("CPUWeightStore dequantizes q4 group64 tensors to logical shape")
    func cpuWeightStoreDequantizesQ4Group64() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device")
            return
        }

        let blockSize = AffineQ4Group64Format().bytesPerBlock
        let buffer = try #require(device.makeBuffer(length: blockSize, options: .storageModeShared))
        let basePointer = buffer.contents()

        basePointer.storeBytes(
            of: Float16(1).bitPattern,
            as: UInt16.self
        )
        basePointer.advanced(by: 2).storeBytes(
            of: Float16(0).bitPattern,
            as: UInt16.self
        )
        let quantizedBytes = basePointer.advanced(by: 4).assumingMemoryBound(to: UInt8.self)
        for index in 0..<32 {
            let lowNibble = UInt8((index * 2) % 16)
            let highNibble = UInt8((index * 2 + 1) % 16)
            quantizedBytes[index] = lowNibble | (highNibble << 4)
        }

        let store = STAFWeightStore(
            buffer: buffer,
            entries: [
                "dense.0.weight": STAFTensorEntry(
                    name: "dense.0.weight",
                    payloadOffset: 0,
                    payloadSize: blockSize,
                    schemeIdentifier: .q4Group64ScaleF16,
                    semanticRole: .other,
                    shape: [1, 8],
                    blockSize: 64,
                    groupSize: 64,
                    bufferOffset: 0
                )
            ],
            metadata: .empty,
            specializedBufferAccesses: [:]
        )

        let weightStore = CPUWeightStore(weights: store)
        let shape = try weightStore.shape(named: "dense.0.weight")
        let values = try weightStore.floatTensor(named: "dense.0.weight")

        #expect(shape == [1, 64])
        #expect(values.count == 64)
        #expect(values[0] == 0)
        #expect(values[1] == 1)
        #expect(values[2] == 2)
        #expect(values[3] == 3)
        #expect(values[15] == 15)
        #expect(values[16] == 0)
    }

    private func makeTemporaryDirectory() throws -> URL {
        let base = FileManager.default.temporaryDirectory
        let directory = base.appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        return directory
    }

    private func createDirectory(_ url: URL) throws {
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
    }

    private func writeJSON(_ string: String, to url: URL) throws {
        let data = try #require(string.data(using: .utf8))
        try data.write(to: url)
    }

    private func dummyTextConfig() -> ModelConfig {
        ModelConfig(
            hiddenSize: 8,
            layerCount: 1,
            intermediateSize: 16,
            vocabSize: 32,
            attentionHeads: 1,
            kvHeads: 1,
            headDim: 8,
            attentionBias: false,
            mlpBias: false,
            normEps: 1e-6,
            normKind: .rmsNorm,
            ropeTheta: 10_000.0,
            ropeDimension: 8,
            ropeScaling: nil,
            tiedEmbeddings: true,
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
    }
}
