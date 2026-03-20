import Foundation
import Metal
import Hub
import Jinja
import Tokenizers
import LMArchitecture
import MetalCompiler
import ModelDeclarations

/// Loads a model from HuggingFace and compiles it for Metal inference.
///
/// Optimizations:
/// - STAF cache: safetensors → STAF conversion happens once, cached on disk
/// - Zero-copy mmap: STAF payload mapped directly as MTLBuffer
/// - Parallel loading: tokenizer and STAF load concurrently
/// - Metal binary cache: compiled GPU pipelines cached via MTLBinaryArchive
///
/// ```swift
/// let container = try await ModelBundleLoader().load(repo: "Qwen/Qwen2.5-0.5B-Instruct")
/// ```
public struct ModelBundleLoader: Sendable {

    public init() {}

    /// Load a model from a HuggingFace repository.
    public func load(
        repo: String,
        progress: Progress? = nil
    ) async throws -> ModelContainer {
        let hubApi = HubApi()
        let repoId = Hub.Repo(id: repo)
        let directory = try await hubApi.snapshot(from: repoId, matching: [
            "config.json", "tokenizer.json", "tokenizer_config.json",
            "*.safetensors", "special_tokens_map.json",
            "chat_template.jinja"
        ])
        return try await load(directory: directory)
    }

    /// Load a model from a local directory.
    public func load(directory: URL) async throws -> ModelContainer {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw ModelBundleLoaderError.noMetalDevice
        }

        let startTime = CFAbsoluteTimeGetCurrent()

        // 1. Parse config (fast — small JSON file)
        let configData = try Data(contentsOf: directory.appendingPathComponent("config.json"))
        let config = try HFConfigDecoder().decode(from: configData)
        let modelType = try HFConfigDecoder().modelType(from: configData)

        // 2. Parallel: tokenizer + chat template + STAF
        // These are independent and can run concurrently.
        async let tokenizerTask = AutoTokenizer.from(modelFolder: directory)
        let safetensorsURLs = try findSafetensorsFiles(in: directory)
        let stafMetadata = try STAFModelBundleMetadataBuilder().build(
            directory: directory,
            modelType: modelType,
            config: config,
            configData: configData,
            safetensorsURLs: safetensorsURLs
        )
        async let weightStoreTask = loadOrConvertSTAF(
            directory: directory,
            device: device,
            safetensorsURLs: safetensorsURLs,
            metadata: stafMetadata
        )

        let tokenizer = try await tokenizerTask
        let weightStore = try await weightStoreTask
        let chatTemplate = loadChatTemplate(from: directory)

        let loadTime = CFAbsoluteTimeGetCurrent() - startTime
        print("[ModelBundleLoader] loaded: tokenizer + STAF [\(String(format: "%.3f", loadTime))s]")

        // 3. Build IR + resolve parameter bindings
        let compileStart = CFAbsoluteTimeGetCurrent()
        let graph = try resolveModelGraph(modelType: modelType, config: config)
        let convention = namingConvention(for: modelType)
        let resolvedGraph = ParameterResolver().resolve(graph: graph, convention: convention)

        // 4. Compile IR → MetalDispatchPlan (includes Metal pipeline compilation)
        // The current decode-specialized kernels benchmark best with the
        // aggressive optimizer once fused SwiGLU uses the input=2048 family.
        let compiler = MetalInferenceCompiler(optimizer: AggressiveOptimizer())
        let plan = try compiler.compile(
            graph: resolvedGraph,
            hiddenSize: config.hiddenSize,
            intermediateSize: config.intermediateSize,
            vocabSize: config.vocabSize,
            stafWeightStore: weightStore,
            device: device
        )

        // 4b. Compile prefill plan (sequence graph — step count independent of token count)
        let prefillPlan = try compiler.compilePrefill(
            graph: resolvedGraph,
            hiddenSize: config.hiddenSize,
            intermediateSize: config.intermediateSize,
            vocabSize: config.vocabSize,
            maximumSequenceLength: 4096,
            stafWeightStore: weightStore,
            sharedKVCache: plan.buffers.kvCache,
            sharedConvState: plan.buffers.convState,
            sharedConvStateDimension: plan.buffers.convStateDimension,
            sharedConvStateKernelSize: plan.buffers.convStateKernelSize,
            device: device
        )

        let compileTime = CFAbsoluteTimeGetCurrent() - compileStart
        print("[ModelBundleLoader] compiled: \(plan.fusedEntryCount) dispatches, prefill \(prefillPlan.stepCount) steps [\(String(format: "%.3f", compileTime))s]")

        // 5. Assemble ModelContainer
        var inferenceModel = try MetalInferenceModel(plan: plan, device: device)
        inferenceModel.prefillPlan = prefillPlan

        var modelConfig = ModelConfiguration(name: modelType)
        if let eosId = tokenizer.eosTokenId {
            modelConfig.eosTokenIds.insert(eosId)
        }

        let totalTime = CFAbsoluteTimeGetCurrent() - startTime
        print("[ModelBundleLoader] ready: \(modelConfig.name) [\(String(format: "%.3f", totalTime))s]")

        return ModelContainer(
            inferenceModel: inferenceModel,
            tokenizer: tokenizer,
            configuration: modelConfig,
            chatTemplate: chatTemplate
        )
    }

    // MARK: - STAF Loading

    /// Load STAF from cache, or convert from safetensors if needed.
    private func loadOrConvertSTAF(
        directory: URL,
        device: MTLDevice,
        safetensorsURLs: [URL],
        metadata: STAFFileMetadata
    ) async throws -> STAFWeightStore {
        let stafURL = directory.appendingPathComponent("model.staf")

        let converter = STAFConverter()
        let needsConversion: Bool
        if FileManager.default.fileExists(atPath: stafURL.path) {
            needsConversion = !(try converter.isValid(
                stafURL: stafURL,
                safetensorsURLs: safetensorsURLs,
                expectedMetadata: metadata
            ))
        } else {
            needsConversion = true
        }

        if needsConversion {
            try converter.convert(
                safetensorsURLs: safetensorsURLs,
                outputURL: stafURL,
                metadata: metadata
            )
        }
        return try STAFLoader().load(at: stafURL, device: device)
    }

    // MARK: - Chat Template

    private func loadChatTemplate(from directory: URL) -> Template? {
        let jinjaURL = directory.appendingPathComponent("chat_template.jinja")
        if let templateString = try? String(contentsOf: jinjaURL, encoding: .utf8) {
            return try? Template(templateString)
        }

        let tokenizerConfigURL = directory.appendingPathComponent("tokenizer_config.json")
        if let data = try? Data(contentsOf: tokenizerConfigURL),
           let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
           let templateString = json["chat_template"] as? String {
            return try? Template(templateString)
        }

        return nil
    }

    // MARK: - Model Resolution

    private func resolveModelGraph(modelType: String, config: ModelConfig) throws -> ModelGraph {
        switch modelType.lowercased() {
        case "llama", "qwen2", "qwen3", "mistral", "gemma", "gemma2",
             "phi", "phi3", "starcoder2", "gpt_neox", "internlm2",
             "deepseek", "yi", "baichuan", "chatglm",
             "mixtral", "qwen2_moe", "deepseek_v2", "arctic", "dbrx":
            return try Transformer(config: config).makeModelGraph()
        case "qwen3_5":
            return try Qwen35(config: config).makeModelGraph()
        case "lfm2", "lfm2_moe":
            return try LFM2(config: config).makeModelGraph()
        case "cohere", "command-r":
            return try Cohere(config: config).makeModelGraph()
        case "nemotron_h":
            throw ModelBundleLoaderError.invalidConfig(
                "nemotron_h (Mamba-2 hybrid) is not yet supported")
        default:
            return try Transformer(config: config).makeModelGraph()
        }
    }

    private func namingConvention(for modelType: String) -> ParameterResolver.WeightNamingConvention {
        switch modelType.lowercased() {
        case "lfm2", "lfm2_moe": return .lfm2Family
        default: return .llamaFamily
        }
    }

    // MARK: - File Discovery

    private func findSafetensorsFiles(in directory: URL) throws -> [URL] {
        let contents = try FileManager.default.contentsOfDirectory(
            at: directory, includingPropertiesForKeys: nil)
        let files = contents
            .filter { $0.pathExtension == "safetensors" }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
        guard !files.isEmpty else {
            throw ModelBundleLoaderError.noSafetensorsFiles(directory.path)
        }
        return files
    }
}

// MARK: - Config Decoder

struct HFConfigDecoder {
    func modelType(from data: Data) throws -> String {
        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let modelType = json["model_type"] as? String else {
            throw ModelBundleLoaderError.invalidConfig("Missing model_type in config.json")
        }
        return modelType
    }

    func decode(from data: Data) throws -> ModelConfig {
        guard let rawJson = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw ModelBundleLoaderError.invalidConfig("config.json is not a JSON object")
        }

        // VLM models (Qwen3.5, etc.) nest text model config under "text_config"
        let json: [String: Any]
        if let textConfig = rawJson["text_config"] as? [String: Any], textConfig["hidden_size"] != nil {
            // Merge: text_config values take priority, but keep top-level keys too
            var merged = rawJson
            for (key, value) in textConfig { merged[key] = value }
            json = merged
        } else {
            json = rawJson
        }

        guard let hiddenSize = json["hidden_size"] as? Int else {
            throw ModelBundleLoaderError.invalidConfig("Missing hidden_size")
        }
        guard let layerCount = json["num_hidden_layers"] as? Int else {
            throw ModelBundleLoaderError.invalidConfig("Missing num_hidden_layers")
        }
        // Intermediate size: prefer "intermediate_size", fall back to "block_ff_dim", then hiddenSize*4.
        // LFM2.5 has "intermediate_size"; LFM2 only has "block_ff_dim".
        // SwiGLU models store the raw dimension; actual weight dim = raw * 2/3 (with rounding).
        let rawIntermediateSize = json["intermediate_size"] as? Int
            ?? json["block_ff_dim"] as? Int
            ?? hiddenSize * 4
        // Only LFM2 configs set block_auto_adjust_ff_dim or block_use_swiglu.
        // Standard models (Llama, Qwen, etc.) store the final dimension in intermediate_size.
        let autoAdjust = json["block_auto_adjust_ff_dim"] as? Bool
            ?? json["block_use_swiglu"] as? Bool
            ?? false
        let intermediateSize: Int
        if autoAdjust {
            var adjusted = rawIntermediateSize * 2 / 3
            if let multiplier = json["block_ffn_dim_multiplier"] as? Double {
                adjusted = Int(multiplier * Double(adjusted))
            }
            let multipleOf = json["block_multiple_of"] as? Int ?? 256
            adjusted = multipleOf * ((adjusted + multipleOf - 1) / multipleOf)
            intermediateSize = adjusted
        } else {
            intermediateSize = rawIntermediateSize
        }
        guard let vocabSize = json["vocab_size"] as? Int else {
            throw ModelBundleLoaderError.invalidConfig("Missing vocab_size")
        }
        let attentionHeads = json["num_attention_heads"] as? Int ?? 32
        let kvHeads = json["num_key_value_heads"] as? Int ?? attentionHeads
        let headDim = json["head_dim"] as? Int ?? (hiddenSize / attentionHeads)
        let normEps = (json["rms_norm_eps"] as? Double
            ?? json["layer_norm_eps"] as? Double
            ?? json["norm_eps"] as? Double
            ?? json["block_norm_eps"] as? Double
            ?? 1e-6)
        // RoPE parameters (Qwen3.5 nests rope_theta, partial_rotary_factor, mrope inside rope_parameters)
        let ropeParams = json["rope_parameters"] as? [String: Any]
        let ropeTheta = json["rope_theta"] as? Double
            ?? (ropeParams?["rope_theta"] as? Double)
            ?? 500000.0
        let tiedEmbeddings = json["tie_word_embeddings"] as? Bool
            ?? json["tie_embedding"] as? Bool
            ?? false

        // M-RoPE axes (Qwen3.5 VLM spatial/temporal rotation)
        let mropeAxes: MRoPEAxes?
        if let sections = ropeParams?["mrope_section"] as? [Int], !sections.isEmpty {
            let interleaved = ropeParams?["mrope_interleaved"] as? Bool ?? false
            mropeAxes = MRoPEAxes(sections: sections, interleaved: interleaved)
        } else {
            mropeAxes = nil
        }

        return ModelConfig(
            hiddenSize: hiddenSize,
            layerCount: layerCount,
            intermediateSize: intermediateSize,
            vocabSize: vocabSize,
            attentionHeads: attentionHeads,
            kvHeads: kvHeads,
            headDim: headDim,
            attentionBias: json["attention_bias"] as? Bool ?? false,
            mlpBias: json["mlp_bias"] as? Bool ?? false,
            normEps: Float(normEps),
            normKind: json["model_type"] as? String == "cohere" ? .layerNorm : .rmsNorm,
            ropeTheta: Float(ropeTheta),
            ropeDimension: json["rope_dim"] as? Int ?? headDim,
            ropeScaling: nil,
            tiedEmbeddings: tiedEmbeddings,
            // LFM2 MoE uses "num_experts"; Mixtral/DeepSeek use "num_local_experts"
            expertCount: json["num_local_experts"] as? Int ?? json["num_experts"] as? Int,
            expertsPerToken: json["num_experts_per_tok"] as? Int,
            moeIntermediateSize: json["moe_intermediate_size"] as? Int,
            // LFM2 always uses QK norm (q_layernorm + k_layernorm in every attention layer)
            qkNorm: json["qk_norm"] as? Bool
                ?? (["lfm2", "lfm2_moe"].contains(json["model_type"] as? String ?? "")),
            fullAttentionInterval: json["full_attention_interval"] as? Int,
            // Qwen3.5 uses linear_num_value_heads / linear_key_head_dim / linear_value_head_dim
            ssmNumHeads: json["ssm_num_heads"] as? Int
                ?? json["linear_num_value_heads"] as? Int,
            ssmGroupCount: json["linear_num_key_heads"] as? Int,
            ssmKeyHeadDim: json["ssm_state_size"] as? Int
                ?? json["linear_key_head_dim"] as? Int,
            ssmValueHeadDim: json["ssm_state_size"] as? Int
                ?? json["linear_value_head_dim"] as? Int,
            convKernelSize: json["conv_kernel_size"] as? Int
                ?? json["linear_conv_kernel_dim"] as? Int,
            convLCache: json["conv_L_cache"] as? Int,
            partialRotaryFactor: (json["partial_rotary_factor"] as? Double
                ?? ropeParams?["partial_rotary_factor"] as? Double).map { Float($0) },
            slidingWindow: json["sliding_window"] as? Int,
            // LFM2.5 has "layer_types"; LFM2 has "full_attn_idxs" instead
            layerTypes: {
                if let types = json["layer_types"] as? [String] { return types }
                if let attnIdxs = json["full_attn_idxs"] as? [Int] {
                    let attnSet = Set(attnIdxs)
                    return (0..<layerCount).map { attnSet.contains($0) ? "full_attention" : "conv" }
                }
                return nil
            }(),
            numDenseLayers: json["num_dense_layers"] as? Int ?? 0,
            mropeAxes: mropeAxes
        )
    }
}

// MARK: - Errors

public enum ModelBundleLoaderError: Error, CustomStringConvertible {
    case noMetalDevice
    case noSafetensorsFiles(String)
    case invalidConfig(String)

    public var description: String {
        switch self {
        case .noMetalDevice:
            return "No Metal GPU device available"
        case .noSafetensorsFiles(let path):
            return "No .safetensors files found in: \(path)"
        case .invalidConfig(let message):
            return "Invalid config.json: \(message)"
        }
    }
}
