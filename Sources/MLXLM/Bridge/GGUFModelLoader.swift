import Foundation
import MLX
import MLXNN
import GGUFParser
import GGUFTokenizer

/// Loads a complete model pipeline from a single GGUF file.
///
/// Orchestrates: GGUF parse → config extraction → model construction →
/// weight loading → tokenizer creation → ModelContext assembly.
public struct GGUFModelLoader {

    public init() {}

    /// Download and load a model from Hugging Face Hub.
    ///
    /// When `filename` is omitted, automatically discovers GGUF files in the
    /// repository and selects the best one (prefers Q4_K_M).
    ///
    /// - Parameters:
    ///   - repo: Hugging Face repository ID (e.g., `"Qwen/Qwen2.5-0.5B-Instruct-GGUF"`).
    ///   - filename: Explicit GGUF filename. When `nil`, auto-selects the best available.
    ///   - mmprojFilename: Optional mmproj GGUF filename for VLM vision encoder.
    ///   - quantization: Quantization settings. `nil` for auto-detect, `.disabled` for F16.
    ///   - token: Optional Hugging Face access token for private models.
    ///   - adapterDirectory: Optional path to LoRA adapter directory.
    ///   - configOverride: Optional model configuration overrides.
    ///   - progress: Optional callback reporting download progress (0.0 to 1.0).
    /// - Returns: A fully initialized `ModelContainer` ready for generation.
    public func load(
        repo: String,
        filename: String? = nil,
        mmprojFilename: String? = nil,
        quantization: QuantizationConfiguration? = nil,
        token: String? = nil,
        adapterDirectory: URL? = nil,
        configOverride: ModelConfigurationOverride? = nil,
        progress: (@Sendable (Double) -> Void)? = nil
    ) async throws -> ModelContainer {
        let downloader = HuggingFaceDownloader()
        let localURL = try await downloader.download(
            repo: repo, filename: filename, token: token, progress: progress)

        var mmprojURL: URL?
        if let mmprojFilename {
            mmprojURL = try await downloader.download(
                repo: repo, filename: mmprojFilename, token: token, progress: progress)
        }

        return try load(
            url: localURL, mmprojURL: mmprojURL, quantization: quantization,
            adapterDirectory: adapterDirectory, configOverride: configOverride)
    }

    /// Load a model from a GGUF file URL.
    ///
    /// - Parameters:
    ///   - url: Path to the GGUF file.
    ///   - mmprojURL: Optional path to mmproj GGUF for VLM vision encoder.
    ///   - quantization: Quantization settings. Pass `nil` to auto-detect from GGUF metadata,
    ///     an explicit config like `.fourBit` to override, or `.disabled` to keep F16 weights.
    ///   - adapterDirectory: Optional path to a directory containing `adapter_config.json` and `adapters.safetensors`.
    ///   - configOverride: Optional overrides for model configuration.
    /// - Returns: A fully initialized `ModelContainer` ready for generation.
    public func load(
        url: URL,
        mmprojURL: URL? = nil,
        quantization: QuantizationConfiguration? = nil,
        adapterDirectory: URL? = nil,
        configOverride: ModelConfigurationOverride? = nil
    ) throws -> ModelContainer {
        let context = try loadContext(
            url: url, mmprojURL: mmprojURL, quantization: quantization,
            adapterDirectory: adapterDirectory, configOverride: configOverride)
        return ModelContainer(context: context)
    }

    /// Load a model context from a GGUF file URL.
    ///
    /// Use this when you need to inspect the context before wrapping in a container.
    public func loadContext(
        url: URL,
        mmprojURL: URL? = nil,
        quantization: QuantizationConfiguration? = nil,
        adapterDirectory: URL? = nil,
        configOverride: ModelConfigurationOverride? = nil
    ) throws -> ModelContext {
        // 1. Parse GGUF
        let file = try GGUFFile.parse(url: url)

        // 2. Determine architecture
        guard let arch = file.architecture else {
            throw GGUFLoadError.missingMetadata("general.architecture")
        }

        // 3. Build model + load weights based on architecture
        let model: any LanguageModel

        // Mixtral uses "llama" as its GGUF architecture string.
        // Detect MoE by checking for expert_count metadata.
        let isMoE = (file.expertCount ?? 0) > 0
        let archKey = arch.lowercased()

        switch archKey {
        case "llama", "qwen2", "qwen3", "mistral", "gemma2", "phi3", "starcoder2":
            let configuration = try GGUFConfigExtractor.extractTransformerConfig(
                from: file, archHint: archKey, isMoE: isMoE)
            let transformerModel = TransformerModel(configuration)

            let mapper: any GGUFTensorNameMapper
            if isMoE {
                mapper = MixtralTensorNameMapper()
            } else if archKey == "gemma2" {
                mapper = Gemma2TensorNameMapper()
            } else {
                mapper = LlamaTensorNameMapper()
            }

            try loadWeightsWithLoRA(into: transformerModel, from: file, mapper: mapper, quantization: quantization)
            model = transformerModel

        case "qwen35", "qwen3_5", "qwen3next":
            let configuration = try GGUFConfigExtractor.extractQwen35Config(from: file)
            let qwen35Model = Qwen35Model(configuration)
            try loadWeightsWithLoRA(into: qwen35Model, from: file, mapper: Qwen35TensorNameMapper(), quantization: quantization)
            model = qwen35Model

        case "command-r", "cohere2":
            let configuration = try GGUFConfigExtractor.extractCohereConfig(from: file)
            let cohereModel = CohereModel(configuration)
            try loadWeightsWithLoRA(into: cohereModel, from: file, mapper: CohereTensorNameMapper(), quantization: quantization)
            model = cohereModel

        case "qwen2vl", "qwen2_5_vl", "qwen25vl":
            let textConfig = try GGUFConfigExtractor.extractTransformerConfig(
                from: file, archHint: "qwen2", isMoE: false)
            let qwen25vlTextConfig = Qwen25VLConfiguration.TextConfiguration(
                hiddenSize: textConfig.hiddenSize,
                hiddenLayers: textConfig.hiddenLayers,
                intermediateSize: textConfig.intermediateSize,
                attentionHeads: textConfig.attentionHeads,
                kvHeads: textConfig.kvHeads,
                vocabularySize: textConfig.vocabularySize,
                normEps: textConfig.normEps,
                ropeTheta: textConfig.ropeTheta,
                tieWordEmbeddings: textConfig.tieWordEmbeddings
            )

            var visionConfig = Qwen25VLConfiguration.VisionConfiguration()
            if let mmprojURL {
                let visionLoader = GGUFVisionLoader()
                let (_, loadedVisionConfig) = try visionLoader.load(url: mmprojURL)
                visionConfig = loadedVisionConfig
            }
            visionConfig.outHiddenSize = qwen25vlTextConfig.hiddenSize

            let vlmConfig = Qwen25VLConfiguration(text: qwen25vlTextConfig, vision: visionConfig)
            let vlmModel = Qwen25VLModel(vlmConfig)

            // Load text weights using Llama mapper (Qwen2 uses llama-style naming in GGUF)
            try loadWeightsWithLoRA(
                into: vlmModel, from: file, mapper: LlamaTensorNameMapper(), quantization: quantization)

            // Load vision weights from mmproj if provided
            if let mmprojURL {
                let visionLoader = GGUFVisionLoader()
                let (loadedEncoder, _) = try visionLoader.load(url: mmprojURL)
                // Transfer loaded vision encoder weights into the model
                let visionParams = loadedEncoder.parameters()
                try vlmModel.visionEncoder.update(parameters: visionParams, verify: .noUnusedKeys)
                eval(vlmModel.visionEncoder)
            }

            model = vlmModel

        default:
            throw GGUFLoadError.unsupportedArchitecture(arch)
        }

        // Apply external adapter if provided
        if let adapterDir = adapterDirectory {
            let container = try LoRAContainer.from(directory: adapterDir)
            try container.fuse(with: model)
            eval(model)
        }

        // 4. Create tokenizer
        let tokenizer = try GGUFTokenizerFactory.create(from: file)

        // 5. Build model configuration
        var eosTokenIds = Set<Int>()
        if let eosId = file.eosTokenID {
            eosTokenIds.insert(eosId)
        }

        var modelConfig = ModelConfiguration(
            name: file.name ?? url.deletingPathExtension().lastPathComponent,
            eosTokenIds: eosTokenIds
        )

        // Apply overrides
        if let override = configOverride {
            if let overrideEos = override.eosTokenIds {
                modelConfig.eosTokenIds = overrideEos
            }
            if let extra = override.extraEOSTokens {
                for token in extra {
                    let ids = tokenizer.encode(text: token)
                    if let id = ids.first {
                        modelConfig.eosTokenIds.insert(id)
                    }
                }
            }
            if let format = override.toolCallFormat {
                modelConfig.toolCallFormat = format
            }
        }

        // 6. Build user input processor
        let chatTemplate = file.chatTemplate
        let isVLM = archKey == "qwen2vl" || archKey == "qwen2_5_vl" || archKey == "qwen25vl"
        let processor: any UserInputProcessor

        if isVLM, let vlmModel = model as? Qwen25VLModel {
            processor = VLMUserInputProcessor(
                tokenizer: tokenizer,
                chatTemplate: chatTemplate,
                bosToken: tokenizer.bosTokenID.flatMap { tokenizer.tokenToString($0) },
                eosToken: tokenizer.eosTokenID.flatMap { tokenizer.tokenToString($0) },
                addBosToken: file.addBosToken ?? false,
                visionConfig: vlmModel.configuration.vision
            )
        } else {
            processor = GGUFUserInputProcessor(
                tokenizer: tokenizer,
                chatTemplate: chatTemplate,
                bosToken: tokenizer.bosTokenID.flatMap { tokenizer.tokenToString($0) },
                eosToken: tokenizer.eosTokenID.flatMap { tokenizer.tokenToString($0) },
                addBosToken: file.addBosToken ?? false
            )
        }

        // 7. Assemble context
        return ModelContext(
            configuration: modelConfig,
            model: model,
            processor: processor,
            tokenizer: tokenizer
        )
    }

    // MARK: - Weight Loading

    /// Load GGUF tensor data into a model, auto-detecting and fusing embedded LoRA weights.
    private func loadWeightsWithLoRA(
        into model: any LanguageModel & Module,
        from file: GGUFFile,
        mapper: some GGUFTensorNameMapper,
        quantization: QuantizationConfiguration?
    ) throws {
        let bridge = GGUFTensorBridge()
        var weights: [String: MLXArray] = [:]

        for tensor in file.tensors {
            guard let mlxName = mapper.mlxName(for: tensor.name) else {
                continue
            }

            let data = try file.tensorData(for: tensor)
            let array = try bridge.convert(tensor: tensor, data: data)
            weights[mlxName] = array
        }

        // Apply model-specific weight sanitization
        let sanitized = model.sanitize(weights: weights)

        // Check for embedded LoRA tensors
        let hasLoRA = sanitized.keys.contains { $0.contains("lora_a") }
        let hasDoRA = sanitized.keys.contains { $0.contains(".m") && sanitized.keys.contains($0.replacingOccurrences(of: ".m", with: ".lora_a")) }

        if hasLoRA {
            // Split base weights and LoRA weights
            var baseWeights: [String: MLXArray] = [:]
            var loraWeights: [String: MLXArray] = [:]

            for (key, value) in sanitized {
                if key.hasSuffix(".lora_a") || key.hasSuffix(".lora_b") {
                    loraWeights[key] = value
                } else if isDoRAMagnitude(key: key, allKeys: sanitized.keys) {
                    loraWeights[key] = value
                } else {
                    baseWeights[key] = value
                }
            }

            // Load base weights
            let baseParameters = ModuleParameters.unflattened(baseWeights)
            try model.update(parameters: baseParameters, verify: .noUnusedKeys)

            // Detect LoRA configuration from tensor shapes
            let loraConfig = detectLoRAConfiguration(
                from: loraWeights, isDora: hasDoRA, model: model)

            // Create container and fuse
            let loraParams = ModuleParameters.unflattened(loraWeights)
            let container = LoRAContainer(configuration: loraConfig, parameters: loraParams)
            try container.fuse(with: model)
        } else {
            // Standard loading path (no LoRA)
            let parameters = ModuleParameters.unflattened(sanitized)
            try model.update(parameters: parameters, verify: .noUnusedKeys)
        }

        // Re-quantize to MLX native format for efficient runtime inference.
        // MLX lazy evaluation fuses the dequant→requant pipeline so the F16
        // intermediate is never fully materialized in memory.
        let resolved = quantization ?? QuantizationConfiguration.detect(from: file)
        if let config = resolved, config.isEnabled {
            quantize(
                model: model,
                groupSize: config.groupSize,
                bits: config.bits
            )
        }

        eval(model)
    }

    /// Check if a key is a DoRA magnitude parameter.
    private func isDoRAMagnitude(key: String, allKeys: Dictionary<String, MLXArray>.Keys) -> Bool {
        guard key.hasSuffix(".m") else { return false }
        let loraKey = String(key.dropLast(2)) + ".lora_a"
        return allKeys.contains(loraKey)
    }

    /// Detect LoRA configuration from embedded tensor shapes.
    private func detectLoRAConfiguration(
        from loraWeights: [String: MLXArray],
        isDora: Bool,
        model: any LanguageModel
    ) -> LoRAConfiguration {
        // Infer rank from the first lora_a tensor
        var rank = 8
        for (key, value) in loraWeights where key.hasSuffix(".lora_a") {
            rank = value.dim(1)
            break
        }

        // Count layers that have LoRA tensors
        var layerIndices = Set<Int>()
        for key in loraWeights.keys {
            // Extract layer index from paths like "model.layers.0.self_attn.q_proj.lora_a"
            let parts = key.split(separator: ".")
            if let layersIdx = parts.firstIndex(of: "layers"),
               layersIdx + 1 < parts.count,
               let idx = Int(parts[layersIdx + 1]) {
                layerIndices.insert(idx)
            }
        }

        let numLayers = layerIndices.count > 0 ? layerIndices.count : model.layerCount

        return LoRAConfiguration(
            numLayers: numLayers,
            fineTuneType: isDora ? .dora : .lora,
            loraParameters: LoRAConfiguration.LoRAParameters(rank: rank, scale: 20.0)
        )
    }
}

/// Optional overrides when loading a model.
public struct ModelConfigurationOverride: Sendable {
    public var eosTokenIds: Set<Int>?
    public var extraEOSTokens: Set<String>?
    public var toolCallFormat: ToolCallFormat?

    public init(
        eosTokenIds: Set<Int>? = nil,
        extraEOSTokens: Set<String>? = nil,
        toolCallFormat: ToolCallFormat? = nil
    ) {
        self.eosTokenIds = eosTokenIds
        self.extraEOSTokens = extraEOSTokens
        self.toolCallFormat = toolCallFormat
    }
}

/// Weight quantization parameters for runtime inference.
///
/// Controls how model weights are stored and computed at runtime.
/// MLX uses `quantizedMatmul` with packed integer weights and per-group
/// scale/bias factors, which reduces memory by 4-8x and accelerates
/// Metal matmul kernels.
///
/// ```swift
/// // Auto-detect from GGUF (default behavior)
/// let container = try loader.load(url: ggufURL)
///
/// // Explicit 4-bit quantization
/// let container = try loader.load(url: ggufURL, quantization: .fourBit)
///
/// // No quantization (keep F16 weights)
/// let container = try loader.load(url: ggufURL, quantization: .disabled)
/// ```
public struct QuantizationConfiguration: Sendable {

    /// Number of bits per quantized weight element.
    ///
    /// MLX supports 2, 4, and 8-bit quantization.
    /// 4-bit is the most common (matches Q4_K_M GGUF files).
    public var bits: Int

    /// Number of elements per quantization group.
    ///
    /// Each group shares one scale factor and one bias value.
    /// Smaller groups preserve more precision but add overhead.
    public var groupSize: Int

    public init(bits: Int = 4, groupSize: Int = 64) {
        self.bits = bits
        self.groupSize = groupSize
    }

    /// 4-bit quantization with group size 64.
    ///
    /// Best for Q4_0, Q4_K, Q5_K GGUF models.
    public static let fourBit = QuantizationConfiguration(bits: 4, groupSize: 64)

    /// 8-bit quantization with group size 32.
    ///
    /// Best for Q8_0 GGUF models or when higher precision is needed.
    public static let eightBit = QuantizationConfiguration(bits: 8, groupSize: 32)

    /// 2-bit quantization with group size 64.
    ///
    /// Aggressive compression for Q2_K GGUF models.
    public static let twoBit = QuantizationConfiguration(bits: 2, groupSize: 64)

    /// Sentinel value indicating quantization should be skipped.
    ///
    /// Weights remain as F16 `Linear` layers. Use this when you need
    /// full precision or the model is already unquantized.
    public static let disabled = QuantizationConfiguration(bits: 0, groupSize: 0)

    /// Whether this configuration actually enables quantization.
    public var isEnabled: Bool { bits > 0 }

    /// Detect appropriate quantization from a GGUF file's tensor metadata.
    ///
    /// Examines the predominant quantization type across all tensors and maps
    /// to the nearest MLX-supported bit width (2, 4, or 8).
    /// Returns `nil` for unquantized (F16/F32/BF16) models.
    public static func detect(from file: GGUFFile) -> QuantizationConfiguration? {
        var typeCounts: [GGUFQuantizationType: Int] = [:]
        for tensor in file.tensors {
            guard !tensor.quantizationType.isUnquantized else { continue }
            typeCounts[tensor.quantizationType, default: 0] += 1
        }

        guard let (mostCommon, _) = typeCounts.max(by: { $0.value < $1.value }) else {
            return nil
        }

        switch mostCommon {
        case .q2_K:
            return .twoBit
        case .q3_K, .q4_0, .q4_1, .q4_K, .q5_0, .q5_1, .q5_K,
             .iq2_XXS, .iq2_XS, .iq2_S, .iq3_XXS, .iq3_S,
             .iq4_NL, .iq4_XS, .iq1_S, .iq1_M, .tq1_0, .tq2_0:
            return .fourBit
        case .q6_K:
            return QuantizationConfiguration(bits: 8, groupSize: 64)
        case .q8_0, .q8_1, .q8_K:
            return .eightBit
        default:
            return .fourBit
        }
    }
}

