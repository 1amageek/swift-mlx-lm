import Foundation
@preconcurrency import MLX
import MLXNN
import MLXCompiler
import GGUFParser
import GGUFTokenizer
import SwiftLM
import Synchronization

/// Work item for parallel tensor conversion.
private struct TensorWorkItem: Sendable {
    let mlxName: String
    let tensor: GGUFTensorInfo
    let data: Data
    let isWeight: Bool
}

/// Accumulated results from parallel tensor conversion.
/// Guarded by Mutex — each MLXArray is independently created per thread.
private struct ConversionState: Sendable {
    var weights: [String: MLXArray] = [:]
    var directQuantInfo: [String: (groupSize: Int, bits: Int)] = [:]
    var errorMessage: String?
}

/// Loads a complete model pipeline from a single GGUF file.
///
/// Orchestrates: GGUF parse → model type resolution → model construction →
/// weight loading → tokenizer creation → ModelContext assembly.
///
/// Model selection is data-driven: each model type inspects the GGUF file's
/// structure (tensor names, metadata) to determine compatibility. No
/// architecture strings are hardcoded in the loader.
public struct GGUFModelLoader {

    private let modelTypes: [any GGUFLoadableModel.Type]

    /// Default initializer with all built-in model types.
    public init() {
        self.modelTypes = Self.defaultModelTypes
    }

    /// Package-internal initializer for custom model type lists.
    package init(modelTypes: [any GGUFLoadableModel.Type]) {
        self.modelTypes = modelTypes
    }

    /// Built-in model types in priority order.
    ///
    /// Each model's `canLoad` is a **sufficient condition** — mutually exclusive
    /// with all others except the universal fallback. The ordering is defense-in-depth
    /// (most-specific first) but correctness does not depend on it.
    package static let defaultModelTypes: [any GGUFLoadableModel.Type] = [
        Qwen35VLModel.self,     // VLM + DeltaNet SSM tensors
        Qwen25VLModel.self,     // VLM + standard Transformer
        Qwen35Model.self,       // DeltaNet SSM tensors (text-only)
        CohereModel.self,       // parallel attn+FFN with QK norm
        TransformerModel.self,  // universal fallback
    ]

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
    ///   - progress: Optional `Progress` for byte-level download tracking.
    /// - Returns: A fully initialized `ModelContainer` ready for generation.
    public func load(
        repo: String,
        filename: String? = nil,
        mmprojFilename: String? = nil,
        quantization: QuantizationConfiguration? = nil,
        token: String? = nil,
        adapterDirectory: URL? = nil,
        configOverride: ModelConfigurationOverride? = nil,
        progress: Progress? = nil
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
        let loadStart = CFAbsoluteTimeGetCurrent()

        // 1. Parse GGUF
        let fileSize = (try? FileManager.default.attributesOfItem(atPath: url.path)[.size] as? Int64) ?? 0
        print("[GGUFModelLoader] loading \(url.lastPathComponent) (\(ByteCountFormatter.string(fromByteCount: fileSize, countStyle: .file)))")
        var t0 = CFAbsoluteTimeGetCurrent()
        let file = try GGUFFile.parse(url: url)
        print("[GGUFModelLoader] parsed: architecture=\(file.architecture ?? "unknown") name=\(file.name ?? "unknown") tensors=\(file.tensors.count) [\(String(format: "%.3f", CFAbsoluteTimeGetCurrent() - t0))s]")

        // 2. Create tokenizer early (needed for VLM token ID resolution)
        t0 = CFAbsoluteTimeGetCurrent()
        let tokenizer = try GGUFTokenizerFactory.create(from: file)
        print("[GGUFModelLoader] tokenizer created [\(String(format: "%.3f", CFAbsoluteTimeGetCurrent() - t0))s]")

        // 3. Build loading context
        let loadContext = GGUFLoadContext(tokenizer: tokenizer, mmprojURL: mmprojURL)

        // 4. Find first model type that can handle this GGUF
        var candidatesEvaluated = 0
        var selectedType: (any GGUFLoadableModel.Type)?
        for modelType in modelTypes {
            candidatesEvaluated += 1
            if modelType.canLoad(from: file, context: loadContext) {
                selectedType = modelType
                break
            }
        }
        guard let modelType = selectedType else {
            print("[GGUFModelLoader] no compatible model type for architecture=\(file.architecture ?? "unknown")")
            throw GGUFLoadError.unsupportedArchitecture(file.architecture ?? "unknown")
        }
        print("[GGUFModelLoader] model type: \(modelType) (candidate \(candidatesEvaluated)/\(modelTypes.count))")

        let modelResolution = LoadReport.ModelResolution(
            selectedType: String(describing: modelType),
            candidatesEvaluated: candidatesEvaluated,
            totalCandidates: modelTypes.count
        )

        // 5. Model self-construction (config + empty model + mapper + processor factory)
        t0 = CFAbsoluteTimeGetCurrent()
        let result = try modelType.load(from: file, context: loadContext)
        print("[GGUFModelLoader] model constructed [\(String(format: "%.3f", CFAbsoluteTimeGetCurrent() - t0))s]")

        // 6. Load weights (shared for all models)
        t0 = CFAbsoluteTimeGetCurrent()
        let weightReport = try loadWeightsWithLoRA(
            into: result.model, from: file, mapper: result.mapper,
            quantization: quantization)
        print("[GGUFModelLoader] weights loaded [\(String(format: "%.3f", CFAbsoluteTimeGetCurrent() - t0))s]")
        print("[GGUFModelLoader] total loadContext [\(String(format: "%.3f", CFAbsoluteTimeGetCurrent() - loadStart))s]")

        // 7. Load vision encoder if VLM
        if let visionLoader = result.visionLoader, let mmprojURL {
            try visionLoader(mmprojURL)
        }

        // 8. Apply external adapter if provided
        if let adapterDir = adapterDirectory {
            let container = try LoRAContainer.from(directory: adapterDir)
            try container.fuse(with: result.model)
            eval(result.model)
        }

        // 9. Build model configuration
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

        // 10. Build user input processor
        let chatTemplate = file.chatTemplate
        let bosToken = tokenizer.bosTokenID.flatMap { tokenizer.tokenToString($0) }
        let eosToken = tokenizer.eosTokenID.flatMap { tokenizer.tokenToString($0) }
        let addBosToken = file.addBosToken ?? false

        let processor = result.makeProcessor(
            tokenizer, chatTemplate, bosToken, eosToken, addBosToken)

        // 11. Assemble load report
        let loadReport = LoadReport(
            modelResolution: modelResolution,
            weightLoading: weightReport.weightLoading,
            quantization: weightReport.quantization
        )

        // 12. Assemble context
        print("[GGUFModelLoader] ready: \(modelConfig.name) mapped=\(weightReport.weightLoading.mappedCount) skipped=\(weightReport.weightLoading.skippedCount) quant=\(weightReport.quantization.source.rawValue) \(weightReport.quantization.bits)bit eos=\(modelConfig.eosTokenIds)")
        return ModelContext(
            configuration: modelConfig,
            model: result.model,
            processor: processor,
            tokenizer: tokenizer,
            loadReport: loadReport
        )
    }

    // MARK: - Weight Loading

    /// Internal result from weight loading, used to build `LoadReport`.
    private struct WeightLoadingResult {
        let weightLoading: LoadReport.WeightLoading
        let quantization: LoadReport.QuantizationApplied
    }

    /// Load GGUF tensor data into a model using native MLX quantization.
    ///
    /// For quantization types that map directly to MLX's format (Q4_0, Q4_1, Q4_K, Q5_K,
    /// Q6_K, Q8_0, Q8_1), weights are packed into MLX's native UInt32 format with
    /// scales/biases. This avoids any GGUF → F16 → MLX re-quantization step.
    ///
    /// Remaining unsupported quantization types fall back to dense weights unless the caller
    /// explicitly requests MLX-side quantization via `quantization`.
    ///
    /// Returns structured facts about the loading process for diagnostic reporting.
    private func loadWeightsWithLoRA(
        into model: any LanguageModel,
        from file: GGUFFile,
        mapper: some GGUFTensorNameMapper,
        quantization: QuantizationConfiguration?
    ) throws -> WeightLoadingResult {
        var t0 = CFAbsoluteTimeGetCurrent()

        let bridge = GGUFTensorBridge()
        var weights: [String: MLXArray] = [:]
        var skippedTensors: [String] = []
        let preserveDenseWeights = quantization?.isEnabled == false

        // Track per-module quantization info for modules that were directly packed.
        var directQuantInfo: [String: (groupSize: Int, bits: Int)] = [:]

        // Phase 1: Collect work items (sequential — name mapping + data slicing)
        var workItems: [TensorWorkItem] = []
        for tensor in file.tensors {
            guard let mlxName = mapper.mlxName(for: tensor.name) else {
                skippedTensors.append(tensor.name)
                continue
            }
            let data = try file.tensorData(for: tensor)
            let isWeight = mlxName.hasSuffix(".weight") && !preserveDenseWeights
            workItems.append(TensorWorkItem(
                mlxName: mlxName, tensor: tensor, data: data, isWeight: isWeight))
        }

        // Phase 2: Convert tensors in parallel across CPU cores
        let count = workItems.count

        let state = Mutex<ConversionState>(.init())

        DispatchQueue.concurrentPerform(iterations: count) { i in
            let item = workItems[i]
            do {
                let result: ConvertedTensor
                if item.isWeight {
                    result = try bridge.convertDirect(
                        tensor: item.tensor, data: item.data)
                } else {
                    result = .float16(
                        try bridge.convert(tensor: item.tensor, data: item.data))
                }
                state.withLock { s in
                    switch result {
                    case .float16(let array):
                        s.weights[item.mlxName] = array
                    case .quantized(let weight, let scales, let biases, let groupSize, let bits):
                        let modulePath = String(item.mlxName.dropLast(".weight".count))
                        s.weights[item.mlxName] = weight
                        s.weights[modulePath + ".scales"] = scales
                        s.weights[modulePath + ".biases"] = biases
                        s.directQuantInfo[modulePath] = (groupSize: groupSize, bits: bits)
                    }
                }
            } catch {
                state.withLock { s in
                    if s.errorMessage == nil { s.errorMessage = "\(error)" }
                }
            }
        }

        let merged = state.withLock { $0 }
        if let msg = merged.errorMessage {
            throw GGUFLoadError.invalidData(msg)
        }
        weights = merged.weights
        directQuantInfo = merged.directQuantInfo

        print("[loadWeights] convertDirect loop: \(weights.count) weights, \(directQuantInfo.count) quantized [\(String(format: "%.3f", CFAbsoluteTimeGetCurrent() - t0))s]")

        let mappedCount = weights.count - directQuantInfo.count * 2 // scales+biases are extra

        // Apply model-specific weight sanitization
        t0 = CFAbsoluteTimeGetCurrent()
        let sanitized = model.sanitize(weights: weights)
        print("[loadWeights] sanitize [\(String(format: "%.3f", CFAbsoluteTimeGetCurrent() - t0))s]")

        // Check for embedded LoRA tensors
        let hasLoRA = sanitized.keys.contains { $0.contains("lora_a") }
        let hasDoRA = sanitized.keys.contains { $0.contains(".m") && sanitized.keys.contains($0.replacingOccurrences(of: ".m", with: ".lora_a")) }

        var embeddedAdapter: LoadReport.EmbeddedAdapter?

        // Install modules that can consume the pre-packed GGUF representation directly.
        // This avoids calling `MLX.quantized()` as a placeholder-construction step, which
        // would re-quantize from dense weights and rejects GGUF-native layouts like Q6_K
        // (`groupSize = 16`).
        t0 = CFAbsoluteTimeGetCurrent()
        if !directQuantInfo.isEmpty {
            let moduleUpdates = try makeDirectQuantizedModuleUpdates(
                model: model,
                weights: sanitized,
                directQuantInfo: directQuantInfo
            )
            if !moduleUpdates.isEmpty {
                try model.update(modules: ModuleChildren.unflattened(moduleUpdates), verify: .none)
            }
        }
        print("[loadWeights] directQuant module updates [\(String(format: "%.3f", CFAbsoluteTimeGetCurrent() - t0))s]")

        t0 = CFAbsoluteTimeGetCurrent()
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

            // Record adapter facts
            embeddedAdapter = LoadReport.EmbeddedAdapter(
                type: hasDoRA ? .dora : .lora,
                rank: loraConfig.loraParameters.rank,
                layerCount: loraConfig.numLayers,
                tensorCount: loraWeights.count
            )

            // Create container and fuse
            let loraParams = ModuleParameters.unflattened(loraWeights)
            let container = LoRAContainer(configuration: loraConfig, parameters: loraParams)
            try container.fuse(with: model)
        } else {
            // Standard loading path (no LoRA)
            let parameters = ModuleParameters.unflattened(sanitized)
            try model.update(parameters: parameters, verify: .noUnusedKeys)
        }
        print("[loadWeights] model.update(parameters:) [\(String(format: "%.3f", CFAbsoluteTimeGetCurrent() - t0))s]")

        // Determine effective quantization for remaining (non-directly-packed) Linear modules.
        // Automatic GGUF-type detection is intentionally not used here: mixed GGUF models
        // must not be dequantized to F16 and then re-quantized to a different MLX bit width.
        let requantConfig: QuantizationConfiguration?
        if let explicit = quantization {
            requantConfig = explicit.isEnabled ? explicit : nil
        } else {
            requantConfig = nil
        }

        // Re-quantize any remaining Linear modules whose weights fell back to F16.
        // Modules already converted to QuantizedLinear (directly-packed) are skipped
        // automatically by MLXNN's quantize function.
        if let config = requantConfig {
            quantize(model: model, groupSize: config.groupSize, bits: config.bits)
        }

        // Report quantization status
        let quantizationReport: LoadReport.QuantizationApplied
        if !directQuantInfo.isEmpty {
            // Some or all layers were directly packed
            let mostCommon = Dictionary(grouping: directQuantInfo.values, by: { "\($0.bits)_\($0.groupSize)" })
                .max(by: { $0.value.count < $1.value.count })?.value.first
            quantizationReport = LoadReport.QuantizationApplied(
                source: .autoDetected,
                bits: mostCommon?.bits ?? 4,
                groupSize: mostCommon?.groupSize ?? 32
            )
        } else if let config = requantConfig {
            // No layers were directly packed — all re-quantized from F16
            quantizationReport = LoadReport.QuantizationApplied(
                source: quantization != nil ? .userSpecified : .autoDetected,
                bits: config.bits, groupSize: config.groupSize)
        } else {
            quantizationReport = LoadReport.QuantizationApplied(
                source: .disabled, bits: 0, groupSize: 0
            )
        }

        t0 = CFAbsoluteTimeGetCurrent()
        eval(model)
        print("[loadWeights] eval(model) [\(String(format: "%.3f", CFAbsoluteTimeGetCurrent() - t0))s]")

        return WeightLoadingResult(
            weightLoading: LoadReport.WeightLoading(
                mappedCount: mappedCount,
                skippedCount: skippedTensors.count,
                skippedTensors: skippedTensors,
                embeddedAdapter: embeddedAdapter
            ),
            quantization: quantizationReport
        )
    }

    // MARK: - Inference Compiler Path

    /// Load a compiled model as a `ModelContainer`, ready for the standard generation pipeline.
    ///
    /// This is the primary entry point for the compiled inference path. It produces
    /// a `ModelContainer` with a `CompiledLanguageModel` that works through the
    /// standard `TokenIterator` → `generate()` pipeline.
    ///
    /// - Parameters:
    ///   - repo: Hugging Face repository ID.
    ///   - filename: Explicit GGUF filename. When `nil`, auto-selects the best available.
    ///   - token: Optional Hugging Face access token.
    ///   - configOverride: Optional model configuration overrides.
    ///   - progress: Optional `Progress` for byte-level download tracking.
    /// - Returns: A `ModelContainer` using the compiled inference path.
    public func loadCompiled(
        repo: String,
        filename: String? = nil,
        token: String? = nil,
        configOverride: ModelConfigurationOverride? = nil,
        progress: Progress? = nil
    ) async throws -> ModelContainer {
        let downloader = HuggingFaceDownloader()
        let localURL = try await downloader.download(
            repo: repo, filename: filename, token: token, progress: progress)
        let context = try loadCompiledContext(url: localURL, configOverride: configOverride)
        return ModelContainer(context: context)
    }

    /// Load a compiled model context from a GGUF file URL.
    ///
    /// Returns a `ModelContext` with a `CompiledLanguageModel` adapter that wraps
    /// `MLXLoweredInferenceModel` and exposes it through the `LanguageModel` protocol.
    /// This enables use through the standard `TokenIterator` → `generate()` pipeline.
    ///
    /// - Parameters:
    ///   - url: Path to the GGUF file.
    ///   - configOverride: Optional overrides for model configuration.
    /// - Returns: A `ModelContext` with the compiled inference model.
    public func loadCompiledContext(
        url: URL,
        configOverride: ModelConfigurationOverride? = nil
    ) throws -> ModelContext {
        let fileSize = (try? FileManager.default.attributesOfItem(atPath: url.path)[.size] as? Int64) ?? 0
        print("[GGUFModelLoader] loading compiled \(url.lastPathComponent) (\(ByteCountFormatter.string(fromByteCount: fileSize, countStyle: .file)))")

        let file = try GGUFFile.parse(url: url)
        print("[GGUFModelLoader] parsed: architecture=\(file.architecture ?? "unknown") tensors=\(file.tensors.count)")

        // Create tokenizer (needed for ModelContext)
        let tokenizer = try GGUFTokenizerFactory.create(from: file)
        let loadContext = GGUFLoadContext(tokenizer: tokenizer, mmprojURL: nil)

        // Find a compilable model type that matches this GGUF
        var selectedType: (any GGUFCompilableModel.Type)?
        for modelType in modelTypes {
            guard let compilableType = modelType as? any GGUFCompilableModel.Type,
                  compilableType.canLoad(from: file, context: loadContext)
            else { continue }
            selectedType = compilableType
            break
        }

        guard let compilableType = selectedType else {
            throw GGUFLoadError.unsupportedArchitecture(
                "No GGUFCompilableModel found for architecture=\(file.architecture ?? "unknown")")
        }
        print("[GGUFModelLoader] compilable type: \(compilableType)")

        // Build ModelComponent declaration and compile
        let result = try compilableType.load(from: file, context: loadContext)
        let declaration = try compilableType.makeModelDeclaration(from: file)
        let lowered = try compileModel(
            from: file, declaration: declaration, mapper: result.mapper,
            compilableType: compilableType)

        // Wrap in CompiledLanguageModel adapter
        let compiledModel = CompiledLanguageModel(lowered: lowered)

        // Build model configuration
        var eosTokenIds = Set<Int>()
        if let eosId = file.eosTokenID {
            eosTokenIds.insert(eosId)
        }

        var modelConfig = ModelConfiguration(
            name: file.name ?? url.deletingPathExtension().lastPathComponent,
            eosTokenIds: eosTokenIds
        )

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

        // Build user input processor
        let chatTemplate = file.chatTemplate
        let bosToken = tokenizer.bosTokenID.flatMap { tokenizer.tokenToString($0) }
        let eosToken = tokenizer.eosTokenID.flatMap { tokenizer.tokenToString($0) }
        let addBosToken = file.addBosToken ?? false

        let processor = result.makeProcessor(
            tokenizer, chatTemplate, bosToken, eosToken, addBosToken)

        print("[GGUFModelLoader] compiled ready: \(modelConfig.name) cacheSlots=\(lowered.metadata.cacheSlotCount) eos=\(modelConfig.eosTokenIds)")
        return ModelContext(
            configuration: modelConfig,
            model: compiledModel,
            processor: processor,
            tokenizer: tokenizer
        )
    }

    /// Compile a model graph from GGUF tensor data.
    ///
    /// Internal helper shared by `loadCompiledContext` and direct compilation paths.
    private func compileModel(
        from file: GGUFFile,
        declaration: some ModelComponent,
        mapper: some GGUFTensorNameMapper,
        compilableType: any GGUFCompilableModel.Type
    ) throws -> MLXLoweredInferenceModel {
        // 1. Build ModelGraph from declaration
        let normalized = try declaration.makeNormalizedModel()
        let graph = normalized.graph
        print("[SwiftLM] architecture dump:\n\(ModelGraphDumper.dump(normalized))")

        // 2. Convert GGUF tensors to RawWeights (keyed by MLX weight paths)
        let rawWeights = try buildRawWeights(from: file, mapper: mapper)

        // 3. Apply model-specific weight sanitization (compiled path equivalent of sanitize(weights:))
        let sanitizedTensors = compilableType.sanitizeCompiledWeights(rawWeights.tensors)
        let sanitizedWeights = RawWeights(tensors: sanitizedTensors)

        // 4. Bind weights to ParameterSlots
        let binder = MLXWeightPathBinder()
        let boundWeights = try binder.bind(sanitizedWeights, to: graph)

        // 5. Compile
        let compiler = MLXInferenceCompiler()
        return try compiler.compile(graph: graph, weights: boundWeights)
    }

    /// Convert GGUF tensors to `RawWeights` keyed by MLX weight paths.
    ///
    /// Each tensor is converted to `MLXTensorStorage` (preserving quantization)
    /// and wrapped in `TensorData`. The GGUF tensor name is mapped to its
    /// MLX weight path via the provided mapper.
    private func buildRawWeights(
        from file: GGUFFile,
        mapper: some GGUFTensorNameMapper
    ) throws -> RawWeights {
        let bridge = GGUFTensorBridge()
        var tensors: [String: TensorData] = [:]

        for tensor in file.tensors {
            guard let mlxName = mapper.mlxName(for: tensor.name) else {
                continue
            }

            let data = try file.tensorData(for: tensor)
            let storage = try bridge.convertToTensorStorage(tensor: tensor, data: data)
            let shape = tensor.dimensions.reversed().map { Int($0) }

            let dtype: DTypeHint
            switch storage {
            case .dense:
                dtype = .float16
            case .affineQuantized(let qt):
                switch qt.bits {
                case 2: dtype = .int2
                case 3: dtype = .int3
                case 4: dtype = .int4
                case 5: dtype = .int5
                case 6: dtype = .int6
                case 8: dtype = .int8
                default: dtype = .int4
                }
            }

            tensors[mlxName] = TensorData(
                shape: shape,
                dtype: dtype,
                storage: storage
            )
        }

        return RawWeights(tensors: tensors)
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

extension GGUFModelLoader {

    private func makeDirectQuantizedModuleUpdates(
        model: any LanguageModel,
        weights: [String: MLXArray],
        directQuantInfo: [String: (groupSize: Int, bits: Int)]
    ) throws -> [(String, Module)] {
        let leafModules = Dictionary(uniqueKeysWithValues: model.leafModules().flattened())
        var updates: [(String, Module)] = []
        updates.reserveCapacity(directQuantInfo.count)

        for (path, info) in directQuantInfo.sorted(by: { $0.key < $1.key }) {
            guard let module = leafModules[path] else {
                throw GGUFLoadError.tensorNotFound(path)
            }
            let weightKey = path + ".weight"
            let scalesKey = path + ".scales"
            let biasesKey = path + ".biases"
            guard let weight = weights[weightKey] else {
                throw GGUFLoadError.tensorNotFound(weightKey)
            }
            guard let scales = weights[scalesKey] else {
                throw GGUFLoadError.tensorNotFound(scalesKey)
            }
            guard let biases = weights[biasesKey] else {
                throw GGUFLoadError.tensorNotFound(biasesKey)
            }

            let replacement = try makeDirectQuantizedModule(
                module: module,
                weight: weight,
                scales: scales,
                biases: biases,
                groupSize: info.groupSize,
                bits: info.bits,
                path: path
            )
            updates.append((path, replacement))
        }

        return updates
    }

    private func makeDirectQuantizedModule(
        module: Module,
        weight: MLXArray,
        scales: MLXArray,
        biases: MLXArray,
        groupSize: Int,
        bits: Int,
        path: String
    ) throws -> Module {
        if let linear = module as? MLXNN.Linear {
            if groupSize >= 32 {
                return QuantizedLinear(
                    weight: weight,
                    bias: linear.bias,
                    scales: scales,
                    biases: biases,
                    groupSize: groupSize,
                    bits: bits,
                    mode: .affine
                )
            }

            return DirectQuantizedLinear(
                weight: weight,
                bias: linear.bias,
                scales: scales,
                biases: biases,
                groupSize: groupSize,
                bits: bits,
                mode: .affine
            )
        }

        if module is MLXNN.Embedding {
            return DirectQuantizedEmbedding(
                weight: weight,
                scales: scales,
                biases: biases,
                groupSize: groupSize,
                bits: bits,
                mode: .affine
            )
        }

        throw GGUFLoadError.dimensionMismatch(
            "Direct quantized module at \(path) must be Linear or Embedding, got \(type(of: module))"
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
    /// to the nearest MLX-supported bit width. MLX supports 2, 3, 4, 5, 6, 8-bit
    /// quantization natively.
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
        // 2-bit types
        case .q2_K, .iq2_XXS, .iq2_XS, .iq2_S:
            return .twoBit
        // 3-bit types → re-quantize to 4-bit (precision improves)
        case .q3_K, .iq3_XXS, .iq3_S:
            return .fourBit
        // 4-bit types
        case .q4_0, .q4_1, .q4_K, .iq4_NL, .iq4_XS:
            return .fourBit
        // 5-bit types → re-quantize to 8-bit (preserves more precision than 4-bit)
        case .q5_0, .q5_1, .q5_K:
            return .eightBit
        // 6-bit types → re-quantize to 8-bit
        case .q6_K:
            return .eightBit
        // 8-bit types
        case .q8_0, .q8_1, .q8_K:
            return .eightBit
        // 1-bit / ternary types → re-quantize to 4-bit
        case .iq1_S, .iq1_M, .tq1_0, .tq2_0:
            return .fourBit
        default:
            return .fourBit
        }
    }
}

// MARK: - Dummy Tokenizer for Auto-Detection

/// Minimal tokenizer used only during `loadCompiled(url:)` auto-detection.
///
/// The `canLoad(from:context:)` checks only inspect tensor names and metadata,
/// never using the tokenizer. This avoids building a real tokenizer just for
/// model type detection.
private struct DummyTokenizer: Tokenizer {
    func encode(text: String) -> [Int] { [] }
    func decode(tokens: [Int]) -> String { "" }
    var bosTokenID: Int? { nil }
    var eosTokenID: Int? { nil }
    var vocabularySize: Int { 0 }
    func tokenToString(_ id: Int) -> String? { nil }
    func tokenID(for string: String) -> Int? { nil }
}
