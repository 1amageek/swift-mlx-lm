import Foundation
@preconcurrency import MLX
import MLXCompiler
import SwiftLM
import MetalPerformanceShadersGraph

/// Unified loading facade that constructs a `ModelContext` from any `ModelBundle`.
///
/// Supports the compiled inference path (IR → MLXInferenceCompiler → MLXLoweredInferenceModel).
/// HF bundles flow through:
///
/// ```
/// bundle.configuration()  → ModelConfig
/// bundle.architecture()   → DetectedArchitecture
/// IRGraphAssembler.assemble() → ModelGraph
/// bundle.loadWeights()    → WeightManifest → RawWeights
/// bind + compile          → MLXLoweredInferenceModel
/// bundle.tokenizer()      → Tokenizer
/// bundle.chatTemplate()   → ChatTemplateRenderer
/// → ModelContext
/// ```
public struct ModelBundleLoader: Sendable {

    public init() {}

    /// Load a compiled model from any bundle.
    ///
    /// Uses the IR → compiler → lowered model path for optimized inference.
    ///
    /// - Parameters:
    ///   - bundle: Model bundle (HF directory).
    ///   - configOverride: Optional overrides for model configuration.
    /// - Returns: A fully initialized `ModelContext` ready for generation.
    public func loadCompiled(
        bundle: any ModelBundle,
        configOverride: ModelConfigurationOverride? = nil
    ) throws -> ModelContext {
        let loadStart = CFAbsoluteTimeGetCurrent()

        // 1. Extract config and detect architecture
        var t0 = CFAbsoluteTimeGetCurrent()
        let config = try bundle.configuration()
        let architecture = try bundle.architecture()
        print("[ModelBundleLoader] config: hiddenSize=\(config.hiddenSize) layers=\(config.layerCount) arch=\(architecture) [\(String(format: "%.3f", CFAbsoluteTimeGetCurrent() - t0))s]")

        // 2. Assemble IR + load weights
        var t1 = CFAbsoluteTimeGetCurrent()
        let assembler = IRGraphAssembler()
        let irGraph = try assembler.assemble(config: config, architecture: architecture)
        print("[ModelBundleLoader] IR: \(irGraph.rootRegion.operations.count) ops [\(String(format: "%.3f", CFAbsoluteTimeGetCurrent() - t1))s]")

        t1 = CFAbsoluteTimeGetCurrent()
        let manifest = try bundle.loadWeights()
        let rawWeights = convertToRawWeights(manifest: manifest)
        let sanitized = WeightSanitizer.filterRotaryEmbeddings(rawWeights.tensors)
        let naming: WeightNamingConvention = architecture == .hybridConvAttention
            ? .lfm2Family : .llamaFamily
        let boundWeights = try MLXWeightPathBinder(naming: naming)
            .bind(RawWeights(tensors: sanitized), to: irGraph)
        print("[ModelBundleLoader] weights: \(manifest.weights.count) tensors [\(String(format: "%.3f", CFAbsoluteTimeGetCurrent() - t1))s]")

        // 3. Compile — MPSGraph for dense transformers, MLX for everything else
        t1 = CFAbsoluteTimeGetCurrent()
        let model: any LanguageModel
        let isQuantized = manifest.quantizationInfo.values.contains { $0.bits <= 8 }

        switch architecture {
        case .transformer where !isQuantized, .parallelAttentionMLP where !isQuantized:
            do {
                let compiled = try MPSGraphCompiler().compile(graph: irGraph, weights: boundWeights)
                model = MPSGraphLanguageModel(compiled: compiled)
                print("[ModelBundleLoader] MPSGraph compiled [\(String(format: "%.3f", CFAbsoluteTimeGetCurrent() - t1))s]")
            } catch {
                print("[ModelBundleLoader] MPSGraph failed (\(error.localizedDescription)), using MLX")
                let lowered = try MLXInferenceCompiler().compile(graph: irGraph, weights: boundWeights)
                model = CompiledLanguageModel(lowered: lowered)
                print("[ModelBundleLoader] MLX compiled [\(String(format: "%.3f", CFAbsoluteTimeGetCurrent() - t1))s]")
            }
        default:
            let lowered = try MLXInferenceCompiler().compile(graph: irGraph, weights: boundWeights)
            model = CompiledLanguageModel(lowered: lowered)
            print("[ModelBundleLoader] MLX compiled [\(String(format: "%.3f", CFAbsoluteTimeGetCurrent() - t1))s]")
        }

        // 4. Create tokenizer
        t0 = CFAbsoluteTimeGetCurrent()
        let tokenizer = try bundle.tokenizer()
        print("[ModelBundleLoader] tokenizer created [\(String(format: "%.3f", CFAbsoluteTimeGetCurrent() - t0))s]")

        // 5. Build model configuration
        let modelConfig = makeModelConfiguration(
            config: config, tokenizer: tokenizer, configOverride: configOverride)

        // 6. Build user input processor
        let chatTemplate = try bundle.chatTemplate()
        let processor = makeUserInputProcessor(
            tokenizer: tokenizer, chatTemplate: chatTemplate)

        let totalTime = CFAbsoluteTimeGetCurrent() - loadStart
        print("[ModelBundleLoader] ready: \(modelConfig.name) eos=\(modelConfig.eosTokenIds) [\(String(format: "%.3f", totalTime))s]")

        return ModelContext(
            configuration: modelConfig,
            model: model,
            processor: processor,
            tokenizer: tokenizer
        )
    }

    // MARK: - Private

    /// Load a compiled model and wrap in a `ModelContainer`.
    ///
    /// Convenience method that calls `loadCompiled(bundle:configOverride:)`
    /// and wraps the result in a `ModelContainer`.
    public func load(
        bundle: any ModelBundle,
        configOverride: ModelConfigurationOverride? = nil
    ) throws -> ModelContainer {
        let context = try loadCompiled(bundle: bundle, configOverride: configOverride)
        return ModelContainer(context: context)
    }

    /// Download and load a model from a HuggingFace repository.
    ///
    /// Downloads the full HF bundle (config.json + weights + tokenizer).
    ///
    /// - Parameters:
    ///   - repo: HuggingFace repository ID (e.g., `"Qwen/Qwen2.5-0.5B-Instruct"`).
    ///   - token: Optional Hugging Face access token.
    ///   - configOverride: Optional model configuration overrides.
    ///   - progress: Optional `Progress` for download tracking.
    /// - Returns: A fully initialized `ModelContainer`.
    public func load(
        repo: String,
        token: String? = nil,
        configOverride: ModelConfigurationOverride? = nil,
        progress: Progress? = nil
    ) async throws -> ModelContainer {
        let downloader = HuggingFaceDownloader()
        let directory = try await downloader.downloadBundle(
            repo: repo, token: token, progress: progress)
        let bundle = try HFDirectoryBundle(directory: directory)
        return try load(bundle: bundle, configOverride: configOverride)
    }

    // MARK: - Private

    /// Convert a `WeightManifest` to `RawWeights` for the compiled path.
    ///
    /// Each MLXArray is wrapped in `MLXTensorStorage.dense` and stored as `TensorData`.
    /// For pre-quantized weights (from mlx-community HF repos), the quantization info
    /// in the manifest is used to construct `AffineQuantizedTensor` representations.
    private func convertToRawWeights(manifest: WeightManifest) -> RawWeights {
        var tensors: [String: TensorData] = [:]

        // Track which module paths were successfully consumed as quantized.
        // A weight is "consumed" only if it has quantization info AND both
        // scales and biases are present. If either is missing, the weight
        // falls through to dense and its scales/biases must also be kept.
        var consumedQuantizedModules = Set<String>()

        // First pass: process .weight tensors and build quantized set
        for (path, array) in manifest.weights {
            if path.hasSuffix(".weight") {
                let modulePath = String(path.dropLast(".weight".count))
                if let spec = manifest.quantizationInfo[path],
                   let scales = manifest.weights[modulePath + ".scales"],
                   let biases = manifest.weights[modulePath + ".biases"] {
                    // Pre-quantized weight — wrap as affine quantized
                    let qt = AffineQuantizedTensor(
                        logicalShape: array.shape.map { Int($0) },
                        packedWeight: array,
                        scales: scales,
                        zeroBiases: biases,
                        groupSize: spec.groupSize,
                        bits: spec.bits,
                        origin: .mlxQuantized
                    )
                    let storage = MLXTensorStorage.affineQuantized(qt)
                    tensors[path] = TensorData(
                        shape: array.shape.map { Int($0) },
                        dtype: dtypeHint(for: spec.bits),
                        storage: storage
                    )
                    consumedQuantizedModules.insert(modulePath)
                    continue
                }
            }

            // Skip scales/biases for now — handled in second pass
            if path.hasSuffix(".scales") || path.hasSuffix(".biases") {
                continue
            }

            // Dense weight or non-weight tensor
            let storage = MLXTensorStorage.dense(array)
            tensors[path] = TensorData(
                shape: array.shape.map { Int($0) },
                dtype: .float16,
                storage: storage
            )
        }

        // Second pass: include scales/biases that were NOT consumed by quantized weights
        for (path, array) in manifest.weights {
            guard path.hasSuffix(".scales") || path.hasSuffix(".biases") else { continue }

            let basePath: String
            if path.hasSuffix(".scales") {
                basePath = String(path.dropLast(".scales".count))
            } else {
                basePath = String(path.dropLast(".biases".count))
            }

            // Skip only if the module was actually consumed as quantized
            if consumedQuantizedModules.contains(basePath) {
                continue
            }

            // Not consumed — keep as dense tensor
            let storage = MLXTensorStorage.dense(array)
            tensors[path] = TensorData(
                shape: array.shape.map { Int($0) },
                dtype: .float16,
                storage: storage
            )
        }

        return RawWeights(tensors: tensors)
    }

    /// Map quantization bits to DTypeHint.
    private func dtypeHint(for bits: Int) -> DTypeHint {
        switch bits {
        case 2: return .int2
        case 3: return .int3
        case 4: return .int4
        case 5: return .int5
        case 6: return .int6
        case 8: return .int8
        default: return .int4
        }
    }

    /// Build a `ModelConfiguration` from config, tokenizer, and overrides.
    private func makeModelConfiguration(
        config: ModelConfig,
        tokenizer: any Tokenizer,
        configOverride: ModelConfigurationOverride?
    ) -> ModelConfiguration {
        var eosTokenIds = Set<Int>()
        if let eosId = tokenizer.eosTokenID {
            eosTokenIds.insert(eosId)
        }

        var modelConfig = ModelConfiguration(
            name: "model",
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

        return modelConfig
    }

    /// Build a user input processor from tokenizer and chat template.
    private func makeUserInputProcessor(
        tokenizer: any Tokenizer,
        chatTemplate: String?
    ) -> any UserInputProcessor {
        let bosToken = tokenizer.bosTokenID.flatMap { tokenizer.tokenToString($0) }
        let eosToken = tokenizer.eosTokenID.flatMap { tokenizer.tokenToString($0) }

        return ChatTemplateInputProcessor(
            tokenizer: tokenizer,
            chatTemplate: chatTemplate,
            bosToken: bosToken,
            eosToken: eosToken,
            addBosToken: false
        )
    }
}

// MARK: - Configuration Override

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

// MARK: - Errors

/// Errors specific to `ModelBundleLoader`.
public enum ModelBundleLoaderError: Error, CustomStringConvertible {
    case unknownFormat(repo: String)

    public var description: String {
        switch self {
        case .unknownFormat(let repo):
            return "Cannot determine model format for repository '\(repo)'. No safetensors files found."
        }
    }
}
