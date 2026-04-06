import Foundation
import Metal
import Hub
import Tokenizers
import MetalCompiler

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
            "chat_template.jinja", "preprocessor_config.json", "processor_config.json"
        ])
        return try await load(directory: directory)
    }

    /// Load a model from a local directory.
    public func load(directory: URL) async throws -> ModelContainer {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw ModelBundleLoaderError.noMetalDevice
        }

        let startTime = CFAbsoluteTimeGetCurrent()
        let inspector = ModelBundleInspector()
        let resources = try inspector.inspect(directory: directory)
        // 2. Parallel: tokenizer + chat template + STAF
        async let tokenizerTask = AutoTokenizer.from(modelFolder: directory)
        async let weightStoreTask = STAFCacheLoader().load(resources: resources, device: device)

        let tokenizer = try await tokenizerTask
        let weightStore = try await weightStoreTask
        let visionRuntime = try QwenVisionRuntime.makeIfSupported(resources: resources, device: device)
        let gemma4Runtime = try Gemma4Runtime.makeIfSupported(
            resources: resources,
            tokenizer: tokenizer,
            weights: weightStore
        )

        let loadTime = CFAbsoluteTimeGetCurrent() - startTime
        print("[ModelBundleLoader] loaded: tokenizer + STAF [\(String(format: "%.3f", loadTime))s]")

        // 3. Build IR + resolve parameter bindings
        let compileStart = CFAbsoluteTimeGetCurrent()
        let graphResolver = ModelGraphResolver()
        let graph = try graphResolver.resolveModelGraph(
            modelType: resources.modelType,
            config: resources.config
        )
        let convention = graphResolver.namingConvention(for: resources.modelType)
        let resolvedGraph = ParameterResolver().resolve(graph: graph, convention: convention)

        // 4. Compile IR → MetalCompiledModel (includes Metal pipeline compilation)
        // The current decode-specialized kernels benchmark best with the
        // aggressive optimizer once fused SwiGLU uses the input=2048 family.
        let compiler = MetalInferenceCompiler(optimizer: AggressiveOptimizer())
        var compiledModel = try compiler.compile(
            graph: resolvedGraph,
            hiddenSize: resources.config.hiddenSize,
            intermediateSize: resources.config.intermediateSize,
            vocabSize: resources.config.vocabSize,
            stafWeightStore: weightStore,
            device: device
        )

        // 4b. Compile prefill plan (sequence graph — step count independent of token count)
        let prefillPlan = try compiler.compilePrefill(
            graph: resolvedGraph,
            hiddenSize: resources.config.hiddenSize,
            intermediateSize: resources.config.intermediateSize,
            vocabSize: resources.config.vocabSize,
            maximumSequenceLength: 4096,
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

        let compileTime = CFAbsoluteTimeGetCurrent() - compileStart
        print("[ModelBundleLoader] compiled: \(compiledModel.fusedEntryCount) dispatches, prefill \(prefillPlan.stepCount) steps [\(String(format: "%.3f", compileTime))s]")

        // 5. Assemble ModelContainer
        let inferenceModel = try MetalInferenceModel(plan: compiledModel, device: device)

        var modelConfig = ModelConfiguration(
            name: resources.modelType,
            inputCapabilities: resources.inputCapabilities,
            executionCapabilities: ModelExecutionCapabilities(
                supportsTextGeneration: true,
                supportsPromptStateReuse: true,
                supportsImagePromptPreparation:
                    resources.inputCapabilities.supportsImages &&
                    (
                        (visionRuntime != nil &&
                         QwenVisionSupport.supportsImagePromptPreparation(
                            vision: resources.visionConfiguration
                         ))
                        || (gemma4Runtime != nil &&
                            Gemma4Support.supportsImagePromptPreparation(
                                vision: resources.visionConfiguration
                            ))
                    ),
                supportsImageExecution:
                    resources.inputCapabilities.supportsImages &&
                    (
                        (visionRuntime != nil &&
                         QwenVisionSupport.supportsImagePromptPreparation(
                            vision: resources.visionConfiguration
                         ))
                        || (gemma4Runtime != nil &&
                            Gemma4Support.supportsImagePromptPreparation(
                                vision: resources.visionConfiguration
                            ))
                    ),
                supportsVideoPromptPreparation:
                    visionRuntime != nil &&
                    resources.inputCapabilities.supportsVideo &&
                    QwenVisionSupport.supportsVideoPromptPreparation(
                        vision: resources.visionConfiguration
                    ),
                supportsVideoExecution:
                    visionRuntime != nil &&
                    resources.inputCapabilities.supportsVideo &&
                    QwenVisionSupport.supportsVideoPromptPreparation(
                        vision: resources.visionConfiguration
                    )
            ),
            vision: resources.visionConfiguration
        )
        if let eosId = tokenizer.eosTokenId {
            modelConfig.eosTokenIds.insert(eosId)
        }

        let totalTime = CFAbsoluteTimeGetCurrent() - startTime
        print("[ModelBundleLoader] ready: \(modelConfig.name) [\(String(format: "%.3f", totalTime))s]")

        return ModelContainer(
            inferenceModel: inferenceModel,
            tokenizer: tokenizer,
            configuration: modelConfig,
            chatTemplate: resources.chatTemplate,
            visionRuntime: visionRuntime,
            gemma4Runtime: gemma4Runtime
        )
    }
}

// MARK: - Errors

public enum ModelBundleLoaderError: Error, CustomStringConvertible {
    /// No Metal GPU device was available for model compilation or execution.
    case noMetalDevice
    /// The model directory did not contain any `.safetensors` files.
    case noSafetensorsFiles(String)
    /// The model bundle metadata was missing required fields or was otherwise invalid.
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
