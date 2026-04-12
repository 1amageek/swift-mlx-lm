import Foundation
import ModelDeclarations

struct SentenceTransformerMetadata: Sendable {
    enum PoolingStrategy: Sendable {
        case mean
        case cls
        case max
        case lastToken
    }

    struct Pooling: Sendable {
        let strategy: PoolingStrategy
        let includePrompt: Bool
    }

    enum DenseActivation: Sendable {
        case identity
        case tanh
        case relu
        case gelu

        init(name: String) throws {
            let normalized = name.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
            if normalized.contains("identity") {
                self = .identity
            } else if normalized.contains("tanh") {
                self = .tanh
            } else if normalized.hasSuffix(".relu") || normalized.contains("relu") {
                self = .relu
            } else if normalized.contains("gelu") {
                self = .gelu
            } else {
                throw ModelBundleLoaderError.invalidConfig(
                    "Unsupported sentence-transformers dense activation: \(name)"
                )
            }
        }
    }

    struct DenseLayerSpec: Sendable {
        let weightName: String
        let biasName: String
        let inputDimension: Int?
        let outputDimension: Int?
        let activation: DenseActivation
    }

    let prompts: [String: String]
    let defaultPromptName: String?
    let similarityFunctionName: String?
    let pooling: Pooling
    let denseLayers: [DenseLayerSpec]
    let postprocessors: [SentenceEmbeddingPostprocessor]

    var availablePromptNames: [String] {
        prompts.keys.sorted()
    }

    static func load(from resources: ModelBundleResources) throws -> SentenceTransformerMetadata {
        guard let modulesData = resources.modulesData else {
            throw ModelBundleLoaderError.invalidConfig(
                "modules.json is required for sentence-transformers embeddings"
            )
        }

        let modules = try parseModules(from: modulesData)
        let promptConfig = try parsePromptConfig(from: resources.sentenceTransformersConfigData)

        var sawTransformer = false
        var pooling: Pooling?
        var denseLayers: [DenseLayerSpec] = []
        var denseIndex = 0
        var postprocessors: [SentenceEmbeddingPostprocessor] = []

        for module in modules {
            switch module.kind {
            case .transformer:
                sawTransformer = true
            case .pooling:
                pooling = try loadPooling(
                    from: resources.directory,
                    modelType: resources.modelType,
                    path: module.path
                )
            case .dense:
                denseLayers.append(
                    try loadDenseLayerSpec(
                        from: resources.directory,
                        modelType: resources.modelType,
                        path: module.path,
                        denseIndex: denseIndex
                    )
                )
                denseIndex += 1
            case .normalize:
                if postprocessors.contains(.l2Normalize) == false {
                    postprocessors.append(.l2Normalize)
                }
            }
        }

        guard sawTransformer else {
            throw ModelBundleLoaderError.invalidConfig(
                "sentence-transformers bundle is missing a Transformer module"
            )
        }
        guard let pooling else {
            throw ModelBundleLoaderError.invalidConfig(
                "sentence-transformers bundle is missing a Pooling module"
            )
        }

        return SentenceTransformerMetadata(
            prompts: promptConfig.prompts,
            defaultPromptName: promptConfig.defaultPromptName,
            similarityFunctionName: promptConfig.similarityFunctionName,
            pooling: pooling,
            denseLayers: denseLayers,
            postprocessors: postprocessors
        )
    }

    private struct PromptConfig: Sendable {
        let prompts: [String: String]
        let defaultPromptName: String?
        let similarityFunctionName: String?
    }

    private enum ModuleKind {
        case transformer
        case pooling
        case dense
        case normalize
    }

    private struct ModuleDescriptor {
        let idx: Int
        let path: String
        let kind: ModuleKind
    }

    private static func parsePromptConfig(from data: Data?) throws -> PromptConfig {
        guard let data else {
            return PromptConfig(prompts: [:], defaultPromptName: nil, similarityFunctionName: nil)
        }
        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw ModelBundleLoaderError.invalidConfig(
                "config_sentence_transformers.json is not a JSON object"
            )
        }
        let prompts = json["prompts"] as? [String: String] ?? [:]
        return PromptConfig(
            prompts: prompts,
            defaultPromptName: json["default_prompt_name"] as? String,
            similarityFunctionName: json["similarity_fn_name"] as? String
        )
    }

    private static func parseModules(from data: Data) throws -> [ModuleDescriptor] {
        let rawObject = try JSONSerialization.jsonObject(with: data)

        let rawModules: [Any]
        if let array = rawObject as? [Any] {
            rawModules = array
        } else if let json = rawObject as? [String: Any] {
            if let modules = json["modules"] as? [Any] {
                rawModules = modules
            } else {
                rawModules = Array(json.values)
            }
        } else {
            throw ModelBundleLoaderError.invalidConfig("modules.json has an unsupported shape")
        }

        let modules = try rawModules.compactMap(parseModule)
        guard modules.isEmpty == false else {
            throw ModelBundleLoaderError.invalidConfig("modules.json does not contain any modules")
        }
        return modules.sorted { $0.idx < $1.idx }
    }

    private static func parseModule(_ rawValue: Any) throws -> ModuleDescriptor? {
        guard let json = rawValue as? [String: Any],
              let idx = json["idx"] as? Int,
              let type = json["type"] as? String else {
            return nil
        }

        let path = json["path"] as? String ?? ""
        let normalizedType = type.lowercased()
        let kind: ModuleKind
        if normalizedType.hasSuffix(".transformer") {
            kind = .transformer
        } else if normalizedType.hasSuffix(".pooling") {
            kind = .pooling
        } else if normalizedType.hasSuffix(".dense") {
            kind = .dense
        } else if normalizedType.hasSuffix(".normalize") {
            kind = .normalize
        } else {
            throw ModelBundleLoaderError.invalidConfig(
                "Unsupported sentence-transformers module type: \(type)"
            )
        }

        return ModuleDescriptor(idx: idx, path: path, kind: kind)
    }

    private static func loadPooling(
        from directory: URL,
        modelType: String,
        path: String
    ) throws -> Pooling {
        let configURL = directory
            .appendingPathComponent(path)
            .appendingPathComponent("config.json")
        guard FileManager.default.fileExists(atPath: configURL.path) else {
            guard let defaults = SentenceTransformerModuleDefaultsResolver.poolingDefaults(
                modelType: modelType,
                modulePath: path
            ) else {
                throw ModelBundleLoaderError.invalidConfig(
                    "Missing pooling config: \(configURL.path)"
                )
            }
            return Pooling(
                strategy: poolingStrategy(from: defaults.strategy),
                includePrompt: defaults.includePrompt
            )
        }

        let data = try Data(contentsOf: configURL)
        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw ModelBundleLoaderError.invalidConfig(
                "Pooling config is not a JSON object: \(configURL.lastPathComponent)"
            )
        }

        let includePrompt = json["include_prompt"] as? Bool ?? true
        let meanTokens = json["pooling_mode_mean_tokens"] as? Bool ?? false
        let clsToken = json["pooling_mode_cls_token"] as? Bool ?? false
        let maxTokens = json["pooling_mode_max_tokens"] as? Bool ?? false
        let lastToken = json["pooling_mode_lasttoken"] as? Bool ?? false
        let meanSqrtLenTokens = json["pooling_mode_mean_sqrt_len_tokens"] as? Bool ?? false
        let weightedMeanTokens = json["pooling_mode_weightedmean_tokens"] as? Bool ?? false

        if meanSqrtLenTokens || weightedMeanTokens {
            throw ModelBundleLoaderError.invalidConfig(
                "Unsupported sentence-transformers pooling mode in \(configURL.path)"
            )
        }

        let enabledStrategies = [
            (meanTokens, PoolingStrategy.mean),
            (clsToken, PoolingStrategy.cls),
            (maxTokens, PoolingStrategy.max),
            (lastToken, PoolingStrategy.lastToken),
        ].filter { $0.0 }

        guard enabledStrategies.count == 1,
              let strategy = enabledStrategies.first?.1 else {
            throw ModelBundleLoaderError.invalidConfig(
                "Exactly one sentence-transformers pooling mode must be enabled in \(configURL.path)"
            )
        }

        return Pooling(strategy: strategy, includePrompt: includePrompt)
    }

    private static func loadDenseLayerSpec(
        from directory: URL,
        modelType: String,
        path: String,
        denseIndex: Int
    ) throws -> DenseLayerSpec {
        let configURL = directory
            .appendingPathComponent(path)
            .appendingPathComponent("config.json")
        guard FileManager.default.fileExists(atPath: configURL.path) else {
            guard let defaults = SentenceTransformerModuleDefaultsResolver.denseDefaults(
                modelType: modelType,
                modulePath: path,
                denseIndex: denseIndex
            ) else {
                throw ModelBundleLoaderError.invalidConfig(
                    "Missing dense config: \(configURL.path)"
                )
            }
            return DenseLayerSpec(
                weightName: "dense.\(denseIndex).weight",
                biasName: "dense.\(denseIndex).bias",
                inputDimension: nil,
                outputDimension: nil,
                activation: try DenseActivation(name: defaults.activationFunctionName)
            )
        }

        let data = try Data(contentsOf: configURL)
        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw ModelBundleLoaderError.invalidConfig(
                "Dense config is not a JSON object: \(configURL.lastPathComponent)"
            )
        }

        let activationName = json["activation_function"] as? String
            ?? "torch.nn.modules.linear.Identity"
        return DenseLayerSpec(
            weightName: "dense.\(denseIndex).weight",
            biasName: "dense.\(denseIndex).bias",
            inputDimension: json["in_features"] as? Int,
            outputDimension: json["out_features"] as? Int,
            activation: try DenseActivation(name: activationName)
        )
    }

    private static func poolingStrategy(
        from strategy: SentenceTransformerModuleDefaultsResolver.PoolingStrategy
    ) -> PoolingStrategy {
        switch strategy {
        case .mean:
            return .mean
        case .cls:
            return .cls
        case .max:
            return .max
        case .lastToken:
            return .lastToken
        }
    }
}
