import Foundation
import MLX
import MLXNN

/// Configuration for LoRA or DoRA adapter application.
///
/// Compatible with `adapter_config.json` produced by mlx-lm training.
public struct LoRAConfiguration: Sendable, Codable {

    public enum FineTuneType: String, Sendable, Codable {
        case lora
        case dora
    }

    public struct LoRAParameters: Sendable, Codable {

        public let rank: Int
        public let scale: Float
        public let keys: [String]?

        public init(rank: Int = 8, scale: Float = 20.0, keys: [String]? = nil) {
            self.rank = rank
            self.scale = scale
            self.keys = keys
        }
    }

    public let numLayers: Int
    public let fineTuneType: FineTuneType
    public let loraParameters: LoRAParameters

    public init(
        numLayers: Int = 16,
        fineTuneType: FineTuneType = .lora,
        loraParameters: LoRAParameters = .init()
    ) {
        self.numLayers = numLayers
        self.fineTuneType = fineTuneType
        self.loraParameters = loraParameters
    }

    enum CodingKeys: String, CodingKey {
        case numLayers = "num_layers"
        case fineTuneType = "fine_tune_type"
        case loraParameters = "lora_parameters"
    }
}

// MARK: - Errors

public enum LoRAError: Error {
    case incompatibleModelType
}

// MARK: - LoRAContainer

/// Manages LoRA/DoRA adapter lifecycle: load, fuse, and unload.
public struct LoRAContainer: @unchecked Sendable {

    public let configuration: LoRAConfiguration
    public let parameters: ModuleParameters

    public init(
        configuration: LoRAConfiguration,
        parameters: consuming ModuleParameters
    ) {
        self.configuration = configuration
        self.parameters = parameters
        eval(self.parameters)
    }

    /// Create a container by applying LoRA adapters to a model and capturing trainable parameters.
    public static func from(
        model: any LanguageModel,
        configuration: LoRAConfiguration = .init()
    ) throws -> LoRAContainer {
        guard let lora = model as? LoRAModel else {
            throw LoRAError.incompatibleModelType
        }

        model.freeze()
        let layers = lora.loraLayers.suffix(configuration.numLayers)
        let keys = configuration.loraParameters.keys ?? lora.loraDefaultKeys
        replaceLayers(layers: layers, keys: keys) { (layer: Module) in
            createReplacementLayer(target: layer, configuration: configuration)
        }

        return LoRAContainer(
            configuration: configuration,
            parameters: model.trainableParameters()
        )
    }

    /// Load a container from a directory containing `adapter_config.json` and `adapters.safetensors`.
    public static func from(directory: URL) throws -> LoRAContainer {
        let configURL = directory.appending(component: "adapter_config.json")
        let configData = try Data(contentsOf: configURL)
        let configuration = try JSONDecoder().decode(LoRAConfiguration.self, from: configData)

        let weightsURL = directory.appending(component: "adapters.safetensors")
        let weights = try MLX.loadArrays(url: weightsURL)
        let parameters = ModuleParameters.unflattened(weights)

        return LoRAContainer(configuration: configuration, parameters: parameters)
    }

    /// Apply adapter layers to the model and load adapter weights.
    public func load(into model: any LanguageModel) throws {
        guard let lora = model as? LoRAModel else {
            throw LoRAError.incompatibleModelType
        }

        let layers = lora.loraLayers.suffix(configuration.numLayers)
        let keys = configuration.loraParameters.keys ?? lora.loraDefaultKeys
        replaceLayers(layers: layers, keys: keys) { (layer: Module) in
            createReplacementLayer(target: layer, configuration: configuration)
        }

        try model.update(parameters: parameters, verify: .noUnusedKeys)
    }

    /// Permanently fuse adapter weights into the model's base layers.
    public func fuse(with model: any LanguageModel) throws {
        guard let lora = model as? LoRAModel else {
            throw LoRAError.incompatibleModelType
        }

        try load(into: model)
        let layers = lora.loraLayers.suffix(configuration.numLayers)
        let keys = configuration.loraParameters.keys ?? lora.loraDefaultKeys
        replaceLayers(layers: layers, keys: keys) { (loraLayer: LoRALayer) in
            loraLayer.fused()
        }
    }

    /// Remove adapter layers and restore the model to its original form.
    public func unload(from model: any LanguageModel) {
        guard let lora = model as? LoRAModel else {
            return
        }

        let layers = lora.loraLayers.suffix(configuration.numLayers)
        let keys = configuration.loraParameters.keys ?? lora.loraDefaultKeys
        replaceLayers(layers: layers, keys: keys) { (loraLayer: LoRALayer) in
            loraLayer.reverted()
        }
    }
}

// MARK: - Private Helpers

/// Create an adapter replacement layer for a given module.
private func createReplacementLayer(
    target: Module,
    configuration: LoRAConfiguration
) -> LoRALayer? {
    switch (target, configuration.fineTuneType) {
    case (let linear as Linear, .lora):
        return LoRALinear.from(
            linear: linear,
            rank: configuration.loraParameters.rank,
            scale: configuration.loraParameters.scale
        )
    case (let linear as Linear, .dora):
        return DoRALinear.from(
            linear: linear,
            rank: configuration.loraParameters.rank,
            scale: configuration.loraParameters.scale
        )
    default:
        return nil
    }
}

/// Traverse model layers and replace matching modules using a transformation closure.
private func replaceLayers<T>(
    layers: ArraySlice<Module>,
    keys: [String],
    transforming transform: (T) -> Module?
) {
    for layer in layers {
        var update: [(String, Module)] = []
        for (key, child) in layer.namedModules() where keys.contains(key) {
            if let child = child as? T, let transformed = transform(child) {
                update.append((key, transformed))
            }
        }

        if !update.isEmpty {
            layer.update(modules: .unflattened(update))
        }
    }
}
