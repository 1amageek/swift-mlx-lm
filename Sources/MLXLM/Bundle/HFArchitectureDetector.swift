import Foundation

/// Detects `DetectedArchitecture` from HuggingFace config.json `model_type`.
///
/// Unlike `GGUFArchitectureDetector` which inspects tensor name patterns,
/// this detector uses the explicit `model_type` string from config.json.
/// For unknown model types, falls back to inspecting config fields
/// (e.g., `layer_types` for hybrid attention).
public struct HFArchitectureDetector: Sendable {

    public init() {}

    /// Detect architecture from `model_type` string and config fields.
    ///
    /// - Parameters:
    ///   - modelType: The `model_type` value from config.json (e.g., "llama", "qwen3_5")
    ///   - config: The full config dictionary for field-level inspection
    /// - Returns: Detected architecture variant
    public func detect(modelType: String, config: [String: Any] = [:]) -> DetectedArchitecture {
        // Check explicit model_type mapping first
        if let arch = modelTypeMapping[modelType.lowercased()] {
            return arch
        }

        // For VLM configs, check text_config's model_type
        if let textConfig = config["text_config"] as? [String: Any],
           let textModelType = textConfig["model_type"] as? String {
            if let arch = textModelTypeMapping[textModelType.lowercased()] {
                return arch
            }
        }

        // Fallback: inspect config fields
        return detectFromFields(config)
    }

    /// Detect architecture from config.json data directly.
    ///
    /// Convenience method that parses JSON and extracts model_type.
    public func detect(from data: Data) throws -> DetectedArchitecture {
        guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw HFConfigError.invalidJSON
        }
        guard let modelType = root["model_type"] as? String else {
            throw HFConfigError.missingField("model_type")
        }
        return detect(modelType: modelType, config: root)
    }

    // MARK: - Model Type Mappings

    /// Maps top-level `model_type` → architecture.
    private let modelTypeMapping: [String: DetectedArchitecture] = [
        // Standard transformer family
        "llama": .transformer,
        "qwen2": .transformer,
        "qwen2_moe": .moe,
        "mistral": .transformer,
        "gemma": .transformer,
        "gemma2": .transformer,
        "phi": .transformer,
        "phi3": .transformer,
        "starcoder2": .transformer,
        "gpt_neox": .transformer,
        "internlm2": .transformer,
        "deepseek": .transformer,
        "deepseek_v2": .moe,
        "yi": .transformer,
        "baichuan": .transformer,
        "chatglm": .transformer,

        // Parallel attention + MLP family
        "cohere": .parallelAttentionMLP,
        "command-r": .parallelAttentionMLP,

        // MoE family
        "mixtral": .moe,
        "arctic": .moe,
        "dbrx": .moe,

        // Hybrid DeltaNet / attention family
        "qwen3_5": .hybridDeltaNetAttention,
    ]

    /// Maps `text_config.model_type` for VLM wrappers.
    private let textModelTypeMapping: [String: DetectedArchitecture] = [
        "qwen3_5_text": .hybridDeltaNetAttention,
        "qwen2_vl": .transformer,
    ]

    /// Fallback detection from config field patterns.
    private func detectFromFields(_ config: [String: Any]) -> DetectedArchitecture {
        let inspectConfig: [String: Any]
        if let textConfig = config["text_config"] as? [String: Any] {
            inspectConfig = textConfig
        } else {
            inspectConfig = config
        }

        // Check for hybrid attention indicators
        if let layerTypes = inspectConfig["layer_types"] as? [String] {
            if layerTypes.contains("linear_attention") {
                return .hybridDeltaNetAttention
            }
        }
        if inspectConfig["full_attention_interval"] != nil
            && inspectConfig["linear_num_value_heads"] != nil {
            return .hybridDeltaNetAttention
        }

        // Check for MoE indicators
        if inspectConfig["num_local_experts"] != nil {
            return .moe
        }

        // Check for parallel attention + MLP indicators
        if let _ = inspectConfig["use_qk_norm"] as? Bool,
           inspectConfig["ffn_norm"] == nil {
            return .parallelAttentionMLP
        }

        // Default: standard transformer
        return .transformer
    }
}
