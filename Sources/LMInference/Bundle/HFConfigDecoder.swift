import Foundation
import SwiftLM

/// Decodes HuggingFace config.json into a format-agnostic `ModelConfig`.
///
/// Handles two structural patterns:
/// - **Flat config** (Llama, Qwen2, Phi3, Mistral): All fields at top level
/// - **Nested config** (Qwen3.5 VLM): Text model fields inside `text_config`
///
/// Unlike mlx-swift-lm's per-model `Configuration` structs, this decoder
/// produces a single universal `ModelConfig` because the IR-based pipeline
/// doesn't need model-specific Swift types.
public struct HFConfigDecoder: Sendable {

    public init() {}

    /// Decode config.json data into ModelConfig.
    ///
    /// - Parameter data: Raw JSON data from config.json
    /// - Returns: Populated ModelConfig
    /// - Throws: `HFConfigError` if required fields are missing
    public func decode(from data: Data) throws -> ModelConfig {
        guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw HFConfigError.invalidJSON
        }

        // Determine the text config dict: either nested `text_config` or the root itself
        let textConfig: [String: Any]
        if let nested = root["text_config"] as? [String: Any] {
            textConfig = nested
        } else {
            textConfig = root
        }

        // Core dimensions (required)
        let hiddenSize = try requireInt(textConfig, "hidden_size")
        let layerCount = try requireInt(textConfig, "num_hidden_layers")
        let attentionHeads = try requireInt(textConfig, "num_attention_heads")
        let intermediateSize = try requireInt(textConfig, "intermediate_size")
        let vocabSize = try requireInt(textConfig, "vocab_size")

        // Attention
        let kvHeads = optionalInt(textConfig, "num_key_value_heads") ?? attentionHeads
        let headDim = optionalInt(textConfig, "head_dim") ?? (hiddenSize / attentionHeads)
        let attentionBias = optionalBool(textConfig, "attention_bias") ?? false
        let mlpBias = optionalBool(textConfig, "mlp_bias") ?? false

        // Normalization
        let normEps: Float
        let normKind: ModelConfig.NormKind
        if let rmsEps = optionalFloat(textConfig, "rms_norm_eps") {
            normEps = rmsEps
            normKind = .rmsNorm
        } else if let layerNormEps = optionalFloat(textConfig, "layer_norm_eps")
                    ?? optionalFloat(textConfig, "layer_norm_epsilon") {
            normEps = layerNormEps
            normKind = .layerNorm
        } else if let eps = optionalFloat(textConfig, "norm_eps") {
            // LFM2 family uses "norm_eps" for RMS norm
            normEps = eps
            normKind = .rmsNorm
        } else {
            normEps = 1e-5
            normKind = .rmsNorm
        }

        // RoPE
        let ropeConfig = extractRoPEConfig(from: textConfig)

        // Output
        let tiedEmbeddings = optionalBool(textConfig, "tie_word_embeddings")
            ?? optionalBool(root, "tie_word_embeddings")
            ?? optionalBool(textConfig, "tie_embedding")
            ?? optionalBool(root, "tie_embedding")
            ?? false

        // MoE
        let expertCount = optionalInt(textConfig, "num_local_experts")
        let expertsPerToken = optionalInt(textConfig, "num_experts_per_tok")

        // QK norm (Cohere-style parallel attention)
        let qkNorm = optionalBool(textConfig, "use_qk_norm") ?? false

        // Hybrid DeltaNet / Attention
        let fullAttentionInterval = optionalInt(textConfig, "full_attention_interval")
        let layerTypes = optionalStringArray(textConfig, "layer_types")

        // DeltaNet-specific (Qwen3.5 linear_attention parameters)
        let ssmNumHeads = optionalInt(textConfig, "linear_num_value_heads")
        let ssmGroupCount = optionalInt(textConfig, "linear_num_key_heads")
        let ssmKeyHeadDim = optionalInt(textConfig, "linear_key_head_dim")
        let ssmValueHeadDim = optionalInt(textConfig, "linear_value_head_dim")
        let convKernelSize = optionalInt(textConfig, "linear_conv_kernel_dim")

        // Short convolution (LFM2 family)
        let convLCache = optionalInt(textConfig, "conv_L_cache")

        // Sliding window
        let slidingWindow = optionalInt(textConfig, "sliding_window")

        // M-RoPE (from rope_parameters or top-level)
        let mropeAxes = extractMRoPEAxes(from: textConfig)

        // RoPE dimension: explicit from config, or derived from partial_rotary_factor,
        // or defaults to headDim (full rotary). This matches GGUF's ropeDimensionCount.
        let ropeDimension: Int
        if let dim = ropeConfig.dimension {
            ropeDimension = dim
        } else if let factor = ropeConfig.partialRotaryFactor {
            ropeDimension = Int(Float(headDim) * factor)
        } else {
            ropeDimension = headDim
        }

        return ModelConfig(
            hiddenSize: hiddenSize,
            layerCount: layerCount,
            intermediateSize: intermediateSize,
            vocabSize: vocabSize,
            attentionHeads: attentionHeads,
            kvHeads: kvHeads,
            headDim: headDim,
            attentionBias: attentionBias,
            mlpBias: mlpBias,
            normEps: normEps,
            normKind: normKind,
            ropeTheta: ropeConfig.theta,
            ropeDimension: ropeDimension,
            ropeScaling: ropeConfig.scaling,
            tiedEmbeddings: tiedEmbeddings,
            expertCount: expertCount,
            expertsPerToken: expertsPerToken,
            qkNorm: qkNorm,
            fullAttentionInterval: fullAttentionInterval,
            ssmNumHeads: ssmNumHeads,
            ssmGroupCount: ssmGroupCount,
            ssmKeyHeadDim: ssmKeyHeadDim,
            ssmValueHeadDim: ssmValueHeadDim,
            convKernelSize: convKernelSize,
            convLCache: convLCache,
            partialRotaryFactor: ropeConfig.partialRotaryFactor,
            slidingWindow: slidingWindow,
            layerTypes: layerTypes,
            mropeAxes: mropeAxes
        )
    }

    /// Extract the `model_type` string from config.json data.
    ///
    /// For nested VLM configs, returns the top-level `model_type` (e.g., "qwen3_5"),
    /// not the nested `text_config.model_type`.
    public func modelType(from data: Data) throws -> String {
        guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw HFConfigError.invalidJSON
        }
        guard let modelType = root["model_type"] as? String else {
            throw HFConfigError.missingField("model_type")
        }
        return modelType
    }

    // MARK: - RoPE Extraction

    private struct RoPEConfig {
        let theta: Float
        let dimension: Int?
        let scaling: RoPEScaling?
        let partialRotaryFactor: Float?
    }

    private func extractRoPEConfig(from config: [String: Any]) -> RoPEConfig {
        // Check nested rope_parameters first (Qwen3.5 style)
        if let ropeParams = config["rope_parameters"] as? [String: Any] {
            let theta = optionalFloat(ropeParams, "rope_theta")
                ?? optionalFloat(config, "rope_theta")
                ?? 10000.0
            let partialFactor = optionalFloat(ropeParams, "partial_rotary_factor")
                ?? optionalFloat(config, "partial_rotary_factor")
            let scaling = extractRoPEScaling(from: ropeParams)
            return RoPEConfig(theta: theta, dimension: nil, scaling: scaling,
                              partialRotaryFactor: partialFactor)
        }

        // Flat config
        let theta = optionalFloat(config, "rope_theta") ?? 10000.0
        let partialFactor = optionalFloat(config, "partial_rotary_factor")
        let scaling = extractRoPEScaling(from: config)
        return RoPEConfig(theta: theta, dimension: nil, scaling: scaling,
                          partialRotaryFactor: partialFactor)
    }

    private func extractRoPEScaling(from config: [String: Any]) -> RoPEScaling? {
        guard let scalingDict = config["rope_scaling"] as? [String: Any] else {
            return nil
        }
        guard let typeStr = scalingDict["rope_type"] as? String
                ?? scalingDict["type"] as? String else {
            return nil
        }

        let kind: RoPEScalingKind
        switch typeStr.lowercased() {
        case "linear": kind = .linear
        case "dynamic": kind = .dynamic
        case "yarn": kind = .yarn
        case "su", "longrope": kind = .custom("su")
        case "default": return nil
        default: return nil
        }

        let factor = optionalFloat(scalingDict, "factor") ?? 1.0
        let origMaxPos = optionalInt(scalingDict, "original_max_position_embeddings")
        return RoPEScaling(kind: kind, factor: factor, originalMaxPositions: origMaxPos)
    }

    // MARK: - M-RoPE Extraction

    private func extractMRoPEAxes(from config: [String: Any]) -> MRoPEAxes? {
        // Qwen3.5: nested in rope_parameters
        if let ropeParams = config["rope_parameters"] as? [String: Any] {
            if let section = ropeParams["mrope_section"] as? [Int], !section.isEmpty {
                let interleaved = optionalBool(ropeParams, "mrope_interleaved") ?? false
                return MRoPEAxes(sections: section, interleaved: interleaved)
            }
        }

        // Qwen2-VL: nested in rope_scaling (e.g., { "type": "mrope", "mrope_section": [16, 24, 24] })
        if let ropeScaling = config["rope_scaling"] as? [String: Any] {
            if let section = ropeScaling["mrope_section"] as? [Int], !section.isEmpty {
                let interleaved = optionalBool(ropeScaling, "mrope_interleaved") ?? false
                return MRoPEAxes(sections: section, interleaved: interleaved)
            }
        }

        // Top-level mrope_section (fallback)
        if let section = config["mrope_section"] as? [Int], !section.isEmpty {
            return MRoPEAxes(sections: section, interleaved: false)
        }

        return nil
    }

    // MARK: - JSON Helpers

    private func requireInt(_ dict: [String: Any], _ key: String) throws -> Int {
        if let v = dict[key] as? Int { return v }
        if let v = dict[key] as? Double { return Int(v) }
        if let v = dict[key] as? NSNumber { return v.intValue }
        throw HFConfigError.missingField(key)
    }

    private func optionalInt(_ dict: [String: Any], _ key: String) -> Int? {
        if let v = dict[key] as? Int { return v }
        if let v = dict[key] as? Double { return Int(v) }
        if let v = dict[key] as? NSNumber, !(dict[key] is Bool) { return v.intValue }
        return nil
    }

    private func optionalFloat(_ dict: [String: Any], _ key: String) -> Float? {
        if let v = dict[key] as? Float { return v }
        if let v = dict[key] as? Double { return Float(v) }
        if let v = dict[key] as? NSNumber, !(dict[key] is Bool) { return v.floatValue }
        return nil
    }

    private func optionalBool(_ dict: [String: Any], _ key: String) -> Bool? {
        dict[key] as? Bool
    }

    private func optionalStringArray(_ dict: [String: Any], _ key: String) -> [String]? {
        dict[key] as? [String]
    }
}

// MARK: - Errors

/// Errors during HuggingFace config.json decoding.
public enum HFConfigError: Error, CustomStringConvertible {
    case invalidJSON
    case missingField(String)
    case invalidValue(field: String, message: String)

    public var description: String {
        switch self {
        case .invalidJSON:
            return "config.json is not valid JSON"
        case .missingField(let field):
            return "Required field '\(field)' missing in config.json"
        case .invalidValue(let field, let message):
            return "Invalid value for '\(field)' in config.json: \(message)"
        }
    }
}
