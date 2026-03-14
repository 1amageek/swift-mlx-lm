@preconcurrency import MLX

// MARK: - ModelBundle

/// Unified abstraction over model distribution formats.
///
/// A bundle provides everything needed to construct a `ModelContext`:
/// configuration, architecture detection, weights, tokenizer, and prompt formatting.
///
/// Primary implementation: `HFDirectoryBundle` (config.json + safetensors + tokenizer.json).
public protocol ModelBundle: Sendable {

    /// Extract model configuration from the bundle.
    func configuration() throws -> ModelConfig

    /// Detect architecture from bundle contents.
    func architecture() throws -> DetectedArchitecture

    /// Load weights as a flat dictionary with optional quantization metadata.
    ///
    /// Keys use MLX parameter paths (e.g., `model.layers.0.self_attn.q_proj.weight`).
    func loadWeights() throws -> WeightManifest

    /// Create a tokenizer from the bundle contents.
    func tokenizer() throws -> any Tokenizer

    /// Jinja2 chat template string. Nil if not available.
    func chatTemplate() throws -> String?

    /// Vision preprocessor configuration. Nil for text-only models.
    func visionConfig() throws -> VisionPreprocessorConfig?
}

// MARK: - WeightManifest

/// Encapsulates loaded weights with optional per-tensor quantization metadata.
///
/// Produced by `ModelBundle.loadWeights()` and consumed by the weight binder.
/// Weights are typically fp16/bf16 with empty quantization info,
/// or pre-quantized by mlx-community with per-tensor quantization specs.
public struct WeightManifest: @unchecked Sendable {

    /// Flat dictionary of weight arrays keyed by MLX parameter paths.
    public let weights: [String: MLXArray]

    /// Per-tensor quantization metadata (groupSize, bits).
    /// Empty for dense (unquantized) weights.
    public let quantizationInfo: [String: QuantizationSpec]

    /// Tensor name mapping applied during load (original → MLX path).
    /// For diagnostics and debugging.
    public let nameMapping: [String: String]

    public init(
        weights: [String: MLXArray],
        quantizationInfo: [String: QuantizationSpec] = [:],
        nameMapping: [String: String] = [:]
    ) {
        self.weights = weights
        self.quantizationInfo = quantizationInfo
        self.nameMapping = nameMapping
    }
}

/// Quantization parameters for a single tensor.
public struct QuantizationSpec: Sendable, Equatable {
    public let groupSize: Int
    public let bits: Int

    public init(groupSize: Int, bits: Int) {
        self.groupSize = groupSize
        self.bits = bits
    }
}

// MARK: - VisionPreprocessorConfig

/// Vision preprocessor configuration extracted from `preprocessor_config.json`.
///
/// Contains image/video preprocessing parameters needed for VLM models.
/// Nil for text-only models.
public struct VisionPreprocessorConfig: Sendable {
    public let imageSize: Int
    public let patchSize: Int
    public let meanValues: [Float]
    public let stdValues: [Float]
    public let rescaleFactor: Float?
    public let maxPixels: Int?
    public let minPixels: Int?

    public init(
        imageSize: Int,
        patchSize: Int,
        meanValues: [Float] = [0.48145466, 0.4578275, 0.40821073],
        stdValues: [Float] = [0.26862954, 0.26130258, 0.27577711],
        rescaleFactor: Float? = nil,
        maxPixels: Int? = nil,
        minPixels: Int? = nil
    ) {
        self.imageSize = imageSize
        self.patchSize = patchSize
        self.meanValues = meanValues
        self.stdValues = stdValues
        self.rescaleFactor = rescaleFactor
        self.maxPixels = maxPixels
        self.minPixels = minPixels
    }
}

// MARK: - ModelFormat

/// Format of a model distribution.
public enum ModelFormat: Sendable, Equatable {
    /// Auto-detect from repository or directory contents.
    case auto
    /// HuggingFace directory format (config.json + safetensors).
    case safetensors
}
