import Foundation
import Hub
@preconcurrency import MLX
import SwiftLM
import Tokenizers

/// ModelBundle implementation for HuggingFace directory format.
///
/// Loads from a local directory containing:
/// - `config.json` — Architecture definition (required)
/// - `tokenizer.json` — Tokenizer body (required)
/// - `tokenizer_config.json` — Special tokens (required)
/// - `chat_template.jinja` — Prompt formatting (optional, fallback: tokenizer_config.json)
/// - `*.safetensors` — Weights (required, may be sharded)
/// - `model.safetensors.index.json` — Shard index (for sharded weights)
/// - `preprocessor_config.json` — VLM image/video config (optional)
public struct HFDirectoryBundle: ModelBundle {

    private let directory: URL

    /// Cached config.json data to avoid repeated file reads.
    private let configData: Data

    public init(directory: URL) throws {
        self.directory = directory
        let configURL = directory.appendingPathComponent("config.json")
        self.configData = try Data(contentsOf: configURL)
    }

    // MARK: - ModelBundle

    public func configuration() throws -> ModelConfig {
        let decoder = HFConfigDecoder()
        return try decoder.decode(from: configData)
    }

    public func architecture() throws -> DetectedArchitecture {
        let detector = HFArchitectureDetector()
        return try detector.detect(from: configData)
    }

    public func loadWeights() throws -> WeightManifest {
        let config = try configuration()
        let mapper = HFTensorNameMapper(tiedEmbeddings: config.tiedEmbeddings)
        let safetensorsURLs = try findSafetensorsFiles()

        var weights: [String: MLXArray] = [:]
        var quantInfo: [String: QuantizationSpec] = [:]
        var nameMap: [String: String] = [:]

        for url in safetensorsURLs {
            let fileWeights = try MLX.loadArrays(url: url)
            for (hfName, array) in fileWeights {
                guard let mlxPath = mapper.mlxPath(for: hfName) else {
                    continue
                }
                nameMap[hfName] = mlxPath
                weights[mlxPath] = array
            }
        }

        // Detect per-layer quantization from config.json
        if let quantConfig = extractQuantizationConfig() {
            for (path, _) in weights where path.hasSuffix(".weight") {
                if let spec = quantConfig.specForPath(path) {
                    quantInfo[path] = spec
                }
            }
        }

        return WeightManifest(
            weights: weights,
            quantizationInfo: quantInfo,
            nameMapping: nameMap
        )
    }

    public func tokenizer() throws -> any Tokenizer {
        let config = try configuration()
        let hfTokenizer = try loadHFTokenizer()
        return HFTokenizerAdapter(wrapped: hfTokenizer, vocabularySize: config.vocabSize)
    }

    public func chatTemplate() throws -> String? {
        // Priority 1: Standalone chat_template.jinja file
        let jinjaURL = directory.appendingPathComponent("chat_template.jinja")
        if FileManager.default.fileExists(atPath: jinjaURL.path) {
            return try String(contentsOf: jinjaURL, encoding: .utf8)
        }

        // Priority 2: Extract from tokenizer_config.json
        let tokenizerConfigURL = directory.appendingPathComponent("tokenizer_config.json")
        if FileManager.default.fileExists(atPath: tokenizerConfigURL.path) {
            let data = try Data(contentsOf: tokenizerConfigURL)
            if let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
               let template = json["chat_template"] as? String {
                return template
            }
        }

        return nil
    }

    public func visionConfig() throws -> VisionPreprocessorConfig? {
        let preprocessorURL = directory.appendingPathComponent("preprocessor_config.json")
        guard FileManager.default.fileExists(atPath: preprocessorURL.path) else { return nil }

        let data = try Data(contentsOf: preprocessorURL)
        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return nil
        }

        guard let imageSize = json["size"] as? [String: Any],
              let height = imageSize["height"] as? Int ?? imageSize["shortest_edge"] as? Int else {
            return nil
        }

        let patchSize = json["patch_size"] as? Int ?? 14
        let meanValues = (json["image_mean"] as? [Double])?.map(Float.init)
            ?? [0.48145466, 0.4578275, 0.40821073]
        let stdValues = (json["image_std"] as? [Double])?.map(Float.init)
            ?? [0.26862954, 0.26130258, 0.27577711]
        let rescaleFactor = (json["rescale_factor"] as? Double).map(Float.init)
        let maxPixels = json["max_pixels"] as? Int
        let minPixels = json["min_pixels"] as? Int

        return VisionPreprocessorConfig(
            imageSize: height,
            patchSize: patchSize,
            meanValues: meanValues,
            stdValues: stdValues,
            rescaleFactor: rescaleFactor,
            maxPixels: maxPixels,
            minPixels: minPixels
        )
    }

    // MARK: - HF-Specific Accessors

    /// The local directory URL.
    public var directoryURL: URL { directory }

    // MARK: - Private

    /// Find all safetensors files, respecting shard index if present.
    private func findSafetensorsFiles() throws -> [URL] {
        // Check for shard index first
        let indexURL = directory.appendingPathComponent("model.safetensors.index.json")
        if FileManager.default.fileExists(atPath: indexURL.path) {
            let data = try Data(contentsOf: indexURL)
            guard let index = try JSONSerialization.jsonObject(with: data) as? [String: Any],
                  let weightMap = index["weight_map"] as? [String: String] else {
                throw HFBundleError.invalidShardIndex
            }
            // Collect unique shard filenames
            let shardFiles = Set(weightMap.values)
            return shardFiles.sorted().map { directory.appendingPathComponent($0) }
        }

        // Single file
        let singleURL = directory.appendingPathComponent("model.safetensors")
        if FileManager.default.fileExists(atPath: singleURL.path) {
            return [singleURL]
        }

        // Glob for any *.safetensors files
        let contents = try FileManager.default.contentsOfDirectory(
            at: directory, includingPropertiesForKeys: nil)
        let safetensorsFiles = contents
            .filter { $0.pathExtension == "safetensors" }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }

        guard !safetensorsFiles.isEmpty else {
            throw HFBundleError.noWeightFiles(directory: directory.path)
        }

        return safetensorsFiles
    }

    /// Load HF tokenizer from tokenizer.json + tokenizer_config.json.
    private func loadHFTokenizer() throws -> Tokenizers.Tokenizer {
        let tokenizerURL = directory.appendingPathComponent("tokenizer.json")
        let tokenizerConfigURL = directory.appendingPathComponent("tokenizer_config.json")

        guard let tokenizerData = try JSONSerialization.jsonObject(
            with: Data(contentsOf: tokenizerURL)) as? [NSString: Any] else {
            throw HFBundleError.missingFile(
                name: "tokenizer.json", directory: directory.path)
        }
        guard let tokenizerConfig = try JSONSerialization.jsonObject(
            with: Data(contentsOf: tokenizerConfigURL)) as? [NSString: Any] else {
            throw HFBundleError.missingFile(
                name: "tokenizer_config.json", directory: directory.path)
        }

        return try AutoTokenizer.from(
            tokenizerConfig: Config(tokenizerConfig),
            tokenizerData: Config(tokenizerData)
        )
    }

    /// Extract quantization config from config.json (for mlx-community quantized models).
    private func extractQuantizationConfig() -> HFQuantizationConfig? {
        guard let root = try? JSONSerialization.jsonObject(with: configData) as? [String: Any],
              let quantDict = root["quantization"] as? [String: Any] else {
            return nil
        }

        let defaultGroupSize = quantDict["group_size"] as? Int ?? 64
        let defaultBits = quantDict["bits"] as? Int ?? 4

        return HFQuantizationConfig(
            defaultGroupSize: defaultGroupSize,
            defaultBits: defaultBits
        )
    }
}

// MARK: - HFTokenizerAdapter

/// Bridges swift-transformers `Tokenizers.Tokenizer` to our `Tokenizer` protocol.
struct HFTokenizerAdapter: LMInference.Tokenizer {

    private let wrapped: Tokenizers.Tokenizer
    let vocabularySize: Int

    init(wrapped: Tokenizers.Tokenizer, vocabularySize: Int) {
        self.wrapped = wrapped
        self.vocabularySize = vocabularySize
    }

    func encode(text: String) -> [Int] {
        wrapped.encode(text: text)
    }

    func decode(tokens: [Int]) -> String {
        wrapped.decode(tokens: tokens)
    }

    var bosTokenID: Int? {
        wrapped.bosTokenId
    }

    var eosTokenID: Int? {
        wrapped.eosTokenId
    }

    func tokenToString(_ id: Int) -> String? {
        wrapped.convertIdToToken(id)
    }

    func tokenID(for string: String) -> Int? {
        wrapped.convertTokenToId(string)
    }
}

// MARK: - HFQuantizationConfig

/// Quantization config parsed from config.json for mlx-community quantized models.
private struct HFQuantizationConfig {
    let defaultGroupSize: Int
    let defaultBits: Int

    func specForPath(_ path: String) -> QuantizationSpec? {
        // Only weight tensors (not norms/biases) are quantized
        guard path.hasSuffix(".weight") else { return nil }
        // Skip norm weights (layernorm, rmsnorm — no quantization)
        if path.contains("layernorm") || path.contains("norm.weight") { return nil }
        // Note: embed_tokens and lm_head ARE quantized in mlx-community models
        // (they have .scales and .biases companion tensors). If a weight turns out
        // not to have scales/biases, convertToRawWeights falls through to dense safely.
        return QuantizationSpec(groupSize: defaultGroupSize, bits: defaultBits)
    }
}

// MARK: - Errors

/// Errors specific to HuggingFace directory bundle loading.
public enum HFBundleError: Error, CustomStringConvertible {
    case noWeightFiles(directory: String)
    case invalidShardIndex
    case missingFile(name: String, directory: String)

    public var description: String {
        switch self {
        case .noWeightFiles(let dir):
            return "No safetensors files found in \(dir)"
        case .invalidShardIndex:
            return "Invalid model.safetensors.index.json format"
        case .missingFile(let name, let dir):
            return "Required file '\(name)' not found in \(dir)"
        }
    }
}
