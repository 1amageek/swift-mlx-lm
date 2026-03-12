import Foundation
import GGUFParser
import MLX
import MLXNN

/// Loads a vision encoder from a GGUF mmproj file.
///
/// Parses the mmproj GGUF to extract ``VisionConfig`` from metadata,
/// detects the encoder architecture from tensor patterns, and loads weights.
///
/// Architecture detection:
/// - **Windowed (Conv3d + SwiGLU)**: mmproj has `v.blk.*.ffn_gate.*` tensors.
/// - **Full attention (Conv2d + GELU)**: no gate tensors in the MLP.
struct GGUFVisionLoader {

    /// Result of loading a vision encoder.
    struct LoadResult {
        let encoder: any VisionEncoder
        let config: VisionConfig
        let imageProcessor: VisionImageProcessor
    }

    /// Load vision encoder from an mmproj GGUF file.
    ///
    /// Automatically detects the encoder architecture and returns a unified result.
    func load(url: URL, textHiddenSize: Int) throws -> LoadResult {
        let file = try GGUFFile.parse(url: url)
        let usesWindowAttention = file.tensors.contains { $0.name.contains("ffn_gate") }

        if usesWindowAttention {
            return try loadWindowedEncoder(file: file, textHiddenSize: textHiddenSize)
        } else {
            return try loadFullAttentionEncoder(file: file, textHiddenSize: textHiddenSize)
        }
    }

    // MARK: - Windowed Encoder (Conv3d + SwiGLU + window/full attention hybrid)

    private func loadWindowedEncoder(
        file: GGUFFile, textHiddenSize: Int
    ) throws -> LoadResult {
        var config = try extractWindowedConfig(from: file)
        config.outHiddenSize = textHiddenSize

        let encoder = WindowedVisionTransformer(config)
        try loadWindowedWeights(into: encoder, from: file)
        eval(encoder)

        let processor = VisionImageProcessor(config: config)
        return LoadResult(encoder: encoder, config: config, imageProcessor: processor)
    }

    private func extractWindowedConfig(from file: GGUFFile) throws -> VisionConfig {
        let (hiddenSize, depth, numHeads) = try extractRequiredFields(from: file)

        let intermediateSize = visionMeta(file, "feed_forward_length")?.intValue ?? (hiddenSize * 4)
        let outHiddenSize = visionMeta(file, "projection_dim")?.intValue ?? hiddenSize
        let patchSize = visionMeta(file, "patch_size")?.intValue ?? 14
        let windowSize = visionMeta(file, "image_size")?.intValue ?? (patchSize * 8)
        let spatialMergeSize = file.metadata["clip.vision.spatial_merge_size"]?.intValue ?? 2
        let normEps = visionMeta(file, "attention.layer_norm_epsilon")?.float32Value ?? 1e-6
        let fullAttBlocks = parseFullAttBlockIndexes(from: file, depth: depth)

        let imageMean = extractFloatArray(from: file, key: "clip.vision.image_mean",
                                          fallback: [0.48145466, 0.4578275, 0.40821073])
        let imageStd = extractFloatArray(from: file, key: "clip.vision.image_std",
                                         fallback: [0.26862954, 0.26130258, 0.27577711])

        return VisionConfig(
            hiddenSize: hiddenSize,
            intermediateSize: intermediateSize,
            depth: depth,
            numHeads: numHeads,
            outHiddenSize: outHiddenSize,
            patchSize: patchSize,
            spatialMergeSize: spatialMergeSize,
            temporalPatchSize: 2,
            normEps: normEps,
            windowSize: windowSize,
            fullAttBlockIndexes: fullAttBlocks,
            imageMean: imageMean,
            imageStd: imageStd
        )
    }

    private func loadWindowedWeights(
        into encoder: WindowedVisionTransformer,
        from file: GGUFFile
    ) throws {
        let mapper = WindowedVisionTensorNameMapper()
        let bridge = GGUFTensorBridge()
        var weights: [String: MLXArray] = [:]

        for tensor in file.tensors {
            guard let mlxName = mapper.mlxName(for: tensor.name) else { continue }
            let data = try file.tensorData(for: tensor)
            weights[mlxName] = try bridge.convert(tensor: tensor, data: data)
        }

        weights = fuseQKVIfNeeded(weights, depth: encoder.config.depth)
        weights = fusePatchEmbedIfNeeded(weights)
        weights = sanitizeConv3dWeights(weights)

        let parameters = ModuleParameters.unflattened(weights)
        try encoder.update(parameters: parameters, verify: .noUnusedKeys)
    }

    // MARK: - Full Attention Encoder (Conv2d + GELU)

    private func loadFullAttentionEncoder(
        file: GGUFFile, textHiddenSize: Int
    ) throws -> LoadResult {
        var config = try extractFullAttentionConfig(from: file)
        config.outHiddenSize = textHiddenSize

        let encoder = FullAttentionVisionTransformer(config)
        try loadFullAttentionWeights(into: encoder, from: file)
        eval(encoder)

        let processor = VisionImageProcessor(config: config)
        return LoadResult(encoder: encoder, config: config, imageProcessor: processor)
    }

    private func extractFullAttentionConfig(from file: GGUFFile) throws -> VisionConfig {
        let (hiddenSize, depth, numHeads) = try extractRequiredFields(from: file)

        let intermediateSize = visionMeta(file, "feed_forward_length")?.intValue ?? (hiddenSize * 4)
        let outHiddenSize = visionMeta(file, "projection_dim")?.intValue ?? hiddenSize
        let patchSize = visionMeta(file, "patch_size")?.intValue ?? 16
        let spatialMergeSize = file.metadata["clip.vision.spatial_merge_size"]?.intValue ?? 2
        let normEps = visionMeta(file, "attention.layer_norm_epsilon")?.float32Value ?? 1e-6

        let imageMean = extractFloatArray(from: file, key: "clip.vision.image_mean",
                                          fallback: [0.48145466, 0.4578275, 0.40821073])
        let imageStd = extractFloatArray(from: file, key: "clip.vision.image_std",
                                         fallback: [0.26862954, 0.26130258, 0.27577711])

        return VisionConfig(
            hiddenSize: hiddenSize,
            intermediateSize: intermediateSize,
            depth: depth,
            numHeads: numHeads,
            outHiddenSize: outHiddenSize,
            patchSize: patchSize,
            spatialMergeSize: spatialMergeSize,
            temporalPatchSize: 1,
            normEps: normEps,
            imageMean: imageMean,
            imageStd: imageStd
        )
    }

    private func loadFullAttentionWeights(
        into encoder: FullAttentionVisionTransformer,
        from file: GGUFFile
    ) throws {
        let mapper = FullAttentionVisionTensorNameMapper()
        let bridge = GGUFTensorBridge()
        var weights: [String: MLXArray] = [:]

        for tensor in file.tensors {
            guard let mlxName = mapper.mlxName(for: tensor.name) else { continue }
            let data = try file.tensorData(for: tensor)
            weights[mlxName] = try bridge.convert(tensor: tensor, data: data)
        }

        weights = fuseQKVIfNeeded(weights, depth: encoder.config.depth)
        weights = sanitizeConv2dWeights(weights)

        let parameters = ModuleParameters.unflattened(weights)
        try encoder.update(parameters: parameters, verify: .noUnusedKeys)
    }

    // MARK: - Shared Helpers

    private func extractRequiredFields(
        from file: GGUFFile
    ) throws -> (hiddenSize: Int, depth: Int, numHeads: Int) {
        guard let hiddenSize = visionMeta(file, "embedding_length")?.intValue else {
            throw GGUFLoadError.missingMetadata("clip.vision.embedding_length")
        }
        guard let depth = visionMeta(file, "block_count")?.intValue else {
            throw GGUFLoadError.missingMetadata("clip.vision.block_count")
        }
        guard let numHeads = visionMeta(file, "head_count")?.intValue else {
            throw GGUFLoadError.missingMetadata("clip.vision.head_count")
        }
        return (hiddenSize, depth, numHeads)
    }

    private func visionMeta(_ file: GGUFFile, _ key: String) -> GGUFMetadataValue? {
        file.metadata["clip.vision.\(key)"]
    }

    private func parseFullAttBlockIndexes(from file: GGUFFile, depth: Int) -> [Int] {
        if let pattern = file.metadata["clip.vision.n_wa_pattern"]?.intValue, pattern > 0 {
            var indexes = [Int]()
            for i in stride(from: pattern - 1, to: depth, by: pattern) {
                indexes.append(i)
            }
            return indexes
        }
        let interval = max(1, depth / 4)
        return Array(stride(from: interval - 1, to: depth, by: interval))
    }

    private func extractFloatArray(from file: GGUFFile, key: String, fallback: [Float]) -> [Float] {
        guard let value = file.metadata[key],
              case .array(let elements) = value else {
            return fallback
        }
        let floats = elements.compactMap { $0.float32Value }
        return floats.isEmpty ? fallback : floats
    }

    // MARK: - QKV Fusion

    private func fuseQKVIfNeeded(_ weights: [String: MLXArray], depth: Int) -> [String: MLXArray] {
        var result = weights

        for i in 0..<depth {
            let prefix = "blocks.\(i).attn"
            let qKey = "\(prefix).q_proj.weight"
            let kKey = "\(prefix).k_proj.weight"
            let vKey = "\(prefix).v_proj.weight"
            let qkvKey = "\(prefix).qkv.weight"

            if let q = result[qKey], let k = result[kKey], let v = result[vKey],
               result[qkvKey] == nil {
                result[qkvKey] = concatenated([q, k, v], axis: 0)
                result.removeValue(forKey: qKey)
                result.removeValue(forKey: kKey)
                result.removeValue(forKey: vKey)
            }

            let qBias = "\(prefix).q_proj.bias"
            let kBias = "\(prefix).k_proj.bias"
            let vBias = "\(prefix).v_proj.bias"
            let qkvBias = "\(prefix).qkv.bias"

            if let qb = result[qBias], let kb = result[kBias], let vb = result[vBias],
               result[qkvBias] == nil {
                result[qkvBias] = concatenated([qb, kb, vb], axis: 0)
                result.removeValue(forKey: qBias)
                result.removeValue(forKey: kBias)
                result.removeValue(forKey: vBias)
            }
        }

        return result
    }

    // MARK: - Weight Sanitization

    /// Combine two Conv2d weights into a single Conv3d weight (llama.cpp split format).
    private func fusePatchEmbedIfNeeded(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var result = weights

        let w0Key = "patch_embed.proj.weight"
        let w1Key = "patch_embed.proj.weight.1"

        if let w0 = result[w0Key], let w1 = result[w1Key] {
            let combined = stacked([w0, w1], axis: 1)
            result[w0Key] = combined
            result.removeValue(forKey: w1Key)
        }

        return result
    }

    /// Sanitize Conv3d weights: PyTorch [O, I, D, H, W] → MLX [O, D, H, W, I].
    private func sanitizeConv3dWeights(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var result = weights

        for key in Array(result.keys) {
            guard let w = result[key] else { continue }
            if key.contains("patch_embed.proj.weight") && w.ndim == 5 {
                let shape = w.shape
                if shape[1] == 3 || shape[1] == 1 {
                    result[key] = w.transposed(0, 2, 3, 4, 1)
                }
            }
        }

        return result
    }

    /// Sanitize Conv2d weights: PyTorch [O, I, H, W] → MLX [O, H, W, I].
    private func sanitizeConv2dWeights(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var result = weights

        for key in Array(result.keys) {
            guard let w = result[key] else { continue }
            if key.contains("patch_embed.proj.weight") && w.ndim == 4 {
                let shape = w.shape
                if shape[1] == 3 || shape[1] == 1 {
                    result[key] = w.transposed(0, 2, 3, 1)
                }
            }
        }

        return result
    }
}
