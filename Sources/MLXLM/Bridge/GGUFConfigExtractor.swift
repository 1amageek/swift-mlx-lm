import GGUFParser

/// Extracts model configuration from GGUF metadata.
struct GGUFConfigExtractor {

    // MARK: - Common Helpers

    private static func extractBaseFields(from file: GGUFFile) throws
        -> (embeddingLength: Int, blockCount: Int, headCount: Int, feedForwardLength: Int,
            kvHeads: Int, rmsNormEps: Float, ropeTheta: Float, vocabSize: Int,
            tieWordEmbeddings: Bool, hasAttentionBias: Bool, hasMlpBias: Bool)
    {
        guard let embeddingLength = file.embeddingLength else {
            throw GGUFLoadError.missingMetadata("embedding_length")
        }
        guard let blockCount = file.blockCount else {
            throw GGUFLoadError.missingMetadata("block_count")
        }
        guard let headCount = file.headCount else {
            throw GGUFLoadError.missingMetadata("head_count")
        }

        let feedForwardLength = file.feedForwardLength ?? (embeddingLength * 4)
        let kvHeads = file.headCountKV ?? headCount
        let rmsNormEps = file.attentionLayerNormRMSEpsilon ?? 1e-5
        let ropeTheta = file.ropeFreqBase ?? 10_000.0
        let vocabSize = file.vocabularySize ?? file.vocabularyLength ?? 0

        let tieWordEmbeddings: Bool = {
            if file.metadata["output.weight"] != nil {
                return false
            }
            return file.tensors.contains { $0.name == "output.weight" } ? false : true
        }()

        let hasAttentionBias = file.tensors.contains { $0.name == "blk.0.attn_q.bias" }
        let hasMlpBias = file.tensors.contains { $0.name == "blk.0.ffn_gate.bias" }

        return (embeddingLength, blockCount, headCount, feedForwardLength,
                kvHeads, rmsNormEps, ropeTheta, vocabSize,
                tieWordEmbeddings, hasAttentionBias, hasMlpBias)
    }

    /// Build RoPE scaling dictionary from GGUF metadata.
    private static func extractRopeScaling(from file: GGUFFile) -> [String: StringOrNumber]? {
        guard let ropeType = file.ropeScalingType else { return nil }

        var config: [String: StringOrNumber] = [
            "type": .string(ropeType)
        ]

        if let factor = file.ropeScalingFactor {
            config["factor"] = .float(factor)
        }
        if let origMax = file.ropeScalingOriginalMaxPositionEmbeddings {
            config["original_max_position_embeddings"] = .int(origMax)
        }

        // Llama 3 specific
        if let lowFreq = file.ropeScalingLowFreqFactor {
            config["low_freq_factor"] = .float(lowFreq)
        }
        if let highFreq = file.ropeScalingHighFreqFactor {
            config["high_freq_factor"] = .float(highFreq)
        }

        // Su/LongRoPE specific
        if let attnFactor = file.ropeScalingAttnFactor {
            config["attn_factor"] = .float(attnFactor)
        }
        if let shortFactor = file.ropeScalingShortFactor {
            config["short_factor"] = .floats(shortFactor)
        }
        if let longFactor = file.ropeScalingLongFactor {
            config["long_factor"] = .floats(longFactor)
        }

        return config
    }

    // MARK: - Unified Transformer Config

    /// Extract a unified TransformerConfiguration from GGUF metadata.
    ///
    /// Covers Llama, Qwen2, Mistral, Phi-3, StarCoder2, Gemma 2, and Mixtral.
    /// Architecture-specific features are expressed as configuration flags.
    static func extractTransformerConfig(
        from file: GGUFFile,
        archHint: String,
        isMoE: Bool
    ) throws -> TransformerConfiguration {
        let base = try extractBaseFields(from: file)
        let ropeScaling = extractRopeScaling(from: file)

        // Architecture-specific overrides
        var activation: TransformerConfiguration.ActivationType = .silu
        var hasPostNorm = false
        var attnLogitSoftcap: Float? = nil
        var queryPreAttnScalar: Float? = nil
        var finalLogitSoftcap: Float? = nil
        var slidingWindowPattern: TransformerConfiguration.SlidingWindowPattern = .none
        var embedScale: Float? = nil
        var expertCount: Int? = nil
        var expertUsedCount: Int? = nil
        var headDimensions: Int? = file.headDimension

        switch archHint {
        case "gemma2":
            let headDim = file.attentionKeyLength ?? base.embeddingLength / base.headCount
            headDimensions = headDim
            activation = .gelu
            hasPostNorm = true
            attnLogitSoftcap = file.attnLogitSoftcapping
            finalLogitSoftcap = file.finalLogitSoftcapping
            queryPreAttnScalar = Float(headDim)
            embedScale = Float(base.embeddingLength).squareRoot()
            if file.slidingWindow != nil {
                slidingWindowPattern = .evenLayers
            }

        case "starcoder2":
            if file.slidingWindow != nil {
                slidingWindowPattern = .allLayers
            }

        default:
            break
        }

        if isMoE {
            guard let ec = file.expertCount else {
                throw GGUFLoadError.missingMetadata("expert_count")
            }
            expertCount = ec
            expertUsedCount = file.expertUsedCount ?? 2
        }

        return TransformerConfiguration(
            hiddenSize: base.embeddingLength,
            hiddenLayers: base.blockCount,
            intermediateSize: base.feedForwardLength,
            attentionHeads: base.headCount,
            headDimensions: headDimensions,
            vocabularySize: base.vocabSize,
            kvHeads: base.kvHeads,
            normEps: base.rmsNormEps,
            hasPostNorm: hasPostNorm,
            activation: activation,
            attentionBias: base.hasAttentionBias,
            mlpBias: base.hasMlpBias,
            attnLogitSoftcap: attnLogitSoftcap,
            queryPreAttnScalar: queryPreAttnScalar,
            ropeTheta: base.ropeTheta,
            ropeTraditional: false,
            ropeScaling: ropeScaling,
            maxPositionEmbeddings: file.contextLength,
            ropeDimensionCount: (archHint == "phi3") ? file.ropeDimensionCount : nil,
            slidingWindow: file.slidingWindow,
            slidingWindowPattern: slidingWindowPattern,
            expertCount: expertCount,
            expertUsedCount: expertUsedCount,
            tieWordEmbeddings: base.tieWordEmbeddings,
            finalLogitSoftcap: finalLogitSoftcap,
            embedScale: embedScale
        )
    }

    // MARK: - Qwen 3.5 (Hybrid DeltaNet)

    static func extractQwen35Config(from file: GGUFFile) throws -> Qwen35Configuration {
        let base = try extractBaseFields(from: file)
        let ropeScaling = extractRopeScaling(from: file)

        guard let linearKeyHeads = file.linearKeyHeadCount else {
            throw GGUFLoadError.missingMetadata("linear.key_head_count")
        }
        guard let linearValueHeads = file.linearValueHeadCount else {
            throw GGUFLoadError.missingMetadata("linear.value_head_count")
        }

        let linearKeyHeadDim = file.linearKeyHeadDim ?? 128
        let linearValueHeadDim = file.linearValueHeadDim ?? 128
        let convKernelSize = file.linearConvKernelSize ?? 4
        let fullAttentionInterval = file.fullAttentionInterval ?? 4
        let partialRotaryFactor = file.partialRotaryFactor ?? 0.25
        let headDim = file.attentionKeyLength ?? base.embeddingLength / base.headCount

        return Qwen35Configuration(
            hiddenSize: base.embeddingLength,
            hiddenLayers: base.blockCount,
            intermediateSize: base.feedForwardLength,
            vocabularySize: base.vocabSize,
            normEps: base.rmsNormEps,
            tieWordEmbeddings: base.tieWordEmbeddings,
            attentionHeads: base.headCount,
            kvHeads: base.kvHeads,
            headDim: headDim,
            ropeTheta: base.ropeTheta,
            ropeTraditional: false,
            ropeScaling: ropeScaling,
            maxPositionEmbeddings: file.contextLength,
            partialRotaryFactor: partialRotaryFactor,
            linearKeyHeads: linearKeyHeads,
            linearValueHeads: linearValueHeads,
            linearKeyHeadDim: linearKeyHeadDim,
            linearValueHeadDim: linearValueHeadDim,
            convKernelSize: convKernelSize,
            fullAttentionInterval: fullAttentionInterval
        )
    }

    // MARK: - Cohere / Command-R

    static func extractCohereConfig(from file: GGUFFile) throws -> CohereConfiguration {
        let base = try extractBaseFields(from: file)
        let ropeScaling = extractRopeScaling(from: file)
        let layerNormEps = file.attentionLayerNormEpsilon ?? base.rmsNormEps
        let hasQKNorm = file.tensors.contains { $0.name == "blk.0.attn_q_norm.weight" }
        let logitScale = file.logitScale

        return CohereConfiguration(
            hiddenSize: base.embeddingLength,
            hiddenLayers: base.blockCount,
            intermediateSize: base.feedForwardLength,
            attentionHeads: base.headCount,
            headDimensions: file.headDimension,
            layerNormEps: layerNormEps,
            vocabularySize: base.vocabSize,
            kvHeads: base.kvHeads,
            maxPositionEmbeddings: file.contextLength,
            ropeTheta: base.ropeTheta,
            ropeTraditional: false,
            ropeScaling: ropeScaling,
            tieWordEmbeddings: base.tieWordEmbeddings,
            useQKNorm: hasQKNorm,
            logitScale: logitScale
        )
    }

}

/// Errors during GGUF model loading.
enum GGUFLoadError: Error {
    case missingMetadata(String)
    case unsupportedArchitecture(String)
    case unsupportedQuantization(UInt32)
    case tensorNotFound(String)
    case dimensionMismatch(String)
}
