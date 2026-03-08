/// Typed convenience accessors for common GGUF metadata keys.
///
/// Architecture-specific keys use the `{arch}.xxx` convention,
/// where `arch` comes from `general.architecture`.
extension GGUFFile {

    // MARK: - General

    /// Model architecture identifier (e.g., "llama", "qwen2").
    public var architecture: String? {
        metadata["general.architecture"]?.stringValue
    }

    /// Model name.
    public var name: String? {
        metadata["general.name"]?.stringValue
    }

    /// File type (quantization level).
    public var fileType: Int? {
        metadata["general.file_type"]?.intValue
    }

    // MARK: - Architecture-Specific Config

    /// Look up an architecture-prefixed metadata key.
    public func architectureMetadata(_ key: String) -> GGUFMetadataValue? {
        guard let arch = architecture else { return nil }
        return metadata["\(arch).\(key)"]
    }

    /// Maximum context length.
    public var contextLength: Int? {
        architectureMetadata("context_length")?.intValue
    }

    /// Embedding dimension.
    public var embeddingLength: Int? {
        architectureMetadata("embedding_length")?.intValue
    }

    /// Number of transformer blocks.
    public var blockCount: Int? {
        architectureMetadata("block_count")?.intValue
    }

    /// Number of attention heads.
    public var headCount: Int? {
        architectureMetadata("attention.head_count")?.intValue
    }

    /// Number of key-value attention heads.
    public var headCountKV: Int? {
        architectureMetadata("attention.head_count_kv")?.intValue
    }

    /// Dimension per attention head.
    ///
    /// Prefers explicit `attention.key_length` metadata (e.g. Qwen3.5),
    /// falls back to `embedding_length / head_count`.
    public var headDimension: Int? {
        if let explicit = attentionKeyLength {
            return explicit
        }
        if let embed = embeddingLength, let heads = headCount, heads > 0 {
            return embed / heads
        }
        return nil
    }

    /// RoPE base frequency.
    public var ropeFreqBase: Float? {
        architectureMetadata("rope.freq_base")?.float32Value
            ?? architectureMetadata("rope.freq_base")?.doubleValue.map(Float.init)
    }

    /// RoPE dimension count.
    public var ropeDimensionCount: Int? {
        architectureMetadata("rope.dimension_count")?.intValue
    }

    /// RoPE scaling type.
    public var ropeScalingType: String? {
        architectureMetadata("rope.scaling.type")?.stringValue
    }

    /// FFN intermediate size (feed-forward hidden dimension).
    public var feedForwardLength: Int? {
        architectureMetadata("feed_forward_length")?.intValue
    }

    /// Layer norm epsilon.
    public var attentionLayerNormRMSEpsilon: Float? {
        architectureMetadata("attention.layer_norm_rms_epsilon")?.float32Value
            ?? architectureMetadata("attention.layer_norm_rms_epsilon")?.doubleValue.map(Float.init)
    }

    /// Vocabulary size from architecture metadata.
    public var vocabularyLength: Int? {
        architectureMetadata("vocab_size")?.intValue
    }

    // MARK: - Tokenizer

    /// Tokenizer model type (e.g., "llama", "gpt2").
    public var tokenizerModel: String? {
        metadata["tokenizer.ggml.model"]?.stringValue
    }

    /// Token vocabulary as an array of strings.
    public var tokens: [String]? {
        metadata["tokenizer.ggml.tokens"]?.arrayValue?.compactMap(\.stringValue)
    }

    /// Token scores (SentencePiece).
    public var tokenScores: [Float]? {
        metadata["tokenizer.ggml.scores"]?.arrayValue?.compactMap(\.float32Value)
    }

    /// Token types.
    public var tokenTypes: [Int]? {
        metadata["tokenizer.ggml.token_type"]?.arrayValue?.compactMap(\.intValue)
    }

    /// BPE merge rules.
    public var merges: [String]? {
        metadata["tokenizer.ggml.merges"]?.arrayValue?.compactMap(\.stringValue)
    }

    /// Pre-tokenizer type.
    public var preTokenizer: String? {
        metadata["tokenizer.ggml.pre"]?.stringValue
    }

    /// BOS token ID.
    public var bosTokenID: Int? {
        metadata["tokenizer.ggml.bos_token_id"]?.intValue
    }

    /// EOS token ID.
    public var eosTokenID: Int? {
        metadata["tokenizer.ggml.eos_token_id"]?.intValue
    }

    /// Padding token ID.
    public var paddingTokenID: Int? {
        metadata["tokenizer.ggml.padding_token_id"]?.intValue
    }

    /// Whether to automatically add BOS token.
    public var addBosToken: Bool? {
        metadata["tokenizer.ggml.add_bos_token"]?.boolValue
    }

    /// Whether to automatically add EOS token.
    public var addEosToken: Bool? {
        metadata["tokenizer.ggml.add_eos_token"]?.boolValue
    }

    /// Jinja2 chat template string.
    public var chatTemplate: String? {
        metadata["tokenizer.chat_template"]?.stringValue
    }

    // MARK: - Sliding Window Attention

    /// Sliding window attention size.
    public var slidingWindow: Int? {
        architectureMetadata("attention.sliding_window")?.intValue
    }

    /// Number of layers that use full (non-sliding-window) attention from the bottom.
    public var maxWindowLayers: Int? {
        architectureMetadata("attention.max_window_layers")?.intValue
    }

    // MARK: - MoE (Mixture of Experts)

    /// Total number of MoE experts.
    public var expertCount: Int? {
        architectureMetadata("expert_count")?.intValue
    }

    /// Number of experts activated per token (top-k).
    public var expertUsedCount: Int? {
        architectureMetadata("expert_used_count")?.intValue
    }

    // MARK: - Logit Soft-Capping (Gemma 2)

    /// Attention logit soft-capping value (e.g. 50.0 for Gemma 2).
    public var attnLogitSoftcapping: Float? {
        architectureMetadata("attn_logit_softcapping")?.float32Value
            ?? architectureMetadata("attn_logit_softcapping")?.doubleValue.map(Float.init)
    }

    /// Final logit soft-capping value (e.g. 30.0 for Gemma 2).
    public var finalLogitSoftcapping: Float? {
        architectureMetadata("final_logit_softcapping")?.float32Value
            ?? architectureMetadata("final_logit_softcapping")?.doubleValue.map(Float.init)
    }

    // MARK: - RoPE Scaling

    /// RoPE scaling factor.
    public var ropeScalingFactor: Float? {
        architectureMetadata("rope.scaling.factor")?.float32Value
            ?? architectureMetadata("rope.scaling.factor")?.doubleValue.map(Float.init)
    }

    /// Original max position embeddings for RoPE scaling.
    public var ropeScalingOriginalMaxPositionEmbeddings: Int? {
        architectureMetadata("rope.scaling.original_max_position_embeddings")?.intValue
    }

    /// Low frequency factor for Llama 3 RoPE.
    public var ropeScalingLowFreqFactor: Float? {
        architectureMetadata("rope.scaling.low_freq_factor")?.float32Value
            ?? architectureMetadata("rope.scaling.low_freq_factor")?.doubleValue.map(Float.init)
    }

    /// High frequency factor for Llama 3 RoPE.
    public var ropeScalingHighFreqFactor: Float? {
        architectureMetadata("rope.scaling.high_freq_factor")?.float32Value
            ?? architectureMetadata("rope.scaling.high_freq_factor")?.doubleValue.map(Float.init)
    }

    /// Attention factor for Su/LongRoPE scaling.
    public var ropeScalingAttnFactor: Float? {
        architectureMetadata("rope.scaling.attn_factor")?.float32Value
            ?? architectureMetadata("rope.scaling.attn_factor")?.doubleValue.map(Float.init)
    }

    /// Short factor array for Su/LongRoPE scaling.
    public var ropeScalingShortFactor: [Float]? {
        architectureMetadata("rope.scaling.short_factor")?.arrayValue?.compactMap(\.float32Value)
    }

    /// Long factor array for Su/LongRoPE scaling.
    public var ropeScalingLongFactor: [Float]? {
        architectureMetadata("rope.scaling.long_factor")?.arrayValue?.compactMap(\.float32Value)
    }

    // MARK: - Attention Dimensions

    /// Per-head key dimension (when different from value dimension).
    public var attentionKeyLength: Int? {
        architectureMetadata("attention.key_length")?.intValue
    }

    /// Per-head value dimension.
    public var attentionValueLength: Int? {
        architectureMetadata("attention.value_length")?.intValue
    }

    /// Layer norm epsilon (non-RMS variant).
    public var attentionLayerNormEpsilon: Float? {
        architectureMetadata("attention.layer_norm_epsilon")?.float32Value
            ?? architectureMetadata("attention.layer_norm_epsilon")?.doubleValue.map(Float.init)
    }

    /// Logit scale factor (Cohere).
    public var logitScale: Float? {
        architectureMetadata("logit_scale")?.float32Value
            ?? architectureMetadata("logit_scale")?.doubleValue.map(Float.init)
    }

    // MARK: - DeltaNet / Hybrid Attention (Qwen 3.5)

    /// Number of linear attention heads (DeltaNet groups).
    public var linearKeyHeadCount: Int? {
        architectureMetadata("ssm.group_count")?.intValue
    }

    /// Number of linear attention value heads (same as key heads).
    public var linearValueHeadCount: Int? {
        architectureMetadata("ssm.group_count")?.intValue
    }

    /// Per-head key dimension for linear attention (SSM state size).
    public var linearKeyHeadDim: Int? {
        architectureMetadata("ssm.state_size")?.intValue
    }

    /// Per-head value dimension for linear attention (SSM state size).
    public var linearValueHeadDim: Int? {
        architectureMetadata("ssm.state_size")?.intValue
    }

    /// Conv1D kernel size for linear attention.
    public var linearConvKernelSize: Int? {
        architectureMetadata("ssm.conv_kernel")?.intValue
    }

    /// Interval for full attention layers (e.g. 4 = every 4th layer).
    public var fullAttentionInterval: Int? {
        architectureMetadata("full_attention_interval")?.intValue
    }

    /// Partial rotary factor for RoPE (e.g. 0.25 = only 25% of head_dim is rotated).
    public var partialRotaryFactor: Float? {
        architectureMetadata("rope.partial_rotary_factor")?.float32Value
            ?? architectureMetadata("rope.partial_rotary_factor")?.doubleValue.map(Float.init)
    }

    /// SSM inner size (total DeltaNet projection size).
    public var ssmInnerSize: Int? {
        architectureMetadata("ssm.inner_size")?.intValue
    }

    /// SSM time step rank.
    public var ssmTimeStepRank: Int? {
        architectureMetadata("ssm.time_step_rank")?.intValue
    }

    // MARK: - Vocabulary Size

    /// Vocabulary size derived from token list or architecture metadata.
    public var vocabularySize: Int? {
        if let count = tokens?.count { return count }
        return vocabularyLength
    }
}
