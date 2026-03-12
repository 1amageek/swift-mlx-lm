// MARK: - Int Keys

extension GGUFMetadataKey where T == Int {

    package init(path: String, scope: Scope = .architecture) {
        self.init(path: path, scope: scope) { $0.intValue }
    }

    // General
    package static let fileType = Self(path: "file_type", scope: .global)

    // Architecture dimensions
    package static let contextLength = Self(path: "context_length")
    package static let embeddingLength = Self(path: "embedding_length")
    package static let blockCount = Self(path: "block_count")
    package static let headCount = Self(path: "attention.head_count")
    package static let headCountKV = Self(path: "attention.head_count_kv")
    package static let feedForwardLength = Self(path: "feed_forward_length")
    package static let vocabularyLength = Self(path: "vocab_size")

    // Attention dimensions
    package static let attentionKeyLength = Self(path: "attention.key_length")
    package static let attentionValueLength = Self(path: "attention.value_length")

    // Sliding window
    package static let slidingWindow = Self(path: "attention.sliding_window")
    package static let maxWindowLayers = Self(path: "attention.max_window_layers")

    // MoE
    package static let expertCount = Self(path: "expert_count")
    package static let expertUsedCount = Self(path: "expert_used_count")

    // RoPE
    package static let ropeDimensionCount = Self(path: "rope.dimension_count")
    package static let ropeScalingOriginalMaxPositionEmbeddings = Self(
        path: "rope.scaling.original_max_position_embeddings")

    // DeltaNet / Hybrid (Qwen 3.5)
    package static let ssmGroupCount = Self(path: "ssm.group_count")
    package static let ssmStateSize = Self(path: "ssm.state_size")
    package static let ssmConvKernel = Self(path: "ssm.conv_kernel")
    package static let ssmInnerSize = Self(path: "ssm.inner_size")
    package static let ssmTimeStepRank = Self(path: "ssm.time_step_rank")
    package static let fullAttentionInterval = Self(path: "full_attention_interval")

    // Tokenizer
    package static let bosTokenID = Self(path: "bos_token_id", scope: .tokenizer)
    package static let eosTokenID = Self(path: "eos_token_id", scope: .tokenizer)
    package static let paddingTokenID = Self(path: "padding_token_id", scope: .tokenizer)
}

// MARK: - Float Keys

extension GGUFMetadataKey where T == Float {

    /// Extract a Float from either float32 or float64 metadata values.
    package init(path: String, scope: Scope = .architecture) {
        self.init(path: path, scope: scope) { value in
            if let f = value.float32Value { return f }
            if let d = value.doubleValue { return Float(d) }
            return nil
        }
    }

    // Norms
    package static let attentionLayerNormRMSEpsilon = Self(
        path: "attention.layer_norm_rms_epsilon")
    package static let attentionLayerNormEpsilon = Self(
        path: "attention.layer_norm_epsilon")

    // RoPE
    package static let ropeFreqBase = Self(path: "rope.freq_base")
    package static let ropeScalingFactor = Self(path: "rope.scaling.factor")
    package static let ropeScalingLowFreqFactor = Self(path: "rope.scaling.low_freq_factor")
    package static let ropeScalingHighFreqFactor = Self(path: "rope.scaling.high_freq_factor")
    package static let ropeScalingAttnFactor = Self(path: "rope.scaling.attn_factor")
    package static let partialRotaryFactor = Self(path: "rope.partial_rotary_factor")

    // Logit soft-capping
    package static let attnLogitSoftcapping = Self(path: "attn_logit_softcapping")
    package static let finalLogitSoftcapping = Self(path: "final_logit_softcapping")

    // Cohere
    package static let logitScale = Self(path: "logit_scale")
}

// MARK: - String Keys

extension GGUFMetadataKey where T == String {

    package init(path: String, scope: Scope = .architecture) {
        self.init(path: path, scope: scope) { $0.stringValue }
    }

    // General
    package static let architecture = Self(path: "architecture", scope: .global)
    package static let name = Self(path: "name", scope: .global)

    // RoPE
    package static let ropeScalingType = Self(path: "rope.scaling.type")

    // Tokenizer
    package static let tokenizerModel = Self(path: "model", scope: .tokenizer)
    package static let preTokenizer = Self(path: "pre", scope: .tokenizer)
    package static let chatTemplate = Self(path: "tokenizer.chat_template", scope: .raw)
}

// MARK: - Bool Keys

extension GGUFMetadataKey where T == Bool {

    package init(path: String, scope: Scope = .tokenizer) {
        self.init(path: path, scope: scope) { $0.boolValue }
    }

    package static let addBosToken = Self(path: "add_bos_token")
    package static let addEosToken = Self(path: "add_eos_token")
}

// MARK: - Array Keys

extension GGUFMetadataKey where T == [String] {

    package static let tokens = Self(
        path: "tokens", scope: .tokenizer
    ) { $0.arrayValue?.compactMap(\.stringValue) }

    package static let merges = Self(
        path: "merges", scope: .tokenizer
    ) { $0.arrayValue?.compactMap(\.stringValue) }
}

extension GGUFMetadataKey where T == [Float] {

    package static let tokenScores = Self(
        path: "scores", scope: .tokenizer
    ) { $0.arrayValue?.compactMap(\.float32Value) }

    package static let ropeScalingShortFactor = Self(
        path: "rope.scaling.short_factor", scope: .architecture
    ) { $0.arrayValue?.compactMap(\.float32Value) }

    package static let ropeScalingLongFactor = Self(
        path: "rope.scaling.long_factor", scope: .architecture
    ) { $0.arrayValue?.compactMap(\.float32Value) }
}

extension GGUFMetadataKey where T == [Int] {

    package static let tokenTypes = Self(
        path: "token_type", scope: .tokenizer
    ) { $0.arrayValue?.compactMap(\.intValue) }

    /// M-RoPE section sizes (e.g. [16,24,24] for Qwen 2.5-VL).
    package static let ropeScalingSections = Self(
        path: "rope.scaling.sections", scope: .architecture
    ) { $0.arrayValue?.compactMap(\.intValue) }
}
