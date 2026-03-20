import Foundation

enum STAFMetadataKey {
    static let sourceFormat = "general.source_format"
    static let converterVersion = "staf.converter_version"
    static let sourceShardCount = "staf.source_shard_count"
    static let metadataSchemaVersion = "staf.metadata_schema_version"

    static let modelArchitectureFamily = "model.architecture_family"
    static let modelHiddenSize = "model.hidden_size"
    static let modelLayerCount = "model.layer_count"
    static let modelIntermediateSize = "model.intermediate_size"
    static let modelVocabSize = "model.vocab_size"
    static let modelAttentionHeads = "model.attention_heads"
    static let modelKVHeads = "model.kv_heads"
    static let modelHeadDimension = "model.head_dim"
    static let modelTiedEmbeddings = "model.tied_embeddings"
    static let modelRopeDimension = "model.rope_dimension"
    static let modelRopeTheta = "model.rope_theta"

    static let sourceConfigHash = "source.config_hash"
    static let sourceTokenizerHash = "source.tokenizer_hash"
    static let sourceTokenizerConfigHash = "source.tokenizer_config_hash"
    static let sourceSpecialTokensHash = "source.special_tokens_hash"
    static let sourceChatTemplateHash = "source.chat_template_hash"
    static let sourceChatTemplateSource = "source.chat_template_source"
    static let sourceTokenizerPresent = "source.tokenizer_present"
    static let sourceTokenizerConfigPresent = "source.tokenizer_config_present"
    static let sourceSpecialTokensPresent = "source.special_tokens_present"
    static let sourceChatTemplatePresent = "source.chat_template_present"
    static let sourceSafetensorsManifestHash = "source.safetensors_manifest_hash"

    static let chatTemplateJinjaSource = "chat_template.jinja"
    static let tokenizerConfigSource = "tokenizer_config.json"
}
