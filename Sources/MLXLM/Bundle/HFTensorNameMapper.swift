import Foundation

/// Maps HuggingFace safetensors tensor names to IR-compatible MLX parameter paths.
///
/// HF safetensors names are close to MLX-compatible dotted paths but may need
/// minor transformations:
///
/// - **VLM models**: Strip `language_model.` prefix (e.g.,
///   `language_model.model.layers.0.self_attn.q_proj.weight` → `model.layers.0.self_attn.q_proj.weight`)
/// - **Vision tensors**: Skip `vision_tower.*` (not part of text decoder IR)
/// - **Tied embeddings**: Skip `lm_head.weight` when config says `tie_word_embeddings`
/// - **Rotary embeddings**: Skip `rotary_emb.inv_freq`
public struct HFTensorNameMapper: Sendable {

    /// Whether the model uses tied embeddings (lm_head shares embed_tokens weight).
    public let tiedEmbeddings: Bool

    public init(tiedEmbeddings: Bool = false) {
        self.tiedEmbeddings = tiedEmbeddings
    }

    /// Map a safetensors tensor name to its MLX parameter path.
    ///
    /// Returns nil for tensors that should be skipped.
    public func mlxPath(for hfName: String) -> String? {
        // Skip vision encoder tensors (handled separately for VLMs)
        if hfName.hasPrefix("vision_tower.") || hfName.hasPrefix("visual.") {
            return nil
        }

        // Skip rotary embedding frequency tensors
        if hfName.contains("rotary_emb") { return nil }

        // Strip VLM text model prefix: "language_model.model." → "model."
        var mapped = hfName
        if mapped.hasPrefix("language_model.") {
            mapped = String(mapped.dropFirst("language_model.".count))
        }

        // For tied embeddings, skip lm_head (embed_tokens weight is used instead)
        if tiedEmbeddings && mapped == "lm_head.weight" {
            return nil
        }

        return mapped
    }
}
