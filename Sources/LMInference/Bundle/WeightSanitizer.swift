import SwiftLM

/// Weight sanitization utilities for HF safetensors models.
///
/// Removes unused tensors (rotary embeddings) that are present in weight files
/// but not needed at inference time.
public enum WeightSanitizer {

    /// Remove rotary embedding inverse frequency tensors (unused at inference time).
    ///
    /// Safe to apply to any format — HF safetensors may also contain these.
    public static let filterRotaryEmbeddings: @Sendable ([String: TensorData]) -> [String: TensorData] = { weights in
        weights.filter { !$0.key.contains("rotary_emb.inv_freq") }
    }
}
