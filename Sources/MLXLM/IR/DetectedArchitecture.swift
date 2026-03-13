/// Architecture variant detected from model metadata or tensor patterns.
///
/// Used to dispatch the correct IR assembly strategy in `IRGraphAssembler`.
public enum DetectedArchitecture: Sendable, Equatable {

    /// Standard transformer (Llama, Qwen2, Mistral, Gemma, Phi, StarCoder2).
    /// Sequential residual blocks: [norm+attn] → [norm+mlp].
    case transformer

    /// Shared-norm parallel attention + MLP transformer.
    /// No separate FFN norm. QK norm weights present.
    case parallelAttentionMLP

    /// MoE transformer (Mixtral): standard transformer with expert FFN.
    /// Has `ffn_gate_inp` (router) and per-expert `ffn_gate.{e}` tensors.
    case moe

    /// Hybrid DeltaNet / full-attention decoder.
    /// Has DeltaNet tensors (ssm_beta, ssm_alpha, ssm_conv1d) on some layers.
    case hybridDeltaNetAttention
}
