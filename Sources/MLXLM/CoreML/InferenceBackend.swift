import Foundation

/// Inference backend selection for LLM execution.
///
/// MPSGraph is the target default for standard transformer architectures
/// (graph compilation with kernel fusion, no Python dependency).
/// MLX is used as fallback for all architectures and as the current default
/// while MPSGraph integration is in progress.
public enum InferenceBackend: Sendable {
    /// Automatically select the best backend.
    /// Currently defaults to MLX. Will switch to MPSGraph when integration is complete.
    case auto

    /// MPSGraph (Metal Performance Shaders Graph) execution.
    /// Compiles the full model into a fused execution plan.
    /// No Python dependency — pure Swift + Metal.
    case mpsgraph

    /// MLX Metal execution. Always available.
    /// Uses fused RMSNorm, fused SDPA, QKV packing, flat decode plan.
    case mlx
}
