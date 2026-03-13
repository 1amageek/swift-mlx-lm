import GGUFParser
import MLXLM

// MARK: - GGUFArchitectureDetector

/// Detects architecture from GGUF tensor name patterns.
///
/// Each pattern check is a sufficient condition. Detection order
/// (most specific first):
/// 1. hybridDeltaNetAttention — DeltaNet tensors
/// 2. parallelAttentionMLP — QK norm + no FFN norm
/// 3. moe — expert tensors
/// 4. transformer — universal fallback
public struct GGUFArchitectureDetector: Sendable {

    public init() {}

    /// Detect architecture from tensor name patterns in the GGUF file.
    public func detect(file: GGUFFile) -> DetectedArchitecture {
        let names = Set(file.tensors.map(\.name))
        return detect(tensorNames: names)
    }

    /// Detect architecture from a set of tensor names.
    ///
    /// Exposed for testing with synthetic tensor name sets.
    public func detect(tensorNames names: Set<String>) -> DetectedArchitecture {
        // Priority 1: hybrid DeltaNet / full-attention decoder
        if names.contains("blk.0.ssm_beta.weight") {
            return .hybridDeltaNetAttention
        }

        // Priority 2: shared-norm parallel attention + MLP
        if names.contains("blk.0.attn_q_norm.weight")
            && !names.contains("blk.0.ffn_norm.weight")
        {
            return .parallelAttentionMLP
        }

        // Priority 3: MoE (expert routing gate)
        if names.contains("blk.0.ffn_gate_inp.weight") {
            return .moe
        }

        // Fallback: standard transformer
        return .transformer
    }
}
