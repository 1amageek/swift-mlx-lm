/// Maps GGUF tensor names to MLX model weight paths.
public protocol GGUFTensorNameMapper {
    /// Convert a GGUF tensor name to the corresponding MLX weight key.
    func mlxName(for ggufName: String) -> String?
}

/// Tensor name mapper for standard transformer decoders.
///
/// Maps GGUF tensor names (e.g. `blk.0.attn_q.weight`) to
/// MLX model weight paths (e.g. `model.layers.0.self_attn.q_proj.weight`).
struct TransformerTensorNameMapper: GGUFTensorNameMapper {

    init() {}

    func mlxName(for ggufName: String) -> String? {
        // Global tensors
        switch ggufName {
        case "token_embd.weight":
            return "model.embed_tokens.weight"
        case "output_norm.weight":
            return "model.norm.weight"
        case "output.weight":
            return "lm_head.weight"
        default:
            break
        }

        // Block tensors: blk.{i}.xxx → model.layers.{i}.xxx
        guard ggufName.hasPrefix("blk.") else { return nil }

        let parts = ggufName.split(separator: ".", maxSplits: 2)
        guard parts.count == 3,
              let layerIndex = Int(parts[1])
        else { return nil }

        let suffix = String(parts[2])
        let prefix = "model.layers.\(layerIndex)"

        switch suffix {
        case "attn_norm.weight":
            return "\(prefix).input_layernorm.weight"
        case "ffn_norm.weight":
            return "\(prefix).post_attention_layernorm.weight"
        case "attn_q.weight":
            return "\(prefix).self_attn.q_proj.weight"
        case "attn_k.weight":
            return "\(prefix).self_attn.k_proj.weight"
        case "attn_v.weight":
            return "\(prefix).self_attn.v_proj.weight"
        case "attn_output.weight":
            return "\(prefix).self_attn.o_proj.weight"
        case "ffn_gate.weight":
            return "\(prefix).mlp.gate_proj.weight"
        case "ffn_up.weight":
            return "\(prefix).mlp.up_proj.weight"
        case "ffn_down.weight":
            return "\(prefix).mlp.down_proj.weight"

        // Bias variants
        case "attn_q.bias":
            return "\(prefix).self_attn.q_proj.bias"
        case "attn_k.bias":
            return "\(prefix).self_attn.k_proj.bias"
        case "attn_v.bias":
            return "\(prefix).self_attn.v_proj.bias"
        case "attn_output.bias":
            return "\(prefix).self_attn.o_proj.bias"
        case "ffn_gate.bias":
            return "\(prefix).mlp.gate_proj.bias"
        case "ffn_up.bias":
            return "\(prefix).mlp.up_proj.bias"
        case "ffn_down.bias":
            return "\(prefix).mlp.down_proj.bias"

        // LoRA A weights
        case "attn_q.loraA.weight":
            return "\(prefix).self_attn.q_proj.lora_a"
        case "attn_k.loraA.weight":
            return "\(prefix).self_attn.k_proj.lora_a"
        case "attn_v.loraA.weight":
            return "\(prefix).self_attn.v_proj.lora_a"
        case "attn_output.loraA.weight":
            return "\(prefix).self_attn.o_proj.lora_a"
        case "ffn_gate.loraA.weight":
            return "\(prefix).mlp.gate_proj.lora_a"
        case "ffn_up.loraA.weight":
            return "\(prefix).mlp.up_proj.lora_a"
        case "ffn_down.loraA.weight":
            return "\(prefix).mlp.down_proj.lora_a"

        // LoRA B weights
        case "attn_q.loraB.weight":
            return "\(prefix).self_attn.q_proj.lora_b"
        case "attn_k.loraB.weight":
            return "\(prefix).self_attn.k_proj.lora_b"
        case "attn_v.loraB.weight":
            return "\(prefix).self_attn.v_proj.lora_b"
        case "attn_output.loraB.weight":
            return "\(prefix).self_attn.o_proj.lora_b"
        case "ffn_gate.loraB.weight":
            return "\(prefix).mlp.gate_proj.lora_b"
        case "ffn_up.loraB.weight":
            return "\(prefix).mlp.up_proj.lora_b"
        case "ffn_down.loraB.weight":
            return "\(prefix).mlp.down_proj.lora_b"

        default:
            return nil
        }
    }
}

/// Tensor name mapper for post-normalized transformer decoders.
///
/// Extends the standard transformer mapping with post-normalization weights.
struct PostNormTransformerTensorNameMapper: GGUFTensorNameMapper {

    private let base = TransformerTensorNameMapper()

    func mlxName(for ggufName: String) -> String? {
        // Check Gemma 2 specific tensors first
        if ggufName.hasPrefix("blk.") {
            let parts = ggufName.split(separator: ".", maxSplits: 2)
            guard parts.count == 3, let layerIndex = Int(parts[1]) else {
                return base.mlxName(for: ggufName)
            }

            let suffix = String(parts[2])
            let prefix = "model.layers.\(layerIndex)"

            switch suffix {
            case "attn_post_norm.weight":
                return "\(prefix).post_attention_layernorm.weight"
            case "ffn_pre_norm.weight":
                return "\(prefix).pre_feedforward_layernorm.weight"
            case "ffn_post_norm.weight":
                return "\(prefix).post_feedforward_layernorm.weight"
            default:
                break
            }
        }

        return base.mlxName(for: ggufName)
    }
}

/// Tensor name mapper for shared-norm parallel attention/MLP transformers.
///
/// Adds QK normalization weight mappings.
struct ParallelAttentionMLPTensorNameMapper: GGUFTensorNameMapper {

    private let base = TransformerTensorNameMapper()

    func mlxName(for ggufName: String) -> String? {
        if ggufName.hasPrefix("blk.") {
            let parts = ggufName.split(separator: ".", maxSplits: 2)
            guard parts.count == 3, let layerIndex = Int(parts[1]) else {
                return base.mlxName(for: ggufName)
            }

            let suffix = String(parts[2])
            let prefix = "model.layers.\(layerIndex)"

            switch suffix {
            case "attn_q_norm.weight":
                return "\(prefix).self_attn.q_norm.weight"
            case "attn_q_norm.bias":
                return "\(prefix).self_attn.q_norm.bias"
            case "attn_k_norm.weight":
                return "\(prefix).self_attn.k_norm.weight"
            case "attn_k_norm.bias":
                return "\(prefix).self_attn.k_norm.bias"
            default:
                break
            }
        }

        return base.mlxName(for: ggufName)
    }
}

/// Tensor name mapper for MoE transformer decoders.
///
/// Maps expert weights (ffn_gate_exps, ffn_up_exps, ffn_down_exps) and
/// the router gate (ffn_gate_inp) to MLX model paths.
struct MoETensorNameMapper: GGUFTensorNameMapper {

    private let base = TransformerTensorNameMapper()

    func mlxName(for ggufName: String) -> String? {
        if ggufName.hasPrefix("blk.") {
            let parts = ggufName.split(separator: ".", maxSplits: 2)
            guard parts.count == 3, let layerIndex = Int(parts[1]) else {
                return base.mlxName(for: ggufName)
            }

            let suffix = String(parts[2])
            let prefix = "model.layers.\(layerIndex)"

            // Router gate
            if suffix == "ffn_gate_inp.weight" {
                return "\(prefix).block_sparse_moe.gate.weight"
            }

            // Individual expert weights: ffn_gate.{e}.weight -> experts.{e}.gate_proj.weight
            if let match = parseMoEExpertSuffix(suffix) {
                return "\(prefix).block_sparse_moe.experts.\(match.expertIndex).\(match.projName).weight"
            }
        }

        return base.mlxName(for: ggufName)
    }

    private struct MoEExpertMatch {
        let expertIndex: Int
        let projName: String
    }

    private func parseMoEExpertSuffix(_ suffix: String) -> MoEExpertMatch? {
        // Pattern: ffn_{gate|up|down}.{expertIndex}.weight
        let patterns: [(String, String)] = [
            ("ffn_gate.", "gate_proj"),
            ("ffn_up.", "up_proj"),
            ("ffn_down.", "down_proj"),
        ]

        for (prefix, projName) in patterns {
            guard suffix.hasPrefix(prefix) else { continue }
            let rest = suffix.dropFirst(prefix.count)
            guard rest.hasSuffix(".weight") else { continue }
            let indexStr = rest.dropLast(".weight".count)
            guard let idx = Int(indexStr) else { continue }
            return MoEExpertMatch(expertIndex: idx, projName: projName)
        }

        return nil
    }
}

/// Tensor name mapper for the hybrid DeltaNet / full-attention architecture.
///
/// Maps both DeltaNet layer tensors (linear_attn.*) and full attention layer tensors (self_attn.*).
/// GGUF tensor names are provisional and derived from the current hybrid GGUF layout.
struct HybridDeltaNetAttentionTensorNameMapper: GGUFTensorNameMapper {

    private let base = TransformerTensorNameMapper()

    func mlxName(for ggufName: String) -> String? {
        // Global tensors handled by base mapper
        if !ggufName.hasPrefix("blk.") {
            return base.mlxName(for: ggufName)
        }

        let parts = ggufName.split(separator: ".", maxSplits: 2)
        guard parts.count == 3, let layerIndex = Int(parts[1]) else {
            return base.mlxName(for: ggufName)
        }

        let suffix = String(parts[2])
        let prefix = "model.layers.\(layerIndex)"

        // DeltaNet linear attention tensors
        switch suffix {
        case "attn_qkv.weight":
            return "\(prefix).linear_attn.in_proj_qkv.weight"
        case "attn_gate.weight":
            return "\(prefix).linear_attn.in_proj_z.weight"
        case "ssm_beta.weight":
            return "\(prefix).linear_attn.in_proj_b.weight"
        case "ssm_alpha.weight":
            return "\(prefix).linear_attn.in_proj_a.weight"
        case "ssm_conv1d.weight":
            return "\(prefix).linear_attn.conv1d.weight"
        case "ssm_dt.bias":
            return "\(prefix).linear_attn.dt_bias"
        case "ssm_a":
            return "\(prefix).linear_attn.A_log"
        case "ssm_norm.weight":
            return "\(prefix).linear_attn.norm.weight"
        case "ssm_out.weight":
            return "\(prefix).linear_attn.out_proj.weight"

        // Shared norm (both layer types)
        case "post_attention_norm.weight":
            return "\(prefix).post_attention_layernorm.weight"

        // Full attention with QK norm
        case "attn_q_norm.weight":
            return "\(prefix).self_attn.q_norm.weight"
        case "attn_k_norm.weight":
            return "\(prefix).self_attn.k_norm.weight"

        default:
            break
        }

        // Fall back to standard transformer mapping for shared tensors
        return base.mlxName(for: ggufName)
    }
}
