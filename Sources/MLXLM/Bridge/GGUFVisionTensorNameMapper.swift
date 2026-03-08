/// Maps GGUF mmproj tensor names to MLX vision encoder weight paths.
///
/// Handles the llama.cpp mmproj format where vision encoder tensors
/// use `v.blk.{i}.*` naming and merger tensors use `mm.*` naming.
struct Qwen25VLVisionTensorNameMapper: GGUFTensorNameMapper {

    func mlxName(for ggufName: String) -> String? {
        // Patch embedding
        switch ggufName {
        case "v.patch_embd.weight":
            return "visual.patch_embed.proj.weight"
        case "v.patch_embd.weight.1":
            return "visual.patch_embed.proj.weight.1"  // Second Conv2d for split Conv3d
        case "v.patch_embd.bias":
            return "visual.patch_embed.proj.bias"

        // Merger (PatchMerger)
        case "mm.ln_q.weight":
            return "visual.merger.ln_q.weight"
        case "mm.0.weight":
            return "visual.merger.mlp.0.weight"
        case "mm.0.bias":
            return "visual.merger.mlp.0.bias"
        case "mm.2.weight":
            return "visual.merger.mlp.2.weight"
        case "mm.2.bias":
            return "visual.merger.mlp.2.bias"

        default:
            break
        }

        // Vision transformer blocks: v.blk.{i}.xxx
        guard ggufName.hasPrefix("v.blk.") else { return nil }

        let parts = ggufName.split(separator: ".", maxSplits: 2)
        guard parts.count == 3,
              let layerIndex = Int(parts[1])
        else { return nil }

        let suffix = String(parts[2])
        let prefix = "visual.blocks.\(layerIndex)"

        switch suffix {
        // Norms
        case "ln1.weight":
            return "\(prefix).norm1.weight"
        case "ln2.weight":
            return "\(prefix).norm2.weight"

        // Attention (split QKV in GGUF)
        case "attn_q.weight":
            return "\(prefix).attn.q_proj.weight"
        case "attn_q.bias":
            return "\(prefix).attn.q_proj.bias"
        case "attn_k.weight":
            return "\(prefix).attn.k_proj.weight"
        case "attn_k.bias":
            return "\(prefix).attn.k_proj.bias"
        case "attn_v.weight":
            return "\(prefix).attn.v_proj.weight"
        case "attn_v.bias":
            return "\(prefix).attn.v_proj.bias"
        case "attn_out.weight":
            return "\(prefix).attn.proj.weight"
        case "attn_out.bias":
            return "\(prefix).attn.proj.bias"

        // Fused QKV (some GGUF producers use this)
        case "attn_qkv.weight":
            return "\(prefix).attn.qkv.weight"
        case "attn_qkv.bias":
            return "\(prefix).attn.qkv.bias"

        // FFN (SwiGLU)
        case "ffn_gate.weight":
            return "\(prefix).mlp.gate_proj.weight"
        case "ffn_gate.bias":
            return "\(prefix).mlp.gate_proj.bias"
        case "ffn_up.weight":
            return "\(prefix).mlp.up_proj.weight"
        case "ffn_up.bias":
            return "\(prefix).mlp.up_proj.bias"
        case "ffn_down.weight":
            return "\(prefix).mlp.down_proj.weight"
        case "ffn_down.bias":
            return "\(prefix).mlp.down_proj.bias"

        default:
            return nil
        }
    }
}
