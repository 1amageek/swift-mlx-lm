#!/usr/bin/env python3
"""Extract detailed layer 0 intermediate states from HF Gemma4 for comparison with swift-lm probes."""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "google/gemma-4-E2B-it"
PROMPT = "What is the capital of Japan? Answer with exactly one word."

def main():
    print(f"[Loading] {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16)
    model.eval()

    messages = [{"role": "user", "content": PROMPT}]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(formatted, return_tensors="pt")
    token_ids = inputs["input_ids"][0].tolist()
    print(f"[Tokens] ({len(token_ids)}) {token_ids}")

    lm = model.model.language_model
    layer0 = lm.layers[0]
    attn = layer0.self_attn
    last_pos = len(token_ids) - 1

    def norm_and_first(t, pos=None):
        if pos is not None and t.dim() == 3:
            h = t[0, pos, :].float().numpy()
        elif t.dim() == 2:
            h = t[0, :].float().numpy()
        else:
            h = t.flatten().float().numpy()
        n = np.linalg.norm(h)
        f4 = [f"{v:+.4f}" for v in h[:4]]
        return n, f4

    with torch.no_grad():
        # Step 0: Embedding
        emb_raw = lm.embed_tokens(inputs["input_ids"])
        hidden_size = emb_raw.shape[-1]
        emb_scaled = emb_raw * (hidden_size ** 0.5)
        n, f4 = norm_and_first(emb_scaled, last_pos)
        print(f"\n[step 0] embedding (scaled by sqrt({hidden_size}))")
        print(f"  norm={n:.4f}  first4={f4}")

        # Step 1: input_layernorm
        hidden = emb_scaled.clone()
        ln_out = layer0.input_layernorm(hidden)
        n, f4 = norm_and_first(ln_out, last_pos)
        print(f"\n[step 1] input_layernorm output")
        print(f"  norm={n:.4f}  first4={f4}")

        # Check RMSNorm internals
        x = hidden[0, last_pos, :].float()
        rms = torch.sqrt(x.pow(2).mean() + 1e-6)
        normed = x / rms
        w = layer0.input_layernorm.weight.float()
        result_manual = normed * w
        n_manual = torch.norm(result_manual).item()
        print(f"  [manual check] rms={rms:.4f} normed_norm={torch.norm(normed).item():.4f}")
        print(f"  [manual check] weight mean={w.mean().item():.4f} norm_of_result={n_manual:.4f}")
        print(f"  [manual check] weight*normed first4={[f'{v:+.4f}' for v in (normed*w)[:4].numpy()]}")

        # Step 2-4: Q, K, V projections (from input_layernorm output)
        q = attn.q_proj(ln_out)
        k = attn.k_proj(ln_out)
        v = attn.v_proj(ln_out)
        n_q, f4_q = norm_and_first(q, last_pos)
        n_k, f4_k = norm_and_first(k, last_pos)
        n_v, f4_v = norm_and_first(v, last_pos)
        print(f"\n[step 2] q_proj output: norm={n_q:.4f}  first4={f4_q}")
        print(f"[step 3] k_proj output: norm={n_k:.4f}  first4={f4_k}")
        print(f"[step 4] v_proj output: norm={n_v:.4f}  first4={f4_v}")

        # Step 5: QK norms (reshape to [batch, seq, heads, head_dim])
        head_dim = attn.head_dim
        num_heads = q.shape[-1] // head_dim
        num_kv_heads = k.shape[-1] // head_dim

        q_reshaped = q.view(1, -1, num_heads, head_dim)
        k_reshaped = k.view(1, -1, num_kv_heads, head_dim)
        v_reshaped = v.view(1, -1, num_kv_heads, head_dim)

        q_normed = attn.q_norm(q_reshaped)
        k_normed = attn.k_norm(k_reshaped)
        v_normed = attn.v_norm(v_reshaped)

        n_qn, f4_qn = norm_and_first(q_normed[0, last_pos, 0, :])
        n_kn, f4_kn = norm_and_first(k_normed[0, last_pos, 0, :])
        n_vn, f4_vn = norm_and_first(v_normed[0, last_pos, 0, :])
        print(f"\n[step 5] q_norm output (head 0): norm={n_qn:.4f}  first4={f4_qn}")
        print(f"[step 5] k_norm output (head 0): norm={n_kn:.4f}  first4={f4_kn}")
        print(f"[step 6] v_norm output (head 0): norm={n_vn:.4f}  first4={f4_vn}")

        # QK norm weight analysis
        print(f"\n[QK norm weights]")
        print(f"  q_norm: has_weight={hasattr(attn.q_norm, 'weight') and attn.q_norm.weight is not None}")
        if hasattr(attn.q_norm, 'weight') and attn.q_norm.weight is not None:
            qw = attn.q_norm.weight.float()
            print(f"  q_norm weight: mean={qw.mean().item():.6f} first4={qw[:4].tolist()}")
        print(f"  k_norm: has_weight={hasattr(attn.k_norm, 'weight') and attn.k_norm.weight is not None}")
        if hasattr(attn.k_norm, 'weight') and attn.k_norm.weight is not None:
            kw = attn.k_norm.weight.float()
            print(f"  k_norm weight: mean={kw.mean().item():.6f} first4={kw[:4].tolist()}")
        print(f"  v_norm: has_weight={hasattr(attn.v_norm, 'weight') and attn.v_norm.weight is not None}")

        # Full attention computation
        print(f"\n[Attention computation]")
        print(f"  scaling = {attn.scaling}")

        # Apply RoPE
        position_ids = torch.arange(len(token_ids)).unsqueeze(0)
        cos, sin = lm.rotary_emb(v_normed, position_ids)

        from transformers.models.gemma4.modeling_gemma4 import apply_multidimensional_rope
        q_rope = apply_multidimensional_rope(q_normed, cos, sin, position_ids)
        k_rope = apply_multidimensional_rope(k_normed, cos, sin, position_ids)

        # Transpose: [batch, seq, heads, dim] -> [batch, heads, seq, dim]
        q_t = q_rope.transpose(1, 2)
        k_t = k_rope.transpose(1, 2)
        v_t = v_normed.transpose(1, 2)

        # Attention scores
        scores = torch.matmul(q_t, k_t.transpose(2, 3)) * attn.scaling
        print(f"  QK scores [head 0, last_pos]: min={scores[0, 0, last_pos, :].min().item():.4f} max={scores[0, 0, last_pos, :].max().item():.4f}")

        # Causal mask + softmax
        causal_mask = torch.triu(torch.ones(len(token_ids), len(token_ids)), diagonal=1).bool()
        scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        attn_weights = torch.softmax(scores.float(), dim=-1).to(q_t.dtype)
        print(f"  attn_weights [head 0, last_pos]: min={attn_weights[0, 0, last_pos, :].float().min().item():.6f} max={attn_weights[0, 0, last_pos, :].float().max().item():.6f}")

        # Weighted V
        attn_output = torch.matmul(attn_weights, v_t)
        attn_output = attn_output.transpose(1, 2).reshape(1, -1, num_heads * head_dim).contiguous()

        # o_proj
        o_proj_out = attn.o_proj(attn_output)
        n_o, f4_o = norm_and_first(o_proj_out, last_pos)
        print(f"\n[step 10] o_proj output: norm={n_o:.4f}  first4={f4_o}")

        # post_attention_layernorm
        post_attn_norm_out = layer0.post_attention_layernorm(o_proj_out)
        n_pan, f4_pan = norm_and_first(post_attn_norm_out, last_pos)
        print(f"\n[after post_attn_norm] norm={n_pan:.4f}  first4={f4_pan}")

        # Residual add
        residual_after_attn = post_attn_norm_out + emb_scaled
        n_ra, f4_ra = norm_and_first(residual_after_attn, last_pos)
        print(f"[after residual_add] norm={n_ra:.4f}  first4={f4_ra}")

        # pre_feedforward_layernorm
        pre_ffn_out = layer0.pre_feedforward_layernorm(residual_after_attn)
        n_pf, f4_pf = norm_and_first(pre_ffn_out, last_pos)
        print(f"[after pre_ffn_norm] norm={n_pf:.4f}  first4={f4_pf}")

        # MLP
        gate = layer0.mlp.gate_proj(pre_ffn_out)
        up = layer0.mlp.up_proj(pre_ffn_out)
        act = layer0.mlp.act_fn(gate)
        mlp_intermediate = act * up
        mlp_out = layer0.mlp.down_proj(mlp_intermediate)
        n_mlp, f4_mlp = norm_and_first(mlp_out, last_pos)
        print(f"\n[MLP output] norm={n_mlp:.4f}  first4={f4_mlp}")

        # post_feedforward_layernorm
        post_ffn_norm_out = layer0.post_feedforward_layernorm(mlp_out)
        n_pfn, f4_pfn = norm_and_first(post_ffn_norm_out, last_pos)
        print(f"[after post_ffn_norm] norm={n_pfn:.4f}  first4={f4_pfn}")

        # Residual add
        residual_after_mlp = post_ffn_norm_out + residual_after_attn
        n_rm, f4_rm = norm_and_first(residual_after_mlp, last_pos)
        print(f"[after MLP residual_add] norm={n_rm:.4f}  first4={f4_rm}")

        # PerLayerInput
        per_layer_gate_out = layer0.act_fn(layer0.per_layer_input_gate(residual_after_mlp))
        n_plg, f4_plg = norm_and_first(per_layer_gate_out, last_pos)
        print(f"\n[per_layer_input gate output] norm={n_plg:.4f}  first4={f4_plg}")

        per_layer_proj_out = layer0.per_layer_projection(per_layer_gate_out)
        n_plp, f4_plp = norm_and_first(per_layer_proj_out, last_pos)
        print(f"[per_layer_input projection output] norm={n_plp:.4f}  first4={f4_plp}")

        # post_per_layer_input_norm
        pli_norm_out = layer0.post_per_layer_input_norm(per_layer_proj_out)
        n_plin, f4_plin = norm_and_first(pli_norm_out, last_pos)
        print(f"[post_per_layer_input_norm output] norm={n_plin:.4f}  first4={f4_plin}")

        # Residual add
        residual_after_pli = pli_norm_out + residual_after_mlp
        n_rpli, f4_rpli = norm_and_first(residual_after_pli, last_pos)
        print(f"[after PLI residual_add] norm={n_rpli:.4f}  first4={f4_rpli}")

        # Layer scale
        for n, p in layer0.named_parameters():
            if 'scale' in n.lower() and 'norm' not in n.lower():
                scale_val = p.float().item()
                print(f"\n[layer_scalar] {n} = {scale_val:.6f}")
                scaled = residual_after_pli * scale_val
                # Layer exit = residual + scaled
                layer_exit = emb_scaled + scaled  # NO! layer exit uses the ORIGINAL residual
                break

        # Full forward to get the actual exit
        print(f"\n[Running full model forward for layer exits]")
        outputs = model(**inputs, output_hidden_states=True)
        hs = outputs.hidden_states
        for idx in [0, 1]:
            h = hs[idx][0, last_pos, :].float().numpy()
            label = "embedding" if idx == 0 else f"layer {idx-1} exit"
            n = np.linalg.norm(h)
            f4 = [f"{v:+.4f}" for v in h[:4]]
            print(f"  [{label}] norm={n:.4f} first4={f4}")

        logits = outputs.logits[0, last_pos, :]
        top10 = torch.topk(logits, 10)
        print(f"\n[Top 10 logits]")
        for i in range(10):
            tid = top10.indices[i].item()
            val = top10.values[i].item()
            tok = tokenizer.decode([tid])
            print(f"  id={tid} logit={val:.4f} token={repr(tok)}")


if __name__ == "__main__":
    main()
