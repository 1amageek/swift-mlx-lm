#!/usr/bin/env python3
"""Get Layer 0 intermediate reference values from HuggingFace Gemma4."""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "google/gemma-4-E2B-it"
PROMPT = "What is the capital of Japan? Answer with exactly one word."

def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    messages = [{"role": "user", "content": PROMPT}]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(formatted, return_tensors="pt")
    token_ids = inputs["input_ids"][0].tolist()
    print(f"[Tokens] {len(token_ids)} tokens")
    print(f"[Token IDs] {token_ids}")

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16)
    model.eval()

    lm = model.model.language_model
    layer0 = lm.layers[0]

    captures = {}

    def hook(name):
        def fn(module, inp, out):
            if isinstance(out, tuple):
                captures[name] = out[0].detach().float()
            else:
                captures[name] = out.detach().float()
            # Also capture input
            if isinstance(inp, tuple) and len(inp) > 0:
                captures[name + "_input"] = inp[0].detach().float()
            elif isinstance(inp, torch.Tensor):
                captures[name + "_input"] = inp.detach().float()
        return fn

    hooks = []
    hooks.append(lm.embed_tokens.register_forward_hook(hook("embed_tokens")))
    hooks.append(layer0.input_layernorm.register_forward_hook(hook("input_layernorm")))
    hooks.append(layer0.self_attn.register_forward_hook(hook("self_attn")))
    hooks.append(layer0.post_attention_layernorm.register_forward_hook(hook("post_attn_norm")))
    hooks.append(layer0.pre_feedforward_layernorm.register_forward_hook(hook("pre_ff_norm")))
    hooks.append(layer0.mlp.register_forward_hook(hook("mlp")))
    hooks.append(layer0.post_feedforward_layernorm.register_forward_hook(hook("post_ff_norm")))

    if hasattr(layer0, 'per_layer_input_gate'):
        hooks.append(layer0.per_layer_input_gate.register_forward_hook(hook("per_layer_gate")))
        hooks.append(layer0.per_layer_projection.register_forward_hook(hook("per_layer_proj")))
        hooks.append(layer0.post_per_layer_input_norm.register_forward_hook(hook("post_pli_norm")))

    # Also hook the full layer to get final output
    hooks.append(layer0.register_forward_hook(hook("layer0_output")))

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    for h in hooks:
        h.remove()

    last_pos = inputs["input_ids"].shape[1] - 1
    print(f"\n[Last token position] {last_pos}")

    # Print all captured values at last token position
    order = [
        "embed_tokens",
        "input_layernorm",
        "self_attn",
        "post_attn_norm",
        "pre_ff_norm",
        "mlp",
        "post_ff_norm",
        "per_layer_gate",
        "per_layer_proj",
        "post_pli_norm",
        "layer0_output",
    ]

    for name in order:
        if name in captures:
            t = captures[name]
            if t.dim() == 3:
                h = t[0, last_pos, :].numpy()
            elif t.dim() == 2:
                h = t[0, :].numpy()
            else:
                print(f"[{name}] unexpected shape: {t.shape}")
                continue
            norm = np.linalg.norm(h)
            max_val = np.max(np.abs(h))
            first8 = [f"{v:.6f}" for v in h[:8]]
            print(f"[{name}] norm={norm:.4f} max={max_val:.4f} dim={h.shape[0]} first8={first8}")

    # Also print residual points (computed manually)
    if "embed_tokens" in captures and "post_attn_norm" in captures:
        emb = captures["embed_tokens"][0, last_pos, :].numpy()
        post_attn = captures["post_attn_norm"][0, last_pos, :].numpy()
        after_attn_residual = emb + post_attn
        norm = np.linalg.norm(after_attn_residual)
        first8 = [f"{v:.6f}" for v in after_attn_residual[:8]]
        print(f"[after_attn_residual_add] norm={norm:.4f} first8={first8}")

    if "post_attn_norm" in captures and "post_ff_norm" in captures and "embed_tokens" in captures:
        emb = captures["embed_tokens"][0, last_pos, :].numpy()
        post_attn = captures["post_attn_norm"][0, last_pos, :].numpy()
        after_attn_res = emb + post_attn
        post_ff = captures["post_ff_norm"][0, last_pos, :].numpy()
        after_ff_residual = after_attn_res + post_ff
        norm = np.linalg.norm(after_ff_residual)
        first8 = [f"{v:.6f}" for v in after_ff_residual[:8]]
        print(f"[after_ff_residual_add] norm={norm:.4f} first8={first8}")

    # Top logits
    logits = outputs.logits[0, last_pos, :]
    top_values, top_indices = torch.topk(logits, 5)
    print(f"\n[Top 5 logits]")
    for i in range(5):
        tid = top_indices[i].item()
        val = top_values[i].item()
        decoded = tokenizer.decode([tid])
        print(f"  id={tid} logit={val:.4f} token={repr(decoded)}")

    # Layer scale
    print(f"\n[layer_scalar] {layer0.layer_scalar.item():.8f}")

if __name__ == "__main__":
    main()
