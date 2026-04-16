#!/usr/bin/env python3
"""Extract per-layer hidden states from HuggingFace Gemma4 for comparison with swift-lm."""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "google/gemma-4-E2B-it"
PROMPT = "What is the capital of Japan? Answer with exactly one word."

def main():
    print(f"[Loading tokenizer] {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    print("[Applying chat template]")
    messages = [{"role": "user", "content": PROMPT}]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    print(f"[Formatted prompt] {repr(formatted)}")

    inputs = tokenizer(formatted, return_tensors="pt")
    token_ids = inputs["input_ids"][0].tolist()
    print(f"[Token IDs] ({len(token_ids)} tokens) {token_ids}")

    print(f"\n[Loading model] {MODEL_ID}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16
    )
    model.eval()

    # Hook into layer 0 to capture intermediate states
    intermediate_captures = {}

    def make_hook(name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                intermediate_captures[name] = output[0].detach()
            else:
                intermediate_captures[name] = output.detach()
        return hook

    # Get the language model layers
    lm = model.model.language_model

    # Register hooks on layer 0 sub-modules
    hooks = []
    layer0 = lm.layers[0]
    for sub_name, sub_module in layer0.named_children():
        hooks.append(sub_module.register_forward_hook(make_hook(f"layer0.{sub_name}")))

    # Also hook the embedding layer
    hooks.append(lm.embed_tokens.register_forward_hook(make_hook("embed_tokens")))

    print("[Running forward pass with output_hidden_states=True]")
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    # Remove hooks
    for h in hooks:
        h.remove()

    hidden_states = outputs.hidden_states

    last_token_pos = inputs["input_ids"].shape[1] - 1

    # Print intermediate layer 0 captures
    print(f"\n{'='*80}")
    print(f"[Layer 0 intermediate states (last token position = {last_token_pos})]")
    print(f"{'='*80}")
    all_names = sorted(intermediate_captures.keys())
    for name in all_names:
        if name in intermediate_captures:
            t = intermediate_captures[name]
            if t.dim() == 3:
                h = t[0, last_token_pos, :].float().numpy()
            elif t.dim() == 2:
                h = t[0, :].float().numpy()
            else:
                print(f"  [{name}] unexpected shape: {t.shape}")
                continue
            norm = np.linalg.norm(h)
            max_val = np.max(np.abs(h))
            print(f"  [{name}] norm={norm:.4f}  max={max_val:.4f}  dim={h.shape[0]}")
            print(f"    first 8: {[f'{v:.6f}' for v in h[:8]]}")
        else:
            print(f"  [{name}] NOT CAPTURED")
    # hidden_states[0] = embedding output (before any transformer layer)
    # hidden_states[i] = output of layer i-1 (1-indexed: hidden_states[1] = after layer 0)
    # hidden_states[-1] = after final layer (= input to final norm)
    print(f"[Hidden states] {len(hidden_states)} entries (embedding + {len(hidden_states)-1} layers)")

    last_token_pos = inputs["input_ids"].shape[1] - 1
    print(f"[Last token position] {last_token_pos}")

    # Extract last-token hidden at key checkpoints
    checkpoints = [0, 1, 2, 5, 10, 15, 20, 25, 30, 34, 35]
    checkpoints = [c for c in checkpoints if c < len(hidden_states)]

    print(f"\n{'='*80}")
    print(f"[Per-layer hidden state norms and values (last token position)]")
    print(f"{'='*80}")

    for idx in checkpoints:
        h = hidden_states[idx][0, last_token_pos, :].float().numpy()
        norm = np.linalg.norm(h)
        max_val = np.max(np.abs(h))
        label = "embedding" if idx == 0 else f"layer {idx-1} output"
        if idx == len(hidden_states) - 1:
            label += " (final, pre-norm)"

        print(f"\n[{label}] idx={idx}")
        print(f"  norm={norm:.4f}  max={max_val:.4f}  shape={h.shape}")
        print(f"  first 16 values: {[f'{v:.6f}' for v in h[:16]]}")

    # Also compute logits manually for verification
    print(f"\n{'='*80}")
    print(f"[Final logits verification]")
    print(f"{'='*80}")

    logits = outputs.logits[0, last_token_pos, :]
    top_k = 10
    top_values, top_indices = torch.topk(logits, top_k)
    print(f"\n[Top {top_k} logits (raw, before softcapping)]")
    for i in range(top_k):
        tid = top_indices[i].item()
        val = top_values[i].item()
        decoded = tokenizer.decode([tid])
        print(f"  id={tid} logit={val:.4f} token={repr(decoded)}")

    # Check the RMSNorm weight values
    print(f"\n{'='*80}")
    print(f"[Layer 0 RMSNorm weight inspection]")
    print(f"{'='*80}")
    layer0 = lm.layers[0]
    for norm_name in ["input_layernorm", "post_attention_layernorm",
                      "pre_feedforward_layernorm", "post_feedforward_layernorm"]:
        if hasattr(layer0, norm_name):
            w = getattr(layer0, norm_name).weight.float().numpy()
            w_norm = np.linalg.norm(w)
            w_mean = np.mean(np.abs(w))
            w_max = np.max(np.abs(w))
            print(f"  [{norm_name}] norm={w_norm:.4f}  mean_abs={w_mean:.4f}  max_abs={w_max:.4f}")
            print(f"    first 8: {[f'{v:.6f}' for v in w[:8]]}")

    # Check embedding scale
    print(f"\n{'='*80}")
    print(f"[Embedding scale check]")
    print(f"{'='*80}")
    hidden_size = model.config.text_config.hidden_size
    print(f"  hidden_size={hidden_size}")
    print(f"  sqrt(hidden_size)={hidden_size**0.5:.4f}")

    # Check layer scale weight
    if hasattr(layer0, 'model_layer_scale'):
        scale_val = layer0.model_layer_scale.float().item()
        print(f"  layer_scale[0] = {scale_val:.6f}")
    for n, p in layer0.named_parameters():
        if 'scale' in n.lower():
            print(f"  param {n}: shape={list(p.shape)} first={p.flatten()[:4].tolist()}")

if __name__ == "__main__":
    main()
