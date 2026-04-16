#!/usr/bin/env python3
"""Dump all parameters from a HuggingFace safetensors model.

Outputs: name, shape, dtype, norm (L2), mean, std, min, max, first 8 values.
Used to record ground-truth parameter values for comparison with swift-lm.
"""

import sys
import argparse
from pathlib import Path
import numpy as np
from safetensors import safe_open


def stats_for_tensor(name: str, t) -> dict:
    """Compute statistics for a tensor. t is a torch tensor."""
    import torch
    # Cast to float32 for stable stats
    if t.dtype in (torch.bfloat16, torch.float16, torch.float64):
        f = t.float()
    elif t.dtype in (torch.int8, torch.uint8, torch.int16, torch.int32, torch.int64):
        f = t.float()
    else:
        f = t.float() if t.dtype != torch.float32 else t
    flat = f.reshape(-1)
    n = flat.numel()
    norm = float(torch.linalg.vector_norm(flat).item())
    mean = float(flat.mean().item())
    std = float(flat.std().item()) if n > 1 else 0.0
    mn = float(flat.min().item())
    mx = float(flat.max().item())
    first8 = [float(v) for v in flat[:8].tolist()]
    return {
        "name": name,
        "shape": tuple(t.shape),
        "dtype": str(t.dtype).replace("torch.", ""),
        "numel": n,
        "norm": norm,
        "mean": mean,
        "std": std,
        "min": mn,
        "max": mx,
        "first8": first8,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", help="Path to HuggingFace model snapshot directory")
    parser.add_argument("--output", "-o", help="Output file path (default: stdout)")
    parser.add_argument("--format", choices=["text", "md"], default="md")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    safetensor_files = sorted(model_dir.glob("*.safetensors"))
    if not safetensor_files:
        print(f"No safetensors found in {model_dir}", file=sys.stderr)
        sys.exit(1)

    all_stats = []
    total_params = 0
    for st_file in safetensor_files:
        with safe_open(str(st_file), framework="pt") as f:
            for key in f.keys():
                t = f.get_tensor(key)
                s = stats_for_tensor(key, t)
                s["file"] = st_file.name
                all_stats.append(s)
                total_params += s["numel"]

    # Sort by name
    all_stats.sort(key=lambda s: s["name"])

    def emit(line):
        if args.output:
            fh.write(line + "\n")
        else:
            print(line)

    fh = open(args.output, "w") if args.output else None
    try:
        if args.format == "md":
            emit(f"# Parameter Dump: {model_dir.name}")
            emit("")
            emit(f"- **Path**: `{model_dir}`")
            emit(f"- **Files**: {', '.join(f.name for f in safetensor_files)}")
            emit(f"- **Total parameters**: {total_params:,} ({total_params / 1e9:.3f}B)")
            emit(f"- **Tensor count**: {len(all_stats)}")
            emit("")
            emit("| # | Name | Shape | Dtype | Norm | Mean | Std | Min | Max | First 8 |")
            emit("|---|------|-------|-------|------|------|-----|-----|-----|---------|")
            for i, s in enumerate(all_stats):
                shape = "×".join(str(d) for d in s["shape"])
                first8 = ", ".join(f"{v:.4g}" for v in s["first8"])
                emit(f"| {i} | `{s['name']}` | {shape} | {s['dtype']} | {s['norm']:.4g} | {s['mean']:.4g} | {s['std']:.4g} | {s['min']:.4g} | {s['max']:.4g} | {first8} |")
        else:
            emit(f"# {model_dir.name}")
            emit(f"# Total params: {total_params:,}")
            for s in all_stats:
                shape = "x".join(str(d) for d in s["shape"])
                first8 = ", ".join(f"{v:.6g}" for v in s["first8"])
                emit(f"{s['name']:<80s} shape={shape:<30s} dtype={s['dtype']:<10s} norm={s['norm']:.6g} mean={s['mean']:.6g} std={s['std']:.6g} min={s['min']:.6g} max={s['max']:.6g} first8=[{first8}]")
    finally:
        if fh:
            fh.close()

    if args.output:
        print(f"Wrote {len(all_stats)} tensors ({total_params:,} params) to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
