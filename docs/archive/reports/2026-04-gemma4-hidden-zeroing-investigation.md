# Gemma4 Prefill Hidden Zeroing Investigation Report

Date: 2026-04-16

## Problem

Gemma4 E2B model produces garbage output (`" عج<pad><pad>..."`) instead of `"Tokyo"` for the prompt `"What is the capital of Japan? Answer with exactly one word."`.

## Investigation Timeline

### 1. Full-Step Probe Artifact

The initial probe (capturing all 589 steps individually) reported hidden zeroing at step 27 (layer 1, `kv_cache_fill_seq_f32`). This was a **probe mechanism artifact**.

**Root cause of artifact**: `captureLastTokenHiddenSnapshots` runs each step in its own `withCompute` call (= independent MTL4CommandBuffer + fresh MTL4ArgumentTable). In normal execution, multiple steps share the same encoder and argument table. Per-step probe creates a fresh table where bindings from prior steps (e.g., runtime constant buffer for `sequenceLength`) are absent, causing abnormal kernel behavior.

**Verification**: A targeted probe capturing only steps 20-34 showed hidden at step 27 was normal (norm=169.92). `kv_cache_fill` does not affect hidden.

### 2. Actual Failure Point: Layer 26-27 (Non-deterministic)

Coarse-grained layer-boundary probe (capturing only 5-way synthesized steps) identified the true failure:

| Run | Last Normal Layer | First Zero Layer | Zeroing Kernel |
|-----|-------------------|------------------|----------------|
| 1st | Layer 25 (step 451, norm=227.03) | Layer 26 (step 466, norm=0) | `synthesized_5way` |
| 2nd | Layer 26 (step 466, norm=550.38) | Layer 27 (step 472, norm=0) | `synthesized_4way` |

**Critical**: The failure point shifts between runs. This indicates a **non-deterministic issue** — race condition, missing barrier, or uninitialized buffer read.

### 3. Buffer Configuration

Buffer aliasing ruled out:

| Buffer | Storage Mode | Hazard Tracking | Separate MTLBuffer |
|--------|-------------|-----------------|-------------------|
| hidden (prefill) | `storageModeShared` | ON (default) | Yes |
| residual (prefill) | `storageModePrivate` | ON (default) | Yes |
| scratch (prefill) | `storageModePrivate` | ON (default) | Yes |
| KV cache | `storageModePrivate` | ON (default) | Yes |

All allocated via independent `device.makeBuffer()` calls. No memory address overlap.

### 4. Layer 26 Internal Step Structure

Layers 15+ are KV-cache-sharing layers. No `kv_cache_fill` step; flash_attn references shared KV cache.

```
step=451  layer=25  synthesized_5way (layer boundary)   norm=236.31 OK
step=452  layer=-   gemm (q_proj)                       norm=236.31 OK
step=453  layer=26  qk_rms_norm                         norm=236.31 OK
step=454  layer=26  rope                                norm=236.31 OK
step=455  layer=26  flash_attn_batch                    norm=236.31 OK
step=456  layer=-   gemm (o_proj)                       norm=48.98  OK
step=457  layer=26  synthesized_4way (sandwich norm)    norm=22.14  OK
step=458  layer=26  gemm (gate_proj)                    norm=22.14  OK
step=459  layer=26  gemm (up_proj)                      norm=22.14  OK
step=460  layer=26  geglu                               norm=22.14  OK
step=461  layer=-   gemm (down_proj)                    norm=30.53  OK
step=462  layer=26  synthesized_3way (post-MLP)         norm=3507.15 OK
step=463  layer=-   gemm (per_layer gate)               norm=3507.15 OK
step=464  layer=26  per_layer_input_modulation          norm=3507.15 OK
step=465  layer=-   gemm (per_layer proj)               norm=9157.37 OK
step=466  layer=26  synthesized_5way (layer boundary)   norm=550.38 OK (ZERO in 1st run)

step=467  layer=-   gemm (q_proj)                       norm=550.38 OK
step=468  layer=27  qk_rms_norm                         norm=550.38 OK
step=469  layer=27  rope                                norm=550.38 OK
step=470  layer=27  flash_attn_batch                    norm=550.38 OK
step=471  layer=-   gemm (o_proj)                       norm=55.63  OK
step=472  layer=27  synthesized_4way (sandwich norm)    norm=0.0000 ZERO (OK in 1st run)
```

### 5. Ruled Out Hypotheses

| Hypothesis | Status | Reason |
|------------|--------|--------|
| Buffer aliasing | Ruled out | All buffers are independent MTLBuffer objects |
| kv_cache_fill corrupts hidden | Ruled out | kv_cache_fill does not bind hidden; confirmed by targeted probe |
| Computation logic error | Unlikely | Same fragments used across all layers; early layers are correct |
| Missing weight data | Unlikely | Failure point is non-deterministic (weight issues cause fixed-position failures) |
| Full-step probe reliability | Unreliable | Fresh argument table artifact confirmed |

## Leading Hypothesis

**GPU synchronization/barrier issue**: Non-deterministic failure strongly suggests insufficient barriers in Metal 4's concurrent compute encoder.

Prefill buffers do not use `hazardTrackingModeUntracked` (automatic hazard tracking is ON), but `MTL4ComputeCommandEncoder` defaults to concurrent dispatch. Possible issues:
- Automatic hazard tracking may not correctly handle offset-based buffer access (scratch slots at different offsets within the same MTLBuffer)
- Synthesized kernels' compound buffer access patterns may not be fully tracked
- Explicit barrier policies on some steps may conflict with or be insufficient alongside automatic tracking

## Next Actions

1. **Examine barrier policies on synthesized_4way/5way kernels** — verify sufficient barriers exist when these fused kernels read results from prior steps
2. **Verify Metal 4 concurrent dispatch + hazard tracking interaction** — determine if explicit barriers are needed for the prefill path
3. **Reproducibility test** — run the same probe multiple times to record failure point distribution
4. **HuggingFace reference comparison** — confirm hidden states through layer 25 match HF output, validating computation logic correctness
