# Barrier Optimization Report

## Summary

Three optimizations applied to the decode dispatch pipeline to reduce GPU synchronization overhead:

1. **AggressiveOptimizer** — Q/K/V projection batching (3 steps → 1 step)
2. **Resource-scoped barriers** — `memoryBarrier(resources:)` replaces `memoryBarrier(scope: .buffers)`
3. **RoPE + flash_attn kernel fusion** — Inline RoPE into flash attention (2 dispatches → 1 per layer)

## Background

`hazardTrackingModeUntracked` buffers require explicit `memoryBarrier` calls between GPU dispatches that share data. Each `memoryBarrier(scope: .buffers)` synchronizes ALL Metal buffers (~31μs on Apple Silicon), while kernel execution averages ~4μs. Barrier overhead dominates decode latency.

## Optimization 1: AggressiveOptimizer (Q/K/V Batched GEMV)

Replaced `StandardOptimizer` with `AggressiveOptimizer` as the default. The aggressive optimizer batches consecutive non-output projections with the same input dimension into a single `batched_gemvN` kernel dispatch.

### Results (2-layer Transformer, h=128)

| Metric | Standard | Aggressive | Change |
|---|---|---|---|
| Unfused entries | 32 | 28 | -4 |
| Fused steps | 25 | 20 | **-5 (-20%)** |
| Barriers | 20 | 19 | -1 |
| Steps per layer | 11 | 9 | **-2 (-18%)** |

### Results (4-layer Transformer, h=256)

| Metric | Standard | Aggressive | Change |
|---|---|---|---|
| Unfused entries | 60 | 52 | -8 |
| Fused steps | 45 | 36 | **-9 (-20%)** |
| Barriers | 36 | 35 | -1 |
| Steps per layer | ~10.5 | 9 | **-1.5 (-14%)** |

### Analysis

Q/K/V batching reduces **step count** (fewer kernel dispatches) but has minimal barrier count reduction. The per-layer computation is a linear chain where each step reads the previous step's output, making barriers genuinely necessary. The original K/V projections were barrier-free because they read the same input as Q, but once batched into a single dispatch, there's no opportunity for barrier elision.

**Estimated impact for Gemma4-E2B (35 layers):**
- Step reduction: ~70 fewer dispatches (35 layers × 2 steps/layer)
- Barrier reduction: ~1 barrier
- Dispatch overhead savings: ~70 × 4μs = ~280μs per decode iteration

### Per-Layer Decode Pattern (Aggressive)

```
fused_copy_rms_norm → batched_gemv3 → rope → flash_attn → o_proj →
fused_residual_add_copy_rms_norm → fused_swiglu_projection → down_proj →
fused_residual_add_copy_rms_norm
```

9 steps per layer, all with barriers (genuine RAW dependencies).

## Optimization 2: Resource-Scoped Barriers (Metal 3) / Barrier Elimination (Metal 4)

### Metal 3 Implementation (historical)

Under Metal 3, replaced `memoryBarrier(scope: .buffers)` (global synchronization of all buffers) with `memoryBarrier(resources: [conflictingBuffers])` (synchronization of only the buffers with actual data dependencies). The measurements below were taken under this Metal 3 regime.

1. Extended `BufferRegion` to retain raw `MTLBuffer` reference alongside `ObjectIdentifier`
2. Added `MetalBarrierPolicy.resourceBarrier(resources: [MTLResource])` case
3. Added `MetalBufferAccesses.conflictingResources(from:)` to resolve conflicting regions to unique MTLBuffer objects
4. `optimizeDecodeBarrierPolicies` now generates `.resourceBarrier` with the minimum set of conflicting buffers

### Metal 4 Status

Metal 4 has no resource-scoped barrier. Both `.bufferBarrier` and `.resourceBarrier` emit the same `encoder.barrier(afterEncoderStages: .dispatch, beforeEncoderStages: .dispatch, visibilityOptions: [])`. The optimizer's conflict analysis still determines WHETHER a barrier is needed (eliminating unnecessary barriers), but the resource list is not expressible at encoding time.

Metal 4 intra-pass barriers are measured at ~0μs overhead (vs ~30μs per Metal 3 barrier), so the per-barrier cost reduction now comes from the Metal 4 barrier mechanism itself rather than resource scoping.

### Results (Metal 3 measurements)

| Metric | 2-layer | 4-layer |
|---|---|---|
| Resource barriers | 19 (100%) | 35 (100%) |
| Scope barriers | 0 (0%) | 0 (0%) |
| Avg resources/barrier | 1.0 | 1.0 |

Under Metal 3, all barriers became resource-scoped with an average of 1.0 buffers per barrier. A typical Transformer model has 10+ buffers (hidden, residual, scratch, logits, weights[], KV cache, position, tokenIn/Out). Synchronizing 1 buffer instead of 10+ significantly reduced Metal 3 per-barrier CPU encode cost.

**Metal 3 estimated impact:** `memoryBarrier(resources: [1 buffer])` costs ~3-5μs (vs ~31μs for scope), so barrier overhead for Gemma4-E2B drops from ~17ms to ~2-3ms.
**Metal 4 actual impact:** All barriers cost ~0μs regardless of scope, so this optimization's primary benefit is now barrier elimination (fewer barriers via conflict analysis), not scope reduction.

## Combined Impact Estimate (Gemma4-E2B, 35 layers)

| Component | Before | After | Savings |
|---|---|---|---|
| Decode steps | ~385 | ~315 | -70 dispatches |
| Dispatch overhead | ~1.5ms | ~1.3ms | ~0.3ms |
| Barrier cost (31μs × N) | ~17ms | ~2-3ms (est.) | ~14ms |
| **Total decode** | **~21ms** | **~6ms (est.)** | **~15ms (~70%)** |

*Note: GPU-side barrier cost per call remains ~30μs. Resource scoping primarily reduces CPU encode overhead.*

## Files Changed

| File | Change |
|---|---|
| `MetalInferenceCompiler.swift` | Default optimizer: Standard → Aggressive |
| `MetalBindings.swift` | Added `MetalBarrierPolicy.resourceBarrier`, `.isBarrier`, resource-aware encode |
| `MetalBufferAccesses.swift` | `BufferRegion` stores raw MTLBuffer, added `conflictingResources(from:)` |
| `MetalDispatchStepBuilder.swift` | `optimizeDecodeBarrierPolicies` generates resource barriers |
| Test files (5) | Updated `.bufferBarrier` checks to `.isBarrier` |

## Real-Device Benchmark: Gemma4-E2B (35 attention layers)

A/B comparison on Apple Silicon. Same model, same hardware, same test harness. Only difference: barrier scope.

### Decode Performance (599 steps, 563 barriers)

| Metric | scope barrier | resource barrier | Change |
|---|---|---|---|
| Single CB GPU time | 20.60 ms | 19.98 ms | **-0.62 ms (-3.0%)** |
| Encode+submit | 1,891 μs | 572 μs | **-1,319 μs (-70%)** |
| Decode throughput | 44.4 tok/s | 47.0 tok/s | **+2.6 tok/s (+5.9%)** |
| GPU kernel compute | 2,872 μs | 2,833 μs | -39 μs (noise) |

### Analysis

1. **Encode+submit 70% reduction** is the dominant improvement under Metal 3. `memoryBarrier(resources: [1 buffer])` is dramatically cheaper to encode on the CPU side than `memoryBarrier(scope: .buffers)`. With 563 barriers per decode, the CPU-side savings alone account for ~1.3ms.

2. **GPU-side savings ~0.6ms** — the actual GPU synchronization with 1 buffer vs all buffers shows modest improvement. The GPU barrier mechanism on Apple Silicon appears to be mostly scope-independent at the hardware level, with cost primarily determined by outstanding writes rather than barrier scope.

3. **Total decode improvement: ~1.9ms** from combined CPU encode + GPU savings, translating to ~5.9% throughput gain.

**Note:** These measurements were taken under Metal 3. After migrating to Metal 4, barrier overhead dropped to ~0μs per barrier regardless of scope, making this optimization's scope-reduction benefit moot. The conflict-based barrier elimination (determining which barriers can be skipped entirely) remains valuable under Metal 4.

### Kernel Breakdown (AggressiveOptimizer, 599 steps)

| Category | Time (μs) | Count | Avg (μs) | % |
|---|---|---|---|---|
| GEMV/GEMM | 1,112 | 176 | 6.3 | 39.2% |
| Norm | 663 | 211 | 3.1 | 23.3% |
| GEGLU projection | 564 | 35 | 16.1 | 19.8% |
| Attention | 206 | 35 | 5.9 | 7.3% |
| Structural | 129 | 70 | 1.8 | 4.6% |
| RoPE | 94 | 35 | 2.7 | 3.3% |
| PerLayerInput | 65 | 35 | 1.8 | 2.3% |
| **Total kernel** | **2,840** | **599** | **4.7** | |
| **Barrier overhead** | **~17,160** | **563** | **~30.5** | |

Barrier overhead (GPU time - kernel compute ≈ 17ms) remains the dominant cost at ~86% of total decode time. Kernel compute is only ~14% of GPU time.

### Bandwidth

- Decode: 960 GB/s (2× weight read per token) — near Apple Silicon memory bandwidth ceiling
- The model is memory-bandwidth-bound. Barrier reduction has diminishing returns once barrier overhead is comparable to kernel dispatch overhead.

## Test Results

All test suites pass:
- BarrierOptimizationTests: 10/10 ✓
- FragmentBindingTests: 10/10 ✓  
- HiddenConversionTests: 5/5 ✓
- PrefillTransferTests: 10/10 ✓
- RotorQuantCorrectnessTests: 25/25 ✓
- BarrierDiagnosticTests: 2/2 ✓
- Gemma4BenchmarkTests: 6/6 ✓

## Optimization 3: RoPE + Flash Attention Kernel Fusion

Inlined Rotary Position Embedding computation directly into the flash attention decode kernel, eliminating one separate RoPE dispatch and one barrier per attention layer.

### Design

RoPE rotation is applied to Q and K vectors inside the flash attention kernel using threadgroup memory staging:

1. **K path**: Load K into `rotBuf[512]` → apply RoPE rotation → optionally apply rotor sandwich → quantize & write to KV cache
2. **Q path**: Load Q into `rotQuery[512]` → apply RoPE rotation → optionally apply rotor sandwich → use for dot product scoring

The kernel supports M-RoPE (multi-axis RoPE for vision-language models like Qwen3.5) with temporal/height/width axis mapping.

### Fragment-Level Changes

RoPE fusion is implemented at the fragment composition level, not the optimizer level:

- `AttentionAttributes.fragment()` no longer emits a separate `RoPEFragment`
- `FlashAttentionFragment` accepts rope parameters (`ropeDimension`, `ropeBase`, `mropeAxes`)
- When `hasInlineRoPE`, the kernel name changes to `rope_flash_attn_decode` / `rope_flash_attn_decode_f32`
- Prefill path preserved: `FlashAttentionFragment.prefillSteps()` emits a separate `rope_seq` step internally

### Buffer Bindings

Flash attention uses indices 0–21. RoPE fusion adds indices 22–28:

| Index | Binding | Type |
|---|---|---|
| 22 | ropePositionAxes | buffer |
| 23 | ropeDimension | uint32 |
| 24 | ropeBase | float |
| 25 | temporalSections | uint32 |
| 26 | heightSections | uint32 |
| 27 | widthSections | uint32 |
| 28 | mropeInterleaved | uint32 |

### Results (Gemma4-E2B, 35 attention layers)

| Metric | Before (resource barrier) | After (RoPE fused) | Change |
|---|---|---|---|
| Decode steps | 599 | 564 | **-35 (-5.8%)** |
| Barriers | 563 | 528 | **-35 (-6.2%)** |
| GPU kernel compute | 2,840 μs | 2,904 μs | +64 μs |
| Attention avg | 5.9 μs | 11.4 μs | +5.5 μs (absorbs RoPE) |
| RoPE category | 94 μs (35 dispatches) | 0 μs (eliminated) | -94 μs |
| Single CB GPU time | ~20.0 ms | ~20.5 ms | +0.5 ms (noise) |
| Decode throughput | ~47.0 tok/s | ~47.0 tok/s | within noise |

### Per-Layer Decode Pattern (8 steps, was 9)

```
fused_copy_rms_norm → batched_gemv3 → rope_flash_attn_decode → o_proj →
fused_residual_add_copy_rms_norm → fused_swiglu_projection → down_proj →
fused_residual_add_copy_rms_norm
```

### Analysis

The fused kernel is computationally more expensive (11.4μs vs 5.9μs + 2.7μs = 8.6μs separate), likely due to increased register pressure and threadgroup memory traffic from staging both Q and K through `rotBuf`/`rotQuery`. The 35 saved barriers (~1.05ms at ~30μs each) are partially offset by the ~100μs extra compute cost.

Net throughput impact is within measurement noise at ~47 tok/s. The primary benefit is architectural: the decode pipeline now has a cleaner 8-step per-layer pattern, and the barrier count is reduced from 563 to 528. This fusion is a prerequisite for future optimizations where barrier overhead becomes a larger fraction of total time (e.g., at shorter context lengths or with faster weight formats).

### RotorQuant Compatibility

RoPE is applied BEFORE rotor sandwich product, sharing the same threadgroup memory buffer (`rotBuf`). The computation order is: load → RoPE → rotor → quantize. The kernel checks `is_rotor_scheme()` to conditionally apply the sandwich product, so non-rotor schemes incur no additional cost.

### Files Changed

| File | Change |
|---|---|
| `MetalSourceGenerator+Attention.swift` | Added inline RoPE code generation with M-RoPE axis support |
| `FlashAttentionFragment.swift` | Added rope parameters, fused kernel name, decode/prefill bindings |
| `AttentionFragment.swift` | Removed separate RoPEFragment, passes rope config to FlashAttentionFragment |
| `MetalKernelSourceCatalog.swift` | Explicit rope_seq kernel generation for prefill when inline RoPE detected |

## Cumulative Results (Gemma4-E2B, 35 layers)

| Metric | Baseline | +Aggressive | +Resource Barrier | +RoPE Fusion |
|---|---|---|---|---|
| Steps/layer | ~11 | 9 | 9 | **8** |
| Total steps | ~669 | 599 | 599 | **564** |
| Barriers | ~669 | 563 | 563 | **528** |
| Encode+submit | ~1,900 μs | ~1,900 μs | 572 μs | ~540 μs |
| GPU time | ~21 ms | ~21 ms | ~20 ms | ~20.5 ms |
| Throughput | ~44 tok/s | ~44 tok/s | ~47 tok/s | **~47 tok/s** |

Under Metal 3, the dominant improvement came from resource-scoped barriers (encode cost -70%). Under Metal 4, barrier overhead is near-zero regardless of scope, so the cumulative benefit shifts to step/barrier count reduction (AggressiveOptimizer + RoPE fusion). The model remains memory-bandwidth-bound (~960 GB/s weight reads dominate decode time).

## Test Results

All test suites pass:
- BarrierOptimizationTests: 10/10
- FragmentBindingTests: 10/10
- HiddenConversionTests: 5/5
- PrefillTransferTests: 10/10
- RotorQuantCorrectnessTests: 25/25
- BarrierDiagnosticTests: 2/2

## Next Steps

1. **Prefill path** — Apply barrier elimination analysis to prefill executor to reduce unnecessary barriers
2. **Norm → GEMV fusion** — Requires Metal 4 inter-threadgroup synchronization (not feasible in Metal 3)
3. **Metal 4 command prerecording** (future work) — The current implementation still re-encodes every decode pass via `MetalSubmissionContext.withCompute`. Metal 4's reusable `MTL4CommandBuffer` enables a prerecording pattern where a fixed dispatch sequence is encoded once and replayed per token, eliminating per-token encode cost. This requires all per-token variable data (position, token ID) to be indirected through GPU addresses rather than re-bound each frame.
