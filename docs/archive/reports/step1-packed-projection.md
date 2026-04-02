# Step 1: PackedProjection — Projection Packing Fusion

**Date**: 2026-03-11
**Module**: `MLXCompiler` (swift-mlx-lm)
**Hardware**: M4 Max

---

## Background

Compiled inference path executes Q/K/V projections as 3 separate `matmul` (or `quantizedMM`) dispatches per attention layer, and Gate/Up as 2 separate dispatches per MLP layer. For a 32-layer model, this means 32 × 5 = 160 projection dispatches per decode token.

Baseline benchmarks (Step 0) showed layer scaling of 4L→32L = 1.7x instead of expected 8x, confirming that **Metal kernel dispatch overhead is dominant** at small model dimensions.

### Insight from mlx-swift ANE Reports

The ANE investigation reports (`mlx-swift/reports/`) provided key insights applicable to Metal optimization:

1. **Dispatch overhead is the primary bottleneck** — ANE reports showed ~0.1ms per dispatch, and Metal op-by-op FFN was 2-12x slower than fused kernels due to accumulated dispatch costs
2. **DRAM bandwidth dominates** — intermediate values written to VRAM between ops create 10x bandwidth overhead vs on-chip retention
3. **Reducing kernel count is the highest-leverage optimization** — ANE FFN fusion (15 ops → 1 dispatch) achieved 2-12x speedup; the same principle applies to Metal matmul packing

---

## Design

### PackedProjection

Concatenate multiple projection weights along axis=0 (output dimension), perform a single matmul/quantizedMM, then split the result along the last axis to recover individual outputs.

```
Before (3 dispatches):
  Q = matmul(x, Wq.T)    ← dispatch 1, reads x from VRAM
  K = matmul(x, Wk.T)    ← dispatch 2, reads x from VRAM again
  V = matmul(x, Wv.T)    ← dispatch 3, reads x from VRAM again

After (1 dispatch):
  QKV = matmul(x, [Wq; Wk; Wv].T)   ← dispatch 1, reads x once
  Q, K, V = split(QKV, indices)       ← near-zero cost (view operation)
```

### Kernel Variant Support

Packing requires all projections to use the same kernel variant:

| Variant | Packing Method | Constraint |
|---------|---------------|------------|
| `.dense` | Concat weights along axis=0 | None |
| `.affineQuantized` | Concat packedWeight, scales, zeroBiases along axis=0 | Same bits & groupSize |
| `.dequantizeMatmul` | Same as affineQuantized | Same bits & groupSize |
| Mixed | **Not packable** → fallback to individual projections | N/A |

### GQA Support

GQA models have `qDim ≠ kvDim` (e.g., Q=128, K=64, V=64). PackedProjection handles this via non-uniform split indices:

```
splitIndices = [128, 192]  →  split produces [128], [64], [64]
```

---

## Implementation

### New File

**`Sources/MLXCompiler/Lowered/PackedProjection.swift`**

```swift
public struct PackedProjection: @unchecked Sendable {
    public let kernel: ProjectionKernel
    public let packedBias: MLXArray?
    public let splitIndices: [Int]

    public func apply(_ x: MLXArray) -> [MLXArray]
    public static func pack(_ projections: [LoweredProjection]) -> PackedProjection?
}
```

- `pack()` returns `nil` if packing is not possible (mixed variants, incompatible quantization)
- `apply()` dispatches single matmul + split
- Bias handling: if any projection has bias, all are concatenated (nil → zero-filled)

### Modified Files

**`Sources/MLXCompiler/Lowered/LoweredAttention.swift`**

- Added `qkvPacked: PackedProjection?` field
- Individual `qProj/kProj/vProj` become optional (nil when packed)
- Two init paths: packed (preferred) and individual (fallback)
- `apply()` branches on `qkvPacked != nil`

**`Sources/MLXCompiler/Lowered/LoweredMLP.swift`**

- Added `gateUpPacked: PackedProjection?` field
- Individual `gateProj/upProj` become optional (nil when packed)
- Two init paths: packed (preferred) and individual (fallback)
- `apply()` branches on `gateUpPacked != nil`

**`Sources/MLXCompiler/MLXInferenceCompiler.swift`**

- `lowerAttention()`: creates individual projections, then attempts `PackedProjection.pack([q, k, v])`. Falls back to individual init on failure.
- `lowerMLP()`: for gated variants, attempts `PackedProjection.pack([gate, up])`. Falls back to individual init on failure.

### New Test File

**`Tests/MLXCompilerTests/PackedProjectionTests.swift`** — 11 tests

| Test | What it verifies |
|------|-----------------|
| Dense QKV packed | Packed output == individual output (exact match) |
| Dense Gate+Up packed | Packed output == individual output (exact match) |
| Dense GQA (H≠KVH) | Non-uniform split indices, correct output shapes |
| Dense with biases | Bias concatenation and correct application |
| Dense mixed bias | Some projections with bias, some without |
| Single projection → nil | Requires >= 2 projections |
| Empty list → nil | Edge case |
| AffineQuantized packing | Same bits/groupSize → pack succeeds, correct metadata |
| Mixed kernel → nil | Dense + quantized → pack fails, returns nil |
| Attention integration | Packed vs individual attention produce same output |
| MLP integration | Packed vs individual MLP produce same output |

---

## Benchmark Results

### Test Configuration

- Model: BenchTransformer (Llama-style), D=64, headCount=4, headDim=16, intermediateSize=128, vocabSize=32
- Benchmark: warmup=10, iterations=100 (decode), warmup=5, iterations=30 (prefill)
- Hardware: M4 Max

### Decode Latency

| Config | Before (Step 0) | After (Step 1) | Improvement |
|--------|-----------------|----------------|-------------|
| 8L D=64 decode | 23.7 ms | 20.5 ms | **-13.5%** |
| 16L D=64 decode | 29.6 ms | 24.9 ms | **-15.9%** |
| GQA 8L H=8 KVH=2 | — | 21.0 ms | (new baseline) |

### Prefill Latency

| Config | Before (Step 0) | After (Step 1) | Improvement |
|--------|-----------------|----------------|-------------|
| 64tok 8L D=64 | 38.0 ms | 35.2 ms | **-7.4%** |

### Dispatch Count Reduction

| Per layer | Before | After | Reduction |
|-----------|--------|-------|-----------|
| Attention projections | 3 (Q, K, V) | 1 (QKV packed) | -2 |
| MLP projections | 2 (Gate, Up) | 1 (GateUp packed) | -1 |
| **Total per layer** | **5** | **2** | **-3 (-60%)** |

For a 32-layer model: 160 → 64 projection dispatches per decode token.

---

## Analysis

### Why Decode Benefits More Than Prefill

- **Decode (B=1, L=1)**: matmul is tiny, dispatch overhead dominates. Reducing dispatch count has maximum relative impact.
- **Prefill (B=1, L=64)**: matmul is larger, compute starts to dominate over dispatch overhead. Packing still helps via input VRAM read reduction (x read once instead of 3 times).

### Comparison with ANE Reports

The ANE reports showed that for D=1024 FFN:
- Metal op-by-op: ~0.9ms/layer
- ANE fused (15 ops → 1 dispatch): ~0.3ms/layer → **3.3x speedup**

Our PackedProjection achieves a more modest improvement because:
1. We're only reducing 5→2 dispatches (not 15→1)
2. MLX's lazy evaluation already coalesces some element-wise ops
3. The benchmark model is very small (D=64), making dispatch overhead proportionally larger

For production models (D=2048-4096), the dispatch overhead fraction is smaller, but the VRAM bandwidth savings from reading input `x` once instead of 3 times becomes more significant.

---

## Correctness Verification

All tests verify packed output against individual projection output:

| Test Type | Tolerance | Result |
|-----------|-----------|--------|
| Dense (exact) | < 1e-5 | PASS |
| Dense with bias | < 1e-5 | PASS |
| GQA dimensions | < 1e-5 | PASS |
| Attention integration | < 1e-4 | PASS |
| MLP integration | < 1e-4 | PASS |
| Quantized metadata | exact match | PASS |

143 existing compiler tests + 11 new tests = **154 tests PASS**.

---

## Files

| File | Change |
|------|--------|
| `Sources/MLXCompiler/Lowered/PackedProjection.swift` | New: PackedProjection struct |
| `Sources/MLXCompiler/Lowered/LoweredAttention.swift` | Modified: qkvPacked field, dual init |
| `Sources/MLXCompiler/Lowered/LoweredMLP.swift` | Modified: gateUpPacked field, dual init |
| `Sources/MLXCompiler/MLXInferenceCompiler.swift` | Modified: auto-packing in lowerAttention/lowerMLP |
| `Tests/MLXCompilerTests/PackedProjectionTests.swift` | New: 11 tests |

---

## Next Steps

Step 1 confirms that dispatch count reduction improves performance. The next optimizations build on this:

| Step | Description | Expected Impact |
|------|-------------|-----------------|
| **Step 2**: FusedBlock | Recognize IR patterns (`.residual([.rmsNorm, .attention])`) and emit fused execution blocks | Eliminate FlatStep switch overhead, reduce residual stack allocations |
| **Step 3**: Layer Template Reuse | `MLX.compile()` for `.repeating(N)` → compile once, execute N times | Metal command buffer reuse, compiled graph 1 instead of N |
| **Step 4**: Loop-Invariant Hoisting | Hoist attention mask and RoPE offset computation out of layer loop | Remove redundant per-layer computation |
| **Step 5**: KV Cache Prefix Reuse | Fix P2/P3 gaps in CompiledKVCache | Enable PrefixCachePool for compiled path |
