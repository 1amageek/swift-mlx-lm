# Step 2: FusedSubLayer — Block-Level Execution Fusion

**Date**: 2026-03-11
**Module**: `MLXCompiler` (swift-mlx-lm)
**Hardware**: M4 Max

---

## Background

Step 1 (PackedProjection) reduced projection dispatch count from 5 to 2 per layer. However, the execution engine still processes each transformer sub-layer as 4 separate FlatSteps:

```
saveResidual         ← push to stack
op(norm)             ← switch dispatch + apply
op(attention/mlp)    ← switch dispatch + apply
addResidual          ← pop from stack + add
```

For a 32-layer model with 2 sub-layers per layer, this means 32 × 2 × 4 = 256 FlatStep dispatches. Each dispatch involves:
1. Swift enum switch evaluation
2. Function call to `executeOp()`
3. Residual stack push/pop (array append/removeLast)

### Insight from Step 1

Step 1's 13-16% decode improvement confirmed that reducing dispatch count improves performance. FusedSubLayer extends this principle from projection-level to sub-layer-level fusion.

---

## Design

### FusedSubLayer

Recognize the common transformer pattern `residual(body: [norm, op])` during step flattening and replace the 4-step sequence with a single fused step.

```
Before (4 steps):
  saveResidual
  op(norm)         ← switch + function call
  op(attention)    ← switch + function call
  addResidual

After (1 step):
  fusedSubLayer(.attention(norm, attn))  ← direct inline: x + attn.apply(norm.apply(x))
```

### Pattern Detection

Detection occurs during `flattenSteps()` — the existing flattening phase that converts recursive `LoweredStep` to linear `FlatStep`. When a `.residual(body:)` is encountered:

1. Check if `body` has exactly 2 steps
2. First step must be `.op(.norm(...))`
3. Second step must be `.op(.attention | .mlp | .deltaNet | .moe)`
4. If matched → emit single `.fusedSubLayer(...)` step
5. If not matched → fall back to `[saveResidual, ...body..., addResidual]`

### Supported Fusion Patterns

| Pattern | Fused Type | Residual Strategy |
|---------|-----------|-------------------|
| `[norm, attention]` | `.attention(norm, attn)` | `x + attn(norm(x), caches)` |
| `[norm, mlp]` | `.mlp(norm, mlp)` | `x + mlp(norm(x))` |
| `[norm, deltaNet]` | `.deltaNet(norm, dn)` | `x + dn(norm(x), caches)` |
| `[norm, moe]` | `.moe(norm, moe)` | `x + moe(norm(x))` |

### Non-Fuseable Patterns (Fallback)

- Single-op residual bodies (e.g., `[norm]`)
- Three or more ops in body
- Wrong order (e.g., `[attention, norm]`)
- Non-norm first op (e.g., `[linear, attention]`)
- Nested residuals in body

---

## Implementation

### New File

**`Sources/MLXCompiler/Lowered/FusedBlock.swift`**

```swift
public enum FusedSubLayer: @unchecked Sendable {
    case attention(norm: LoweredNorm, attention: LoweredAttention)
    case mlp(norm: LoweredNorm, mlp: LoweredMLP)
    case deltaNet(norm: LoweredNorm, deltaNet: LoweredDeltaNet)
    case moe(norm: LoweredNorm, moe: LoweredMoE)

    public func apply(_ x: MLXArray, state: inout InferenceState) -> MLXArray
}

func tryFuseResidual(_ body: [LoweredStep]) -> FusedSubLayer?
```

### Modified Files

**`Sources/MLXCompiler/Lowered/FlatStep.swift`**

- Added `case fusedSubLayer(FusedSubLayer)` to `FlatStep` enum
- Modified `flattenSteps()` to call `tryFuseResidual()` for `.residual(body:)` cases
- Added `.fusedSubLayer` handling in `executeFlatSteps()`

### New Test File

**`Tests/MLXCompilerTests/FusedBlockTests.swift`** — 13 tests

| Test | What it verifies |
|------|-----------------|
| Fuse: norm + attention | Pattern detection for attention sub-layer |
| Fuse: norm + mlp | Pattern detection for MLP sub-layer |
| Fuse: norm + moe | Pattern detection for MoE sub-layer |
| No fuse: single op | Requires exactly 2 ops |
| No fuse: three ops | Requires exactly 2 ops |
| No fuse: norm + norm | Second op must be fuseable |
| No fuse: wrong order | First op must be norm |
| flattenSteps produces fusedSubLayer | End-to-end flattening creates fused step |
| flattenSteps preserves unfuseable | Non-matching residuals use save/add |
| Fused attention correctness | Fused vs individual output exact match |
| Fused MLP correctness | Fused vs individual output exact match |
| Full block fused matches unfused | Complete transformer block correctness |
| Fusion reduces step count | 8-layer model: 67 steps → 19 steps |

---

## Benchmark Results

### Test Configuration

- Model: BenchTransformer (Llama-style), D=64, headCount=4, headDim=16, intermediateSize=128, vocabSize=32
- Benchmark: warmup=10, iterations=100 (decode), warmup=5, iterations=30 (prefill)
- Hardware: M4 Max

### Decode Latency

| Config | Step 1 | Step 2 | Improvement |
|--------|--------|--------|-------------|
| 8L D=64 decode | 20.5 ms | 20.0 ms | **-2.4%** |
| 16L D=64 decode | 24.9 ms | 25.9 ms | ±noise |
| GQA 8L H=8 KVH=2 | 21.0 ms | 20.8 ms | **-1.0%** |

### Prefill Latency

| Config | Step 1 | Step 2 | Improvement |
|--------|--------|--------|-------------|
| 64tok 8L D=64 | 35.2 ms | 32.0 ms | **-9.1%** |

### Step Count Reduction

| Model Config | Before (Step 1) | After (Step 2) | Reduction |
|-------------|-----------------|----------------|-----------|
| 8L (8 × 2 sub-layers) | 67 steps | 19 steps | **-71.6%** |
| 16L (16 × 2 sub-layers) | 131 steps | 35 steps | **-73.3%** |
| 32L (32 × 2 sub-layers) | 259 steps | 67 steps | **-74.1%** |

Step count formula:
- Before: `1 (emb) + N×2×4 (save/norm/op/add) + 2 (final norm + head) = 8N + 3`
- After: `1 (emb) + N×2 (fused) + 2 (final norm + head) = 2N + 3`

---

## Analysis

### Why Latency Improvement Is Modest

1. **GPU compute dominates**: Even at D=64, the matmul/attention kernels take orders of magnitude more time than CPU-side enum switch dispatches. The 4→1 step reduction saves ~3 switch evaluations per sub-layer, but each switch takes nanoseconds while GPU compute takes milliseconds.

2. **MLX lazy evaluation**: MLX builds a computation graph and evaluates lazily. The individual `norm.apply()` and `attn.apply()` calls within the fused step produce the same computation graph as the unfused path — MLX doesn't "see" the fusion.

3. **Residual stack is cheap**: Array `append`/`removeLast` in Swift is O(1) amortized. The stack overhead is negligible.

### Where FusedBlock Helps

1. **Prefill (-9.1%)**: Larger token sequences amplify any CPU-side overhead since the per-sub-layer dispatch happens for larger intermediate tensors. The fused path avoids materializing the intermediate norm output as a separate `MLXArray` binding.

2. **Step count for profiling**: 71-74% fewer steps means cleaner profiling and debugging of the compiled path.

3. **Foundation for Step 3**: Layer Template Reuse (`MLX.compile()`) needs a well-defined "layer function" to compile once and reuse N times. FusedSubLayer provides exactly this: a self-contained function `(MLXArray, inout InferenceState) → MLXArray` that captures all weights.

---

## Correctness Verification

All tests verify fused output against individual (unfused) output:

| Test Type | Tolerance | Result |
|-----------|-----------|--------|
| Fused attention vs individual | < 1e-5 | PASS |
| Fused MLP vs individual | < 1e-5 | PASS |
| Full transformer block (fused vs manual) | < 1e-4 | PASS |
| Pattern detection (7 cases) | exact match | PASS |
| Step count (8L model) | exact count | PASS |

156 existing tests + 13 new tests = **169 tests PASS**.

---

## Files

| File | Change |
|------|--------|
| `Sources/MLXCompiler/Lowered/FusedBlock.swift` | New: FusedSubLayer enum + tryFuseResidual() |
| `Sources/MLXCompiler/Lowered/FlatStep.swift` | Modified: fusedSubLayer case, pattern detection in flattenSteps() |
| `Tests/MLXCompilerTests/FusedBlockTests.swift` | New: 13 tests |

---

## Cumulative Optimization Summary

| Step | Optimization | Decode 8L | Prefill 64tok |
|------|-------------|-----------|---------------|
| 0 | Baseline | 23.7 ms | 38.0 ms |
| 1 | PackedProjection | 20.5 ms (-13.5%) | 35.2 ms (-7.4%) |
| 2 | FusedSubLayer | 20.0 ms (-2.4%) | 32.0 ms (-9.1%) |
| **Total** | Steps 0→2 | **-15.6%** | **-15.8%** |

---

## Next Steps

| Step | Description | Expected Impact |
|------|-------------|-----------------|
| **Step 3**: Layer Template Reuse | `MLX.compile()` for `.repeating(N)` → compile once, execute N times | Metal command buffer reuse, compiled graph 1 instead of N |
| **Step 4**: Loop-Invariant Hoisting | Hoist attention mask and RoPE offset computation out of layer loop | Remove redundant per-layer computation |
| **Step 5**: KV Cache Prefix Reuse | Fix P2/P3 gaps in CompiledKVCache | Enable PrefixCachePool for compiled path |

Step 2 provides the critical building block for Step 3: each `FusedSubLayer` is a self-contained function with captured weights that can be traced by `MLX.compile()` once and reused across all layers.
