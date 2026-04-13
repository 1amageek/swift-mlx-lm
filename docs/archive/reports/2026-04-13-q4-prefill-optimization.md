# EmbeddingGemma Q4 Prefill Optimization

**Date**: 2026-04-13
**Module**: `MetalCompiler` (swift-lm)
**Hardware**: Apple M4 Max (36GB Unified Memory, 546 GB/s bandwidth)
**Model**: EmbeddingGemma 300M (Q4 group64, seqLen=6-12 tokens)

---

## Background

EmbeddingGemma 300M Q4 prefill: swift-lm 43.5 emb/s vs MLX 62 emb/s (0.70x). Goal was to close this gap by routing Q4 prefill projections through direct Q4 GEMM kernels instead of dequant→AMX pipeline.

The hypothesis: dequant→AMX path expands Q4 to BF16 before AMX matmul2d, negating the 4x bandwidth advantage of Q4. Direct Q4 GEMM should restore this advantage.

---

## Changes

### 1. Projection Batching in StandardOptimizer

`AggressiveOptimizer` was force-disabled in `MetalInferenceCompiler.swift:33-38`, replacing it with `StandardOptimizer`. This disabled Q+K+V projection batching.

Added Rule 2 to `StandardOptimizer`: batch consecutive `.gemv` projections with same `inputDimension`. Stops before the last projection in the composite to preserve `markLastProjectionAsOutput()`.

```
Before (StandardOptimizer):
  Rule 1: gate_proj + up_proj + SwiGLU → fusedSwiGLUProjection (3→1)

After (StandardOptimizer):
  Rule 1: gate_proj + up_proj + SwiGLU → fusedSwiGLUProjection (3→1)
  Rule 2: Consecutive .gemv projections with same inputDimension → batchedProjection (N→1)
```

Result: Q+K+V batched (3→1), step count 506 → 482 (24 fewer dispatches).

**Changed files**:

| File | Change |
|------|--------|
| `Sources/MetalCompiler/Optimization/StandardOptimizer.swift` | Added projection batching rule (same logic as AggressiveOptimizer Rule 1) |

### 2. Batched Q4 GEMM Kernel Generation

New Metal kernels for batched Q4 GEMM: `batched_gemm_q4_g{groupSize}_{count}` (count=2 for gate+up, count=3 for Q+K+V). Each kernel reads packed Q4 weights and performs multiply-accumulate with register-level dequantization.

Both kernels require `threadgroup float inputTile[TILE_ELEMENTS]` for shared input caching.

**Changed files**:

| File | Change |
|------|--------|
| `Sources/MetalCompiler/Fragments/MetalSourceGenerator+Quantized.swift` | Added `generateBatchedQuantizedGEMM_Q4_2` and `generateBatchedQuantizedGEMM_Q4_3` |
| `Sources/MetalCompiler/MetalKernelSourceCatalog.swift` | Pre-expansion pass registers batched Q4 kernels from batchedProjection and fusedSwiGLU entries |
| `Sources/MetalCompiler/MetalKernelNameResolver.swift` | `batchedQuantizedGEMMKernelName` returns `"batched_gemm_q4_g{groupSize}_{count}"` |

### 3. threadgroupMemoryLength Crash Fix

Q4 GEMM kernels declare `threadgroup float inputTile[256]` requiring 1024 bytes. `DispatchHeuristics.config()` returns `sharedMemoryBytes: 0` for ALL dispatch dimensions. Setting `threadgroupMemoryLength(0, ...)` when the kernel accesses threadgroup memory causes an immediate GPU fault that crashes the entire Mac.

Added `q4ThreadgroupMemoryLength()` helper that computes the correct value: `max(groupSize * 2, 256) * MemoryLayout<Float>.size`.

Applied at two call sites in the step builder: `.batchedProjection` fallback path and `.projection` case.

**Changed files**:

| File | Change |
|------|--------|
| `Sources/MetalCompiler/MetalPrefillStepBuilder.swift` | Added `q4ThreadgroupMemoryLength(for:)` and `q4GroupSize(for:)` helpers; applied at both dispatch sites |

### 4. Hybrid Q4/AMX Routing Strategy

Initial all-Q4 approach showed minimal improvement (43.5 → 44.0 emb/s). Analysis revealed AMX matmul2d hardware outperforms Q4 software GEMM for short sequences.

Implemented hybrid strategy:
- **Batched entries** (gate+up, Q+K+V): Direct Q4 GEMM (saves dispatch count)
- **Individual projections** (o_proj, down_proj): dequant→AMX (faster per-step)

**Changed files**:

| File | Change |
|------|--------|
| `Sources/MetalCompiler/MetalPrefillStepBuilder.swift` | `useFusedQ4 = false` for individual projections; dequant→AMX for single projections |

---

## Benchmark Results

### Test Configuration

- **Hardware**: Apple M4 Max (36GB Unified Memory, 546 GB/s bandwidth)
- **Test**: `EmbeddingGemmaPerformanceTests` (GPU path via `captureEmbeddingVector`)
- **Model**: EmbeddingGemma 300M Q4 group64 (community4Bit)
- **Sequence length**: 6-12 tokens (short embedding workload)

### Throughput

| Configuration | emb/s | vs MLX |
|---|---|---|
| swift-lm BF16 (before) | 50.8 | 1.04x |
| swift-lm Q4 (before) | 43.5 | 0.70x |
| **swift-lm Q4 (after)** | **44.3** | **0.71x** |
| swift-lm BF16 (after) | 50.7 | 1.04x |
| MLX Q4 | 62.0 | baseline |
| MLX BF16 | 48.8 | 0.79x |

### Step Count

| Configuration | Steps |
|---|---|
| Before (no Q+K+V batching) | ~506 |
| After (Q+K+V batched) | 482 |
| Reduction | 24 fewer dispatches |

### Correctness

`EmbeddingGpuCpuParityTests` cosine similarity = 1.000000 (perfect match between GPU and CPU reference).

---

## Analysis

### Why the 62 emb/s Target Was Not Reached

#### Initial Hypothesis Was Wrong

The plan assumed Q4 bandwidth advantage would restore throughput once dequant overhead was removed. This was incorrect for short-sequence prefill.

Per-step timing comparison:

| Path | Per-step avg | Mechanism |
|---|---|---|
| BF16 MPP (AMX matmul2d) | ~39μs | Hardware matrix multiply |
| Q4 fused GEMM | ~48μs | Software block unpack + multiply |
| Q4 dequant→AMX | ~48μs | 1 dequant dispatch + 1 AMX dispatch |

For sequences of 6-12 tokens, this is essentially batched GEMV — each step processes a tiny matrix. AMX matmul2d hardware is so efficient at BF16 GEMM that it compensates for reading 4x more data. The bandwidth advantage of Q4 only dominates at longer sequences where weight read is the bottleneck.

#### Dispatch Architecture Is the Real Ceiling

~500 dispatches × ~45μs/step ≈ 23ms per embedding. Reducing dispatch count by 24 (Q+K+V batching) saves only ~1ms. The fundamental architecture (individual kernel dispatches with barriers) limits throughput.

MLX achieves 62 emb/s via graph compilation → far fewer fused dispatches.

#### Why MLX Q4 > MLX BF16

MLX graph compilation amortizes dispatch overhead across the entire forward pass. With dispatch overhead nearly eliminated, the 4x bandwidth advantage of Q4 directly translates to throughput improvement (62 vs 48.8 emb/s).

In swift-lm, per-dispatch overhead (~30μs barrier + ~15μs encode) dominates per-step compute time (~45μs), so Q4's bandwidth advantage is masked.

### Timing Diagnostic Caveat

`EmbeddingGemmaTimingDiagnosticsTests` measures the CPU fallback path (`runtime.embed(hiddenStates:)`) which includes a naive O(n²) matrix-vector multiply taking ~400ms. This is NOT the production GPU path (`captureEmbeddingVector`). The diagnostic showing `cpuPost=92.2%` reflects the wrong code path.

---

## Baseline for Future Optimization

| Metric | Value |
|--------|-------|
| Model | EmbeddingGemma 300M Q4 group64 |
| Input | Short text (6-12 tokens) |
| Throughput | **44.3 emb/s** (Q4), **50.7 emb/s** (BF16) |
| Step count | 482 (Q4 with projection batching) |
| Per-step avg | ~45μs |
| Dispatch overhead | ~30μs/barrier × ~450 barriers ≈ 13.5ms |
| Hardware | Apple M4 Max (36GB, 546 GB/s) |
| MLX reference | 62.0 emb/s (Q4), 48.8 emb/s (BF16) |

### Future Optimization Candidates

| Approach | Expected Impact | Rationale |
|---|---|---|
| Metal 4 concurrent dispatch + stage barrier | Medium | Reduce barrier overhead from ~30μs to potentially ~5μs |
| Layer-level kernel fusion (1 layer = 1 dispatch) | High | ~500 → ~24 dispatches, eliminates per-dispatch overhead |
| Q4 AMX kernel (if Metal 4 supports) | High | Q4 bandwidth + AMX compute = best of both |
| Graph compilation (MLX-style) | High | Order-of-magnitude dispatch reduction |

---

## Files

| File | Change |
|------|--------|
| `Sources/MetalCompiler/Optimization/StandardOptimizer.swift` | Modified: Added projection batching rule |
| `Sources/MetalCompiler/Fragments/MetalSourceGenerator+Quantized.swift` | Modified: Batched Q4 GEMM kernel generation (count=2, count=3) |
| `Sources/MetalCompiler/MetalKernelSourceCatalog.swift` | Modified: Batched Q4 GEMM kernel registration in pre-expansion pass |
| `Sources/MetalCompiler/MetalKernelNameResolver.swift` | Modified: `batchedQuantizedGEMMKernelName` helper |
| `Sources/MetalCompiler/MetalPrefillStepBuilder.swift` | Modified: threadgroupMemoryLength fix, hybrid Q4/AMX routing, q4 helpers |
