# GPU-Accelerated Weight Packing & groupSize=32 Re-quantization

**Date**: 2026-03-11
**Module**: `MLXLM` (swift-mlx-lm)
**Hardware**: M4 Max

---

## Background

GGUF quantized weights must be repacked from GGUF block format to MLX native affine format (UInt32 packed weight + Float16 scales/biases) before inference. This repacking involves per-element bit manipulation — extracting quantized values from GGUF's interleaved block layouts, applying scale/bias transformations, and packing into MLX's contiguous format.

Two optimizations were implemented:

1. **GPU-accelerated packing**: Metal compute kernels for 10 quantization types, dispatched for tensors above a size threshold
2. **groupSize=32 re-quantization**: Q6_K/Q2_K/Q3_K upgraded from groupSize=16 to groupSize=32, enabling `quantizedMatmul` for all types

## GPU Packing Kernels

### Supported Types

| Type | Block Size | Elements/Block | Kernel Strategy |
|------|-----------|----------------|-----------------|
| Q4_0 | 18 bytes | 32 | Direct pack |
| Q4_1 | 20 bytes | 32 | Direct pack |
| Q5_0 | 22 bytes | 32 | Direct pack |
| Q5_1 | 24 bytes | 32 | Direct pack |
| Q8_0 | 34 bytes | 32 | Direct pack |
| Q8_1 | 36 bytes | 32 | Direct pack |
| Q4_K | 144 bytes | 256 | Direct pack (8 groups/super-block) |
| Q5_K | 176 bytes | 256 | Direct pack (8 groups/super-block) |
| Q6_K | 210 bytes | 256 | Re-quantize (decode → groupSize=32) |
| Q8_K | 292 bytes | 256 | Direct pack (8 groups/super-block) |

### Dispatch Threshold

`gpuThreshold = 2048` blocks. Tensors below this threshold fall back to CPU packing. For 256-element super-block types (Q4_K, Q5_K, etc.), the threshold is `2048 / 8 = 256` super-blocks (65,536 elements).

### Implementation

Each kernel is a `MLXFast.metalKernel()` with 1 GPU thread per output group. Helper functions (`read_f16`, `read_f32`) are placed in the `header` parameter (not `source`) to avoid nested function definitions in Metal C++.

## groupSize=32 Re-quantization

### Problem

MLX `quantizedMatmul` Metal kernel requires `groupSize >= 32`. Three GGUF types originally produced `groupSize = 16`:
- Q6_K: 16 sub-groups of 16 elements per super-block
- Q2_K: 16 sub-groups of 16 elements per super-block
- Q3_K: 16 sub-groups of 16 elements per super-block

These fell back to `dequantize + matmul` — slower and defeating the purpose of quantization.

### Solution

Adjacent pairs of 16-element sub-groups are merged into 32-element groups via `requantizeGroup32()`:

1. Decode 2 × 16 float values from GGUF format
2. Find min/max across 32 values
3. Compute `scale = (max - min) / (2^bits - 1)`, `bias = min`
4. Re-quantize each value to `[0, 2^bits - 1]`
5. Pack into UInt32 words

This introduces minimal precision loss (re-quantization error) but enables hardware-accelerated `quantizedMatmul` for all types.

### Result

All GGUF quantization types now produce `groupSize >= 32`. The `.dequantizeMatmul` fallback path in `LoweredProjection` is effectively unused.

## Benchmark

**Test**: `PackingBenchmarkTests.weightPackingThroughput`
**Model**: Qwen3.5-0.8B-Q4_K_M (508MB GGUF)
**Methodology**: All tensor data preloaded into memory (I/O excluded). Metal kernel compilation excluded (warm-up pass). Each tensor converted via `convertDirect()` and `eval()`'d to force Metal execution.

| Metric | Value |
|--------|-------|
| Weight tensors | 205 |
| Total elements | 752.3M |
| Pack time | 126 ms |
| Throughput | 5,991 M elements/s |

**508MB of quantized weights are repacked in 126ms** (pure compute, excluding I/O and kernel compilation).

## Full Pipeline Profile

**Test**: `LoaderProfilingTests.standardPathProfile`
**Model**: Qwen3.5-0.8B-Q4_K_M (508MB GGUF)
**Path**: Standard (`loadContext()`)

| Stage | Time (ms) | % | Notes |
|-------|-----------|---|-------|
| GGUF parse | 334 | 33% | mmap + metadata read (tokens/merges/scores arrays ~150K entries each) |
| Tokenizer creation | 432 | 43% | Dictionary construction: tokenToID (151K), mergeRanks (151K), control tokens |
| Model construction | 64 | 6% | Config extraction + empty MLXNN module tree |
| Weight conversion | 40 | 4% | convertDirect() + GPU pack (parallel across CPU cores) |
| Sanitize | 1 | 0% | Weight key filtering |
| DirectQuant modules | 13 | 1% | Swap Linear → QuantizedLinear |
| model.update() | 5 | 1% | Inject weight arrays into module tree |
| eval(model) | 97 | 10% | Force Metal graph evaluation |
| **Total** | **1,003** | **100%** | |

### Bottleneck Analysis

Weight packing (the GPU kernel optimization target) is only **4% of total load time**. The dominant bottlenecks are:

1. **Tokenizer creation (432ms, 43%)**: Building `[String: Int]` dictionaries from ~151K vocabulary entries and ~151K merge rules. Each merge string is split and inserted into a hash map. This is pure CPU dictionary construction with no I/O.

2. **GGUF parse (334ms, 33%)**: Memory-mapping is fast, but reading metadata arrays (tokens, merges, scores, token_types — each ~151K entries) requires iterating through binary data and allocating Swift `String` / `[GGUFMetadataValue]` arrays. The `readString()` per-token overhead dominates.

3. **eval(model) (97ms, 10%)**: Forces lazy MLX computation graphs to execute on Metal. This is the first GPU synchronization point after weight loading.

### Optimization Opportunities

| Target | Current | Potential Approach | Expected Impact |
|--------|---------|-------------------|-----------------|
| Tokenizer init | 432ms | Lazy dictionary build, or pre-serialized binary cache | -300ms |
| GGUF metadata arrays | 334ms | Bulk string table read (avoid per-element `readString`) | -200ms |
| eval(model) | 97ms | Already minimal | — |
| Weight conversion | 40ms | Already GPU-accelerated | — |

## Files

| File | Change |
|------|--------|
| `Sources/MLXLM/Bridge/GGUFGPUPackKernels.swift` | GPU Metal kernels for 10 types |
| `Sources/MLXLM/Bridge/GGUFTensorBridge.swift` | `requantizeGroup32()`, rewritten Q6_K/Q2_K/Q3_K CPU paths |
| `Sources/MLXCompiler/Lowered/LoweredProjection.swift` | Documentation update |
| `Tests/MLXLMTests/GPUPackCorrectnessTests.swift` | 10 GPU vs CPU correctness tests |
| `Tests/MLXLMDiagnosticTests/PackingBenchmarkTests.swift` | Throughput benchmark |
