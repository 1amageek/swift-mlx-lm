# Metal 4 Integration Design

## Background

### Current hybrid prefill status

Hybrid models with convolution or recurrent state are correctness-gated before
they can use sequence prefill. `MetalPrefillPlan.requiresSequentialPromptIngestion`
is no longer driven by state-buffer presence alone, but it still returns `true`
when the plan contains an SSM sequence kernel whose output has not matched
decode-equivalent ingestion:

- `ssm_recurrence_seq_*`

Q3 prefill projection and embedding lookup also remain an explicit unsupported
sequence-prefill case. BF16 `conv1d_causal_seq` is enabled for LFM-style
short-convolution plans after matching decode-equivalent short traces, while
Qwen DeltaNet/SSM prompt ingestion still runs token by token until the SSM
sequence kernel is fixed and tested.

Before any Metal 4 command-buffer reuse or MPP prefill work is treated as a
performance win for hybrid models, the stateful sequence path must produce the
same first token and short decode trace as `prefillPlan = nil` sequential
ingestion on the same prompt for the model family being claimed.

### Profile Data (LFM2.5-1.2B, current implementation)

```
Decode (single token):  ~110 tok/s, 8.4 ms/tok
  GEMV:      94.1% (7889 us) — memory bandwidth bound (~516 GB/s)
  FlashAttn:  1.2%
  Fused Norm: 2.2%
  Other:      2.5%

Prefill (64 tokens):  ~316 tok/s, 3.16 ms/tok
  GEMM: dominant (naive row-by-row implementation)
```

### Key Constraints

- **Decode GEMV is at memory bandwidth ceiling.** Kernel-level optimization (threadgroup cache, vectorized loads) showed no improvement or regression. Apple Silicon L2 cache already handles input vector reuse.
- **Prefill GEMM is naive.** Each row computed independently via simd_sum. No tiling, no AMX utilization.
- **Metal 4 matmul2d** uses Apple's internal optimized paths (likely AMX). Supports BF16×BF16→float natively.

## Strategy: Metal 4 Prefill GEMM

Replace the naive `generateGEMM` kernel with Metal 4 `matmul2d_descriptor` + `tensor_inline`.

### Why Prefill GEMM Only

| Path | Bottleneck | Metal 4 Benefit |
|---|---|---|
| **Decode GEMV** | Memory bandwidth (N=1) | None — matmul2d requires N≥32 tile. GEMV is bandwidth-bound regardless of compute. |
| **Prefill GEMM** | Compute (naive kernel) | **High** — matmul2d uses Apple's optimized AMX tiling. seqLen provides the N dimension. |

### API: matmul2d_descriptor

```metal
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;

// C[M×N] = A[M×K] × B[K×N]  (+ C for accumulate mode)
constexpr auto desc = matmul2d_descriptor(
    M_tile,              // output rows per threadgroup
    N_tile,              // output cols per threadgroup
    dynamic_length_v<int>, // K: read from tensor extents at runtime
    transpose_A,         // false: A is [K, M] column-major
    transpose_B,         // false: B is [N, K] column-major
    relaxed_precision,   // false for correctness, true for speed
    matmul2d_descriptor::mode::multiply  // C = A×B (not accumulate)
);

matmul2d<desc, execution_simdgroups<4>> op;
op.run(sliceA, sliceB, sliceC);
```

### Supported Type Combinations (from MPPTensorOpsMatMul2d.h)

```
A (input)    B (weight)   C (output)
---------    ----------   ----------
bfloat       bfloat       float      ← BF16 model, F32 prefill buffer ✓
half         half         float      ← F16 model, F32 prefill buffer
half         half         half       ← F16 model, F16 decode buffer
bfloat       bfloat       bfloat     ← BF16 model, BF16 output
```

### Current vs Metal 4 GEMM

**Current (MetalSourceGenerator.generateGEMM):**
```metal
kernel void gemm_bf16_f32s(
    device const float* input,     // [seqLen × inputDim]
    device const uint16_t* weight, // [outputDim × inputDim]
    device float* output,          // [seqLen × outputDim]
    constant uint& inputDim, constant uint& outputDim, constant uint& seqLen,
    uint2 gid, uint tiisg, uint sgitg
) {
    // 1 row per simdgroup, 2 simdgroups per threadgroup
    const uint row = gid.x * 2 + sgitg;
    const uint seqPos = gid.y;
    float sum = 0.0f;
    for (uint j = tiisg; j < inputDim; j += SIMD_WIDTH)
        sum += bf16_to_float(weight[row * inputDim + j]) * input[seqPos * inputDim + j];
    sum = simd_sum(sum);
    if (tiisg == 0) output[seqPos * outputDim + row] = float(sum);
}
```

**Metal 4 (proposed):**
```metal
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;

kernel void gemm_mpp(
    device bfloat* input,          // [seqLen × inputDim] — need bfloat type
    device bfloat* weight,         // [outputDim × inputDim]
    device float* output,          // [seqLen × outputDim]
    constant uint& inputDim,
    constant uint& outputDim,
    constant uint& seqLen,
    uint2 tgid [[threadgroup_position_in_grid]]
) {
    // Wrap raw buffers as tensor_inline (zero-copy)
    auto A = tensor<device bfloat, dextents<int32_t, 2>, tensor_inline>(
        input, dextents<int32_t, 2>(inputDim, seqLen));
    auto B = tensor<device bfloat, dextents<int32_t, 2>, tensor_inline>(
        weight, dextents<int32_t, 2>(outputDim, inputDim));
    auto C = tensor<device float, dextents<int32_t, 2>, tensor_inline>(
        output, dextents<int32_t, 2>(outputDim, seqLen));

    constexpr auto desc = matmul2d_descriptor(
        64, 32, dynamic_length_v<int>,
        false, true,  // A: not transposed, B: transposed (row-major weight)
        false,
        matmul2d_descriptor::mode::multiply);

    matmul2d<desc, execution_simdgroups<4>> op;

    auto mA = A.slice(0, tgid.y * 64);
    auto mB = B.slice(tgid.x * 32, 0);
    auto mC = C.slice(tgid.x * 32, tgid.y * 64);
    op.run(mA, mB, mC);
}
```

## Architecture Changes

### 1. MetalSourceGenerator: Metal 4 GEMM Variant

```
MetalSourceGenerator
  ├── generateGEMM(...)          ← existing, Metal 3 fallback
  └── generateMPPGEMM(...)       ← new, Metal 4 matmul2d
```

The compiler selects based on Metal 4 availability at compile time.

### 2. Compilation: Metal Language Version

```swift
// Current
compileOptions.languageVersion = .version3_0

// Metal 4 path
compileOptions.languageVersion = .version4_0

// Need framework search path for MetalPerformancePrimitives
compileOptions.preprocessorMacros = ["USE_MPP": NSNumber(value: 1)]
```

**Critical**: `MetalPerformancePrimitives.h` is a framework header, not stdlib. The Metal compiler needs `-framework MetalPerformancePrimitives` or equivalent include path. The example repo pre-compiles to metallib via `xcrun metal` to avoid JIT issues.

### 3. Pre-compiled Metallib vs JIT

Current swift-lm compiles MSL source at runtime via `device.makeLibrary(source:options:)`. For Metal 4 MPP kernels, there are two options:

**Option A: JIT with framework include path**
```swift
let options = MTLCompileOptions()
options.languageVersion = .version4_0
// Need to set include path for MetalPerformancePrimitives
options.preprocessorMacros = ["__HAVE_TENSOR__": NSNumber(value: 1)]
let library = try device.makeLibrary(source: source, options: options)
```

**Option B: Pre-compiled metallib (recommended)**
```bash
xcrun metal -std=metal4.0 -framework MetalPerformancePrimitives -c shader.metal -o shader.air
xcrun metallib -o mpp_kernels.metallib shader.air
```
Ship `mpp_kernels.metallib` as a resource. Load at runtime via `device.makeLibrary(URL:)`.

**Recommendation**: Option B. Pre-compilation avoids runtime header resolution issues and ensures the Apple-optimized paths are properly compiled. The metallib can be generated as a build step.

### 4. Dispatch Configuration

```
// Metal 4 matmul2d with tile size 64×32, 4 simdgroups:
let simdWidth = pipeline.threadExecutionWidth  // 32
let threadsPerThreadgroup = MTLSize(width: simdWidth * 4, height: 1, depth: 1)
let threadgroups = MTLSize(
    width: (outputDim + 31) / 32,   // N tiles
    height: (seqLen + 63) / 64,     // M tiles
    depth: 1
)
```

### 5. Buffer Type Compatibility

**Problem**: Current prefill buffers are `device float*` (F32). Metal 4 matmul2d wants `device bfloat*` for input (BF16 model) and `device float*` for output.

**Solution**: The input to GEMM in prefill is the hidden state (F32 after norm), not raw weight. The weight is BF16 (from STAF). So:
- Input (hidden/scratch): F32 — matmul2d supports `float × bfloat → float`
- Weight (STAF): BF16 raw bytes — cast to `device bfloat*`
- Output (scratch/hidden): F32

This maps to the supported type combination: **float × bfloat → float** ✓

### 6. Tensor Memory Layout

matmul2d expects column-major layout (innermost dimension first in `tensor_inline` extents):
- A `[M×K]`: extents = `(K, M)` where K is inner (column stride)
- B `[K×N]`: extents = `(N, K)` where N is inner
- C `[M×N]`: extents = `(N, M)` where N is inner

Current GEMM buffer layout:
- input: `[seqLen × inputDim]` — row-major, stride = inputDim
- weight: `[outputDim × inputDim]` — row-major, stride = inputDim
- output: `[seqLen × outputDim]` — row-major, stride = outputDim

For tensor_inline, row-major `[M × K]` with stride K = column-major with extents `(K, M)`. This matches.

## Module Structure

```
MetalCompiler/
  ├── Fragments/
  │   ├── MetalSourceGenerator.swift     ← add generateMPPGEMM()
  │   └── Primitives/
  │       └── LinearFragment.swift       ← add Metal 4 kernel name variant
  ├── Metal4/                            ← new directory
  │   ├── MPPKernelSource.swift          ← Metal 4 MSL source for MPP matmul
  │   └── Metal4Availability.swift       ← runtime Metal 4 detection
  └── MetalInferenceCompiler.swift       ← select Metal 3/4 path
```

## Scope

### Phase 1: Prefill GEMM with matmul2d
- Pre-compile MPP GEMM kernel to metallib
- Runtime: detect Metal 4, load metallib, use for prefill projections
- Fallback: existing naive GEMM for Metal 3

### Phase 2: MTL4CommandBuffer for Decode
- Reuse command buffer across decode steps (eliminate per-step allocation)
- Use MTL4CommandAllocator with frame rotation

### Phase 3: Fused lm_head + Argmax
- Single kernel: matmul2d tile → cooperative_tensor local argmax → atomic global argmax
- Eliminates 128KB logits buffer write+read

## Risks

1. **MetalPerformancePrimitives header availability**: JIT compilation may fail if headers aren't found at runtime. Pre-compiled metallib mitigates this.
2. **BF16 type (`bfloat`)**: MSL `bfloat` requires Metal 4.0 language version. Current `uint16_t + bf16_to_float()` approach doesn't work with tensor_inline.
3. **Tile size constraints**: matmul2d requires K to be multiple of 32 (per the example repo's bug report). `dynamic_length_v<int>` handles non-aligned K but may have performance cost.
4. **Minimum seqLen**: matmul2d tile M=64 means seqLen < 64 will use partial tiles. For short prompts (5 tokens), the overhead of tensor setup may negate the benefit.
