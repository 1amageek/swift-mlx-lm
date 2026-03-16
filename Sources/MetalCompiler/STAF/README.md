# STAF — SafeTensor Accelerated Format

## What

STAF is an **executable cache format** that accelerates LLM inference on Apple Silicon Metal while keeping safetensors as the source of truth.

```
*.safetensors (source of truth)
    ↓ STAFConverter (once, offline)
*.staf (GPU-ready cache)
    ↓ mmap → bytesNoCopy
MTLBuffer (zero-copy)
    ↓
Metal kernel reads blocks directly
```

## Why

### Decode is memory bandwidth bound

LLM inference has two distinct phases.

**Prefill** processes the entire prompt at once. For N input tokens, the weights are read once from DRAM and amortized across N tokens of computation. The GPU's compute units stay busy. This phase is close to compute-bound.

**Decode** generates one token at a time. Every step re-reads the full set of weights, but only performs one token's worth of computation. As generation progresses, KV cache reads grow as well. The cost of reading weights is borne by a single token. This phase is memory-bound.

This asymmetry leads to a direct conclusion: **reducing the number of bytes read from DRAM for weights is the primary lever for decode speed.**

### safetensors is not optimized for GPU execution

safetensors excels at safely storing FP16/BF16 dense arrays, but it is not an execution-ready representation.

- Weights are not quantized — the decode bandwidth problem remains.
- No page alignment guarantee — `bytesNoCopy` zero-copy mapping is not possible.
- Tensors must be loaded individually — load-time overhead is high.

Quantizing and converting at every load is expensive and does not fundamentally solve the problem.

For quantized models (e.g., from mlx-community), weight, scales, and biases are stored as **separate tensors**:

```
model.layers.0.self_attn.q_proj.weight  → uint32[4096, 512]
model.layers.0.self_attn.q_proj.scales  → float16[4096, 64]
model.layers.0.self_attn.q_proj.biases  → float16[4096, 64]
```

The GPU must fetch from three disjoint memory regions. Since GEMV is memory bandwidth bound, this random access pattern becomes the bottleneck.

### GGUF is excellent but creates ecosystem friction

llama.cpp's GGUF stores weights as quantization blocks in a near-execution-ready layout. Metal kernels can read blocks directly. This design philosophy is correct.

However, GGUF is not designed to keep safetensors as the source of truth. Tension arises between the original representation's fidelity and execution optimization. Fine-tune delta application, model validation, and HuggingFace ecosystem integration all suffer friction.

### STAF: Separate the original from the executable

STAF generates a GPU-optimized executable cache **alongside** safetensors without replacing them.

- safetensors remains the canonical source. STAF can be deleted and regenerated.
- Quantized weights are repacked into interleaved blocks.
- 4KB alignment guarantees `MTLBuffer(bytesNoCopy:)` works without hacks.
- SHA-256 fingerprint automatically validates consistency with safetensors.

```
safetensors (source of truth)
    ↓ one-time offline conversion
STAF (executable cache, regenerable)
```

This separation ensures the original model's integrity is always preserved. If STAF has a conversion bug, regenerate from safetensors. When a new quantization scheme emerges, safetensors is all you need to produce a new STAF.

## How

### Interleaved Block Format

Three separate tensors (weight + scales + biases) are interleaved into a single contiguous block:

```
safetensors (3 separate tensors):
  weight[4096, 512]:  uint32  ← 4-bit × 8 packed
  scales[4096, 64]:   float16
  biases[4096, 64]:   float16

            ↓ STAFConverter (once)

STAF interleaved blocks:
  row 0: [scale₀ zero₀ qs₀₋₆₃] [scale₁ zero₁ qs₆₄₋₁₂₇] ...
  row 1: [scale₀ zero₀ qs₀₋₆₃] [scale₁ zero₁ qs₆₄₋₁₂₇] ...

1 block = scale (2B) + zero (2B) + packed quants (32B) = 36B for 64 weights
```

The GPU kernel reads these blocks sequentially. Scale and quantized values sit in the same cache line, so a single memory access provides all information needed for dequantization.

### Dequantization Formulas

Each `quant_scheme_id` implies a specific dequantization formula. The kernel must implement exactly this formula.

**Asymmetric (with zero point) — `_Z` suffix variants and MLX affine:**

```
Quantization:   q = round((w - β) / s)      where β = min(w), s = (max(w) - β) / (2^b - 1)
Dequantization: w = s * q + β               (q is unsigned integer 0..2^b-1)

Used by: Q4_G128_SF16_Z, Q4_G64_SF16, Q8_G32_SF16, Q8_G64_SF16, ...
Note: MLX affine quantization is asymmetric. All MLX-community models use this formula.
      The "scale" field stores s, the "zero" field stores β.
```

**Symmetric (no zero point) — future use:**

```
Quantization:   q = round(w / s)             where s = max(|w|) / (2^(b-1) - 1)
Dequantization: w = s * q                    (q is signed integer)

Used by: Q4_G128_SF16 (no _Z suffix)
Note: Requires weights centered around zero. Not used by MLX affine.
```

**Block layout per scheme:**

| Scheme | Header | Quants | Total/block | Dequant |
|--------|--------|--------|-------------|---------|
| Q4_G64_SF16 | scale(2B) + zero(2B) | 32B | 36B / 64 weights | asymmetric |
| Q4_G128_SF16 | scale(2B) | 64B | 66B / 128 weights | symmetric |
| Q4_G128_SF16_Z | scale(2B) + zero(2B) | 64B | 68B / 128 weights | asymmetric |
| Q8_G32_SF16 | scale(2B) + zero(2B) | 32B | 36B / 32 weights | asymmetric |
| Q8_G64_SF16 | scale(2B) + zero(2B) | 64B | 68B / 64 weights | asymmetric |

### Zero-Copy Loading

```swift
let store = try STAFLoader().load(at: stafURL, device: device)
// mmap → page-aligned bytesNoCopy → MTLBuffer
// Zero memcpy. GPU reads directly from the mmap'd region.
```

STAF guarantees 4KB alignment at the file structure level. The page alignment requirement for `MTLBuffer(bytesNoCopy:)` is always satisfied by construction, with no runtime alignment hacks needed.

### Quantization Format Selection

A single byte (`quant_scheme_id`) fully specifies the quantization format — bit width, group size, scale type, and zero-point presence are all encoded in the identifier. The runtime selects the GEMV kernel based solely on this ID, with no type inspection at dispatch time.

```
0x00  FP16_ROWMAJOR        — dense float16
0x40  Q4_G64_SF16          — 4-bit, group=64, scale=float16
0x41  Q4_G128_SF16         — 4-bit, group=128, scale=float16
0x10  Q8_G32_SF16          — 8-bit, group=32, scale=float16
0xFF  PASSTHROUGH           — unknown tensor stored as float16
```

Adding a new quantization format requires only a new `QuantizationFormat` struct. No enum modifications, no compiler changes.

### Cache Invalidation

```swift
let isValid = try STAFConverter().isValid(
    stafURL: cacheURL,
    safetensorsURLs: safetensorsFiles
)
// false → reconvert
```

STAF stores the SHA-256 fingerprint of the source safetensors. When the source changes, the cache automatically becomes invalid. No version management is required. The answer to "should we support old STAF versions?" is always "regenerate."

## Design Decisions and Rationale

### Tensor-level 1:1 conversion only

An early proposal considered fusing tensors at the format level (e.g., `q_proj + k_proj + v_proj → qkv_pack`). This was rejected because it would require the converter to carry architecture-specific lowering rules:

- Rules would proliferate across Llama / Qwen / Gemma / MoE variants.
- GQA (grouped-query attention) with asymmetric head counts would add branching complexity.
- Supporting new architectures would depend on converter updates.

The resolution: **which tensors to combine and how is the runtime's knowledge, not the format's.** The converter only handles `tensor_name → quantization_block`. Fusion is the runtime's responsibility. This keeps the converter universally applicable.

Unknown tensors use `PASSTHROUGH (0xFF)` and are stored as FP16, so the converter never breaks on unrecognized architectures.

### semantic_role as an optional hint

Tensor names (e.g., `model.layers.31.self_attn.q_proj.weight`) vary across model families. Parsing them is fragile. `semantic_role` provides a supplementary hint for each section, but the runtime must function correctly even when all roles are `UNKNOWN`. This simplifies inference code while remaining robust.

### No version field

STAF is deterministically generated from safetensors. When the format specification changes, regenerate. There is no motivation to maintain backward compatibility with old STAF files. `source_fingerprint` is the sole validity proof. A version field would invite the question "should we support old versions?", and the answer is always "no, regenerate."

## Designs Considered but Not Adopted

### Dual layout for prefill and decode

Prefill uses wide GEMM; decode uses narrow GEMV. Separate physical layouts could optimize both. This was rejected because it nearly doubles the model's disk footprint. For local inference on Apple Silicon, disk capacity is a real constraint. STAF v0 uses a single decode-optimized layout.

### Backend subsections (metal / cuda / cpu)

A single logical format producing multiple backend-specific representations was considered. This increases specification complexity, and a Metal-specific design allows deeper optimization. STAF is Metal-only by design.

### GGUF-compatible layout

GGUF's quantization block structure is largely correct. Adopting a compatible layout was considered but rejected for two reasons:

1. GGUF's alignment guarantees are weaker than what `bytesNoCopy` requires.
2. The two-tier architecture (safetensors as source, STAF as cache) does not match GGUF's design philosophy as a standalone format.

## Relationship to Apple Silicon

| STAF Design | Apple Silicon Property |
|---|---|
| 4KB alignment enforcement | VM page boundary = `bytesNoCopy` requirement |
| Zero-copy MTLBuffer | CPU/GPU share unified memory |
| Direct block reads by kernel | decode is memory-bound → reducing DRAM reads has direct impact |
| Decode-optimized layout | Single-token GEMV dominates processing time |
| Extensible to Metal Tensor API | M5/A19+ have dedicated matrix hardware |

### Metal Tensor API (M5/A19+)

Starting with M5 and A19, Apple Silicon includes a dedicated matrix computation unit accessible through the Metal Tensor API. llama.cpp already disables this API for pre-M5/A19 devices (`tensor API disabled for pre-M5 and pre-A19 devices`).

STAF's design accommodates this transition without format changes. The `quant_scheme_id` range `0x70–0xFE` is reserved for future quantization schemes. A Metal Tensor API-optimized format can be introduced as a new scheme ID. The runtime selects between hand-written GEMV kernels (M1–M4) and Metal Tensor API calls (M5+) based on device capability, while STAF's file structure remains identical.

This means the same STAF file can serve both hardware generations — the difference is only in which kernel the runtime dispatches for each `quant_scheme_id`.

## File Structure

```
┌─────────────────────────────────┐  offset 0
│ File Header         (64B)       │  magic, fingerprint, section_count
├─────────────────────────────────┤
│ Section Table       (128B × N)  │  per-tensor metadata
├─────────────────────────────────┤
│ String Table                    │  tensor names (null-terminated)
├─────────────────────────────────┤
│ [padding to 4KB]                │
├─────────────────────────────────┤
│ Payload (4KB aligned)           │  interleaved quantization blocks
│   tensor 0  (256B aligned)      │
│   tensor 1  (256B aligned)      │
│   ...                           │
└─────────────────────────────────┘
```

## File Naming

```
model-00001-of-00008.safetensors   ← source of truth (untouched)
model-00002-of-00008.safetensors
...
model.staf                         ← executable cache (regenerable)
model.staf.lock                    ← exclusive lock during conversion
```

A single STAF file is generated for the entire model, regardless of how many safetensors shards exist.
