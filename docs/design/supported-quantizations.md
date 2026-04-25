# Supported Quantizations

This document defines the **concrete quantization support surface** of `swift-lm`
today: which weight formats and which KV cache schemes are actually validated
per model family, and how they compose.

The broader architectural design for quantization lives in
[`quantization.md`](quantization.md). This document is the ground-truth catalog
for what the repository currently supports, not what it aspires to.

## Two Orthogonal Axes

Quantization in `swift-lm` is two independent axes:

| Axis | Controlled by | Source of truth |
|---|---|---|
| **Weight format** | safetensors + STAF converter | `QuantizationSchemeIdentifier` in `Sources/MetalCompiler/STAF/STAFFormat.swift` |
| **KV cache scheme** | `InferencePolicy.kvCache` at load time | `KVCachePolicy` / `SchemeSelection` |

Weight format is determined by the model bundle on disk. KV cache scheme is a
deployment decision made by the consumer.

## Weight Formats

Enumerated by `QuantizationSchemeIdentifier` and mapped to concrete GEMV/GEMM
kernels via `QuantizationFormatRegistry`.

| Scheme | ID | bits | groupSize | Block struct |
|---|---|---|---|---|
| `fp16RowMajor` | 0x00 | 16 | — | dense |
| `bf16RowMajor` | 0x01 | 16 | — | dense |
| `fp32RowMajor` | 0x02 | 32 | — | dense |
| `q2Group16ScaleF16` | 0x60 | 2 | 16 | `AffineQ2Group16Block` |
| `q2Group32ScaleF16` | 0x61 | 2 | 32 | `AffineQ2Group32Block` |
| `q3Group16ScaleF16` | 0x50 | 3 | 16 | `AffineQ3Group16Block` |
| `q3Group32ScaleF16` | 0x51 | 3 | 32 | `AffineQ3Group32Block` |
| `q3Group64ScaleF16` | 0x52 | 3 | 64 | `AffineQ3Group64Block` |
| `q4Group64ScaleF16` | 0x40 | 4 | 64 | `AffineQ4Group64Block` |
| `q4Group128ScaleF16` | 0x41 | 4 | 128 | `AffineQ4Group128Block` |
| `q4Group128ScaleF16Zero` | 0x42 | 4 | 128 | alias → `AffineQ4Group128Block` |
| `q5Group32ScaleF16` | 0x30 | 5 | 32 | `AffineQ5Group32Block` |
| `q5Group64ScaleF16` | 0x31 | 5 | 64 | `AffineQ5Group64Block` |
| `q6Group16ScaleF16` | 0x20 | 6 | 16 | `AffineQ6Group16Block` |
| `q6Group32ScaleF16` | 0x21 | 6 | 32 | `AffineQ6Group32Block` |
| `q8Group32ScaleF16` | 0x10 | 8 | 32 | `AffineQ8Group32Block` |
| `q8Group64ScaleF16` | 0x11 | 8 | 64 | `AffineQ8Group64Block` |
| `q8Group128ScaleF16` | 0x12 | 8 | 128 | `AffineQ8Group128Block` |

Registry coverage: every block-quantized scheme in this table resolves to a
concrete `QuantizationFormat` via `QuantizationFormatRegistry.format(for:)`.
GEMV/GEMM kernels are generated through the unified `generateUnifiedQuantizedGEMV`
pipeline. Kernel-level correctness is covered by `UnifiedGEMVBitLevelTests`,
`UnifiedGEMVMultiBlockTests`, and `UnifiedGEMVMultiRowTests` in
`Tests/MetalCompilerTests/Core/`. MLX→STAF bit-stream round-trip is covered by
`STAFQuantizedRoundtripTests` and `STAFRoundtripTests`. Full MLX safetensors
→ `STAFConverter` → `STAFLoader` → unified GEMV dispatch is covered
end-to-end by `STAFEndToEndGEMVTests` — one @Test per MLX-reachable
(bits, groupSize) pair.

Real-bundle end-to-end correctness (weight load + decode token quality) is
tracked separately in the "Per-Model Support Matrix" below — registry presence
alone does not imply a published bundle has been validated end-to-end.

## KV Cache Schemes

Declared via `KVCachePolicy.keyScheme` / `KVCachePolicy.valueScheme` on
`InferencePolicy`.

| Scheme | ID | Purpose | Memory vs FP16 |
|---|---|---|---|
| `fp16RowMajor` | 0x00 | Default dense KV cache | 100% |
| `bf16RowMajor` | 0x01 | Selected automatically when weights are BF16 | 100% |
| `rotorQ8Group32ScaleF16` | 0x70 | Clifford rotor + Q8 block quant | 62.5% |
| `rotorQ4Group64ScaleF16` | 0x71 | Clifford rotor + Q4 block quant | 37.5% |

`SchemeSelection.automatic` resolves to `bf16RowMajor` if the weights are
BF16, otherwise `fp16RowMajor`. The loader can additionally default to
`rotorQ4Group64ScaleF16` for pure-attention models without stateful sequence
operators — see `Region.isRotorQuantDefaultCandidate`.

## Per-Model Support Matrix

Support means the scheme has a passing **real-bundle correctness test** or
**real-bundle embedding evaluation** at HEAD. Untested combinations are
considered unsupported — even if the loader accepts them — because there is no
regression guard preventing silent correctness loss.

**Synthetic vs real-bundle coverage.** This matrix only counts *real-bundle*
validation (downloaded weights → decode token quality or embedding checksum
compared against a HuggingFace/MLX reference). The *synthetic* pipeline —
MLX-shaped safetensors → `STAFConverter` → `STAFLoader` → unified GEMV
dispatch — is validated for all 13 MLX-reachable (bits, groupSize) pairs by
`STAFEndToEndGEMVTests`. A ❌ below therefore means "no real bundle validated
for this model × format" and **not** "the kernel path is unverified".

### Gemma4 (text decoder)

Reference bundle: `google/gemma-4-E2B-it` (TestData: `gemma-4-E2B-it`)

Weight formats:

| Format | Status | Evidence |
|---|---|---|
| FP16 | ✅ Validated | `RotorQuantRealBundleBaselineTests` — token "Tokyo" |
| BF16 | ⚠ Loader-supported only | No real-bundle test |
| Q4g64 | ✅ Validated | `Gemma4Q4AgreementTests` — token diversity 28–31/31 |
| Q6g32 | ✅ Validated | `Gemma4Q6AgreementTests` — token diversity 23–29/31 |
| Q8 | ❌ Not tested | No published bundle converted to STAF |

KV cache schemes (FP16 weights, Gemma4-E2B):

| Scheme | Status | Evidence |
|---|---|---|
| FP16 / FP16 | ✅ Validated | `RotorQuantRealBundleBaselineTests` |
| RotorQ8-K + FP16-V | ✅ Validated | `RotorQuantRealBundleKeyPathTests` |
| FP16-K + RotorQ8-V | ✅ Validated | `RotorQuantRealBundleValuePathTests` |
| RotorQ8 / RotorQ8 | ✅ Validated | `RotorQuantRealBundleFullTests` (RotorQ8) |
| RotorQ4 / RotorQ4 | ✅ Validated | `RotorQuantRealBundleFullTests` (RotorQ4) |

### EmbeddingGemma (encoder-only)

Reference bundles:
- `mlx-community/embeddinggemma-300m-bf16`
- `mlx-community/embeddinggemma-300m-4bit`

Weight formats:

| Format | Status | Evidence |
|---|---|---|
| BF16 | ✅ Validated | `EmbeddingGemmaVariantCompatibilityTests` + `EmbeddingGemmaPerformanceTests` (66.2 emb/s) |
| Q4g64 | ✅ Validated | Same tests (54.0 emb/s) |

KV cache: N/A — embedding model has no generative KV cache.

### LFM2 (hybrid DeltaNet + attention)

Reference bundle: `LiquidAI/LFM2.5-1.2B-Thinking` (TestData: `LFM2.5-1.2B-Thinking`)

Weight formats:

| Format | Status | Evidence |
|---|---|---|
| BF16 | ✅ Validated | `LFMOutputDiagnosticsTests`, `ReleaseSmokeOutputTests` |
| FP16 / Q4 / Q8 | ❌ Not tested | No published bundle |

KV cache schemes:

| Scheme | Status | Evidence |
|---|---|---|
| FP16 / FP16 | ✅ Validated | Default — all LFM2 tests |
| Rotor* | ❌ Excluded by loader | `ShortConvAttributes` present → `containsStatefulSequenceState` → auto-rotor disabled |

Rotor is structurally incompatible with LFM2's stateful conv layers in the
current implementation. Attempting to fix a rotor scheme on LFM2 loads but is
untested for correctness.

### Qwen3.5 (VLM: vision + language)

Reference bundle: `Qwen/Qwen3.5-0.8B-Base`

Weight formats:

| Format | Status | Evidence |
|---|---|---|
| BF16 | ✅ Validated (text path) | `QwenVisionRealBundleTextTests` |
| BF16 | ✅ Validated (vision path) | `QwenVisionRealBundleImageTests`, `QwenVisionRealBundleVideoTests` |
| Q4 / Q8 | ❌ Not tested | No published bundle |

KV cache schemes: FP16 default only; hybrid DeltaNet + Attention layers make
rotor application partial at best. Not tested with rotor schemes.

## Roll-up

|   | FP16 | BF16 | Q4 | Q6 | Q8 | Rotor KV |
|---|---|---|---|---|---|---|
| Gemma4 text | ✅ | ⚠ | ✅ (g64) | ✅ (g32) | — | ✅ (Q4 / Q8 both K, V, or full) |
| EmbeddingGemma | — | ✅ | ✅ | — | — | N/A |
| LFM2 | — | ✅ | — | — | — | ❌ loader-excluded |
| Qwen3.5 VLM | — | ✅ | — | — | — | ❌ not tested |

Legend: ✅ real-bundle test passes · ⚠ loader accepts but untested · ❌ unsupported · — no bundle

## Rules When Adding a New Scheme or Bundle

1. **Register the scheme identifier** in `QuantizationSchemeIdentifier` with
   its numeric ID.
2. **Provide a `QuantizationFormat` struct** with kernel names and block
   geometry.
3. **Wire it into `QuantizationFormatRegistry.format(for:)`** — otherwise
   loading fails with "unknown scheme".
4. **Add a real-bundle correctness test** before marking the scheme as
   supported in this document. A passing test is the only evidence counted.
5. **Do not silently fall back** to a different scheme if the requested one is
   unavailable. Loader errors must be explicit (per repo rule: no silent
   fallback).

## Rules When Dropping a Scheme

1. Remove the registry entry.
2. Remove the Swift struct.
3. Update this document (strike the row, do not delete history).
4. The STAF scheme ID must not be reused for a different semantic scheme — IDs
   are an append-only contract.
