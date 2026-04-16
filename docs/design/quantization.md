# Quantization Architecture Design

## Status

This document describes the intended quantization architecture for `swift-lm`.

It is a design document, not a claim that every piece described here is already
implemented.

## Why

`swift-lm` has two strong constraints:

- `safetensors` remains the source of truth
- direct Metal execution remains the performance center

That means quantization cannot be treated as a single `bits` flag or as a
backend-local afterthought. It must be part of the runtime contract while still
preserving the repository's current architecture rules:

- declaration layers stay architecture-centric
- `LMIR` stays backend-independent
- storage layout and kernel choice stay in `MetalCompiler`
- `SwiftLM` exposes policy and capability, not Metal details

## External Reading That Informs This Design

Two existing systems are useful reference points.

### Ollama / llama.cpp

Ollama uses `llama.cpp` as a supported backend and inherits a quantization
philosophy where tensor encoding schemes are explicit, format-level contracts.

- Ollama backend note: `llama.cpp`
- GGUF import-oriented model workflow
- `llama.cpp` tensor encoding schemes such as `Q4_K`, `Q8_0`, and related formats

The useful idea is not "use GGUF". The useful idea is:

- quantization should be identified as a concrete scheme
- each scheme may deserve a dedicated kernel family
- fallback paths must be explicit, not accidental

### MLX / MLX Swift / MLX Swift LM

MLX exposes quantized execution as first-class runtime operations:

- `QuantizedLinear`
- `QuantizedEmbedding`
- `quantizedMM`
- `gatherQuantizedMM`

MLX Swift LM also carries quantization through loader/configuration:

- global quantization
- per-layer quantization overrides
- embedding and linear paths treated separately

The useful idea is:

- quantization belongs in the model-loading and runtime contracts
- embedding and projection paths must not be conflated
- per-layer quantization is a first-class configuration axis

## Design Principles

### 1. Quantization is a formal execution contract

Quantization is not just "4-bit" or "8-bit". A runtime needs the full scheme:

- bit width
- group size
- scale representation
- zero-point presence
- signed vs unsigned interpretation
- optional transforms such as RotorQuant

`swift-lm` should continue to model this as a concrete storage/execution
contract, not as an informal hint.

### 2. Storage layout and execution kernel are related but distinct

The same quantization scheme may admit multiple physical layouts and multiple
kernel families.

- storage layout answers: "how is this tensor encoded on disk / in memory?"
- kernel family answers: "how is this tensor consumed on this path?"

This distinction matters because:

- prefill and decode have different cost profiles
- embedding lookup and projection are different operations
- a dense MPP path and a custom packed-Q4 path are not comparable abstractions

### 3. Path specialization is required

Quantized execution should be chosen per path, not globally.

At minimum, `swift-lm` should treat these as different execution classes:

- embedding lookup
- prefill projection
- decode projection
- LM head / vocab projection
- KV cache write
- KV cache read / attention use

### 4. Per-layer policy is first-class

The repository should support heterogeneous quantization. A good policy is often:

- embeddings: conservative or separate scheme
- attention/MLP projections: more aggressive
- output head: conservative if quality-sensitive
- KV cache: separate policy entirely

### 5. The compiler owns selection

Model declarations should never encode quantization tricks.

The compiler should map:

`operation semantics × tensor scheme × path × device capabilities`

to:

- storage layout
- kernel family
- fallback decision

## Proposed Internal Model

### Quantization Scheme

Represent the semantic quantization contract explicitly.

```swift
public enum QuantizationScheme: Sendable, Equatable {
    case dense(DensePrecision)
    case blockAffine(BlockAffineScheme)
    case blockSymmetric(BlockSymmetricScheme)
    case rotorQuant(base: BlockAffineScheme)
}

public enum DensePrecision: Sendable, Equatable {
    case float16
    case bfloat16
    case float32
}

public struct BlockAffineScheme: Sendable, Equatable {
    public var bits: Int
    public var groupSize: Int
    public var scaleType: ScaleType
    public var zeroPoint: ZeroPointMode
}
```

This is the semantic scheme. It is not yet a kernel choice.

### Tensor Storage Layout

Represent the physical encoding separately.

```swift
public enum TensorStorageLayout: Sendable, Equatable {
    case denseRowMajor(DensePrecision)
    case interleavedBlockAffine(InterleavedBlockLayout)
    case interleavedBlockSymmetric(InterleavedBlockLayout)
    case embeddingTableQuantized(InterleavedBlockLayout)
    case projectionPacked(ProjectionPackedLayout)
}
```

This maps naturally onto STAF's existing design direction. STAF should continue
to own execution-oriented physical layout contracts.

### Execution Path

Represent where the tensor is used.

```swift
public enum QuantizedExecutionPath: Sendable, Equatable {
    case embeddingLookup
    case prefillProjection
    case decodeProjection
    case vocabProjection
    case kvWrite
    case kvRead
}
```

### Kernel Family

Represent the algorithm family chosen by the compiler.

```swift
public enum ExecutionKernelFamily: Sendable, Equatable {
    case denseMPP
    case denseCustomGEMM
    case denseCustomGEMV
    case quantizedEmbeddingLookup
    case quantizedPrefillGEMM
    case quantizedDecodeGEMV
    case quantizedVocabProjection
    case rotorQuantKVWrite
    case rotorQuantKVRead
    case dequantizeThenDense
}
```

This must be visible in diagnostics and tests.

### Compiler Output: Quantization Plan

Every compiled model should be able to describe what was selected.

```swift
public struct QuantizationPlanEntry: Sendable, Equatable {
    public var tensorName: String
    public var path: QuantizedExecutionPath
    public var scheme: QuantizationScheme
    public var layout: TensorStorageLayout
    public var kernelFamily: ExecutionKernelFamily
    public var usedFallback: Bool
    public var fallbackReason: String?
}
```

This plan belongs in `MetalCompiler` diagnostics, not in `SwiftLM` public API.

## Public Policy Surface

`SwiftLM` should expose intent, not backend details.

The current `InferencePolicy` is an appropriate place for KV cache policy, but
weight quantization policy should be a separate concept.

```swift
public struct QuantizationPolicy: Sendable, Equatable {
    public var defaultWeightPolicy: WeightQuantizationPolicy
    public var perLayerOverrides: [String: WeightQuantizationPolicy]
    public var embeddingPolicy: WeightQuantizationPolicy?
    public var outputHeadPolicy: WeightQuantizationPolicy?
}
```

Key rule:

- public API asks for policy
- compiler resolves policy into concrete scheme/layout/kernel choices

The public API should not expose raw MSL/kernel names or Metal binding details.

## Integration with Existing swift-lm Structure

### LMIR

No Metal-specific quantization layout should leak into `LMIR`.

`LMIR` may carry semantic attributes that affect eligibility, but not concrete
Metal layout or kernel decisions.

### LMArchitecture / ModelDeclarations

No declaration should say:

- use Q4 here
- use this packed layout
- use this Metal kernel

Declarations remain architecture-level only.

### MetalCompiler

This is the main owner of quantization execution strategy.

It should own:

- scheme resolution
- STAF layout mapping
- device capability checks
- path-specific kernel selection
- fallback selection
- diagnostics

### STAF

STAF remains the execution cache and should continue to formalize:

- quantization scheme identifier
- payload layout contract
- future layout variants for specific kernel families

Important existing rule remains correct:

- tensor-level storage contract belongs to STAF
- architecture-level fusion belongs to runtime/compiler

## Selection Rules

The compiler should choose kernels by rule, not by scattered heuristics.

### Dense

- prefer `denseMPP` for eligible prefill GEMM
- prefer custom decode GEMV for single-token decode
- do not force decode into MPP when shape does not fit

### Q4 / Q8 projections

- prefill should prefer a dedicated quantized GEMM family
- decode should prefer a dedicated quantized GEMV family
- `dequantizeThenDense` is allowed only as an explicit fallback with diagnostics

### Embedding tables

- quantized embedding lookup is its own family
- do not reuse projection kernels for embedding tables without proving it helps

### KV cache

- treat KV quantization independently from weight quantization
- RotorQuant remains a separate family with separate validation

## Validation Requirements

A quantization design is not complete until it has matrix-style validation.

Required validation dimensions:

- model family
- execution path
- quantization scheme
- correctness mode
- performance mode

Minimum matrix:

- `EmbeddingGemma`: dense vs Q4 embedding path
- `Gemma4`: dense vs RotorQuant KV policy
- `LFM2`: dense vs supported weight-quantized path
- `Qwen3.5`: dense vs supported weight-quantized path

For each supported scheme/path pair:

- focused output correctness tests
- focused retrieval/reference tests for embeddings
- benchmark tests
- diagnostic assertions that the intended kernel family actually ran

## What This Means for Current swift-lm Work

Today the repository already has the right direction in two places:

- STAF treats quantization/layout as an owned execution contract
- RotorQuant is treated as a specific runtime strategy, not as a generic flag

What is still missing is a unified architecture that covers:

- Q4/Q8 weight execution families
- embedding-specific quantized paths
- explicit path-based kernel selection
- compiler-visible fallback reporting
- public per-layer policy

## Immediate Implementation Roadmap

1. Introduce explicit internal types for:
   - `QuantizationScheme`
   - `TensorStorageLayout`
   - `QuantizedExecutionPath`
   - `ExecutionKernelFamily`
2. Add compiler diagnostics for:
   - selected kernel family
   - fallback usage
   - path classification
3. Separate benchmark reporting by:
   - embedding
   - prefill
   - decode
   - KV cache
4. Make `Q4` and `Q8` path-specific kernel families explicit instead of implicit
5. Add a public `QuantizationPolicy` surface after the compiler-side model is stable

## Design Outcome

The intended end state is:

- quantization is represented as a concrete contract
- each path can choose the best execution family for its shape and device
- `swift-lm` can support multiple quantization families without collapsing into
  one generic slow fallback path
- public API remains Swift-like and policy-based
- backend specialization remains contained inside `MetalCompiler`
