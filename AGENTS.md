# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working in this repository.

## Project Overview

`swift-lm` is a Swift package for high-performance language model inference on Apple Silicon using direct Metal compute.

The current repository is not a GGUF/MLX runtime. The active architecture is:

- HuggingFace model bundles as input: `config.json`, `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja`, `*.safetensors`
- `safetensors` as the source of truth for weights
- STAF as a regenerable GPU execution cache
- a backend-independent IR and model-declaration DSL
- direct Metal compilation and execution for prefill and decode

Consumer-facing loading starts from `SwiftLM.ModelBundleLoader`, which downloads or opens a HuggingFace model directory, converts weights to STAF when needed, builds a `ModelGraph`, compiles it to a Metal dispatch plan, and returns a `ModelContainer`.

## Build & Test

```bash
# Build
swift build

# Run all tests
xcodebuild test -scheme swift-lm-Package -destination 'platform=macOS'

# Run one test target
xcodebuild test -scheme swift-lm-Package -destination 'platform=macOS' -only-testing:ModelsTests
xcodebuild test -scheme swift-lm-Package -destination 'platform=macOS' -only-testing:SwiftLMTests
xcodebuild test -scheme swift-lm-Package -destination 'platform=macOS' -only-testing:MetalCompilerTests
```

Important:

- Prefer `xcodebuild test` over `swift test` for this repository.
- For the Qwen3.5+ multimodal suites, prefer [`scripts/run-qwen35-vision-tests.sh`](/Users/1amageek/Desktop/swift-lm/scripts/run-qwen35-vision-tests.sh) over a single large `xcodebuild test` invocation. It uses `build-for-testing` once and then runs `test-without-building` suite-by-suite to reduce peak memory pressure.
- For real-model / Metal-heavy / large-bundle tests, do not batch multiple expensive cases into one long `xcodebuild test` process when you are debugging correctness. Prefer `build-for-testing` once, then `test-without-building` one test at a time. This avoids cumulative GPU memory pressure, repeated model loads in a single process, and hard-to-diagnose xctest crashes.
- When validating output quality for a specific model/policy combination, prefer one focused test per invocation over a whole suite. If you need multiple policy comparisons, run them as separate `test-without-building` invocations.
- For repeated real-model loads inside tests/helpers, explicitly scope temporary objects tightly and prefer `autoreleasepool` on synchronous helper boundaries when possible. Do not keep multiple large `ModelContainer` / tokenizer / bundle instances alive longer than needed.
- Metal-dependent tests and generated libraries are exercised via the package Xcode scheme.
- Repository targets currently declare Swift tools `6.2` and platforms `.macOS(.v26)`, `.iOS(.v26)`, `.visionOS(.v26)` in [Package.swift](/Users/1amageek/Desktop/swift-lm/Package.swift).

### Crash-Resistant Real-Model Test Procedure

When correctness work touches Metal execution, real bundles, or large references, use this procedure instead of ad hoc suite runs:

1. Build once with a hard timeout:
   `perl -e 'alarm shift; exec @ARGV' 120 xcodebuild build-for-testing -scheme swift-lm-Package -destination 'platform=macOS'`
2. Run one focused suite or one focused test process at a time:
   `perl -e 'alarm shift; exec @ARGV' 120 xcodebuild test-without-building -scheme swift-lm-Package -destination 'platform=macOS' -only-testing:<Target>/<SuiteOrCase>`
3. After each heavy invocation, inspect whether the process completed normally before starting the next one. Do not queue multiple `xcodebuild` test processes in parallel.
4. If a suite restarts, crashes, or prints `unexpected exit`, stop batching immediately and reduce scope further.
5. Do not add whole-plan snapshot capture to a heavy real-model test unless the narrow failure cannot be localized any other way.

Rules:

- Outer timeout must stay at `120` seconds or less. Prefer `30-60` seconds for lighter checks.
- For correctness debugging, prefer this order:
  1. contract test
  2. focused real-model output test
  3. optimizer equivalence test
- If a suite is known to allocate large intermediates or many snapshots, split it before changing inference code.
- If repeated synchronous helper loops load large models, add `autoreleasepool` boundaries before expanding test coverage.

### Output-Correctness-First Procedure

For Metal / compiler / runtime changes, validate in this order:

1. fragment and planner contracts
2. focused real-model output correctness
3. regression tests across optimizer modes
4. benchmark tests

Rules:

- Do not treat benchmark improvements as success unless output quality is already confirmed for the same model and prompt class.
- If a model emits incorrect text, investigate correctness first. Performance numbers from that build are not meaningful.
- For output verification, prefer deterministic or near-deterministic settings first (`temperature = 0` or fixed sampling state) before broader sampling checks.
- When a bug appears only under sampling, inspect the host-sampling path separately from the argmax path. CPU-readable/shared logits and GPU-only/private logits must not be conflated.
- Compare optimizer modes explicitly (`none`, `standard`, `aggressive` or their effective fallback) before concluding that a fragment or kernel is correct.

### Fragment / Compiler Debugging Procedure

`swift-lm` is assembled from fragments and compiler routing. A single broken fragment contract can corrupt the full model.

When debugging model corruption:

1. verify the declaration-level structure
2. verify fragment expansion for the affected primitive
3. verify dispatch-entry routing and output marking
4. verify prefill and decode buffer-source selection
5. verify shared/private CPU/GPU ownership assumptions
6. only then inspect model-family specific logic

Checklist:

- Sibling projections that are conceptually parallel (`q_proj` / `k_proj` / `v_proj`, `gate_proj` / `up_proj`) must read from the same pre-projection source unless the design explicitly says otherwise.
- Do not assume sequential dispatch entries imply sequential dataflow. The compiler must preserve graph semantics, not emission order.
- Any CPU read path must use CPU-readable storage. `MTLBuffer.contents()` on `storageModePrivate` is invalid.
- Scratch-slot routing must be validated with explicit stride/offset assertions. Hidden-size stride and slot-dimension stride are not interchangeable.
- Residency, ownership, and submission are separate concerns. Do not hide resource lifetime assumptions inside unrelated runtime types.

### Probe and Regression Guidance

- Long-lived probes must be gated by `ENABLE_METAL_PROBES`. Do not tie production diagnostics to `DEBUG`.
- Prefer probes at fragment boundaries, dispatch-entry routing boundaries, hidden/logits transfer points, and sampling entry points so corruption can be localized quickly.
- Add a narrow regression test for every bug that crosses a layer boundary:
  - fragment contract test
  - optimizer-independence test
  - buffer ownership or residency test
  - real-model output test if the bug escaped unit coverage

## Module Structure

The repository is intentionally split into five layers.

- `LMIR`
  - Backend-independent data model.
  - Defines `ModelGraph`, `Region`, `Operation`, `OperationKind`, `OperationAttributes`, `ParameterBinding`, `ModelConfig`.
  - Contains no Metal code and no DSL knowledge.

- `LMArchitecture`
  - Declarative model DSL and validation layer.
  - Defines `ModelComponent`, `PrimitiveComponent`, structural components such as `Residual`, `Parallel`, `Repeat`, `LayerStack`.
  - Normalizes model declarations into `LMIR.ModelGraph`.

- `ModelDeclarations`
  - Product/model-family declarations built on `LMArchitecture`.
  - Current declarations include `Transformer`, `Qwen35`, `LFM2`, and `Cohere`.
  - This layer decides which computation graph to build from `ModelConfig`.

- `MetalCompiler`
  - Metal backend for `LMIR`.
  - Walks IR, lowers primitive attributes into fragment trees, runs dispatch optimization, compiles kernels, allocates buffers, and produces an opaque `MetalCompiledModel` containing decode/prefill runtime state.
  - Also owns STAF, safetensors conversion/loading, weight naming resolution, and execution-side buffer formats.

- `SwiftLM`
  - Public API for loading, tokenization, prompt formatting, and generation.
  - `ModelBundleLoader` is the main loader.
  - `ModelContainer` exposes `prepare`, `generate`, `encode`, `decode`, and `resetCaches`.

Dependency direction:

```text
LMIR  ←──  LMArchitecture  ←──  ModelDeclarations
  │                                     │
  └──────  MetalCompiler                │
                │                       │
                └───────  SwiftLM  ─────┘
```

Rules:

- `LMArchitecture` must not depend on `MetalCompiler`.
- `MetalCompiler` must not depend on `LMArchitecture`; both meet at `LMIR`.
- Backend-specific behavior belongs in backend layers by extending IR attribute types, not by leaking backend details into `LMIR`.

## Runtime Data Flow

Current load path:

```text
HF repo or local directory
  ├─ config.json                → HFConfigDecoder → ModelConfig
  ├─ tokenizer files            → swift-transformers Tokenizer
  ├─ chat_template.jinja or tokenizer_config.json["chat_template"]
  ├─ safetensors                → STAFConverter → model.staf
  ├─ model declaration          → ModelGraph
  ├─ ParameterResolver          → parameter bindings
  ├─ MetalInferenceCompiler     → decode plan + prefill plan
  └─ ModelContainer             → prepare / generate / decode
```

Generation path:

```text
ModelInput
  ├─ await prepare(input:)      → text prompt or chat-template rendering
  ├─ tokenizer.encode(...)      → PreparedInput(tokenIDs:)
  ├─ makeExecutablePrompt(...)  → ExecutablePrompt(tokenIDs:)
  ├─ prefill(tokens:)           → fill KV/conv cache, emit first token
  └─ decodeSync(tokenID:)       → iterative token generation
```

## Design Principles

### 1. IR is semantic, not backend-specific

`LMIR` describes graph structure and value flow, not execution details.

- `OperationAttributes` are opaque to the IR.
- Backends interpret the same IR attributes through protocol conformances and lowering logic.
- Do not add Metal-only fields to `LMIR` types just because the current backend needs them.

### 2. Model declarations describe families, not execution tricks

`ModelDeclarations` should express the computation graph in family-level terms:

- `Attention`
- `MLP`
- `MoE`
- `DeltaNet`
- `ShortConv`
- `RMSNorm` / `LayerNorm`

Do not encode backend shortcuts or weight-layout assumptions in the declaration layer.

### 3. Missing required config is an error

`ModelConfig` is the normalized input contract for graph construction.

- If a model family requires metadata, validate it and throw.
- Do not silently invent defaults for runtime-critical fields.
- Distinguish clearly between:
  - true architectural defaults
  - derived values
  - missing metadata

### 4. safetensors is canonical, STAF is executable cache

The source of truth is the HuggingFace safetensors bundle.

- STAF is a regenerable cache optimized for GPU execution.
- Conversion bugs should be fixed by improving the converter or loader, not by making STAF authoritative.
- Do not add design assumptions that require STAF to become the canonical storage format.

### 5. Decode hot path matters more than convenience abstractions

The current performance center is direct Metal inference on Apple Silicon.

Prefer changes that improve:

- native execution of packed/quantized weights
- lower decode overhead per token
- safe reuse of KV and convolution state
- fused or batched execution for MoE / DeltaNet / hybrid models

Avoid designs that:

- dequantize large weight sets eagerly into dense temporary buffers on the hot path
- add Swift-side per-token graph walking or dynamic branching
- move runtime-critical work from GPU kernels into host loops

## Model Family Boundary

Treat architecture concepts as reusable families and keep product names local to model declarations.

Family-level concepts:

- `Transformer`
- `MoE`
- `DeltaNet`
- `ShortConv`
- `parallel attention + MLP`
- state-space / recurrent layers

Product/model names:

- `Qwen35`
- `LFM2`
- `Cohere`
- `Llama`
- `Gemma`
- `Mixtral`

Repository rules:

- `Sources/LMArchitecture/**` should contain family-level building blocks only.
- `Sources/LMIR/**` should remain product-agnostic.
- `Sources/Models/**` may contain model/product-specific declarations.
- `Sources/MetalCompiler/**` may contain backend-specific lowering and kernels, but family names are still preferred over product names unless the behavior is truly product-specific.

When adding support for a new model:

1. Identify whether it fits an existing family.
2. If not, add the missing family-level component or IR contract first.
3. Enumerate required config fields in `ModelConfig` and validate them explicitly.
4. Only then add the product-specific declaration in `Sources/Models/**`.

## Current Supported Declaration Families

These are the major declaration paths visible in the repository today.

- `Transformer`
  - Standard pre-norm decoder transformer.
  - Covers dense and MoE variants when the graph shape is still transformer-compatible.

- `Qwen35`
  - Hybrid DeltaNet + full-attention stack.
  - Uses explicit scheduling and hybrid-only config fields such as partial rotary and state-space head dimensions.

- `LFM2`
  - Hybrid short-convolution + attention stack.
  - Supports dense and MoE variants and uses per-layer schedules from config.

- `Cohere`
  - Transformer variant with LayerNorm and QK normalization.

## Public API Notes

Current `SwiftLM` public API is intentionally thin.

- `ModelBundleLoader.load(repo:)` downloads a HuggingFace snapshot and delegates to `load(directory:)`.
- `ModelContainer.prepare(input:)` is async and now prepares text/chat prompts plus Qwen-style image-bearing and video-bearing chat prompts.
- `ModelContainer.generate(input:parameters:)` returns `AsyncStream<Generation>`.
- `PreparedInput` carries rendered prompt text, token IDs, and optional multimodal prompt metadata.
- `ExecutablePrompt` carries the validated runtime input accepted by the current Metal execution path, including Qwen-style multimodal execution payloads when supported by the loaded bundle.

Do not reintroduce stale assumptions from older designs:

- no GGUF loader types
- no MLX execution engine
- no async `preparePrefix`
- no `perform(values:operation:)`

If such functionality is reintroduced later, document it from the actual implementation rather than from prior repository history.

## Metal Backend Notes

The current backend is direct Metal compute.

- Decode uses a compiled dispatch plan with preallocated buffers.
- Prefill uses a separate sequence-oriented plan.
- `MetalInferenceModel` owns command queue execution and mutable decode position.
- Prefill and decode have different precision/buffering tradeoffs.

The repository also contains active design work for Metal 4 / MPP-based prefill improvements in [DESIGN-Metal4.md](/Users/1amageek/Desktop/swift-lm/DESIGN-Metal4.md). Treat that document as forward-looking design, not as a statement that the codebase has already switched to that implementation.

## Testing Expectations

Use tests appropriate to the layer being changed.

- `ModelsTests`
  - graph construction validity
  - determinism
  - required metadata failures
  - family-specific structural expectations

- `SwiftLMTests`
  - loading pipeline
  - config decoding
  - safetensors/STAF integration
  - end-to-end graph construction and compile entry points
  - Qwen3.5+ multimodal coverage is split into focused suites:
    - `QwenVisionCapabilityTests`
    - `QwenVisionPromptProcessorTests`
    - `QwenVisionExecutionLayoutTests`
    - `QwenVisionEncoderTests`
    - `QwenVisionExecutionTests`
    - `QwenVisionIntegrationTests`
    - `QwenVisionRealBundleImageTests`
    - `QwenVisionRealBundleVideoTests`
    - `QwenVisionRealBundleMixedTests`
    - `QwenVisionRealBundlePromptStateTests` for optional local snapshot coverage

- `MetalCompilerTests`
  - lowering
  - source generation
  - dispatch planning
  - diagnostics
  - reference comparisons against Python dumps when available

For performance or correctness work on Metal execution:

- prefer focused target/test execution over full-suite runs
- for crash-prone real-bundle checks, prefer `xcodebuild build-for-testing` once and then `xcodebuild test-without-building` per test case
- for the Qwen3.5+ multimodal path, prefer the focused `SwiftLMTests` suites above over `-only-testing:SwiftLMTests`
- keep timeouts in mind
- when a test hangs, suspect cache/state completion, synchronization, or unfinished stream/state transitions before assuming the compiler is wrong
- if a real-model test crashes or exhausts memory, reduce scope further before changing code: one bundle, one suite, one case per process
- use `autoreleasepool` or tighter object lifetimes around repeated bundle loads and synchronous helper loops to avoid misleading crash signatures from retained GPU resources
- add small contract tests for fragments, routing, stride/layout, and ownership rules; do not rely only on end-to-end text assertions
- weak assertions such as `contains("tokyo")` are insufficient for correctness-sensitive regressions; prefer token IDs, exact prefix, or reference-aligned expectations where feasible

## Editing Guidance

When updating architecture support:

- start from `ModelConfig` and declaration requirements
- keep `ModelGraph` backend-independent
- keep naming resolution and weight-layout handling in `MetalCompiler`
- avoid leaking safetensors/STAF details into declaration code

When updating docs:

- prefer describing what the current code does, not what a predecessor repository used to do
- if a design doc is aspirational, mark it as such
- keep README and `AGENTS.md` aligned with `Package.swift` and the code under `Sources/**`
