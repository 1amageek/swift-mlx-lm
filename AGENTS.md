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

Consumer-facing loading starts from `SwiftLM.ModelBundleLoader`, which downloads or opens a HuggingFace model directory, converts weights to STAF when needed, builds a `ModelGraph`, compiles it to a Metal dispatch plan, and returns a `LanguageModelContainer` or `TextEmbeddingContainer`.

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
- For the Qwen3.5+ multimodal suites, prefer [`scripts/benchmarks/run-qwen35-vision-tests.sh`](/Users/1amageek/Desktop/swift-lm/scripts/benchmarks/run-qwen35-vision-tests.sh) over a single large `xcodebuild test` invocation. It uses `build-for-testing` once and then runs `test-without-building` suite-by-suite to reduce peak memory pressure.
- For generation benchmarks, prefer [`scripts/benchmarks/run-generation-pipeline.sh`](/Users/1amageek/Desktop/swift-lm/scripts/benchmarks/run-generation-pipeline.sh) over running the full benchmark file in one `xcodebuild test` process. It builds once and then runs the split benchmark suites sequentially.
- Generation benchmark suites are intentionally split by cost:
  - `SwiftLMTests/GenerationThroughputBenchmarkTests`
  - `SwiftLMTests/GenerationScalingBenchmarkTests` for `50/128/256`
  - `SwiftLMTests/GenerationScalingLongBenchmarkTests` for `512`
  - `SwiftLMTests/GenerationStreamingBenchmarkTests`
- Keep `512`-token scaling isolated in its own suite. Combining it with shorter scaling cases can exceed the 120-second outer timeout even when individual cases are healthy.
- For very long generation-length benchmarks such as `512`, reduce benchmark repetitions before raising the outer timeout. Prefer fewer iterations over a larger timeout so failures still surface quickly.
- For real-model / Metal-heavy / large-bundle tests, do not batch multiple expensive cases into one long `xcodebuild test` process when you are debugging correctness. Prefer `build-for-testing` once, then `test-without-building` one test at a time. This avoids cumulative GPU memory pressure, repeated model loads in a single process, and hard-to-diagnose xctest crashes.
- When validating output quality for a specific model/policy combination, prefer one focused test per invocation over a whole suite. If you need multiple policy comparisons, run them as separate `test-without-building` invocations.
- For repeated real-model loads inside tests/helpers, explicitly scope temporary objects tightly and prefer `autoreleasepool` on synchronous helper boundaries when possible. Do not keep multiple large `LanguageModelContext` / `TextEmbeddingContext` / tokenizer / bundle instances alive longer than needed.
- When `xcodebuild` reports `unexpected exit`, `Restarting after unexpected exit`, or flaky suite-level process failure, rerun with [`scripts/xcodebuild/test-timeout.sh`](/Users/1amageek/Desktop/swift-lm/scripts/xcodebuild/test-timeout.sh) or [`scripts/xcodebuild/test-hang-guard.sh`](/Users/1amageek/Desktop/swift-lm/scripts/xcodebuild/test-hang-guard.sh) before changing inference code.
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
6. If a full suite is still unstable, split it into smaller `@Suite` groups by concern such as output, prompt-state, and capability so `-only-testing:<Target>/<Suite>` remains usable with Swift Testing.

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
  - `LanguageModelContainer` is the primary public entry point for language generation.
  - `LanguageModelContext` exposes mutable runtime state, prompt staging, prompt snapshots, and cache control.
  - `TextEmbeddingContainer` is the primary public entry point for text embeddings.
  - `TextEmbeddingContext` exposes mutable runtime state for reusable embedding execution.

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
  └─ LanguageModelContainer      → generate / prepare / encode / decode
```

GenerationEvent path:

```text
ModelInput
  ├─ await context.prepare(...)               → PreparedPrompt
  ├─ ExecutablePrompt(preparedPrompt:using:)  → ExecutablePrompt
  ├─ context.generate(from:...)               → prefill + decode stream
  └─ PromptSnapshot(from:using:)              → reusable prefixed state
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

- `ModelBundleLoader.load(repo:)` and `load(directory:)` return a `LanguageModelContainer`.
- `ModelBundleLoader.loadTextEmbeddings(repo:)` and `loadTextEmbeddings(directory:)` return a `TextEmbeddingContainer`.
- `LanguageModelContainer` is the primary public entry point for language generation.
- `LanguageModelContext` is the mutable runtime state for explicit context ownership, prompt snapshots, or staged execution.
- `TextEmbeddingContainer` is the primary public entry point for text embeddings.
- `TextEmbeddingContext` is the mutable runtime state for explicit embedding-context ownership.
- `ModelInput` is the primary public request value for language-model APIs.
- `TextEmbeddingInput` is the primary public request value for embedding APIs.
- `PromptPreparationOptions` holds prompt/render-time configuration only.
- `GenerationParameters` holds generation/output-time configuration only.
- `PreparedPrompt` carries rendered prompt text, token IDs, and optional multimodal prompt metadata.
- `ExecutablePrompt` carries the validated runtime input accepted by the current Metal execution path, including Qwen-style multimodal execution payloads when supported by the loaded bundle.

Public API design rules:

- Prefer `Container -> Context` as the top-level shape of the API.
- Use the container as the recommended entry point for most app code.
- Use the context when the caller explicitly needs mutable runtime ownership.
- Prefer request-value types over parallel argument lists.
  - language generation uses `ModelInput`
  - embeddings use `TextEmbeddingInput`
- Keep prompt/render-time options separate from generation-time options.
  - prompt/template controls belong in `PromptPreparationOptions`
  - sampling and output controls belong in `GenerationParameters`
- Keep staged APIs available, but treat them as advanced APIs.
  - `PreparedPrompt` and `ExecutablePrompt` are not the default path
  - one-shot container APIs should remain the easiest path to use
- When adding new options, first decide whether they are:
  - structural request data
  - prompt/render-time options
  - generation/output-time options
  - advanced staged/runtime-only details
- Do not add public APIs that leak backend-specific Metal execution details into `SwiftLM`.
- Keep language-model APIs and embedding APIs directionally aligned unless there is a clear reason not to.

Do not reintroduce stale assumptions from older designs:

- no GGUF loader types
- no MLX execution engine
- no `InferenceSession`
- no async `preparePrefix`
- no `perform(values:operation:)`

If such functionality is reintroduced later, document it from the actual implementation rather than from prior repository history.

## Metal Backend Notes

The current backend is direct Metal compute.

- Decode uses a compiled dispatch plan with preallocated buffers.
- Prefill uses a separate sequence-oriented plan.
- `MetalInferenceModel` owns command queue execution and mutable decode position.
- Prefill and decode have different precision/buffering tradeoffs.

The repository also contains active design work for Metal 4 / MPP-based prefill improvements in [docs/design/metal4.md](/Users/1amageek/Desktop/swift-lm/docs/design/metal4.md). Treat that document as forward-looking design, not as a statement that the codebase has already switched to that implementation.

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

## Production Readiness Gates

Do not describe `swift-lm` as production-ready unless all of the following are true.

### 1. Output correctness is stable

- Major supported model families have focused real-model output tests.
- Reference-aligned or exact-prefix assertions exist for critical prompts.
- Output quality is confirmed before any benchmark result is treated as meaningful.
- Inference-policy differences (`none`, `standard`, `aggressive`, RotorQuant, fallback paths) are checked for behavioral equivalence where they should match.

### 2. Crash and memory behavior are understood

- Repeated model load / generate / reset / snapshot-restore loops have been exercised without xctest restarts or unexplained process exits.
- Heavy real-model tests are split into focused invocations to avoid false confidence from suite-level batching.
- Large synchronous helper loops use tight lifetimes and `autoreleasepool` boundaries where appropriate.
- Known crash-prone paths have probes or diagnostics that make the failure boundary obvious.

### 3. Performance is measured only after correctness

- Baseline benchmark numbers are recorded for representative models and policies.
- Regressions are judged on both total tok/s and decode tok/s.
- Benchmark diagnostics confirm that expected optimizer features remain active.
- Performance claims are not made from builds that have unresolved output-quality issues.

### 4. Public API direction is stable

- `Container -> Context` remains the top-level API shape.
- Request-value types remain the preferred public request shape.
- Prompt/render options remain separate from generation/output options.
- Advanced staged APIs remain available but are not the default recommended path.
- New public APIs are reviewed against these rules before being added.

### 5. Capability reporting is accurate

- `ModelConfiguration.inputCapabilities` and `executionCapabilities` reflect actual supported behavior.
- Unsupported modalities or template configurations fail with explicit errors.
- README, DocC, tests, and actual runtime behavior stay aligned.

### 6. Release process is repeatable

- Focused correctness suites and smoke benchmarks are runnable with documented commands.
- Supported-model notes and known limitations are updated for each release.
- A release should not be cut while correctness regressions, unexplained crashes, or broken focused suites remain open.

Recommended execution order for readiness work:

1. narrow contract tests
2. focused real-model correctness tests
3. policy / optimizer equivalence checks
4. smoke benchmarks
5. longer benchmarks and release validation

### Production Validation Matrix

Use the following matrix as the default readiness checklist. Prefer focused
`build-for-testing` once, then `test-without-building` per suite or per case.

- LFM2 / LFM2.5
  - declaration / graph validity
    - `ModelsTests/ModelDeclarationTests`
  - compiler / fragment / routing contracts
    - `MetalCompilerTests/ComponentDispatchTests`
    - `MetalCompilerTests/BarrierOptimizationTests`
    - `MetalCompilerTests/PrefillTransferTests`
  - reference-aligned correctness
    - `MetalCompilerTests/ReferenceComparisonTests`
    - `SwiftLMTests/ReleaseSmokePromptStateTests`
  - chat-template correctness
    - `SwiftLMTests/ChatTemplateRenderingTests`
  - performance smoke
    - `SwiftLMTests/GenerationThroughputBenchmarkTests`
    - `SwiftLMTests/GenerationScalingBenchmarkTests`
    - `SwiftLMTests/GenerationStreamingBenchmarkTests`
    - `MetalCompilerTests/BenchmarkDiagnosticsTests`
  - pass criteria
    - reference comparisons pass or skip only because local reference assets are absent
    - prompt-state direct vs restored generation stays equivalent
    - benchmark diagnostics do not show broken optimizer configuration

- Gemma4
  - declaration / graph validity
    - `ModelsTests/ModelDeclarationTests`
    - `MetalCompilerTests/Gemma4CompilerTests`
  - real-bundle correctness
    - `SwiftLMTests/Gemma4RealBundleTests`
  - chat-template / thinking correctness
    - `SwiftLMTests/ChatTemplateRenderingTests`
  - performance smoke
    - `MetalCompilerTests/Gemma4BenchmarkTests`
    - `MetalCompilerTests/RotorQuantBenchmarkTests`
  - pass criteria
    - factual greedy output starts with the expected prefix
    - prompt-state and direct generation remain consistent
    - image prompt preparation and execution pass when local assets are available
    - RotorQuant quality checks remain within the expected tolerance

- Qwen3.5 / Qwen vision
  - declaration / graph validity
    - `ModelsTests/ModelDeclarationTests`
    - `SwiftLMTests/DimensionValidatorTests`
  - multimodal preparation / execution
    - `SwiftLMTests/QwenVisionCapabilityTests`
    - `SwiftLMTests/QwenVisionPromptProcessorTests`
    - `SwiftLMTests/QwenVisionExecutionLayoutTests`
    - `SwiftLMTests/QwenVisionExecutionTests`
    - `SwiftLMTests/QwenVisionIntegrationTests`
  - real-bundle checks
    - `SwiftLMTests/QwenVisionRealBundleImageTests`
    - `SwiftLMTests/QwenVisionRealBundleVideoTests`
    - `SwiftLMTests/QwenVisionRealBundleMixedTests`
    - `SwiftLMTests/QwenVisionRealBundlePromptStateTests`
  - pass criteria
    - placeholder expansion, multimodal token typing, and executable layout all agree
    - prompt-state reuse remains valid for multimodal prompts
    - unsupported modality combinations fail explicitly, not implicitly

- Text embeddings
  - metadata / runtime contracts
    - `SwiftLMTests/TextEmbeddingRuntimeTests`
    - `SwiftLMTests/TextEmbeddingContainerIsolationTests`
  - real-bundle correctness
    - `SwiftLMTests/EmbeddingGemmaRealBundleTests`
    - `SwiftLMTests/EmbeddingGemmaReferenceParityTests`
  - pass criteria
    - output dimension and normalization behavior match bundle expectations
    - retrieval/reference parity remains within the stored tolerance

- RotorQuant / KV cache policy
  - correctness / sizing
    - `MetalCompilerTests/RotorQuantCorrectnessTests`
    - `MetalCompilerTests/RotorQuantBenchmarkTests`
  - pass criteria
    - policy-specific correctness checks pass before throughput claims are used
    - memory-size expectations and quality checks remain stable

Suggested command pattern:

1. `perl -e 'alarm shift; exec @ARGV' 120 xcodebuild build-for-testing -scheme swift-lm-Package -destination 'platform=macOS'`
2. `perl -e 'alarm shift; exec @ARGV' 120 xcodebuild test-without-building -scheme swift-lm-Package -destination 'platform=macOS' -only-testing:<Target>/<SuiteOrCase>`

When running release validation, do not batch multiple heavy real-bundle suites in
one test process. Run them one at a time and inspect the result before starting the
next suite.

### Readiness Tracking Rules

When reporting current readiness status, classify each area as exactly one of:

- `done`
- `partial`
- `missing`

Status rules:

- `done`
  - there is direct evidence from recent test or benchmark execution
  - the relevant focused suites or cases passed
  - no known open correctness or crash issue blocks the claim
- `partial`
  - some coverage or evidence exists, but not enough to make a production claim
  - local-asset-dependent suites were skipped or not all focused suites were run
  - performance data exists without complete correctness evidence, or vice versa
- `missing`
  - no current evidence exists
  - the required suite, benchmark, or real-bundle check has not been run
  - the area is still design-only or known-broken

Evidence rules:

- Do not mark an area `done` from code inspection alone.
- Do not mark an area `done` from benchmark numbers alone.
- Do not use passing unit tests to claim real-model correctness.
- Do not use a suite-level green result if the intended case did not actually run.
- If a real-bundle suite skips because assets are absent, record that as `partial`, not `done`.
- If output correctness is unresolved, benchmark evidence can only raise an area to `partial`.

Recommended reporting format:

- area
  - status: `done|partial|missing`
  - evidence: concrete suite or benchmark names
  - gap: the next missing check required to reach `done`

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
