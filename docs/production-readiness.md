# Production Readiness Gates

This document defines the minimum release gates for `swift-lm`.

## Goal

The release bar is:

- correct first-token and short-trace behavior on real bundles
- no known crash path in prompt-state, sampling, or Metal residency paths
- no material throughput regression against model-specific baselines
- clear probes for fast diagnosis when a Metal execution path breaks
- user-facing docs describe the current public API without stale names or stale flow guidance

## Correctness Gates

These suites must pass before a release:

- `xcodebuild test -scheme swift-lm-Package -destination 'platform=macOS,arch=arm64' -only-testing:SwiftLMTests/ReleaseSmokeOutputTests`
- `xcodebuild test -scheme swift-lm-Package -destination 'platform=macOS,arch=arm64' -only-testing:SwiftLMTests/ReleaseSmokePromptStateTests`
- `xcodebuild test -scheme swift-lm-Package -destination 'platform=macOS,arch=arm64' -only-testing:SwiftLMTests/ReleaseSmokeCapabilityTests`
- `xcodebuild test -scheme swift-lm-Package -destination 'platform=macOS,arch=arm64' -only-testing:SwiftLMTests/RotorQuantRealBundleBaselineTests`
- `xcodebuild test -scheme swift-lm-Package -destination 'platform=macOS,arch=arm64' -only-testing:SwiftLMTests/QwenVisionRealBundleTextTests`
- `xcodebuild test -scheme swift-lm-Package -destination 'platform=macOS,arch=arm64' -only-testing:SwiftLMTests/RotorQuantRealBundleFullTests`
- `xcodebuild test -scheme swift-lm-Package -destination 'platform=macOS,arch=arm64' -only-testing:MetalCompilerTests/PrefillTransferTests`
- `xcodebuild test -scheme swift-lm-Package -destination 'platform=macOS,arch=arm64' -only-testing:MetalCompilerTests/RotorQuantCorrectnessTests`

Required expectations:

- LFM local smoke mentions `Tokyo` (`ReleaseSmokeOutputTests`)
- Gemma4 FP16 real bundle first non-empty chunk starts with `Tokyo` (`RotorQuantRealBundleBaselineTests`)
- Qwen3.5 real bundle first non-empty chunk starts with `Tokyo`
- RotorQuant Gemma4 full K+V paths (RotorQ8, RotorQ4) preserve the same short factual answer shape (`RotorQuantRealBundleFullTests`)
- Hybrid stateful sequence prefill is model-family gated by real-bundle trace
  equivalence. BF16 LFM short-convolution sequence prefill is enabled only while
  its focused short-trace test matches decode-equivalent ingestion. Qwen
  DeltaNet/SSM prefill is not considered enabled until BF16 sequence prefill
  produces the same first token and short decode trace as decode-equivalent
  sequential ingestion. Until then,
  `sequencePrefillFallbackReason == .stateSpaceSequenceKernelNotDecodeEquivalent`
  is the expected behavior for plans containing `ssm_recurrence_seq_*`.

## Performance Gates

These suites must pass and their output must be reviewed against the latest saved baseline:

- `xcodebuild test -scheme swift-lm-Package -destination 'platform=macOS,arch=arm64' -only-testing:MetalCompilerTests/RotorQuantBenchmarkTests`
- `xcodebuild test -scheme swift-lm-Package -destination 'platform=macOS,arch=arm64' -only-testing:MetalCompilerTests/BenchmarkDiagnosticsTests`

Review items:

- Gemma4 decode `RotorQ4` must not regress materially against FP16
- LFM decode `RotorQ4` may be near FP16, but should not show sustained regression beyond normal benchmark variance
- context-length decode scaling must remain monotonic
- host overhead must remain low relative to GPU time

## Model-Aware Policy Gate

The default inference policy is not a universal RotorQuant default.

- `InferencePolicy.default` remains conservative and automatic
- `ModelBundleLoader` resolves the default to `RotorQ4/RotorQ4` only when the graph contains attention-backed KV cache decode
- non-attention graphs must not be forced onto RotorQuant

This behavior is covered by `MetalCompilerTests/RotorQuantCorrectnessTests`.

## Probe Gate

`ENABLE_METAL_PROBES` should remain available for fast failure localization.

Probe coverage must include:

- prefill to decode handoff
- prompt-state save and restore
- sampling logits source selection
- hidden override and deepstack staging paths

Probe output must stay disabled by default.

## Documentation Gate

The public API documentation must stay consistent across:

- `README.md`
- `docs/using-swift-lm.md`
- `Sources/SwiftLM/SwiftLM.docc/`

Required expectations:

- `Container / Context / Input` is the documented API shape for both generation and embeddings
- prompt-time thinking control is documented under `PromptPreparationOptions`
- output-time reasoning visibility is documented under `GenerationParameters.reasoning` / `ReasoningOptions`
- staged generation APIs are described as advanced, not as the default entry point
- embedding docs use `TextEmbeddingInput` as the preferred request value

## Test Execution Rules

To reduce crashes and memory pressure during release validation:

- run `build-for-testing` once
- run `test-without-building` suite by suite
- avoid one large `xcodebuild test` invocation across all real bundle suites
- keep real model scopes tight and prefer autorelease-friendly structure

## Not Yet Release-Blocking

These are still important, but they are not the current release gate:

- async decode pipeline expansion
- prerecording / MTL4 indirect execution work
- model-specific automatic policy tuning beyond the current attention-based default resolver
