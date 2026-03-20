# Compiler Optimization Report

Date: 2026-03-19  
Repository: `swift-lm`  
Hardware: Apple M4 Max  
OS: macOS 26.3 (25D125)

## Scope

This report is limited to compiler and backend optimization.

It covers:

1. dispatch-plan optimization quality
2. emitted kernel hot spots
3. optimizer choice for the compiled Metal path
4. kernel-generation changes for projection-heavy decode
5. where further compiler work is likely to pay off

It does not treat higher-level runtime features such as stream chunking or prompt-state reuse as compiler optimizations.

## Questions

- Does `AggressiveOptimizer` materially improve the compiled decode path?
- Which emitted kernels dominate decode time after current optimizations?
- Is the compiler still leaving obvious backend-side throughput on the table?
- Where should the next compiler optimization effort go?

## Current Status

The most reliable current baseline is the accepted row-major exact-shape BF16 argbuf path after repeated revalidation:

- optimizer comparison:
  - `none`: about `128 tok/s`
  - `standard`: low `130 tok/s`
  - `aggressive`: mid `136 tok/s`
- hot exact-shape GEMV microbench:
  - total: roughly `2.5-2.7 ms`
  - dominant families remain split close to `52/48` between:
    - `gemv_2048_6144_bf16_argbuf`
    - `gemv_2048_sq_bf16_argbuf`
- decode backend state:
  - `147/147` decode steps use encoded argument buffers
  - prepared tail is `0`
  - resident constants are pruned down to `115` steps

Acceptance rule:

- because repeated benchmark runs show a few percent of thermal/noise variation, a kernel change is only accepted if:
  - the target hot-family microbench improves materially, and
  - aggregate optimizer comparison does not regress outside that noise band

This means the compiler is no longer primarily limited by:

- dispatch-count reduction alone
- decode binding overhead
- the remaining argument-buffer migration work

The dominant cost is now inside the hottest exact-shape decode kernels themselves.

Latest accepted kernel-body change:

- `gemv_2048_6144_bf16_argbuf` now uses a row-major `ushort4` contiguous read path with a fixed-iteration loop (`packed4FixedPointerInput`)
- this keeps the generic compiler/store contract unchanged
- it does not introduce a model-specific layout
- parity remains unchanged, including the known decode drift fingerprint (`step 1: Python 2944 / Metal 859`)

## Backend Contract Update

The compiler/backend contract is now clearer:

- decode may request specialized STAF layouts
- prefill must remain on canonical row-major layout

This is now enforced as an execution-phase decision in the weight-access path, instead of being an implicit side effect of whichever specialized buffers happen to exist in the store.

Implication:

- STAF can evolve toward optimized execution layouts without leaking decode-only assumptions into prefill
- specialized layout experiments can be evaluated as backend capabilities, not as model-specific hacks

## Bottleneck Hierarchy

At the current accepted baseline, decode time is concentrated in a very small kernel set:

1. `gemv_2048_6144_bf16_argbuf`
2. `gemv_2048_sq_bf16_argbuf`
3. `fused_swiglu_projection_2048_bf16_argbuf`
4. everything else

The first two kernels are the real bottleneck. They jointly account for most steady-state decode time, while the rest of the backend is already in the noise floor by comparison.

The clearest structural reading from the accepted profiles is:

- exact-shape BF16 GEMV dominates decode
- fused SwiGLU front-half is the only meaningful secondary hot family
- attention, norm, conv-state update, argmax, and output-head kernels are no longer first-order bottlenecks
- planner/backend work has mostly been amortized away by full argbuf coverage

In other words, the compiler frontier has moved from:

- dispatch planning
- buffer binding
- launch-family separation

to:

- weight read path quality
- BF16 conversion cost
- threadgroup-memory pressure
- exact-shape kernel-body design

## Projection Cost Model

To stop treating kernel tuning as blind micro-experimentation, the compiler now has a decode projection cost report that aggregates logical weight bytes, input/output bytes, and estimated FLOPs from the compiled decode entry set itself.

Current report for the accepted row-major BF16 baseline:

- projection steps: `61`
- top projection families by estimated total bytes:
  - `gemv_8192_tiled_bf16`: `16` steps, `8192 -> 2048`, `32.0 MB/step`, `512.3 MB total`, arithmetic intensity `1.00`
  - `gemv_vocab_bf16`: `1` step, `2048 -> 65536`, `256.0 MB/step`, `256.1 MB total`, arithmetic intensity `1.00`
  - `gemv_2048_6144_bf16`: `10` steps, `2048 -> 6144`, `24.0 MB/step`, `240.2 MB total`, arithmetic intensity `1.00`
  - `gemv_2048_sq_bf16`: `22` steps, `2048 -> 2048`, `8.0 MB/step`, `176.2 MB total`, arithmetic intensity `1.00`
  - `gemv_bf16`: `12` steps, `2048 -> 512`, `2.0 MB/step`, `24.1 MB total`, arithmetic intensity `1.00`

Interpretation:

- all dense decode projection families are effectively at `~1 flop/byte`
- this is a memory-traffic regime, not a compute-heavy regime
- `gemv_vocab_bf16` has huge per-step traffic, but only one occurrence per decode step
- the measured steady-state bottleneck remains `gemv_2048_6144_bf16_argbuf` plus `gemv_2048_sq_bf16_argbuf` because they combine large logical traffic with high step multiplicity
- `gemv_8192_tiled_bf16` has even higher total logical traffic, but its tiled structure is currently efficient enough that it is not the primary measured hotspot

This cost model does not replace profiling. It explains why the exact-shape BF16 decode projections are still the right target:

- their arithmetic intensity is already too low for arithmetic micro-tuning to matter much
- the plausible next wins are weight-read structure, address-generation quality, and layout-aware memory behavior
- launch sweeps and small expression rewrites are now structurally lower-yield

## DecodeSync Host-Overhead Diagnosis

To test whether the remaining BF16 gap versus `llama.cpp` was mainly host orchestration rather than kernel quality, the benchmark suite now includes a `decodeSync` breakdown and a direct `decodeSync` vs `decode/flush` throughput comparison.

Current aggressive decode breakdown:

- total wall time: about `8.4 ms/token`
- command-buffer GPU time: about `8.1 ms/token`
- host overhead: about `0.29-0.30 ms/token` (`~3.5%`)
  - CPU token/position writes: negligible (`~0.4 us/token`)
  - encode + submit: about `122 us/token`
  - readback: negligible (`~2 us/token`)

Interpretation:

- the host-side residual is real, but small
- the user-visible public generation path cannot currently recover that residual through the existing `decode` / `flush` pipeline
- a direct aggressive comparison showed `decode/flush` slightly *slower* than `decodeSync` (within noise, but not better)

Implication:

- host orchestration is not the current highest-yield optimization frontier
- the remaining BF16 gap is still better explained by the exact-shape decode kernels themselves than by the command-buffer wrapper
- the next wins should continue to target:
  - `gemv_2048_6144_bf16_argbuf`
  - `gemv_2048_sq_bf16_argbuf`

## Bottleneck Analysis

### 1. The decode path is projection-bound, not attention-bound

Repeated per-step profiles show that `flash_attn_decode_argbuf` is below `1%` of decode time, while exact-shape GEMV families dominate. This removes attention from the critical path for the current model and hardware.

Implication:

- more flash-attention tuning is not the highest-yield next step
- wins must come from projection-heavy decode kernels

### 2. The main bottleneck is now kernel-body quality inside exact-shape BF16 argbuf GEMV

The accepted argbuf migration and resident-constant pruning proved that backend overhead was real, but they also showed a hard limit: after full encoded coverage, the hottest kernels remained:

- `gemv_2048_6144_bf16_argbuf`
- `gemv_2048_sq_bf16_argbuf`

That means the remaining time is spent mostly on:

- weight reads
- BF16-to-float conversion
- inner-loop address generation
- threadgroup-local staging tradeoffs

The latest accepted `2048 -> 6144` change fits this model exactly:

- it does not change launch
- it does not change layout
- it does not change fusion
- it only changes the row-major weight-read structure from `ushort2` pair reads to a contiguous `ushort4` read per lane

This is consistent with the cost model: at `~1.00 flop/byte`, wins are more likely to come from reducing read-path overhead than from arithmetic expression rewrites.

Implication:

- the next useful work is family-specific weight/read-path tuning
- generic launch sweeps are now low-yield

### 2a. The hot `2048 -> 6144` family is semantically narrow

A dedicated decode-binding diagnostic showed that the real `gemv_2048_6144_bf16_argbuf` hot family is not an arbitrary collection of `6144 x 2048` tensors.

It maps specifically to the ten short-convolution input projections:

- `model.layers.0.conv.in_proj.weight`
- `model.layers.1.conv.in_proj.weight`
- `model.layers.3.conv.in_proj.weight`
- `model.layers.4.conv.in_proj.weight`
- `model.layers.6.conv.in_proj.weight`
- `model.layers.7.conv.in_proj.weight`
- `model.layers.9.conv.in_proj.weight`
- `model.layers.11.conv.in_proj.weight`
- `model.layers.13.conv.in_proj.weight`
- `model.layers.15.conv.in_proj.weight`

Implication:

- future layout/read-path experiments for `2048 -> 6144` should be targeted at `conv.in_proj.weight`
- picking an arbitrary `6144 x 2048` tensor is not a valid proxy for the hot family

### 2b. The blocked-layout problem is in the real BF16 read path, not in tensor selection

The blocked `8-row x 128-element` layout was revalidated in three stages:

- synthetic CPU pack order
- synthetic GPU blocked-vs-row-major equivalence
- real-model raw byte packing for `6144 x 2048` BF16 tensors

All three passed.

However, when the same blocked BF16 kernel path was exercised against the actual hot `conv.in_proj.weight` tensors, the blocked path still diverged from the row-major path and produced zeroed outputs.

Implication:

- the remaining blocked-layout issue is not explained by choosing the wrong tensor family
- it is also not explained by the CPU-side pack order alone
- the unresolved part is the real-model BF16 blocked read path itself, or its integration with the generated kernel path

### 3. Threadgroup-memory pressure matters as much as raw reuse

The accepted `6144` buffer-precision staging win and the rejected `square` / `8192` copies of the same idea show that "more float staging" is not a universally good rule on Apple GPU.

What the accepted results suggest:

- reducing threadgroup-local footprint can help more than increasing local reuse
- occupancy and residency effects are visible across neighboring hot kernels
- the right staging policy is family-specific, not global

Implication:

- exact-shape families need explicit staging policy
- future tuning should preserve that policy boundary instead of collapsing back to one generic GEMV path

### 4. Dispatch count still matters, but only after kernel quality is good enough

The optimizer comparison moved over time:

- early on, fewer dispatches were the main visible win
- later, `standard` could outperform `aggressive`
- after full argbuf coverage and exact-shape BF16 improvements, `aggressive` recovered and became best again

Interpretation:

- dispatch reduction is necessary but not sufficient
- once the dominant kernels are under-tuned, more fusion can hide or amplify the wrong costs
- once kernel-body quality improves enough, the lower-dispatch plan wins again

Implication:

- optimizer strategy should be evaluated after kernel-family changes, not in isolation
- the next frontier is still kernel quality first, optimizer policy second

### 5. The output head is no longer a top-tier bottleneck

Earlier in the report, `gemv_vocab_bf16` was a major hot step. After shape-family specialization and argbuf migration, it dropped behind the exact-shape `2048` projection families by a large margin.

Implication:

- `lm_head` is no longer the best next target
- time spent there is less likely to beat work on `2048 -> 6144` and `2048 -> 2048`

## Where The Next Wins Are Likely

The highest-probability next compiler work is:

1. family-specific weight/read-path tuning for `gemv_2048_6144_bf16_argbuf`
2. the same for `gemv_2048_sq_bf16_argbuf`
3. only then, a revisit of `fused_swiglu_projection_2048_bf16_argbuf`

The report so far argues against spending the next round on:

- flash attention
- broader launch-width sweeps
- more argument-buffer migration
- global staging-policy changes
- generic exact-shape clones for already-small families

The data points instead to a narrow conclusion:

- the backend is structurally in the right shape
- the remaining performance problem is concentrated in two BF16 exact-shape GEMV kernels
- the likely path forward is weight-layout and weight-read-path work, not more planner work

## Why These Kernels Dominate

The current bottleneck is not accidental. The dominant exact-shape families combine all of the expensive properties at once:

- they run many times per token
- they are dense BF16 projections
- they sit on the decode hot path
- they still pay BF16 weight-conversion cost in the inner loop
- they are large enough to stress bandwidth and local-memory policy, but small enough that occupancy mistakes still matter

In practice that means:

- `gemv_2048_sq_bf16_argbuf`
  - is frequent enough that even modest per-call overhead compounds
- `gemv_2048_6144_bf16_argbuf`
  - has enough output work that each invocation is expensive on its own
- together
  - they dominate both by frequency and by cost per invocation

This is why work on:

- argument-buffer coverage
- launch-family cleanup
- residual/norm cleanup

still left the decode profile dominated by these two kernels. Those earlier changes removed surrounding overhead; they did not change the underlying BF16 projection cost.

## What The Rejected Experiments Mean

The rejected experiments are useful because they narrow the problem.

### 1. The bottleneck is not "just widen launch"

Rejected launch sweeps showed that:

- wider threadgroups can make the hottest kernels slower
- narrower launches can help one kernel but hurt the aggregate
- once exact-shape specialization is in place, launch is already near a local optimum

Meaning:

- the remaining problem is inside the kernel body and memory path, not the gross launch shape

### 2. The bottleneck is not "just use more threadgroup float staging"

The accepted and rejected staging experiments showed a mixed pattern:

- `2048 -> 6144` improved with buffer-precision staging
- `2048 -> 2048` regressed with the same change
- `2048 -> 8192` regressed with the same change
- `8192 -> 2048` preferred the existing tiled float-staged path over broader local staging

Meaning:

- Apple GPU performance here is constrained by occupancy and threadgroup-local footprint as much as by raw reuse
- there is no single global staging rule for decode GEMV
- staging must remain explicit at the exact-shape family level

### 3. The bottleneck is not arithmetic-expression choice alone

Multiple narrow rewrites failed to produce stable aggregate gains:

- `dot(...)` vs scalar multiply-add
- `fma(...)` substitutions
- pointer-increment accumulation
- fixed-iteration loop rewrites
- more aggressive inner-loop unroll on already-hot families

Meaning:

- the cost center is not just ALU instruction selection
- the dominant issue is likely the weight-read / conversion path and the pressure it creates on the memory subsystem

### 4. The bottleneck is no longer binding infrastructure

The argbuf migration was successful:

- decode is fully `argEncoded`
- prepared tail is `0`
- resident constants were pruned from the hottest encoded steps

Meaning:

- binding/backend work produced real gains
- but it has now hit diminishing returns
- the next large win is unlikely to come from more dispatch-plan or binding-table refactoring

## Root-Cause Hypothesis

The current evidence supports a narrow root-cause hypothesis:

- decode throughput is now limited primarily by BF16 dense weight access in the exact-shape `input=2048` families
- the cost is not only reading bytes from memory
- it is the combination of:
  - row-wise BF16 fetch
  - BF16-to-float decode
  - address generation
  - threadgroup-local staging policy
  - occupancy sensitivity on Apple GPU

That hypothesis explains all of the observed behavior:

- exact-shape specialization helps
- argbuf helps
- family-specific staging helps in some shapes and hurts in others
- launch sweeps do not create consistent wins
- arithmetic micro-tuning does not move the aggregate enough

## Next Compiler-Only Experiments

The next experiments should be ordered by how directly they attack the current root cause.

### 1. `gemv_2048_6144_bf16_argbuf` family-specific weight-read mode

Goal:

- keep the same logical tensor layout
- change only how that family reads BF16 weights in the inner loop

Reason:

- this is the hottest single family
- it already proved sensitive to staging policy
- it is the best candidate for a narrow, measurable read-path specialization

### 2. `gemv_2048_sq_bf16_argbuf` family-specific weight-read mode

Goal:

- repeat the same style of experiment for the square family only after `6144` has a clear result

Reason:

- it is the second largest cost center
- but the earlier staging experiments showed it responds differently from `6144`
- it should not inherit the `6144` policy by default

### 3. Family-specific BF16 weight layout, if read-path tuning stalls

Goal:

- test whether the current row-major BF16 STAF layout is itself the limiting factor for the hottest exact-shape decode families

Reason:

- repeated micro-tuning failures suggest the compiler may be near the limit of what it can get from the current layout
- if so, the next real gain must come from how weights are stored, not only how they are read

### 4. Only after that, revisit `fused_swiglu_projection_2048_bf16_argbuf`

Reason:

- it is the largest secondary hot family
- but it is still clearly below the combined cost of the two exact-shape GEMV kernels
- it should not displace work on the true bottleneck

## Architecture Under Test

Compiler path:

- [MetalInferenceCompiler.swift](/Users/1amageek/Desktop/swift-lm/Sources/MetalCompiler/MetalInferenceCompiler.swift)
- [MetalSourceGenerator.swift](/Users/1amageek/Desktop/swift-lm/Sources/MetalCompiler/Fragments/MetalSourceGenerator.swift)
- [StandardOptimizer.swift](/Users/1amageek/Desktop/swift-lm/Sources/MetalCompiler/Optimization/StandardOptimizer.swift)
- [AggressiveOptimizer.swift](/Users/1amageek/Desktop/swift-lm/Sources/MetalCompiler/Optimization/AggressiveOptimizer.swift)

Execution path used for measurement:

- [MetalInferenceModel.swift](/Users/1amageek/Desktop/swift-lm/Sources/MetalCompiler/MetalInferenceModel.swift)
- [BenchmarkTests.swift](/Users/1amageek/Desktop/swift-lm/Tests/MetalCompilerTests/BenchmarkTests.swift)

The key distinction is:

- `StandardOptimizer`
  - moderate fusion
  - keeps more primitive boundaries
- `AggressiveOptimizer`
  - more dispatch collapsing
  - more batched/fused lowering opportunities

## Experiments

### Experiment 1: Raw decode throughput

Command:

```bash
xcodebuild test -scheme swift-lm-Package -destination 'platform=macOS' \
  -only-testing:MetalCompilerTests/BenchmarkTests
```

Purpose:

- Compare compiled decode throughput across optimizer strategies.
- Measure dispatch-count reduction from optimization.

### Experiment 2: Per-step decode profiling

Command:

```bash
xcodebuild test -scheme swift-lm-Package -destination 'platform=macOS' \
  -only-testing:MetalCompilerTests/BenchmarkTests
```

Purpose:

- Identify which generated kernels dominate steady-state decode.
- Distinguish attention cost from projection cost.

### Experiment 3: Optimizer comparison inside the benchmark suite

Command:

```bash
xcodebuild test -scheme swift-lm-Package -destination 'platform=macOS' \
  -only-testing:MetalCompilerTests/BenchmarkTests
```

Purpose:

- Compare `none`, `standard`, and `aggressive` in the same compiler-focused harness.
- Check whether additional fusion still helps after recent refactors.

### Experiment 4: Decode GEMV tile and staging adjustment

Code changes:

- [MetalSourceGenerator.swift](/Users/1amageek/Desktop/swift-lm/Sources/MetalCompiler/Fragments/MetalSourceGenerator.swift)
- [MetalInferenceCompiler.swift](/Users/1amageek/Desktop/swift-lm/Sources/MetalCompiler/MetalInferenceCompiler.swift)

Purpose:

- Align actual compile-time GEMV tile sizes with the intended decode hot path.
- Stage decode GEMV input tiles in buffer precision instead of `float`.
- Give `gemv_large_bf16` a larger tile than ordinary `gemv_bf16`.

Command:

```bash
xcodebuild test -scheme swift-lm-Package -destination 'platform=macOS' \
  -only-testing:MetalCompilerTests/BenchmarkTests

xcodebuild test -scheme swift-lm-Package -destination 'platform=macOS' \
  -only-testing:MetalCompilerTests/ReferenceComparisonTests
```

### Experiment 5: Shape-specific `inputDimension=2048` decode GEMV family

Code changes:

- [MetalSourceGenerator.swift](/Users/1amageek/Desktop/swift-lm/Sources/MetalCompiler/Fragments/MetalSourceGenerator.swift)
- [MetalInferenceCompiler.swift](/Users/1amageek/Desktop/swift-lm/Sources/MetalCompiler/MetalInferenceCompiler.swift)

Purpose:

- Split the common decode projections with `inputDimension=2048` into their own kernel family.
- Stage the full hidden vector once for `2048 -> {2048, 6144, 8192}` dense projections.
- Keep `lm_head` as a separate vocab-sized specialization.

Command:

```bash
xcodebuild test -scheme swift-lm-Package -destination 'platform=macOS' \
  -only-testing:MetalCompilerTests/BenchmarkTests

xcodebuild test -scheme swift-lm-Package -destination 'platform=macOS' \
  -only-testing:MetalCompilerTests/ReferenceComparisonTests
```

### Experiment 6: `inputDimension=8192` tiled decode GEMV family

Code changes:

- [MetalSourceGenerator.swift](/Users/1amageek/Desktop/swift-lm/Sources/MetalCompiler/Fragments/MetalSourceGenerator.swift)
- [MetalInferenceCompiler.swift](/Users/1amageek/Desktop/swift-lm/Sources/MetalCompiler/MetalInferenceCompiler.swift)

Purpose:

- Split the remaining hot `8192 -> 2048` decode projections into a dedicated tiled kernel family.
- Reduce tile-loop barrier count without paying the occupancy cost of staging the full 8192-element input.
- Keep threadgroup memory modest at `1024` elements instead of full-input staging.

Command:

```bash
xcodebuild test -scheme swift-lm-Package -destination 'platform=macOS' \
  -only-testing:MetalCompilerTests/BenchmarkTests

xcodebuild test -scheme swift-lm-Package -destination 'platform=macOS' \
  -only-testing:MetalCompilerTests/ReferenceComparisonTests
```

### Experiment 7: 4-way unroll for specialized dense decode GEMV

Code changes:

- [MetalSourceGenerator.swift](/Users/1amageek/Desktop/swift-lm/Sources/MetalCompiler/Fragments/MetalSourceGenerator.swift)

Purpose:

- Reduce loop overhead inside `gemv_2048(_bf16)` and `gemv_vocab(_bf16)` without changing launch shape or memory layout.
- Keep the optimization local to the specialized dense GEMV family.

Command:

```bash
xcodebuild test -scheme swift-lm-Package -destination 'platform=macOS' \
  -only-testing:MetalCompilerTests/BenchmarkTests

xcodebuild test -scheme swift-lm-Package -destination 'platform=macOS' \
  -only-testing:MetalCompilerTests/ReferenceComparisonTests
```

### Experiment 8: `vocabDense` launch specialization

Code changes:

- [MetalInferenceCompiler.swift](/Users/1amageek/Desktop/swift-lm/Sources/MetalCompiler/MetalInferenceCompiler.swift)

Purpose:

- Separate `vocabDense` launch policy from `input2048Dense`.
- Increase rows per threadgroup for the output head without changing the kernel body.
- Test whether `lm_head` prefers a wider launch than the rest of the `input=2048` family.

Command:

```bash
xcodebuild test -scheme swift-lm-Package -destination 'platform=macOS' \
  -only-testing:MetalCompilerTests/BenchmarkTests

xcodebuild test -scheme swift-lm-Package -destination 'platform=macOS' \
  -only-testing:MetalCompilerTests/ReferenceComparisonTests
```

### Experiment 9: Exact-shape `input=2048` kernels with fixed-input specialization

Code changes:

- [MetalInferenceCompiler.swift](/Users/1amageek/Desktop/swift-lm/Sources/MetalCompiler/MetalInferenceCompiler.swift)
- [MetalSourceGenerator.swift](/Users/1amageek/Desktop/swift-lm/Sources/MetalCompiler/Fragments/MetalSourceGenerator.swift)

Purpose:

- Split the `input=2048` decode family into exact-shape variants for `2048 -> 2048`, `2048 -> 6144`, and `2048 -> 8192`.
- Keep specialization in the family resolver instead of adding planner-local heuristics.
- Let emitted kernels assume `inputDimension == 2048` and remove hot-path bounds checks during input staging and inner-loop accumulation.

Command:

```bash
xcodebuild test -scheme swift-lm-Package -destination 'platform=macOS' \
  -only-testing:MetalCompilerTests/BenchmarkTests

xcodebuild test -scheme swift-lm-Package -destination 'platform=macOS' \
  -only-testing:MetalCompilerTests/ReferenceComparisonTests
```

### Experiment 10: Contiguous-lane accumulation in specialized dense GEMV

Code changes:

- [MetalSourceGenerator.swift](/Users/1amageek/Desktop/swift-lm/Sources/MetalCompiler/Fragments/MetalSourceGenerator.swift)

Purpose:

- Keep the existing shape-family split unchanged.
- Change only the inner-loop access pattern of `generateSpecializedDenseGEMV(...)`.
- Make each SIMD lane consume a contiguous `4`-element chunk instead of four strided elements separated by `SIMD_WIDTH`.
- Improve locality and reduce address-generation overhead in:
  - `gemv_2048_sq(_bf16)`
  - `gemv_2048_6144(_bf16)`
  - `gemv_2048_8192(_bf16)`
  - `gemv_vocab(_bf16)`

Command:

```bash
xcodebuild test -scheme swift-lm-Package -destination 'platform=macOS' \
  -only-testing:MetalCompilerTests/BenchmarkTests

xcodebuild test -scheme swift-lm-Package -destination 'platform=macOS' \
  -only-testing:MetalCompilerTests/ReferenceComparisonTests
```

## Results

Note:

- GPU throughput varies across long benchmark runs.
- The most reliable signal is the structural result and same-run relative comparison, not any single absolute `tok/s` number.

### Raw decode throughput

From [BenchmarkTests.swift](/Users/1amageek/Desktop/swift-lm/Tests/MetalCompilerTests/BenchmarkTests.swift):

| Metric | Result |
|---|---:|
| Standard decode throughput | `116.4 tok/s` |
| Standard decode latency | `8.59 ms/tok` |
| Aggressive decode throughput | `114.5 tok/s` |
| Aggressive decode latency | `8.74 ms/tok` |
| Standard dispatch count | `179` |
| Aggressive dispatch count | `128` |

Interpretation:

- The compiler is successfully trading graph complexity for fewer dispatches.
- `AggressiveOptimizer` still reduces dispatch count by about `28.5%`.
- After the latest kernel-body specialization, fewer dispatches no longer guarantee faster decode.
- In the current compiler state, kernel quality inside the dominant GEMV families matters more than additional dispatch collapsing.

### Optimizer comparison

Also from [BenchmarkTests.swift](/Users/1amageek/Desktop/swift-lm/Tests/MetalCompilerTests/BenchmarkTests.swift):

| Optimizer | Decode dispatches | Prefill steps | Decode tok/s | Prefill tok/s |
|---|---:|---:|---:|---:|
| `none` | `242` | `258` | `114.0` | `114.9` |
| `standard` | `179` | `258` | `116.1` | `116.0` |
| `aggressive` | `128` | `258` | `112.3` | `112.1` |

Interpretation:

- In the current compiler state, `standard` wins in the same-run comparison.
- Structurally, only decode dispatch count changes; prefill step count is unchanged.
- This means the current optimizer work is still primarily decode-side, not prefill-side.
- The latest GEMV specialization interacts better with the more explicit `standard` plan than with the more collapsed `aggressive` plan.

### Per-step decode profile

From [BenchmarkTests.swift](/Users/1amageek/Desktop/swift-lm/Tests/MetalCompilerTests/BenchmarkTests.swift):

| Kernel | Share |
|---|---:|
| `gemv_2048_8192_bf16` | `37.9%` |
| `gemv_8192_tiled_bf16` | `25.6%` |
| `gemv_2048_6144_bf16` | `9.2%` |
| `gemv_vocab_bf16` | `8.8%` |
| `gemv_2048_sq_bf16` | `7.9%` |
| `gemv_bf16` | `3.6%` |
| `fused_residual_add_copy_rms_norm_bf16` | `2.6%` |
| `flash_attn_decode` | `1.1%` |

Top individual step:

- `gemv_vocab_bf16`: `726 us`, `8.8%`

Interpretation:

- Decode is still overwhelmingly projection-bound.
- Attention is not the current bottleneck.
- The compiler has already pushed the system into a regime where further wins will come from GEMV-family lowering, not from flash-attention work.

### Decode GEMV tile/staging adjustment

The latest compiler-side experiment changed two details in emitted decode GEMV kernels:

- ordinary `gemv` kernels now use a larger tile in the actual compile path
- input tiles are staged in the original decode buffer precision rather than widened to `float` in threadgroup memory

`gemv_large_bf16` also now receives a larger tile than ordinary decode GEMV.

Observed result:

- standard decode: `99.5 tok/s -> 100.9 tok/s`
- aggressive decode: `102.6 tok/s -> 105.4 tok/s`
- optimizer comparison: `aggressive 103.8 tok/s`, `standard 100.4 tok/s`, `none 99.2 tok/s`

Correctness result:

- [ReferenceComparisonTests.swift](/Users/1amageek/Desktop/swift-lm/Tests/MetalCompilerTests/ReferenceComparisonTests.swift) still passes
- prefill parity is unchanged
- decode drift behavior is unchanged rather than worsened

Interpretation:

- The previous compiler path still had under-tuned GEMV staging in the real compile path.
- This was a backend-quality issue, not an IR or optimizer issue.
- The gain is modest but real, and it came without changing dispatch count or sacrificing reference behavior.

### Shape-specific `inputDimension=2048` decode GEMV family

The latest compiler-side experiment added a new decode kernel family for dense projections with `inputDimension=2048`:

- `gemv_2048(_bf16)` for the common `2048 -> {2048, 6144, 8192}` decode projections
- `gemv_vocab(_bf16)` kept separately for the output head

The compiler now:

- recognizes this shape during kernel selection
- emits a kernel that stages the full 2048-element hidden vector once into threadgroup memory
- uses a wider threadgroup for that specialized family than for the generic tiled path

Observed result:

- standard decode: `100.9 tok/s -> 103.9 tok/s`
- aggressive decode: `105.4 tok/s -> 108.0 tok/s`
- end-to-end decode benchmark: `9.90 ms/tok -> 9.75 ms/tok`
- per-step total decode time: `9531 us -> 9333 us`

Structural result:

- `64` decode steps moved from generic `gemv_bf16` into `gemv_2048_bf16`
- the hot path is now split into:
  - `gemv_2048_bf16`: `58.3%`
  - `gemv_bf16`: `26.7%`
  - `gemv_vocab_bf16`: `8.6%`

Correctness result:

- [ReferenceComparisonTests.swift](/Users/1amageek/Desktop/swift-lm/Tests/MetalCompilerTests/ReferenceComparisonTests.swift) still passes
- step-0 and step-1 parity signatures are unchanged
- decode drift still originates before the output head rather than in the new GEMV family

Interpretation:

- The previous generic decode GEMV path was still paying unnecessary tile-loop and barrier overhead for the most common hidden-size projections.
- Shape-specific specialization is paying off even without any change to dispatch count.
- The compiler now has evidence that projection families should be selected by tensor shape, not only by quantization format or optimizer pass.

### `inputDimension=8192` tiled decode GEMV family

The next compiler-side experiment targeted the remaining generic `8192 -> 2048` decode projections.

An initial full-input staging design regressed badly because it raised threadgroup memory too far and collapsed occupancy. That design was discarded.

The retained design instead:

- emits `gemv_8192_tiled(_bf16)`
- keeps the generic GEMV structure
- increases tile size to `1024` only for this shape family
- keeps threadgroup memory to `2048` bytes rather than `16384` bytes

Observed result:

- standard decode: `103.7 tok/s -> 104.4 tok/s`
- end-to-end decode benchmark: `9.82 ms/tok -> 9.72 ms/tok`
- optimizer comparison: `aggressive 106.3 tok/s`, `standard 102.4 tok/s`, `none 100.5 tok/s`

Per-step structural result:

- `16` decode steps moved from generic `gemv_bf16` into `gemv_8192_tiled_bf16`
- `gemv_8192_tiled_bf16` now accounts for `22.3%`
- remaining generic `gemv_bf16` dropped to `3.2%`

Correctness result:

- [ReferenceComparisonTests.swift](/Users/1amageek/Desktop/swift-lm/Tests/MetalCompilerTests/ReferenceComparisonTests.swift) still passes
- step-0, step-1, and layerwise parity signatures are unchanged

Interpretation:

- The `8192` family benefits from fewer tile iterations, but not from full-input staging.
- For this shape, the correct tradeoff is lower barrier count with controlled threadgroup memory, not maximum reuse at any cost.
- This is direct evidence that Metal kernel design here is occupancy-limited before it is purely bandwidth-limited.

### 4-way unroll for specialized dense decode GEMV

The next compiler-side experiment kept the same specialized families but changed the inner loop of `generateSpecializedDenseGEMV(...)`:

- from a 2-way unrolled loop
- to a 4-way unrolled loop

This applies to:

- `gemv_2048(_bf16)`
- `gemv_vocab(_bf16)`

Observed result:

- standard decode: `104.4 tok/s -> 105.4 tok/s`
- aggressive decode: `107.0 tok/s -> 109.3 tok/s`
- optimizer comparison: `none 101.0`, `standard 103.8`, `aggressive 108.7 tok/s`

Correctness result:

- [ReferenceComparisonTests.swift](/Users/1amageek/Desktop/swift-lm/Tests/MetalCompilerTests/ReferenceComparisonTests.swift) still passes
- step-0, step-1, step-2, layerwise diagnostic, and output-head diagnostic are unchanged

Interpretation:

- Once the compiler had separated the hot projection families by shape, small kernel-body tuning started to pay off again.
- This gain came without changing dispatch count, proving the current limit is kernel quality inside the existing plan, not planner structure alone.

### `vocabDense` launch specialization

The next compiler-side experiment stopped treating the output head as just another `input=2048` projection family for launch purposes.

The change was simple:

- `input2048Dense` kept `8` simdgroups
- `vocabDense` moved to `16` simdgroups

This changes the output-head launch from:

- `grid=8192, tg=256`

to:

- `grid=4096, tg=512`

Observed result:

- standard decode: `105.4 tok/s -> 106.6 tok/s`
- aggressive decode: `109.3 tok/s -> 110.2 tok/s`
- `gemv_vocab_bf16`: `864 us -> 809 us`

Correctness result:

- [ReferenceComparisonTests.swift](/Users/1amageek/Desktop/swift-lm/Tests/MetalCompilerTests/ReferenceComparisonTests.swift) still passes
- output-head diagnostic remains `argmax`-identical when fed Python `final_hidden`
- decode drift signature is unchanged

Interpretation:

- The output head has enough row parallelism to benefit from a wider launch even when the rest of the `input=2048` family does not want the same policy.
- This validates the current design direction: specialization should live at the family level, not as isolated ad-hoc tuning in the planner.

### Exact-shape `input=2048` kernels with fixed-input specialization

The next compiler-side experiment took the existing `input=2048` family and made the specialization explicit in both the planner and the emitted kernels.

Structural change:

- compiler resolves exact shape families:
  - `gemv_2048_sq(_bf16)`
  - `gemv_2048_6144(_bf16)`
  - `gemv_2048_8192(_bf16)`
- specialized kernels now assume `inputDimension == 2048`
- input staging and inner-loop accumulation no longer pay per-element bounds checks in that family

Observed result:

- per-step decode total: `9336 us -> 9166 us`
- `gemv_2048_8192_bf16`: `3698 us -> 3599 us`
- `gemv_vocab_bf16`: `883 us -> 835 us`
- focused decode benchmark: `104.1 tok/s -> 103.8 tok/s` for `standard`, `109.0 tok/s -> 107.8 tok/s` for `aggressive`
- optimizer comparison remained noisy:
  - `none 101.9`
  - `standard 102.6`
  - `aggressive 107.0`

Correctness result:

- [ReferenceComparisonTests.swift](/Users/1amageek/Desktop/swift-lm/Tests/MetalCompilerTests/ReferenceComparisonTests.swift) still passes
- prefill parity is unchanged
- decode drift signature is unchanged
- output-head diagnostic remains `argmax`-identical when fed Python `final_hidden`

Interpretation:

- This experiment improved the hot kernels themselves, but the end-to-end optimizer comparison is noisy enough that the gain shows up more clearly in per-step profiling than in whole-suite `tok/s`.
- The design change is still valuable because it moves specialization into the compiler’s shape-family model instead of encoding one-off launch exceptions.
- The next productive step is to keep exploiting fixed-shape facts inside those families rather than broadening generic launch policies again.

### Contiguous-lane accumulation in specialized dense GEMV

The next compiler-side experiment kept the same exact-shape families and launch policies, but changed how each SIMD lane walks the staged input:

- previous pattern:
  - one lane consumed four elements separated by `SIMD_WIDTH`
- new pattern:
  - one lane consumes one contiguous `4`-element chunk
  - loop base advances by `SIMD_WIDTH * 4`

This leaves the family split unchanged and only changes the kernel body.

Observed result:

- focused decode benchmark:
  - `standard 104.1 tok/s -> 116.4 tok/s`
  - `aggressive 106.7 tok/s -> 114.5 tok/s`
- end-to-end decode benchmark:
  - `9.31 ms/tok -> 8.21 ms/tok`
- per-step total decode time:
  - `9313 us -> 8215 us`
- hot kernels:
  - `gemv_2048_8192_bf16: 3669 us -> 3111 us`
  - `gemv_2048_6144_bf16: 965 us -> 756 us`
  - `gemv_2048_sq_bf16: 850 us -> 646 us`
  - `gemv_vocab_bf16: 831 us -> 726 us`

Correctness result:

- [ReferenceComparisonTests.swift](/Users/1amageek/Desktop/swift-lm/Tests/MetalCompilerTests/ReferenceComparisonTests.swift) still passes
- prefill parity is unchanged
- decode parity signature is unchanged
  - step 1 argmax remains `Python 2944 / Metal 859`
  - step 1 logits max error remains about `0.536`
  - output-head diagnostic remains `argmax`-identical when fed Python `final_hidden`

Interpretation:

- For the current Apple GPU codegen, contiguous per-lane access beats the earlier strided-per-lane unroll inside these fixed-shape families.
- This is still a compiler-side shape specialization, not an optimizer-pass trick.
- The result also explains why `standard` now outperforms `aggressive` in the compiler harness: once the dominant dense kernels got materially better, extra fusion no longer repaid its own constraints.

### Rejected follow-up tuning

After the accepted exact-shape work, several narrower compiler-side experiments were run and discarded.

Rejected changes:

- widening `input2048 -> 8192` launch width beyond the current family policy
- widening `vocabDense` beyond the current `16` simdgroup launch
- splitting specialized GEMV accumulation into multiple independent scalar accumulators
- increasing `input8192Tiled` tile width from `1024` to `2048`
- staging `gemv_vocab(_bf16)` input as `float`
- splitting `vocabDense` into an exact `65536 x 2048` family
- reducing `vocabDense` launch width from `16` to `8` simdgroups
- vectorizing `gemv_vocab_bf16` BF16 reads as packed 4-lane loads
- switching exact-shape `fused_swiglu_projection_2048` sigmoid to `fast::exp`
- raising `gemv_vocab(_bf16)` unroll from `4` to `8`
- widening the exact `2048 -> 6144` family from `8` to `12` simdgroups
- replacing `gemv_8192_tiled` tile-load loops with explicit 4-way scalar loads
- rewriting the exact-shape `fused_swiglu_projection_2048` outer loop as a fixed iteration-count loop
- rewriting the fixed-shape inner loop in `gemv_8192_tiled` as a fixed iteration-count loop
- raising `gemv_2048_6144(_bf16)` unroll from `4` to `8`
- removing `max(1u, ...)` from `rowsPerThreadgroup` in the hot exact-shape decode families
- replacing `tid % SIMD_WIDTH` with `tiisg == 0` in the fused RMSNorm-family shared-sum writeback
- reducing `input2048ExpandedDense` launch width from `8` to `6` simdgroups
- vectorizing exact-shape `input2048Dense` BF16 reads as packed 4-lane dot products
- pointer-increment tile staging for the exact-shape `input2048Dense` family
- register-caching `inputLane` values inside `fused_swiglu_projection_2048`
- extending pointer-increment accumulation to float-staged exact-shape `input2048Dense` GEMV
- reducing `input20486144Dense` launch width from `8` to `6` simdgroups
- switching exact `gemv_2048_6144(_bf16)` to non-float staged input tiles
- forcing pointer-increment accumulation only for exact `gemv_2048_6144(_bf16)`
- pointer-increment tile staging for `vocabDense`
- replacing the exact `gemv_2048_6144_bf16_argbuf` input-side scalar pair reads with `half2` loads
- prepacking exact `gemv_2048_6144_bf16_argbuf` weights into an 8-row/128-element blocked layout
- repeating that `8-row/128-element` blocked layout through the formal STAF/store-side specialized access path

Observed result:

- none of these changed the structural hot path
- all of them either regressed decode throughput directly or worsened per-step time in:
  - `gemv_2048_8192_bf16`
  - `gemv_8192_tiled_bf16`
  - `gemv_vocab_bf16`
- the formal store-side blocked-layout path also failed correctness immediately:
  - decode step `0` argmax flipped from Python `521` to Metal `2`
  - conv-state drift spiked in the first decode step

Interpretation:

- the remaining headroom is not in broader launch sweeps
- the current compiler has already found a reasonable launch shape for these families
- the fused SwiGLU front-half also does not appear to be limited by transcendental latency alone
- `vocabDense` also does not improve from larger loop unroll, even when the single kernel gets slightly faster in isolation
- the current exact-shape families also do not benefit from replacing simple counted loops with more explicit scalarized control flow
- exact-shape launch-control simplifications also have not translated into stable decode wins
- further wins are more likely to come from kernel-body and data-layout work inside the existing family model than from widening threadgroups again

### Exact-shape fused SwiGLU for `input=2048`

The next compiler-side experiment applied the same family-specialization idea to the fused MLP front-half:

- generic `fused_swiglu_projection(_bf16)` remains available
- new exact-shape kernels:
  - `fused_swiglu_projection_2048`
  - `fused_swiglu_projection_2048_bf16`

Structural change:

- the fused gate/up projection path now stages the full `2048`-element hidden vector once
- both branch accumulations reuse the same staged input tile
- this keeps fusion in the compiler family model rather than reintroducing planner-local heuristics

Observed result:

- aggressive decode benchmark: `121.6 tok/s -> 129.6 tok/s`
- optimizer comparison:
  - `none 120.7`
  - `standard 125.2`
  - `aggressive 130.9`

Correctness result:

- [ReferenceComparisonTests.swift](/Users/1amageek/Desktop/swift-lm/Tests/MetalCompilerTests/ReferenceComparisonTests.swift) still passes
- standard-path prefill and decode parity signatures are unchanged
- output-head and layerwise diagnostics remain in the same range

Interpretation:

- the earlier aggressive regression was not “fusion is bad” in general
- the real issue was that the fused MLP front-half kernel had fallen behind the exact-shape dense GEMV family
- once fusion received the same shape-aware treatment, aggressive planning again became the fastest decode path in the compiler harness

### Selective fused SwiGLU in the standard optimizer

The next step was not to copy aggressive planning wholesale into the standard path.
Instead, the compiler now shares only the exact-shape MLP front-half fusion rule:

- standard still does not emit `batchedProjection` or `batchedFragment`
- standard now applies:
  - `gate_proj + up_proj + SwiGLU -> fusedSwiGLUProjection`
  - plus the existing norm-side graph fusions

Structural change:

- the fused fragment match was extracted into a shared compiler rule
- both optimizers can reuse the same exact-shape fused SwiGLU family
- the difference between `standard` and `aggressive` remains architectural:
  - `standard` keeps a simpler dispatch plan
  - `aggressive` adds batching on top of the same fused MLP front-half

Observed result:

- dispatch count:
  - `standard 210 -> 147`
- focused decode benchmark:
  - `standard 80.9 tok/s`
- end-to-end decode benchmark:
  - `standard 122.2 tok/s`
- optimizer comparison:
  - `none 120.7`
  - `standard 127.9`
  - `aggressive 130.6`

Per-step result:

- `fused_swiglu_projection_2048_bf16` becomes the largest standard-path hot family at `40.3%`
- `gemv_8192_tiled_bf16` remains second at `21.6%`
- `gemv_vocab_bf16` remains a distinct tail cost at `9.9%`

Correctness result:

- [ReferenceComparisonTests.swift](/Users/1amageek/Desktop/swift-lm/Tests/MetalCompilerTests/ReferenceComparisonTests.swift) still passes
- standard-path decode parity signature is unchanged
  - step 1 argmax remains `Python 2944 / Metal 859`
  - step 1 logits max error is `0.5210`
  - step 1 final hidden max error is `0.5742`
- layerwise diagnostics stay in the prior range
  - drift remains concentrated in late-layer MLP output and downstream residual accumulation

Interpretation:

- this is a good example of design-aligned specialization
- the compiler can share a precise fragment rule across optimizer tiers without collapsing them into one strategy
- exact-shape fused MLP front-half is now strong enough that even the simpler optimizer materially benefits from it
- the remaining gap from `standard` to `aggressive` is now mostly in batching strategy, not in fused MLP kernel quality

### Float-staged exact-shape fused SwiGLU

The next accepted change stayed within the same exact-shape fused family and only changed the kernel body:

- `fused_swiglu_projection_2048(_bf16)` now stages the `2048`-element hidden vector as `float`
- the gate and up accumulators both read from that `float` threadgroup tile
- this removes the repeated activation conversion inside the inner accumulation loop

Structural change:

- no new planner rule
- no new dispatch family
- no launch-policy change
- only the kernel body for the already-specialized fused family changed

Observed result:

- focused decode benchmark:
  - `standard 129.6 tok/s`
  - `aggressive 133.5 tok/s`
- end-to-end decode benchmark:
  - `127.6 tok/s`
- optimizer comparison:
  - `none 126.4`
  - `standard 130.0`
  - `aggressive 133.3`

Per-step result:

- total decode time:
  - `7554 us -> 7226 us`
- hot kernels:
  - `fused_swiglu_projection_2048_bf16: 3047 us -> 2930 us`
  - `gemv_8192_tiled_bf16: 1631 us -> 1541 us`
  - `gemv_vocab_bf16: 745 us -> 703 us`

Correctness result:

- [ReferenceComparisonTests.swift](/Users/1amageek/Desktop/swift-lm/Tests/MetalCompilerTests/ReferenceComparisonTests.swift) still passes
- standard-path decode parity signature is unchanged
  - step 1 argmax remains `Python 2944 / Metal 859`
  - step 1 logits max error remains `0.5210`
  - step 1 final hidden max error remains `0.5742`

Interpretation:

- for this fused family, the extra threadgroup footprint is worth paying because the same staged hidden tile is consumed twice
- this is different from the rejected “stage more of everything” experiments on dense GEMV
- the profitable rule is narrower:
  - stage more when reuse is real and local
  - do not widen threadgroup working sets when the kernel still streams each input only once

### Float-staged `gemv_8192_tiled` and zero dynamic threadgroup memory

The next accepted compiler changes targeted the remaining tiled dense projection family and the Metal launch API itself:

- `gemv_8192_tiled(_bf16)` now stages each `1024`-element activation tile as `float`
- the inner loop reads directly from that `float` tile instead of converting each activation on use
- dispatch planning now reports `threadgroupMemoryLength = 0` for GEMV and reduction paths
  - current kernels use statically declared threadgroup storage
  - they do not declare dynamic `[[threadgroup(n)]]` arguments

Structural change:

- no new kernel family
- no wider launch policy
- no new optimizer rule
- one kernel-body cleanup plus one Metal API correction

Observed result:

- focused decode benchmark:
  - `standard 132.5 tok/s`
  - `aggressive 135.8 tok/s`
- end-to-end decode benchmark:
  - `129.5 tok/s`
- optimizer comparison:
  - `none 128.4`
  - `standard 132.1`
  - `aggressive 135.7`

Per-step result:

- total decode time:
  - `7214 us -> 7091 us`
- hot kernels:
  - `fused_swiglu_projection_2048_bf16: 2915 us -> 2877 us`
  - `gemv_8192_tiled_bf16: 1547 us -> 1516 us`
  - `gemv_2048_6144_bf16: 727 us -> 710 us`

Correctness result:

- [ReferenceComparisonTests.swift](/Users/1amageek/Desktop/swift-lm/Tests/MetalCompilerTests/ReferenceComparisonTests.swift) still passes
- standard-path decode parity signature is unchanged
  - step 1 argmax remains `Python 2944 / Metal 859`
  - step 1 logits max error remains `0.5210`
  - step 1 final hidden max error remains `0.5742`
- compiled decode-plan dumps now show `tgmem=0`, which matches the current generated kernels more accurately

Interpretation:

- the `8192` tiled family still benefits from float-staged activations, but less dramatically than the exact-shape fused MLP front-half
- the larger surprise was the launch-side correction:
  - reserving dynamic threadgroup memory for kernels that only use static threadgroup arrays was unnecessary
  - removing that reservation improved throughput and made diagnostics match the real Metal API contract
- this is still backend-local optimization and stays within the repository design:
- no IR change
- no declaration-layer change
- no product-specific shortcut

### Pointer-increment inner loops for `gemv_8192_tiled` and `fused_swiglu_projection_2048`

Hypothesis:

- the current fixed-shape decode families already have the right launch shape
- the remaining waste is in repeated address generation inside the unrolled inner loops
- converting the unrolled loops from `base + j + laneOffset` indexing to pointer-increment style should reduce integer address work without changing memory traffic or dispatch shape

Changes:

- [MetalSourceGenerator.swift](/Users/1amageek/Desktop/swift-lm/Sources/MetalCompiler/Fragments/MetalSourceGenerator.swift)
  - `gemv_8192_tiled(_bf16)` now advances `tileWeight` and `tileInput` pointers per unrolled stride instead of recomputing indexed addresses inside the loop body
  - `fused_swiglu_projection_2048(_bf16)` now advances `gateRow`, `upRow`, and `inputLane` pointers per unrolled stride in the same style
- no changes to:
  - IR
  - optimizer rules
  - dispatch family selection
  - launch sizes

Result:

- benchmark:
  - standard decode: `121.1 tok/s`
  - aggressive decode: `124.1 tok/s`
  - end-to-end decode: `119.1 tok/s`
- per-step profile:
  - total decode time: `8165 us -> 7992 us`
  - `gemv_8192_tiled_bf16: 1760 us -> 1610 us`
  - `gemv_vocab_bf16: 806 us -> 748 us`
  - `fused_swiglu_projection_2048_bf16: 3112 us -> 3178 us`

Correctness result:

- [ReferenceComparisonTests.swift](/Users/1amageek/Desktop/swift-lm/Tests/MetalCompilerTests/ReferenceComparisonTests.swift) passes
- parity signature is unchanged:
  - decode step 1 argmax remains `Python 2944 / Metal 859`
  - step 1 logits max error remains `0.5210`
  - step 1 final hidden max error remains `0.5742`

Interpretation:

- this was a net win because the `8192` tiled family improved more than the fused MLP front-half regressed
- the result supports a narrower next step:
  - keep specializing by backend family
  - but focus on kernel-body address-generation and loop structure, not more shape splitting by itself
- this also explains why several recent exact-shape experiments lost:
  - the launch shape was already good enough
  - the remaining headroom is mostly in the inner loop

### Float-staged `input=2048` specialized GEMV family

Hypothesis:

- the `input=2048` decode projection family is still spending non-trivial work on repeated `half -> float` activation conversion inside the inner loop
- unlike `vocabDense`, these exact-shape projection kernels are smaller and more latency-sensitive
- staging only the `input2048` family as `threadgroup float` may reduce inner-loop conversion cost without paying the larger occupancy penalty seen in `vocabDense`

Changes:

- [MetalSourceGenerator.swift](/Users/1amageek/Desktop/swift-lm/Sources/MetalCompiler/Fragments/MetalSourceGenerator.swift)
  - `generateSpecializedDenseGEMV(...)` now supports `stagesInputAsFloat`
  - `generateInput2048GEMV(...)` enables float staging
  - `generateVocabGEMV(...)` stays on buffer-precision staging
- no changes to:
  - launch policy
  - decode family selection
  - optimizer rules

Result:

- benchmark:
  - standard decode: `120.7 tok/s`
  - aggressive decode: `124.9 tok/s`
  - end-to-end decode: `119.6 tok/s`
- optimizer comparison:
  - `none 118.8 tok/s`
  - `standard 121.1 tok/s`
  - `aggressive 124.3 tok/s`
- per-step profile:
  - total decode time: `7992 us -> 7936 us`
  - `gemv_2048_6144_bf16: 789 us -> 785 us`
  - `gemv_2048_sq_bf16: 758 us -> 679 us`
  - `gemv_vocab_bf16: 748 us -> 731 us`
  - `fused_swiglu_projection_2048_bf16: 3178 us -> 3197 us`
  - `gemv_8192_tiled_bf16: 1610 us -> 1664 us`

Correctness result:

- [ReferenceComparisonTests.swift](/Users/1amageek/Desktop/swift-lm/Tests/MetalCompilerTests/ReferenceComparisonTests.swift) passes
- parity signature is unchanged:
  - decode step 1 argmax remains `Python 2944 / Metal 859`
  - step 1 logits max error remains `0.5210`
  - step 1 final hidden max error remains `0.5742`

Interpretation:

- this is a small but valid family-local win
- unlike `vocabDense`, the `input2048` family appears to benefit from moving activation conversion out of the hot inner loop
- the tradeoff is visible:
  - `gemv_2048_*` improved
  - `fused_swiglu_projection_2048_bf16` and `gemv_8192_tiled_bf16` moved slightly the wrong way
  - overall decode still improved, so the change is worth keeping

### Higher unroll for `fused_swiglu_projection_2048`

Hypothesis:

- `fused_swiglu_projection_2048_bf16` is now the single largest decode family
- after the earlier pointer-increment cleanup and float-staged input reuse, the remaining overhead may be loop-control cost rather than dispatch shape
- because the kernel reads two weight streams from the same fixed `2048` input, a larger unroll factor may increase useful work per iteration without changing the backend family model

Changes:

- [MetalSourceGenerator.swift](/Users/1amageek/Desktop/swift-lm/Sources/MetalCompiler/Fragments/MetalSourceGenerator.swift)
  - `fused_swiglu_projection_2048`
  - `fused_swiglu_projection_2048_bf16`
  - unroll factor increased from `4` to `8`
- [MetalInferenceCompiler.swift](/Users/1amageek/Desktop/swift-lm/Sources/MetalCompiler/MetalInferenceCompiler.swift)
  - dynamic generated-library path uses the same `unrollFactor: 8`
- no changes to:
  - launch policy
  - dispatch count
  - optimizer rules

Result:

- benchmark:
  - standard decode: `126.1 tok/s`
  - aggressive decode: `129.5 tok/s`
  - end-to-end decode: `122.3 tok/s`
- optimizer comparison:
  - `none 121.0 tok/s`
  - `standard 125.8 tok/s`
  - `aggressive 128.7 tok/s`
- per-step profile:
  - total decode time: `7936 us -> 7455 us`
  - `fused_swiglu_projection_2048_bf16: 3197 us -> 2975 us`
  - `gemv_8192_tiled_bf16: 1664 us -> 1590 us`
  - `gemv_2048_6144_bf16: 785 us -> 743 us`
  - `gemv_2048_sq_bf16: 679 us -> 644 us`

Correctness result:

- [ReferenceComparisonTests.swift](/Users/1amageek/Desktop/swift-lm/Tests/MetalCompilerTests/ReferenceComparisonTests.swift) passes
- decode parity remains in the same regime:
  - step 1 argmax remains `Python 2944 / Metal 859`
  - step 1 logits max error moved from `0.5210` to `0.5322`
  - step 1 final hidden max error moved from `0.5742` to `0.5664`

Interpretation:

- this is the clearest compiler-side win in the current round
- unlike recent launch-shape experiments, this change improved the actual dominant hot family directly
- the result supports the current direction:
  - keep family-level specialization
  - prefer kernel-body improvements inside those families over adding more planner heuristics

### Removing the final barrier in `gemv_8192_tiled`

Hypothesis:

- `gemv_8192_tiled_bf16` still pays two threadgroup barriers per tile
- the second barrier is necessary between tiles to prevent early overwrite of `inputTile`
- but after the final tile there is no subsequent shared-memory write, so the last barrier is redundant

Changes:

- [MetalSourceGenerator.swift](/Users/1amageek/Desktop/swift-lm/Sources/MetalCompiler/Fragments/MetalSourceGenerator.swift)
  - `generateInput8192TiledGEMV(...)` now skips the trailing `threadgroup_barrier` on the final tile
- no changes to:
  - tile width
  - unroll factor
  - launch shape
  - family selection

Result:

- benchmark:
  - standard decode: `124.3 tok/s`
  - aggressive decode: `126.5 tok/s`
  - end-to-end decode: `123.4 tok/s`
- per-step profile:
  - total decode time: `7852 us -> 7546 us`
  - `gemv_8192_tiled_bf16: 1660 us -> 1579 us`
  - `fused_swiglu_projection_2048_bf16: 3009 us -> 2994 us`
  - `gemv_2048_6144_bf16: 812 us -> 759 us`
  - `gemv_2048_sq_bf16: 725 us -> 644 us`

Correctness result:

- the early reference diagnostics completed successfully before the known allocator-side bus error in later tests
- completed diagnostics remained in the same parity regime:
  - decode step 0 argmax: `521`
  - decode step 1 argmax: `Python 2944 / Metal 859`
  - decode step 1 logits max error: `0.5322`
  - decode step 1 final hidden max error: `0.5664`

Interpretation:

- this is a clean synchronization-side win
- the improvement comes from reducing unnecessary threadgroup synchronization, not from changing launch geometry
- it also reinforces the current pattern:
  - the best remaining wins are inside the hot kernel bodies
  - especially memory reuse and synchronization trimming

### SIMD reduction for fused residual-add RMSNorm

Hypothesis:

- the fused decode RMSNorm family still paid a repeated scalar reduction cost after each simdgroup wrote its partial sum
- replacing the single-thread shared-memory fold with a first-simdgroup SIMD reduction should trim overhead without changing launch shape or routing

Changes:

- [MetalSourceGenerator.swift](/Users/1amageek/Desktop/swift-lm/Sources/MetalCompiler/Fragments/MetalSourceGenerator.swift)
  - updated:
    - `generateFusedCopyRMSNorm(...)`
    - `generateFusedResidualAddCopyRMSNorm(...)`
    - `generateFusedResidualAddRMSNorm(...)`
  - each kernel now receives:
    - `thread_index_in_simdgroup`
    - `simdgroup_index_in_threadgroup`
  - the second-stage reduction now:
    - loads the shared partial sums only in simdgroup `0`
    - reduces them with `simd_sum(...)`
    - writes the final inverse RMS scale from lane `0`
- no changes to:
  - launch geometry
  - buffer bindings
  - family selection

Result:

- benchmark:
  - standard decode: `122.8 tok/s -> 124.6 tok/s`
  - aggressive decode: `124.9 tok/s -> 127.7 tok/s`
  - end-to-end decode: `118.2 tok/s -> 122.2 tok/s`
- per-step profile:
  - total decode time: `8079 us -> 7685 us`
  - `fused_residual_add_copy_rms_norm_bf16: 267 us -> 160 us`
  - `fused_swiglu_projection_2048_bf16: 3187 us -> 3029 us`

Correctness result:

- [ReferenceComparisonTests.swift](/Users/1amageek/Desktop/swift-lm/Tests/MetalCompilerTests/ReferenceComparisonTests.swift) passes
- parity stayed in the same regime:
  - decode step 0 argmax still matches Python
  - decode step 1 argmax remains `Python 2944 / Metal 859`
  - decode step 1 logits max error is about `0.5273`
  - decode step 1 final hidden max error is about `0.5703`
  - decode step 1 final norm kernel max error remains `0.0781`

Interpretation:

- this family was not launch-limited; it was paying a small scalar reduction tax on every decode block
- replacing that scalar fold with a SIMD fold is consistent with the existing family-based compiler design
- this is another confirmation that the remaining headroom is in kernel-body cleanup, not broader planner heuristics

## Technical Conclusions

### 1. Current compiler wins are real, and the center of gravity has shifted

The current optimization stack is working in the intended direction:

- fewer decode dispatches
- more fused decode execution
- lower steady-state decode overhead

But the latest results show that dispatch reduction is no longer the main driver by itself. The larger win came from improving the dominant fixed-shape GEMV families. The prefill path is still not seeing analogous optimizer gains, which is visible in the unchanged prefill step count.

### 2. The backend is memory- and projection-dominated

The emitted kernel profile still shows that almost all decode time is spent in `gemv_2048_sq_bf16`, `gemv_2048_6144_bf16`, `gemv_2048_8192_bf16`, `gemv_8192_tiled_bf16`, and `gemv_vocab_bf16`.

This means:

- the compiler is no longer bottlenecked by scheduler overhead alone
- the next ceiling is kernel quality and data movement in projection-heavy layers

In practical terms, the compiler has reached the point where “better dispatch planning” is less important than “better projection kernels.”

### 3. Shape-family specialization is the right design, and it should continue

The profitable compiler changes now share the same pattern:

- identify a real hot tensor shape
- give it an explicit family in the compiler
- emit a kernel body that assumes those shape facts
- keep launch policy attached to the same family

This is aligned with the repository design:

- specialization lives in the backend compiler
- IR stays semantic
- model declarations stay architecture-level
- backend tuning is driven by concrete workload shape, not product-specific shortcuts

The latest shape-specialized GEMV experiment supports this directly: a kernel-family change moved throughput without any change to graph structure or dispatch count.

### 3. Prefill remains under-optimized from the compiler’s point of view

The optimizer comparison shows:

- decode dispatches shrink materially under stronger optimization
- prefill step count does not

That implies the compiler currently has a richer decode optimization story than prefill optimization story. If prefill latency matters, compiler work must expand beyond decode fusion and into sequence-oriented lowering quality.

### 4. `vocabDense` still benefits from kernel-body tuning, not new family splits

The latest accepted change stayed within the existing `vocabDense` family and only changed the inner loop shape:

- for non-float-staged specialized dense GEMV, the generated kernel now uses pointer-increment addressing inside the unrolled accumulation loop
- this removes repeated `j + lane` address formation on the hot weight/input reads
- no new dispatch family was introduced
- no launch-width change was introduced
- no optimizer rule changed

Observed result:

- focused decode benchmark:
  - `standard 122.8 tok/s`
  - `aggressive 124.4 tok/s`
- end-to-end decode benchmark:
  - `119.6 tok/s`
- optimizer comparison:
  - `none 118.3`
  - `standard 121.8`
  - `aggressive 124.8`

Per-step result:

- total decode time:
  - `8124 us -> 7989 us`
- hot kernels:
  - `gemv_vocab_bf16: 815 us -> 766 us`
  - `fused_swiglu_projection_2048_bf16: 3167 us -> 3116 us`
  - `gemv_8192_tiled_bf16: 1705 us -> 1663 us`

Correctness result:

- [ReferenceComparisonTests.swift](/Users/1amageek/Desktop/swift-lm/Tests/MetalCompilerTests/ReferenceComparisonTests.swift) still passes
- standard-path decode parity signature is unchanged
  - step 1 argmax remains `Python 2944 / Metal 859`
  - step 1 logits max error remains `0.5273`
  - step 1 final hidden max error remains `0.5703`

Interpretation:

- `vocabDense` has resisted family-splitting and launch-width experiments, but it still responds to simpler address-generation cleanup inside the existing family
- this is the same pattern seen elsewhere in the current compiler state:
  - wins are now coming from kernel-body cleanup inside stable families
  - not from adding broader dispatch heuristics

### 5. `input8192Tiled` also responds to simpler tile-load addressing

The next accepted change stayed inside the existing `input8192Tiled` family:

- the activation tile load now uses a source pointer that advances by `threadsPerThreadgroup`
- this removes repeated `base + j` address formation during the tile staging loop
- no new kernel family was introduced
- no launch-width change was introduced
- no optimizer rule changed

Observed result:

- focused decode benchmark:
  - `standard 124.2 tok/s`
  - `aggressive 125.9 tok/s`
- end-to-end decode benchmark:
  - `121.5 tok/s`
- optimizer comparison:
  - `none 118.7`
  - `standard 123.3`
  - `aggressive 126.2`

Per-step result:

- total decode time:
  - `7989 us -> 7805 us`
- hot kernels:
  - `gemv_8192_tiled_bf16: 1663 us -> 1606 us`
  - `gemv_vocab_bf16: 766 us -> 745 us`
  - `fused_swiglu_projection_2048_bf16: 3116 us -> 3035 us`

Correctness result:

- [ReferenceComparisonTests.swift](/Users/1amageek/Desktop/swift-lm/Tests/MetalCompilerTests/ReferenceComparisonTests.swift) still passes
- standard-path decode parity signature is unchanged
  - step 1 argmax remains `Python 2944 / Metal 859`
  - step 1 logits max error remains `0.5273`
  - step 1 final hidden max error remains `0.5703`

Interpretation:

- `input8192Tiled` now shows the same pattern as `vocabDense`
- the remaining wins are in address-generation and read-path cleanup inside a stable family
- broader launch tuning is no longer the easiest source of improvement

## Accepted Tuning: `fused_swiglu_projection_2048` Buffer-Precision Staging

Hypothesis:

- `fused_swiglu_projection_2048_bf16` is now the largest decode kernel.
- Its current `threadgroup float inputTile[2048]` may be over-paying for shared-memory bandwidth and occupancy.
- Because decode activations are already stored as F16 buffers, staging them in buffer precision and converting at accumulation time may reduce pressure enough to win overall.

Compiler change:

- keep the exact-shape `input2048ExpandedDense` family design
- change only `generateInput2048FusedSwiGLUProjection(...)`
- stage the 2048-wide hidden vector in buffer precision instead of `float`
- keep unroll factor, launch shape, optimizer rules, and weight path unchanged

Observed result:

- focused decode benchmark:
  - `standard 123.3 tok/s -> 123.1 tok/s`
  - `aggressive 126.8 tok/s -> 126.4 tok/s`
- optimizer comparison:
  - `none 122.6`
  - `standard 128.0`
  - `aggressive 131.1`
- end-to-end decode benchmark:
  - `120.8 tok/s`

Per-step result:

- total decode time:
  - `7862 us -> 7802 us`
- hot kernels:
  - `fused_swiglu_projection_2048_bf16: 3016 us -> 3044 us`
  - `gemv_8192_tiled_bf16: 1657 us -> 1656 us`
  - `gemv_2048_6144_bf16: 801 us -> 793 us`
  - `gemv_vocab_bf16: 733 us -> 761 us`
  - `gemv_2048_sq_bf16: 759 us -> 704 us`

Correctness result:

- the benchmark suite passes with the new kernel family policy
- the known allocator-side crash in [ReferenceComparisonTests.swift](/Users/1amageek/Desktop/swift-lm/Tests/MetalCompilerTests/ReferenceComparisonTests.swift) still reproduces on the second test and is unrelated to this tuning
- the first reference comparison still passes
  - prefill logits argmax matches Python
  - prefill logits max error remains `0.0508`

Interpretation:

- `fused_swiglu_projection_2048_bf16` is not a pure kernel-local optimization problem anymore
- the family got slightly slower in isolation, but the overall decode path got faster
- this suggests the smaller shared-memory footprint improved scheduler behavior or occupancy enough to outweigh the extra conversion cost
- family-specific staging policy is therefore a valid design axis, and it should remain explicit instead of being folded back into a single dense-kernel policy

## Accepted Tuning: `input20486144Dense` Buffer-Precision Staging

Hypothesis:

- Apple GPU guidance suggests that local fast memory is valuable but limited, and overusing threadgroup-local staging can reduce occupancy.
- `gemv_2048_6144_bf16` still accounted for about `10%` of decode time while sharing the same float-staged policy as the other `input2048` exact-shape families.
- The `6144` output tier may benefit from lower threadgroup footprint more than from float-staged input reuse.

Compiler change:

- keep the exact-shape `input2048Dense` family split
- change only `gemv_2048_6144` / `gemv_2048_6144_bf16`
- stage the 2048-wide hidden vector in buffer precision instead of `float`
- keep launch shape, unroll factor, and optimizer rules unchanged

Observed result:

- focused decode benchmark:
  - `standard 123.1 tok/s -> 127.6 tok/s`
  - `aggressive 126.4 tok/s -> 129.3 tok/s`
- end-to-end decode benchmark:
  - `120.1 tok/s -> 126.1 tok/s`
- optimizer comparison:
  - `none 123.1`
  - `standard 127.9`
  - `aggressive 129.2`

Per-step result:

- total decode time:
  - `7802 us -> 7297 us`
- hot kernels:
  - `fused_swiglu_projection_2048_bf16: 3044 us -> 2961 us`
  - `gemv_8192_tiled_bf16: 1656 us -> 1570 us`
  - `gemv_2048_6144_bf16: 793 us -> 732 us`
  - `gemv_vocab_bf16: 761 us -> 712 us`
  - `gemv_2048_sq_bf16: 704 us -> 637 us`

Correctness result:

- [BenchmarkTests.swift](/Users/1amageek/Desktop/swift-lm/Tests/MetalCompilerTests/BenchmarkTests.swift) passes
- full [ReferenceComparisonTests.swift](/Users/1amageek/Desktop/swift-lm/Tests/MetalCompilerTests/ReferenceComparisonTests.swift) still hits the known allocator crash in its second test
- the first reference test still passes before that crash
  - prefill logits argmax matches Python
  - prefill logits max error remains `0.0508`

Interpretation:

- this is stronger evidence that `input2048Dense` should not be treated as a single staging policy
- for Apple GPUs, the occupancy win from smaller threadgroup-local state can dominate even when the kernel does more per-element conversion work
- the effect is not isolated to one kernel: reducing pressure in `gemv_2048_6144_bf16` also improved neighboring hot kernels, which points to scheduler or residency effects rather than a narrow micro-optimization
- the next design-consistent step is to keep staging policy explicit at the exact-shape family level and continue tuning `gemv_2048_sq_bf16` and `gemv_2048_8192_bf16` independently

## Rejected Tuning: `input2048SquareDense` Buffer-Precision Staging

Hypothesis:

- if `gemv_2048_6144_bf16` improved by reducing threadgroup-local footprint, the same policy might help `gemv_2048_sq_bf16`

Compiler change:

- change only `gemv_2048_sq` / `gemv_2048_sq_bf16`
- stage the 2048-wide hidden vector in buffer precision instead of `float`
- keep launch shape, unroll factor, and optimizer rules unchanged

Observed result:

- focused decode benchmark:
  - `standard 127.6 tok/s -> 123.1 tok/s`
  - `aggressive 129.3 tok/s -> 126.5 tok/s`
- end-to-end decode benchmark:
  - `126.1 tok/s -> 120.4 tok/s`
- optimizer comparison:
  - `none 116.9`
  - `standard 121.5`
  - `aggressive 126.7`

Per-step result:

- total decode time:
  - `7297 us -> 7806 us`
- hot kernels:
  - `gemv_2048_sq_bf16: 637 us -> 700 us`
  - `gemv_2048_6144_bf16: 732 us -> 823 us`
  - `gemv_vocab_bf16: 712 us -> 740 us`

Interpretation:

- `input2048SquareDense` does not share the same pressure profile as `input20486144Dense`
- for the square projection, float staging still wins
- this strengthens the case for exact-shape staging policy, not family-wide policy

## Rejected Tuning: `input20488192Dense` Buffer-Precision Staging

Hypothesis:

- `input20488192Dense` shares the same 2048-wide staged hidden vector as `input20486144Dense`
- if `6144` won by reducing threadgroup-local footprint, `8192` might also benefit

Compiler change:

- change only `gemv_2048_8192` / `gemv_2048_8192_bf16`
- stage the 2048-wide hidden vector in buffer precision instead of `float`
- keep launch shape, unroll factor, and optimizer rules unchanged

Observed result:

- focused decode benchmark:
  - `standard 127.6 tok/s -> 123.1 tok/s`
  - `aggressive 129.3 tok/s -> 125.9 tok/s`
- end-to-end decode benchmark:
  - `126.1 tok/s -> 120.8 tok/s`
- optimizer comparison:
  - `none 116.3`
  - `standard 121.1`
  - `aggressive 123.2`

Per-step result:

- total decode time:
  - `7297 us -> 7877 us`
- hot kernels:
  - `fused_swiglu_projection_2048_bf16: 2961 us -> 3089 us`
  - `gemv_2048_6144_bf16: 732 us -> 838 us`
  - `gemv_vocab_bf16: 712 us -> 751 us`

Interpretation:

- `input20488192Dense` behaves like `input2048SquareDense`, not like `input20486144Dense`
- reducing local staging pressure here hurts the rest of the decode path enough to lose overall
- the `6144` win is therefore a shape-specific effect, not a generic `input2048Dense` rule

## Rejected Tuning: `input20486144Dense` Pointer-Increment Accumulation

Hypothesis:

- after `gemv_2048_6144_bf16` benefited from buffer-precision staging, it might also benefit from pointer-increment accumulation
- with fixed input dimension and bounds fully elided, incrementing lane pointers could reduce address-generation cost

Compiler change:

- keep the accepted `input20486144Dense` buffer-precision staging
- change only `gemv_2048_6144` / `gemv_2048_6144_bf16`
- switch accumulation from indexed reads to pointer-increment reads
- keep launch shape and unroll factor unchanged

Observed result:

- focused decode benchmark:
  - `standard 127.9 tok/s -> 121.2 tok/s`
  - `aggressive 129.2 tok/s -> 124.6 tok/s`
- end-to-end decode benchmark:
  - `126.1 tok/s -> 119.7 tok/s`
- per-step result:
  - total decode time: `7297 us -> 7904 us`
  - `gemv_2048_6144_bf16: 732 us -> 843 us`

Interpretation:

- the accepted `6144` win comes from staging policy, not from a better accumulation form
- pointer-increment accumulation likely adds register pressure or reduces compiler scheduling quality on this exact shape
- keep `input20486144Dense` on indexed accumulation

## Rejected Tuning: `input8192Tiled` Buffer-Precision Staging

Hypothesis:

- Apple GPU guidance suggests overusing threadgroup-local memory can reduce occupancy
- `gemv_8192_tiled_bf16` stages 1024 activation elements per tile in `float`
- shrinking that tile to buffer precision might free enough local memory to improve residency

Compiler change:

- change only `gemv_8192_tiled` / `gemv_8192_tiled_bf16`
- stage the 1024-element tile in buffer precision instead of `float`
- keep tile width, launch shape, and unroll factor unchanged

Observed result:

- focused decode benchmark:
  - `standard 127.9 tok/s -> 125.8 tok/s`
  - `aggressive 129.2 tok/s -> 127.1 tok/s`
- end-to-end decode benchmark:
  - `126.1 tok/s -> 121.1 tok/s`
- optimizer comparison:
  - `none 114.4`
  - `standard 124.7`
  - `aggressive 128.5`

Per-step result:

- total decode time:
  - `7297 us -> 8029 us`
- hot kernels:
  - `gemv_8192_tiled_bf16: 1570 us -> 1703 us`
  - `fused_swiglu_projection_2048_bf16: 2961 us -> 3199 us`
  - `gemv_2048_6144_bf16: 732 us -> 853 us`

Interpretation:

- unlike `input20486144Dense`, this tiled family benefits more from avoiding per-element conversion inside the inner loop than from reducing tile storage
- `input8192Tiled` should stay float-staged

## Rejected Tuning: `vocabDense` Unroll 2

Hypothesis:

- `gemv_vocab_bf16` is still one of the largest single decode steps
- reducing unroll from 4 to 2 might lower register pressure enough to improve the output-head path

Compiler change:

- change only `gemv_vocab` / `gemv_vocab_bf16`
- keep buffer-precision staging and pointer-increment accumulation
- reduce unroll factor from `4` to `2`

Observed result:

- focused decode benchmark:
  - `standard 127.9 tok/s -> 126.9 tok/s`
  - `aggressive 129.2 tok/s -> 130.1 tok/s`
- end-to-end decode benchmark:
  - `126.1 tok/s -> 122.5 tok/s`
- optimizer comparison:
  - `none 119.8`
  - `standard 124.4`
  - `aggressive 126.0`

Per-step result:

- total decode time:
  - `7297 us -> 7765 us`
- hot kernels:
  - `gemv_vocab_bf16: 712 us -> 765 us`
  - `gemv_2048_6144_bf16: 732 us -> 825 us`

Interpretation:

- even if a single short run can look competitive, aggregate throughput is worse
- `vocabDense` should stay at unroll 4

## Rejected Tuning: `gemv_vocab_bf16` Native `bfloat` Weight Read

Hypothesis:

- `gemv_vocab_bf16` is still a major single-step decode cost
- replacing the `uint16_t + bf16_to_float(...)` weight read path with native `bfloat` loads might reduce read-path overhead in the output head

Compiler change:

- change only `gemv_vocab_bf16`
- keep buffer-precision staging and pointer-increment accumulation
- emit the kernel with `device const bfloat* weight`
- convert weights with `float(weight[i])` instead of the generic BF16 read helper

Observed result:

- focused decode benchmark:
  - `standard 127.9 tok/s -> 127.8 tok/s`
  - `aggressive 129.2 tok/s -> 130.2 tok/s`
- end-to-end decode benchmark:
  - `126.1 tok/s -> 115.4 tok/s`
- optimizer comparison:
  - `none 118.5`
  - `standard 123.6`
  - `aggressive 127.0`

Per-step result:

- total decode time:
  - `7297 us -> 7940 us`
- hot kernels:
  - `gemv_vocab_bf16: 712 us -> 741 us`
  - `fused_swiglu_projection_2048_bf16: 2961 us -> 3154 us`
  - `gemv_8192_tiled_bf16: 1570 us -> 1645 us`

Interpretation:

- native `bfloat` loads compile cleanly, but this output-head path does not win in aggregate
- the small single-run gain on aggressive decode is noise; the 5-iteration comparison and end-to-end result both regress
- `gemv_vocab_bf16` should keep the existing BF16 read path for now

## Rejected Tuning: `input20486144Dense` Preferred Simdgroups 6

Hypothesis:

- `gemv_2048_6144_bf16` is now an exact-shape family with its own staging policy
- reducing only this family from `8` to `6` simdgroups might improve occupancy without disturbing the other decode GEMV families

Compiler change:

- change only `DecodeProjectionShapeFamily.input20486144Dense`
- set `preferredSimdgroups` from `8` to `6`
- keep the accepted buffer-precision staging for `gemv_2048_6144(_bf16)`

Observed result:

- focused decode benchmark:
  - `standard 127.9 tok/s -> 117.4 tok/s`
  - `aggressive 129.2 tok/s -> 124.7 tok/s`
- prefill throughput also dropped:
  - `16 tok: 128.7 tok/s -> 100.0 tok/s`
  - `32 tok: 129.1 tok/s -> 108.5 tok/s`
  - `64 tok: 127.5 tok/s -> 111.3 tok/s`

Interpretation:

- this exact family does not want a smaller threadgroup width
- the accepted `input20486144Dense` staging win is not helped by reducing simdgroup count
- `input20486144Dense` should stay at `8` preferred simdgroups

## Rejected Tuning: `gemv_2048_6144_bf16` Native `bfloat` Weight Read

Hypothesis:

- `gemv_2048_6144_bf16` runs ten times per decode step and is one of the largest exact-shape decode families
- replacing its BF16 helper path with native `bfloat` loads might reduce weight-read overhead without changing launch shape or staging policy

Compiler change:

- change only `gemv_2048_6144_bf16`
- keep the accepted buffer-precision staging for `input20486144Dense`
- emit the kernel with `device const bfloat* weight`
- convert weights with `float(weight[i])` instead of `bf16_to_float(...)`

Observed result:

- focused decode benchmark:
  - `standard 127.9 tok/s -> 123.1 tok/s`
  - `aggressive 129.2 tok/s -> 125.9 tok/s`
- end-to-end decode benchmark:
  - `126.1 tok/s -> 120.9 tok/s`
- optimizer comparison:
  - `none 113.7`
  - `standard 122.9`
  - `aggressive 126.0`

Per-step result:

- total decode time:
  - `7297 us -> 7864 us`
- hot kernels:
  - `gemv_2048_6144_bf16: 732 us -> 805 us`
  - `gemv_vocab_bf16: 712 us -> 758 us`
  - `gemv_8192_tiled_bf16: 1570 us -> 1664 us`

Interpretation:

- native `bfloat` loads compile and run, but this exact-shape family also loses in aggregate
- the weight-read helper is not the dominant limiter here
- `gemv_2048_6144_bf16` should keep the existing BF16 read path

## Resident Constant Arena

Hypothesis:

- compiled plans still pay avoidable CPU-side binding overhead when every step carries inline `setBytes` constants or its own tiny resident buffer
- the next low-risk improvement is to keep the plan representation but pack static step constants into a single resident arena per compiled plan

Compiler/runtime change:

- add `MetalConstantBindingAllocator`
- extend `MetalBindingTable` to represent either inline byte constants or resident constant-buffer slices
- convert decode and prefill plans from per-step inline constants to resident constant slices backed by one arena buffer per plan
- keep runtime sequence-length overrides as inline bytes, because they are truly dynamic

Observed result:

- build: success
- benchmark suite: success
- focused decode benchmark:
  - `standard 123.1 tok/s -> 118.6 tok/s` on the one-shot test
  - `aggressive 125.5 tok/s -> 126.9 tok/s` on the one-shot test
- optimizer comparison:
  - `none 119.3`
  - `standard 123.8`
  - `aggressive 135.2`

Per-step result:

- total decode time:
  - `2443 us -> 2282 us`
- hot kernels:
  - `gemv_2048_6144_bf16: 781 us -> 787 us`
  - `gemv_2048_sq_bf16: 697 us -> 693 us`
  - `fused_swiglu_projection_2048_bf16: 369 us -> 256 us`
  - `gemv_vocab_bf16: 60 us -> 53 us`

Interpretation:

- the arena conversion does not regress the compiled path and improves aggregate throughput
- one-shot benchmark noise is still present, but the 5-iteration optimizer comparison and per-step profile both favor the arena form
- the main value is architectural: the dispatch plan can now evolve from inline constant binding toward resident descriptor-style binding without changing kernel families
- this is a good foundation for future argument-table work

## Decode Argument-Buffer Backend: Layout `#0`

Hypothesis:

- after resident constants, the next CPU-side dispatch cost is repeated per-step buffer binding for the dominant decode projection families
- the lowest-risk path is to keep the existing kernel families, but add argument-buffer-capable shader variants for the dominant decode layout `indices=[0,1,2]`

Compiler change:

- add `MetalArgumentBindingAllocator`, `MetalPreparedArgumentBufferAllocator`, and `MetalArgumentTableLayout`
- plan decode steps as argument-table candidates
- materialize prepared argument buffers for candidate steps
- add real `_argbuf` shader variants and encoded promotion for layout `#0`
- first cover:
  - `gemv_2048_sq_bf16`
  - `gemv_2048_6144_bf16`
  - `gemv_8192_tiled_bf16`
  - `gemv_bf16`

Observed result:

- generated-library diagnostics:
  - decode steps: `147`
  - argument-table candidates: `133`
  - encoded steps after layout `#0` promotion: `60`
  - dominant encoded families:
    - `gemv_2048_sq_bf16_argbuf x22`
    - `gemv_8192_tiled_bf16_argbuf x16`
    - `gemv_bf16_argbuf x12`
    - `gemv_2048_6144_bf16_argbuf x10`
- focused decode benchmark:
  - `standard 135.8 tok/s`
  - `aggressive 139.1 tok/s`
- optimizer comparison:
  - `none 130.6`
  - `standard 135.4`
  - `aggressive 138.1`

Per-step result:

- total decode time:
  - `3650 us`
- hot kernels:
  - `gemv_2048_6144_bf16_argbuf: 42.4%`
  - `gemv_2048_sq_bf16_argbuf: 41.5%`
  - `gemv_8192_tiled_bf16_argbuf: 2.0%`
  - `gemv_bf16_argbuf: 0.8%`

Interpretation:

- the first real encoded argument-buffer backend works and preserves parity
- the dominant layout was correctly chosen: most encoded steps were projection kernels
- command-side binding work was reduced without changing the projection lowering model

## Decode Argument-Buffer Backend: Layout `#1`

Hypothesis:

- once layout `#0` is encoded, the next best target is decode layout `indices=[0,1,2,3]`
- this layout covers the repeated pre-norm and MLP-front-half kernels, so encoded promotion should further reduce buffer-binding overhead without touching the math path

Compiler change:

- add `_argbuf` shader variants for layout `#1` families:
  - `fused_residual_add_copy_rms_norm_bf16`
  - `fused_swiglu_projection_2048_bf16`
  - `conv_state_update_bf16`
- switch encoded promotion logic from fragile `layout.id` checks to explicit `layout.indices` matching
- keep constants resident and only move the first four resource buffers into the argument buffer

Observed result:

- generated-library diagnostics:
  - decode steps: `147`
  - argument-table candidates: `133`
  - encoded steps after layout `#1` promotion: `117`
  - remaining prepared steps: `16`
- focused decode benchmark:
  - `standard 123.8 tok/s`
  - `aggressive 124.8 tok/s`
- optimizer comparison:
  - `none 120.3`
  - `standard 126.4`
  - `aggressive 127.6`

Per-step result:

- total decode time:
  - `3850 us`
- hot kernels:
  - `gemv_2048_6144_bf16_argbuf: 41.1%`
  - `gemv_2048_sq_bf16_argbuf: 39.7%`
  - `fused_swiglu_projection_2048_bf16_argbuf: 6.2%`
  - `conv_state_update_bf16_argbuf: 0.7%`

Reference result:

- full `ReferenceComparisonTests` passed
- prefill parity stayed unchanged
- decode drift signature stayed unchanged:
  - step 1 argmax still `Python 2944 / Metal 859`
  - this confirms the argument-buffer backend did not introduce a new correctness regression

Interpretation:

- encoded promotion now covers almost the entire decode mainline
- layout `#1` is worth encoding, but the performance gain is smaller than layout `#0`
- after this point, the remaining wins are more likely to come from:
  - the few still-prepared layouts
  - or kernel body quality, not binding backend alone

## Decode Argument-Buffer Backend: Full Decode Coverage

Hypothesis:

- after layouts `#0` and `#1`, the remaining decode tail should be the small `[0,1]` family
- the most likely members are `qk_rms_norm_bf16`, final `rms_norm_bf16`, and `argmax`
- if decode-only 2-buffer kernels with resident constants move to encoded argument-buffer variants without changing math, decode parity should remain unchanged and command-side binding work should drop to zero inline buffer binds
- if the hottest exact-shape `layout #0` kernels are already fixed-shape, their encoded variants should not need resident constants for `inputDimension` / `outputDimension`
- dropping those unused constant bindings should reduce command-side binding work again without changing math or launch shape

Compiler change:

- add decode `_argbuf` shader variants for the remaining `[0,1]` families:
  - `qk_rms_norm_bf16`
  - `rms_norm_bf16`
  - `argmax`
- allow decode-only `2`-buffer binding tables with resident constants to become argument-table candidates
- keep encoded promotion keyed by `layout.indices`, so the new path is tied to layout `[0,1]` instead of any incidental layout id
- remove unused `inputDimension` / `outputDimension` bindings from the encoded exact-shape variants:
  - `gemv_2048_sq_bf16_argbuf`
  - `gemv_2048_6144_bf16_argbuf`
- strip resident constant bindings from those encoded steps during plan materialization

Observed result:

- generated-library diagnostics:
  - decode steps: `147`
  - argument-table candidates: `147`
  - encoded steps after remaining layout promotion: `147`
  - remaining prepared steps: `0`
  - resident-constant steps after exact-shape pruning: `115`
  - layout reuse:
    - `#0 x69 indices=[0,1,2]`
    - `#1 x58 indices=[0,1,2,3]`
    - `#2 x14 indices=[0,1]`
    - `#3 x6 indices=[0,1,2,3,4,5,6]`
- focused decode benchmark:
  - `standard 137.8 tok/s`
  - `aggressive 136.2 tok/s`
- optimizer comparison:
  - `none 124.9`
  - `standard 137.8`
  - `aggressive 136.2`

Per-step result:

- total decode time:
  - `3447 us`
- hot kernels:
  - `gemv_2048_6144_bf16_argbuf: 42.7%`
  - `gemv_2048_sq_bf16_argbuf: 41.4%`
  - `fused_swiglu_projection_2048_bf16_argbuf: 6.3%`
  - `fused_residual_add_copy_rms_norm_bf16_argbuf: 2.2%`
  - `gemv_8192_tiled_bf16_argbuf: 2.1%`
  - `gemv_vocab_bf16_argbuf: 1.4%`
  - `flash_attn_decode_argbuf: 0.8%`
  - `qk_rms_norm_bf16_argbuf: 0.7%`
  - `conv_state_update_bf16_argbuf: 0.6%`
  - `argmax_argbuf: 0.1%`
  - `rms_norm_bf16_argbuf: 0.1%`

Reference result:

- full `ReferenceComparisonTests` passed
- prefill parity stayed unchanged
- decode drift signature stayed unchanged:
  - step 0 argmax still matches Python
  - step 1 argmax is still `Python 2944 / Metal 859`
  - step 2 divergence pattern is unchanged

Interpretation:

- encoded promotion now covers the entire decode argument-table candidate set
- the prepared tail is fully closed at `0` steps, including the final `[0,1]` norm/argmax family
- decode now uses encoded argument buffers for every compiled decode step in this model
- the hottest exact-shape families no longer carry redundant resident constants
- after this point, further argument-buffer work has sharply diminishing returns
- the main compiler frontier is again kernel body quality, especially `gemv_2048_6144_bf16_argbuf` and `gemv_2048_sq_bf16_argbuf`

## Accepted Tuning: Exact-Shape `input2048` Fixed Row Scheduling

Hypothesis:

- the hottest remaining decode kernels are already exact-shape and already launched with a fixed `256`-threadgroup width
- for `input2048SquareDense` and `input20486144Dense`, `rowsPerThreadgroup` is therefore always `8`
- replacing the runtime expression `max(1u, threadsPerThreadgroup / SIMD_WIDTH)` with a compile-time constant in these exact-shape kernels should reduce address/control overhead without changing math

Compiler change:

- keep the existing exact-shape family split
- in `generateInput2048GEMV(...)` and `generateInput2048GEMVArgumentTableVariant(...)`, add optional fixed row scheduling for exact-shape kernels
- enable `fixedRowsPerThreadgroup = 8` for:
  - `gemv_2048_sq(_bf16)`
  - `gemv_2048_sq(_bf16)_argbuf`
  - `gemv_2048_6144(_bf16)`
  - `gemv_2048_6144(_bf16)_argbuf`
- leave the generic and tiled families unchanged

Observed result:

- reference comparison still passed unchanged:
  - prefill parity unchanged
  - decode drift signature unchanged
  - step 1 remained `Python 2944 / Metal 859`
- focused decode benchmark:
  - `standard 133.7 tok/s`
  - `aggressive 136.5 tok/s`
- optimizer comparison:
  - `none 128.1`
  - `standard 133.7`
  - `aggressive 136.5`

Per-step result:

- total decode time:
  - `3669 us -> 3624 us`
- family-level effects:
  - `gemv_2048_sq_bf16_argbuf: 1544 us -> 1446 us`
  - `gemv_2048_6144_bf16_argbuf: 1573 us -> 1626 us`
- net effect:
  - the square family improved enough to offset the slight regression in `6144`

Interpretation:

- exact-shape launch assumptions can be safely folded into the kernel body when they are already fixed by the compiler’s family policy
- this is a valid use of specialization because the launch contract is explicit at the family level, not an incidental runtime assumption
- the gain is modest, but it confirms that the remaining decode frontier is now dominated by exact-shape kernel-body details rather than binding/backend work

## Accepted Tuning: Exact-Shape BF16 Argument-Buffer Pairwise Weight Read

Hypothesis:

- after full decode argument-buffer coverage, the hottest remaining kernels are the exact-shape BF16 families:
  - `gemv_2048_sq_bf16_argbuf`
  - `gemv_2048_6144_bf16_argbuf`
- these kernels still spend most of their time decoding BF16 weight values one element at a time
- if the exact-shape BF16 argbuf path reads weights in `ushort2` pairs and converts them with a dedicated helper, it should reduce decode-side weight conversion overhead without changing layout or launch policy

Compiler change:

- add `bf16x2_to_float2(...)` to the shared Metal source header
- extend `generateInput2048GEMVArgumentTableVariant(...)` with an opt-in pairwise BF16 read path
- enable that path only for:
  - `gemv_2048_sq_bf16_argbuf`
  - `gemv_2048_6144_bf16_argbuf`
- keep:
  - indexed activation reads
  - exact-shape fixed row scheduling
  - existing staging policies
- do not widen the change to other GEMV families

Observed result:

- reference comparison still passed unchanged:
  - prefill parity unchanged
  - decode drift signature unchanged
  - step 1 remained `Python 2944 / Metal 859`
- focused decode benchmark:
  - `standard 134.5 tok/s`
  - `aggressive 137.3 tok/s`
- optimizer comparison:
  - `none 128.5`
  - `standard 134.1`
  - `aggressive 137.2`
- final shader compile no longer emitted the temporary unused-variable warning from the pairwise path

Per-step result:

- total decode time:
  - `3624 us -> 3467 us`
- family-level effects:
  - `gemv_2048_6144_bf16_argbuf: 1603 us`
  - `gemv_2048_sq_bf16_argbuf: 1306 us`
- the exact-shape BF16 argbuf families remained the dominant decode cost centers, but both moved down enough to establish a new best baseline

Interpretation:

- on Apple GPU, this exact-shape BF16 path benefits from reducing weight conversion overhead more than it benefits from further control-flow simplification
- the win is specific to the argbuf BF16 exact-shape family; earlier pointer-increment experiments showed that more aggressive accumulation changes either regressed or harmed parity
- the compiler should keep treating these families as narrow, measured specializations instead of trying to generalize the technique across all GEMV paths

## Rejected Tuning: Exact-Shape `input81922048Tiled`

Hypothesis:

- after full decode argument-buffer coverage, the remaining hot tiled family might still benefit from one more exact-shape split
- `input=8192, output=2048` appears repeatedly in decode and already dominates the remaining non-2048-family projection time
- if the compiler isolates that shape into its own family, the kernel can drop dimension bindings and use a tighter fixed-shape path

Compiler change:

- add a new decode projection family:
  - `input81922048Tiled`
- emit dedicated kernels:
  - `gemv_8192_2048_tiled(_bf16)`
  - `gemv_8192_2048_tiled(_bf16)_argbuf`
- remove encoded constant bindings for that exact-shape family
- keep the rest of the `input8192Tiled` family unchanged

Observed result:

- generated-library diagnostics showed the new family was active in the decode plan
- reference comparison still passed:
  - prefill parity unchanged
  - decode drift signature unchanged
  - step 1 remained `Python 2944 / Metal 859`
- but aggregate performance regressed:
  - optimizer comparison:
    - `none 127.6`
    - `standard 133.4`
    - `aggressive 136.2`
  - per-step decode total:
    - `7049 us`

Per-step result:

- the new exact-shape tiled family consumed nearly half of decode time:
  - `gemv_8192_2048_tiled_bf16_argbuf: 48.5%`
- the top 16 individual steps were all this new family at roughly `212–215 us` each

Interpretation:

- unlike the successful `input2048` exact-shape splits, `8192 -> 2048` did not benefit from being isolated into its own family
- the specialized tiled path became the dominant decode bottleneck instead of improving it
- for this shape, the existing `gemv_8192_tiled_bf16_argbuf` family is better than a narrower exact-shape clone
- exact-shape splitting is therefore not a universal rule; it must be justified per family by measured wins, not by symmetry with other shapes

## Accepted Tuning: `gemv_2048_6144_bf16_argbuf` Pointer-Input Pairwise Read

Hypothesis:

- after the exact-shape BF16 argbuf family moved to `ushort2` weight reads, `gemv_2048_6144_bf16_argbuf` was still the single largest decode bottleneck
- this family stages input in buffer precision rather than `float`
- if the pairwise BF16 path reads staged input through a moving pointer instead of repeated `inputTile[j + offset]` indexing, it should reduce address-generation overhead without changing layout, launch, or accumulation semantics

Compiler change:

- add `Input2048BF16ArgumentReadPolicy.pairwisePointerInput`
- enable it only for the `2048 -> 6144` BF16 exact-shape family
- in the argbuf kernel variant, keep:
  - row-major weights
  - pairwise `ushort2` BF16 weight reads
  - existing accumulation order
- only replace indexed staged-input reads with `threadgroup const <bt>* inputLane` pointer advancement

Observed result:

- reference comparison still passed unchanged:
  - prefill parity unchanged
  - decode drift signature unchanged
  - step 1 remained `Python 2944 / Metal 859`
- optimizer comparison improved over the previous accepted baseline:
  - `none 116.2`
  - `standard 123.2`
  - `aggressive 125.8`
- per-step decode total moved to:
  - `3774 us`

Per-step result:

- the decode bottleneck remained the same family:
  - `gemv_2048_6144_bf16_argbuf: 43.1%`
  - `gemv_2048_sq_bf16_argbuf: 35.6%`
- the change was still worth keeping because aggregate decode throughput improved while preserving parity

Interpretation:

- for the `6144` exact-shape BF16 family, reducing input-side address generation helps even when the weight layout stays row-major
- this is a narrow family-specific win, not evidence that pointer-style accumulation should be generalized everywhere

## Accepted Tuning: `gemv_2048_sq_bf16_argbuf` Float-Pointer Pairwise Read

Hypothesis:

- after the `6144` family win, the next largest decode bottleneck was `gemv_2048_sq_bf16_argbuf`
- unlike `6144`, the square family stages input as `float`
- if its pairwise BF16 path reads staged input through a moving `float*` rather than repeated indexed accesses, it should cut address-generation cost while preserving the more accurate float-staged activation path

Compiler change:

- add `Input2048BF16ArgumentReadPolicy.pairwisePointerFloatInput`
- enable it only for the BF16 square exact-shape family
- in the argbuf kernel variant:
  - keep row-major weights
  - keep pairwise `ushort2` BF16 weight reads
  - keep float staging
  - replace indexed `inputTile[j + offset]` reads with `threadgroup const float* inputLane`

Observed result:

- reference comparison still passed unchanged:
  - prefill parity unchanged
  - decode drift signature unchanged
  - step 1 remained `Python 2944 / Metal 859`
- focused decode benchmark:
  - `standard 124.5 tok/s`
  - `aggressive 127.9 tok/s`
- optimizer comparison:
  - `none 111.3`
  - `standard 124.7`
  - `aggressive 127.4`

Per-step result:

- total decode time improved again:
  - `3774 us -> 3717 us`
- family-level effects:
  - `gemv_2048_6144_bf16_argbuf: 1603 us`
  - `gemv_2048_sq_bf16_argbuf: 1339 us`
  - `fused_swiglu_projection_2048_bf16_argbuf: 249 us`

Interpretation:

- the square family benefits from the same general idea as `6144`, but it needs a different staging-aware read mode
- this reinforces the main compiler conclusion of the current phase:
  - the remaining decode frontier is exact-shape family policy and kernel-body quality, not generic GEMV tuning

## Rejected Tuning: Rows-8 Row-Major Pairwise Weight Addressing

Hypothesis:

- the exact-shape `rowsPerThreadgroup = 8` families already fix row scheduling at compile time
- if the row-major pairwise BF16 path computes weight addresses as `gid * 8192 + sgitg * 1024 + tiisg * pairCount` instead of `(row * 2048) + ...`, it may reduce address-generation cost further for both:
  - `gemv_2048_sq_bf16_argbuf`
  - `gemv_2048_6144_bf16_argbuf`

Compiler change:

- keep row-major layout and pairwise BF16 weight reads unchanged
- change only the initial `weightLane` address calculation for fixed `rowsPerThreadgroup = 8` exact-shape families

Observed result:

- reference comparison still passed unchanged:
  - prefill parity unchanged
  - decode drift signature unchanged
  - step 1 remained `Python 2944 / Metal 859`
- but aggregate performance regressed:
  - `none 116.9`
  - `standard 120.8`
  - `aggressive 122.9`
- per-step decode total worsened to:
  - `3828 us`

Per-step result:

- `gemv_2048_6144_bf16_argbuf` improved slightly in isolation:
  - `1603 us -> 1577 us`
- but `gemv_2048_sq_bf16_argbuf` and the rest of the decode path worsened enough that the aggregate lost:
  - `1339 us -> 1411 us` for the square family

Interpretation:

- even when the algebra is equivalent, a more explicit rows-8 row-major address formula is not automatically a win on Apple GPU
- this is another example of why exact-shape compiler work has to stay empirical:
  - address expressions that look simpler on paper can still hurt aggregate throughput

## Rejected Tuning: `gemv_2048_sq_bf16_argbuf` BF16 Unroll Reduction

Hypothesis:

- the hot-family microbench showed that `gemv_2048_sq_bf16_argbuf` was still more expensive per output element than `gemv_2048_6144_bf16_argbuf`
- the square family differs mainly by:
  - float-staged input
  - a larger `unrollFactor`
- if BF16 square reduced `unrollFactor` from `8` to `4`, register pressure might fall enough to improve the square family without changing layout or parity

Compiler change:

- keep:
  - row-major layout
  - float staging
  - pairwise pointer-float input reads
- change only the BF16 square exact-shape family:
  - `unrollFactor: 8 -> 4`

Observed result:

- reference comparison still passed unchanged:
  - prefill parity unchanged
  - decode drift signature unchanged
  - step 1 remained `Python 2944 / Metal 859`
- but the hot-family microbench regressed clearly:
  - target-family total:
    - `2783 us -> 3192 us`
  - family split:
    - `gemv_2048_6144_bf16_argbuf: 1504 us -> 1616 us`
    - `gemv_2048_sq_bf16_argbuf: 1279 us -> 1576 us`
- aggregate optimizer comparison also failed to justify the change:
  - `none 117.0`
  - `standard 120.6`
  - `aggressive 123.8`

Interpretation:

- the square family’s higher normalized cost is not explained by excess unroll alone
- reducing unroll weakened both the square family and the broader exact-shape frontier
- this is exactly the type of hypothesis the new hot-family microbench is meant to reject early

## Rejected Tuning: `input20486144Dense` Packed-UInt BF16 Read

Hypothesis:

- `gemv_2048_6144_bf16_argbuf` is the dominant exact-shape family
- it is memory-bound and still uses row-major BF16 weights
- replacing `ushort2` pairwise reads with 32-bit packed BF16 loads should keep the same bytes and layout while reducing the weight-read instruction shape

Change:

- add a packed 32-bit BF16 decode helper in the generated source
- switch only the `2048 -> 6144` BF16 exact-shape argbuf family to the packed-uint read policy
- keep:
  - row-major layout
  - the accepted buffer-precision input staging
  - the existing launch shape and rows-per-threadgroup

Result:

- correctness was preserved:
  - `ReferenceComparisonTests` passed
  - decode drift fingerprint remained the same:
    - `step 1: Python 2944 / Metal 859`
- performance regressed:
  - optimizer comparison:
    - `none 109.8 tok/s`
    - `standard 114.3 tok/s`
    - `aggressive 115.9 tok/s`
  - per-step decode total:
    - `4184 us`
  - hot exact-shape GEMV microbench:
    - total: `3207 us`
    - `gemv_2048_6144_bf16_argbuf: 1746 us`
    - `gemv_2048_sq_bf16_argbuf: 1462 us`

Interpretation:

- for the hot `2048 -> 6144` decode family, changing the read instruction from `ushort2` to packed `uint` does not help even when the layout and bytes are unchanged
- the dominant cost is therefore not just "load two BF16 values with fewer source-level objects"
- the next useful hypotheses should target:
  - family-specific weight-read path structure
  - weight layout at the store/contract level
  - or higher-level memory-system effects, not just scalar-vs-packed read syntax

## Recommended Next Compiler Work

1. Prioritize the shape-specialized decode GEMV families.
   - `gemv_2048_sq_bf16`
   - `gemv_2048_6144_bf16`
   - `gemv_2048_8192_bf16`
   - `gemv_8192_tiled_bf16`
   - `gemv_vocab_bf16`
   - These kernels dominate decode time.
   - Any meaningful compiler-side speedup now has to move this frontier.
   - The latest shape-specialization work already proved this path is still productive.

2. Split optimization strategy by path.
   - Decode and prefill should not share the same optimization assumptions.
   - The evidence shows decode fusion is more mature than prefill fusion.

3. Add compiler-facing prefill metrics.
   - Dispatch counts alone are not enough.
   - The prefill path needs the same visibility that decode already has: hot-step breakdown, emitted kernel quality, and optimizer-specific comparison.

4. Continue launch-shape and lowering work only when tied to projection kernels.
   - Generic tuning is unlikely to beat focused GEMV/GEMM improvements now.
   - Concrete next targets are deeper fixed-shape kernel-body specialization and better `vocabDense` treatment.

## Measurement Discipline Going Forward

The current decode work reached a point where whole-request throughput is noticeably noisier than the hottest kernel families themselves. That changes how compiler experiments should be judged.

Current measurement stack:

- correctness gate:
  - `ReferenceComparisonTests`
- decode-family gate:
  - per-step decode profiling
  - hot exact-shape GEMV family microbench
- aggregate confirmation:
  - optimizer comparison benchmark

Current hot-family microbench result:

- total hot exact-shape GEMV time:
  - `2680 us`
- split:
  - `gemv_2048_6144_bf16_argbuf: 1475 us (55.0%)`
  - `gemv_2048_sq_bf16_argbuf: 1205 us (45.0%)`

Interpretation:

- the dominant frontier is now narrow enough that family-level microbenchmarks are more informative than aggregate request-level throughput alone
- from this point on, compiler tuning should be accepted only when it:
  - preserves reference parity
  - improves the relevant hot-family microbench
  - does not regress aggregate optimizer comparison in a clearly material way

## Latest Failure Boundary

The most important recent result is not a speedup but a narrowing of the failure boundary for specialized decode layouts.

Established facts:

- the blocked `8x128` BF16 pack order is correct
- direct GPU reads of the packed blocked buffer are correct
- the standalone blocked `2048 -> 6144` BF16 GEMV matches row-major on all real hot tensors
- the immediate conv consumer chain also matches row-major
- the full isolated conv operator chain (`in_proj -> conv_state_update -> out_proj`) still matches row-major
- a derived `STAFWeightStore` that registers many specialized blocked accesses still preserves that isolated operator-chain behavior

New compiler-level result:

- after adding a generic `ProjectionWeightAccessPolicyOverride`, forcing only a single hot decode tensor to use blocked layout still changes the first decode token from the baseline (`521 -> 2`)

Interpretation:

- the failure is no longer attributable to:
  - blocked pack order
  - BF16 blocked GEMV math in isolation
  - a single specialized tensor buffer being unreadable
  - the isolated conv operator chain
- the remaining problem lies in full compiled decode orchestration or full-plan specialization interaction
- this is exactly why layout choice must remain a generic compiler/store contract rather than product-specific ad hoc code: the bug boundary is now at the contract level, not at one kernel body

Implication for next work:

- do not continue blind micro-tuning on blocked layout
- use the new generic override path to isolate full-plan specialization interactions
- prioritize row-major hot-family read-path work until the full-plan blocked-layout contract is proven safe

## Latest Contract Fix

The full-plan specialization failure boundary is now narrower again.

Root cause:

- specialized weight access and emitted kernel contract were not using the same effective layout source
- a decode step could request a blocked specialized buffer while still resolving to a row-major exact-shape kernel family name in the encoded argument-buffer path

Fix:

- exact-shape `input2048` source policy is now resolved from the same access-policy resolver used by the `STAFWeightStore`
- the effective layout comes from resolved buffer access, not just the requested preference
- blocked exact-shape kernel names now propagate through argument-buffer variant selection, so `prepared` fallback no longer silently routes them back through the row-major encoded path

Validation:

- `single decode tensor override preserves first decode tokens` now passes
- `GeneratedLibraryTests` reports:
  - `decode steps=147`
  - `argTable=147`
  - `argPrepared=0`
  - `argEncoded=147`
  - `residentConst=115`
- `ReferenceComparisonTests` still preserve the known decode drift fingerprint:
  - `step 1: Python 2944 / Metal 859`

Interpretation:

- the blocked-layout idea itself was not disproven
- the failing component was the generic compiler/store contract that decided which kernel family matched a resolved specialized buffer
- this is an architectural result, not a product-specific workaround: layout selection must remain a backend contract shared by store access, kernel naming, source generation, and argument-buffer encoding

Current performance snapshot after the accepted `2048 -> 6144` row-major contiguous-read change:

- optimizer comparison:
  - `none: 125.0 tok/s`
  - `standard: 130.5 tok/s`
  - `aggressive: 133.0 tok/s`
- per-step decode profile:
  - total `3320 us`
  - `gemv_2048_6144_bf16_argbuf: 1465 us (44.1%)`
  - `gemv_2048_sq_bf16_argbuf: 1215 us (36.6%)`

Latest rejected follow-up:

- applying the same idea to the square family via `packed8PointerFloatInput` preserved parity but regressed both hot-family microbench and aggregate throughput
- rejected result:
  - optimizer comparison:
    - `none: 122.6 tok/s`
    - `standard: 127.6 tok/s`
    - `aggressive: 126.3 tok/s`
  - hot exact-shape GEMV microbench:
    - total `2980 us`
    - `gemv_2048_6144_bf16_argbuf: 1630 us`
    - `gemv_2048_sq_bf16_argbuf: 1350 us`
- interpretation:
  - the contiguous `ushort4` row-major read is a real win for `2048 -> 6144`
  - the square family has a different staging/read balance and should not inherit the same read mode blindly
- a narrower follow-up, `packed4PointerFloatInput`, also failed:
  - parity remained intact
  - but hot exact-shape GEMV microbench regressed to `3112 us`
  - split:
    - `gemv_2048_6144_bf16_argbuf: 1590 us`
    - `gemv_2048_sq_bf16_argbuf: 1522 us`
  - per-step decode total worsened to `3831 us`
  - optimizer comparison became unstable and failed acceptance, so the change was reverted
- a square-only occupancy follow-up, lowering `rowsPerThreadgroup` and `preferredSimdgroups` from `8` to `4`, also failed:
  - parity remained intact
  - but per-step decode total worsened to `3644 us`
  - hot exact-shape GEMV microbench regressed to `2922 us`
  - split:
    - `gemv_2048_6144_bf16_argbuf: 1578 us`
    - `gemv_2048_sq_bf16_argbuf: 1344 us`
  - optimizer comparison collapsed to `none 124.7 / standard 123.0 / aggressive 49.4 tok/s`
  - interpretation:
    - the square family is not limited by insufficient row parallelism at the current launch shape
    - reducing square occupancy hurts aggregate decode even when correctness is preserved
- a formal store-side blocked-layout override for all decode `2048 -> 6144` BF16 projections also failed:
  - in the dedicated diagnostic benchmark, baseline tokens were `[2, 2, 2, 0]`
  - the blocked override diverged immediately as `[2, 521, 521, 1198]`
  - `gemv_2048_6144_bf16_argbuf` hot-family microbench worsened from `2871 us` to `4145 us`
  - delta: `+1274 us` (`+44.4%`)
  - interpretation:
    - the current `blockedRows8Tiles128` layout does not satisfy the full compiled decode contract for these projections
    - even as a pure performance experiment, it is materially slower than row-major on the current kernel family
- a row-major `6144`-only input-tile pointer-increment load also failed:
  - hypothesis:
    - `gemv_2048_6144_bf16_argbuf` is memory-bound, so reducing input-side tile-load address generation might lower integer overhead without changing launch shape or layout
  - implementation:
    - keep the accepted `packed4PointerInput` weight-read path
    - change only the input tile staging loop to a pointer-increment load for the `2048 -> 6144` exact-shape family
  - result:
    - optimizer comparison regressed to `none 107.7 / standard 121.8 / aggressive 119.2 tok/s`
    - per-step decode total regressed to `3842 us`
    - hot exact-shape GEMV microbench regressed to `3020 us`
    - split:
      - `gemv_2048_6144_bf16_argbuf: 1645 us`
      - `gemv_2048_sq_bf16_argbuf: 1375 us`
  - interpretation:
    - for the accepted row-major `6144` family, the remaining cost is not improved by changing input tile staging
    - the next read-path work should stay on the weight side, not the input-side staging loop
- a square-only fixed-iteration pairwise float-input loop also failed:
  - hypothesis:
    - `gemv_2048_sq_bf16_argbuf` might benefit from the same loop-control reduction that helped the `6144` family, while keeping the accepted row-major pairwise float-input read structure unchanged
  - implementation:
    - keep `pairwisePointerFloatInput` semantics
    - change only the square-family BF16 pairwise loop to a fixed-iteration form
  - result:
    - optimizer comparison moved to `none 126.4 / standard 133.3 / aggressive 136.8 tok/s`
    - hot exact-shape GEMV microbench regressed from `2477 us` to `2739 us`
    - split:
      - `gemv_2048_sq_bf16_argbuf: 1374 us`
      - `gemv_2048_6144_bf16_argbuf: 1365 us`
  - interpretation:
    - the square family is not limited by pairwise-loop control overhead in the same way as the `6144` family
    - the accepted `6144` fixed-iteration win does not generalize across exact-shape families
- a `6144`-only split-accumulator variant was tested and rejected:
  - hypothesis:
    - after the accepted fixed-iteration `ushort4` path, the next remaining cost in `gemv_2048_6144_bf16_argbuf` might be the serial add dependency chain
  - implementation:
    - keep the same row-major `ushort4` read path and loop shape
    - split the four MACs per iteration across two accumulators, then merge before `simd_sum`
  - result:
    - single runs sometimes improved slightly, but repeated validation was not stable
    - the hot-family microbench drifted back into the same noise band as the baseline
  - interpretation:
    - the effect is too small and unstable to justify keeping the extra kernel complexity
    - this confirms that the accepted `packed4FixedPointerInput` path is the right stopping point for this branch
- a square-family split-accumulator variant also failed:
  - hypothesis:
    - `gemv_2048_sq_bf16_argbuf` might benefit from the same dependency-chain reduction, while keeping its accepted pairwise float-input read path
  - implementation:
    - keep the same `ushort2` pairwise read path and loop shape
    - split accumulation across two partial sums
  - result:
    - correctness was preserved
    - but the hot exact-shape GEMV microbench regressed from the accepted band into a slower `~2.52 ms` run, with square-family time increasing
  - interpretation:
    - the square family is not bottlenecked by the accumulation dependency chain in the same way
    - further square-family work should focus on read structure, not accumulator count
- a generic vectorized BF16 conversion helper also failed acceptance:
  - hypothesis:
    - if the remaining exact-shape BF16 cost includes per-element BF16-to-F32 conversion overhead, replacing the scalar helper expansion in `bf16x2_to_float2` and `bf16x4_to_float4` with vector bitcast conversion should help the hot row-major decode families without introducing family-specific logic
  - implementation:
    - keep all kernel families, launch shapes, and read policies unchanged
    - change only the shared conversion helpers to use `uint2/uint4 << 16` plus `as_type<float2/float4>`
  - result:
    - correctness was preserved
    - but repeat benchmark runs stayed within noise and did not produce a stable gain over the accepted baseline
    - one repeat measured `Hot Exact-Shape GEMV` at `2699 us`, which is worse than the accepted band
    - aggregate decode throughput also failed to improve materially enough to justify a global helper change
  - interpretation:
    - the current hotspot is not limited by scalar BF16 helper expansion alone
    - the next useful work should stay on family-specific read structure, not generic helper rewriting

Implication for next work:

- the dominant problem remains exact-shape BF16 GEMV kernel-body quality
- optimizer choice is no longer the first-order issue in the current state
- next experiments should stay focused on family-specific row-major read paths for `2048 -> 6144` and `2048 -> 2048`

## Non-Goals For This Report

These topics were explored during the same work session, but they are runtime/API concerns rather than compiler optimizations:

- stream chunk batching
- prompt-state reuse
- public generation-path throughput shaping

They may matter more for end-user latency, but they are intentionally excluded from the compiler conclusions above.

## Summary

- The compiler currently helps most on decode, not prefill.
- Optimizer policy is no longer the first-order issue; current runs put `standard` and `aggressive` close enough that kernel quality matters more than fusion policy.
- The newest decode GEMV tile/staging adjustment improved throughput while preserving reference behavior.
- The next compiler gains are most likely to come from projection kernels, not attention kernels.
- The full decode plan now runs with `147/147` argument-buffer-encoded steps and no prepared tail.
- The generated decode path is dominated by shape-specialized decode GEMV families, not attention.
- The next compiler optimization target should be projection-kernel quality and projection-oriented lowering, not tokenizer, streaming, or attention micro-tuning.

## Post-Metadata Validation Reruns

After the STAF metadata/provenance work landed, `BenchmarkTests` was rerun three times to check whether the lower decode throughput numbers were a real hot-path regression or just benchmark noise.

Observed reruns:

- run 1:
  - optimizer comparison: `none 125.5 / standard 130.1 / aggressive 131.2 tok/s`
  - per-step total: `3175 us`
  - hot exact-shape GEMV: `2518 us`
  - split:
    - `gemv_2048_6144_bf16_argbuf: 1306 us`
    - `gemv_2048_sq_bf16_argbuf: 1212 us`
- run 2:
  - optimizer comparison: `none 122.1 / standard 128.2 / aggressive 125.6 tok/s`
  - per-step total: `3329 us`
  - hot exact-shape GEMV: `2687 us`
  - split:
    - `gemv_2048_6144_bf16_argbuf: 1387 us`
    - `gemv_2048_sq_bf16_argbuf: 1300 us`
- run 3:
  - optimizer comparison: `none 119.9 / standard 120.8 / aggressive 132.4 tok/s`
  - per-step total: `3419 us`
  - hot exact-shape GEMV: `2556 us`
  - split:
    - `gemv_2048_6144_bf16_argbuf: 1306 us`
    - `gemv_2048_sq_bf16_argbuf: 1250 us`

Median view:

- optimizer comparison: `none 122.1 / standard 128.2 / aggressive 131.2 tok/s`
- per-step total: `3329 us`
- hot exact-shape GEMV total: `2556 us`

Interpretation:

- end-to-end decode throughput is currently below the earlier accepted `~133 / ~136 tok/s` band
- but the hot exact-shape GEMV microbench and per-step totals are not worse in the same proportion, and are in some reruns slightly better than the older accepted totals
- this suggests the STAF metadata/provenance work did not create a clear decode-kernel regression
- the throughput drop is real at the suite-output level, but the current evidence points to run-to-run variance or benchmark-state effects outside the exact-shape GEMV hot loop

Working conclusion:

- do not attribute the lower throughput numbers to the metadata implementation without a stronger A/B
- keep the accepted kernel baseline
- continue compiler work on `gemv_2048_6144_bf16_argbuf` and `gemv_2048_sq_bf16_argbuf`
- if throughput acceptance tightens further, add a repeated-run benchmark harness and compare medians rather than single suite outputs

## Follow-Up Row-Stream Rejections

Two additional row-stream hypotheses were tested after the benchmark harness cleanup. Both were rejected and reverted immediately.

- square family `2048 -> 2048`
  - hypothesis:
    - increase `rowsPerThreadgroup` / `preferredSimdgroups` from `8` to `16`
    - amortize float-staged input over more rows
  - result:
    - per-step decode profile: `gemv_2048_sq_bf16_argbuf 1657 us`
    - hot exact-shape GEMV total: `3075 us`
    - optimizer comparison: `none 119.5 / standard 123.5 / aggressive 130.4 tok/s`
    - aggressive single decode: `127.5 tok/s`
  - interpretation:
    - square-family kernel time improved locally, but aggregate decode throughput regressed
    - the larger row group hurt the surrounding decode balance enough to fail acceptance

- expanded family `2048 -> 6144`
  - hypothesis:
    - reduce `rowsPerThreadgroup` / `preferredSimdgroups` from `4` to `2`
    - further lower simultaneous weight-row streams in the memory-bound hot family
  - result:
    - per-step decode profile: `gemv_2048_6144_bf16_argbuf 1874 us`
    - hot exact-shape GEMV total: `3618 us`
    - optimizer comparison: `none 112.4 / standard 121.0 / aggressive 126.2 tok/s`
    - aggressive single decode: `125.0 tok/s`
  - interpretation:
    - `4` rows/threadgroup was already near the local optimum for the current row-major `packed4FixedPointerInput` path
    - dropping to `2` doubled grid pressure and clearly regressed the dominant hot family

Current accepted row-stream policy therefore remains:

- `2048 -> 2048`: `rowsPerThreadgroup = 8`
- `2048 -> 6144`: `rowsPerThreadgroup = 4`

Implication:

- the next useful compiler work should stay on family-specific weight-side read structure
- row-stream count sweeps are now largely exhausted for the two dominant exact-shape BF16 argbuf families

## Square Blocked Boundary

The current `blockedRows8Tiles128` story for the square `2048 -> 2048` family is now localized more precisely.

- isolated hot-family benchmark:
  - overriding all square decode tensors to `blockedRows8Tiles128` still cuts the square-family microbench roughly in half
  - latest clean rerun:
    - baseline: `1318 us`
    - specialized: `630 us`
    - delta: `-52.2%`
- but decode correctness still breaks immediately:
  - baseline tokens: `[2, 521, 859, 1595]`
  - specialized tokens: `[2, 7, 521, 521]`

The important new boundary is that this is not a blanket square-family failure.

- single-tensor override results:
  - a single square `q_proj` override to `blockedRows8Tiles128` preserved the next decode token
  - but the same single square `q_proj` override changed a later decode token within three decode steps
  - a single square `self_attn.out_proj` override changed the next decode token
  - a single square `conv.out_proj` override also changed the next decode token
- grouped override results:
  - overriding all square `q_proj` tensors together also changed the next decode token
  - prefix boundary for square `q_proj` tensors:
    - prefix 1: preserved second token `521`
    - prefix 2: preserved second token `521`
    - prefix 3: preserved second token `521`
    - prefix 4: changed second token to `63436`
    - prefix 5: changed second token to `3421`
    - prefix 6: changed second token to `730`
  - but even grouped prefixes below that boundary were not rollout-safe:
    - prefix 2 changed a later decode token within three decode steps
    - prefix 3 changed a later decode token within three decode steps

Interpretation:

- the unsafe subset is on the square output-projection side, not on `q_proj` alone
- but "safe q_proj" is only true if the acceptance criterion is only the next decode token
- once multi-token decode is required, even a single blocked square `q_proj` override is not stable enough for mainline rollout
- more precisely, the current blocked square contract remains stable for the first three `q_proj` tensors only at the second-token boundary, then breaks either immediately at prefix 4 or later within three decode steps at prefixes 1-3
- this sharply narrows the next investigation:
  - do not treat `blockedRows8Tiles128` as "square-family on/off"
  - split square-family experiments by projection role
  - split further by tensor grouping, not just role
  - but do not promote any blocked square subset into mainline until later-token decode parity is demonstrated, not just second-token parity

Working conclusion:

- `blockedRows8Tiles128` still looks like a real performance opportunity for square decode
- but the contract failure is role-specific
- the currently unsafe subset appears to be square `out_proj`
- and grouped `q_proj` overrides are not rollout-safe
- more importantly, even the single-`q_proj` case is not stable enough over multiple decode steps
- so `blockedRows8Tiles128` is not a decode-mainline candidate today; the mainline path stays row-major exact-shape GEMV until a stronger correctness contract exists

## Accepted: 6144 Threadgroup-Row Base Read Path

The next accepted improvement stayed entirely on the row-major mainline path.

- change:
  - add a dedicated BF16 read mode for the exact-shape `2048 -> 6144` argbuf family
  - instead of deriving the row pointer from `row * 2048`, compute the row-major packed4 base as:
    - `gid * 2048 + sgitg * 512 + tiisg`
  - keep the current fixed-iteration packed4 loop and current launch policy
- scope:
  - only `gemv_2048_6144_bf16_argbuf`
  - no square-family change
  - no layout change
  - no optimizer policy change

Validation:

- `xcodebuild build -scheme swift-lm-Package -destination 'platform=macOS'`
- `xcodebuild test -scheme swift-lm-Package -destination 'platform=macOS' -only-testing:MetalCompilerTests/ReferenceComparisonTests`
- `xcodebuild test -scheme swift-lm-Package -destination 'platform=macOS' -only-testing:MetalCompilerTests/BenchmarkTests`
- `xcodebuild test -scheme swift-lm-Package -destination 'platform=macOS' -only-testing:MetalCompilerTests/BenchmarkDiagnosticsTests`

Result:

- correctness:
  - `ReferenceComparisonTests` full pass
  - known decode drift unchanged: `step 1 = Python 2944 / Metal 859`
- clean benchmark acceptance:
  - optimizer comparison: `none 129.2 / standard 131.8 / aggressive 136.5 tok/s`
  - standard decode: `134.1 tok/s`
  - aggressive decode: `137.1 tok/s`
  - end-to-end decode: `128.7 tok/s`
- diagnostics:
  - per-step decode total: `3074 us`
  - hot exact-shape GEMV total: `2479 us`
  - `gemv_2048_6144_bf16_argbuf`: `1232 us`, `123.2 us/step`
  - `gemv_2048_sq_bf16_argbuf`: `1247 us`, `56.7 us/step`

Interpretation:

- this is the strongest accepted evidence so far that `2048 -> 6144` is still leaving performance on the table in row-major address generation, not only in raw bandwidth
- the gain came without touching square-family behavior, which is important because square blocked remains unsafe for decode rollout
- the remaining primary bottleneck is now effectively split between:
  - `gemv_2048_6144_bf16_argbuf`
  - `gemv_2048_sq_bf16_argbuf`

Next implication:

- keep square blocked out of mainline
- continue row-major family-specific work
- next target should be the analogous weight-side address/read structure for `gemv_2048_sq_bf16_argbuf`

## Rejected: Square Threadgroup-Row Base Read Path

The analogous square-family change did not clear acceptance.

- change:
  - add a square-only BF16 read mode that rewrites the row-major pairwise base to threadgroup-row form
  - intended form:
    - `gid * 8192 + sgitg * 1024 + tiisg * pairCount`
  - scope was limited to `gemv_2048_sq_bf16_argbuf`
- result:
  - `ReferenceComparisonTests` stayed green
  - clean benchmark acceptance regressed slightly on the public path
    - optimizer comparison: `none 127.6 / standard 132.1 / aggressive 135.5 tok/s`
    - aggressive single decode: `135.5 tok/s`
    - end-to-end decode: `128.4 tok/s`
  - this did not beat the accepted baseline that already had the `2048 -> 6144` threadgroup-row base path

Interpretation:

- unlike `2048 -> 6144`, square-family row-base rewriting does not currently pay for itself
- the next square-family work should target a different weight-side read structure, not just a different expression for the same row-major base

## Rejected: Square Fixed-Simdgroup Source Policy

The next square-family attempt kept the same runtime launch shape but encoded the current `8`-simdgroup assumption into the generated kernel source.

- change:
  - set `fixedSimdgroups = 8` for the exact-shape `2048 -> 2048` source policy
  - keep the accepted square-family BF16 read mode:
    - `pairwisePointerFloatInput`
  - keep the accepted runtime planner launch shape unchanged
- rationale:
  - the square decode family already prefers `8` simdgroups in the dispatch planner
  - making the generated kernel use a compile-time `threads_per_threadgroup = 256` should reduce staging-loop control overhead without changing correctness or backend contract
- result:
  - `ReferenceComparisonTests` stayed green
  - known decode drift remained `step 1 = Python 2944 / Metal 859`
  - but clean benchmark acceptance regressed:
    - optimizer comparison: `none 127.0 / standard 130.5 / aggressive 132.2 tok/s`
    - standard decode: `133.4 tok/s`
    - aggressive decode: `136.7 tok/s`
    - end-to-end decode: `130.4 tok/s`
- interpretation:
  - the square family does not benefit from baking the current planner simdgroup count into the source
  - the remaining square-family cost is still in the weight-side address/read structure, not in dynamic `threadsPerThreadgroup` usage

## Rejected: Square `packed4PointerFloatInput + unroll=4`

The next square-family attempt combined the two previously promising axes into a single aggressive read-path change.

- change:
  - square BF16 exact-shape family only
  - switch from:
    - `pairwisePointerFloatInput`
    - `unrollFactor = 8`
  - to:
    - `packed4PointerFloatInput`
    - `unrollFactor = 4`
- rationale:
  - the earlier square `packed4` experiment likely paid too much register pressure at `unroll=8`
  - the earlier square `unroll=4` experiment preserved the old pairwise read structure and did not reduce read instruction count
  - combining `ushort4 × float4` with `unroll=4` was the natural missing interaction test
- result:
  - `ReferenceComparisonTests` full pass
  - known decode drift unchanged: `step 1 = Python 2944 / Metal 859`
  - clean benchmark acceptance was mixed but not good enough:
    - optimizer comparison: `none 129.7 / standard 133.7 / aggressive 137.6 tok/s`
    - standard decode: `134.5 tok/s`
    - aggressive decode: `136.7 tok/s`
    - end-to-end decode: `129.9 tok/s`
  - diagnostics showed the real cost regression:
    - per-step decode total: `3210 us`
    - hot exact-shape GEMV total: `2658 us`
    - `gemv_2048_sq_bf16_argbuf: 1422 us`
    - `gemv_2048_6144_bf16_argbuf: 1236 us`
- interpretation:
  - the square family does not want the `ushort4` read structure, even after unroll pressure is reduced
  - this closes the obvious `packed4` branch for square row-major BF16 argbuf kernels
  - the next square-family work should move away from read-width experiments and back toward a different address/reuse structure

## Rejected: Square Output-Only Source Policy Split

The next square-family attempt split `2048 -> 2048` into two source policies using the existing semantic `isOutput` flag.

- change:
  - keep the accepted square-family baseline for non-output projections:
    - `pairwisePointerFloatInput`
    - `unrollFactor = 8`
  - generate a second exact-shape square family only for output projections:
    - `gemv_2048_sq_out(_bf16)(_argbuf)`
    - `pairwisePointerFloatInput`
    - `unrollFactor = 4`
- rationale:
  - `q_proj`-like square projections and output-side square projections do not have the same downstream interaction pattern
  - a semantic split on `isOutput` is generic and avoids product-specific logic
  - if output-side square projections wanted lower unroll pressure, the split would let us keep the accepted non-output path unchanged
- result:
  - `ReferenceComparisonTests` full pass
  - known decode drift unchanged: `step 1 = Python 2944 / Metal 859`
  - clean benchmark acceptance looked neutral-to-slightly-positive:
    - optimizer comparison: `none 128.7 / standard 132.0 / aggressive 136.5 tok/s`
    - standard decode: `133.0 tok/s`
    - aggressive decode: `134.5 tok/s`
    - end-to-end decode: `130.3 tok/s`
  - diagnostics showed the real regression:
    - per-step decode total: `3302 us`
    - hot exact-shape GEMV total: `2717 us`
    - `gemv_2048_6144_bf16_argbuf: 1273 us`
    - square family total:
      - `gemv_2048_sq_out_bf16_argbuf: 1103 us`
      - `gemv_2048_sq_bf16_argbuf: 353 us`
      - combined square total: `1456 us`
- interpretation:
  - splitting square kernels by `isOutput` does not improve the true bottleneck
  - the throughput headline looked acceptable, but the dominant exact-shape GEMV total got worse than the accepted baseline
  - square-family residual cost is still in the row-major weight-side address/read structure, not in a simple output-vs-non-output source split

## Rejected: Square `out_proj`-Only Source Policy Split

The next square-family attempt narrowed the previous split further and touched only semantic `out_proj` kernels.

- change:
  - keep the accepted square-family baseline for:
    - `q_proj`
    - `o_proj`
  - generate a second exact-shape square family only for `out_proj`:
    - `gemv_2048_sq_outproj(_bf16)(_argbuf)`
    - `pairwisePointerFloatInput`
    - `unrollFactor = 4`
- rationale:
  - square role breakdown shows:
    - `out_proj`: 10 steps
    - `q_proj`: 6 steps
    - `o_proj`: 6 steps
  - the previous `isOutput` split likely failed because it also changed `o_proj`
  - `out_proj` alone was the largest semantic square subset and the cleanest next cut
- result:
  - `ReferenceComparisonTests` full pass
  - known decode drift unchanged: `step 1 = Python 2944 / Metal 859`
  - benchmark regressed badly:
    - optimizer comparison: `none 129.1 / standard 133.7 / aggressive 137.4 tok/s`
    - standard decode: `105.7 tok/s`
    - aggressive decode: `131.8 tok/s`
    - end-to-end decode: `101.5 tok/s`
  - diagnostics showed a broad mainline slowdown:
    - per-step decode total: `7110 us`
    - hot exact-shape GEMV total: `2597 us`
    - `gemv_2048_6144_bf16_argbuf: 1240 us`
    - square family total:
      - `gemv_2048_sq_outproj_bf16_argbuf: 1013 us`
      - `gemv_2048_sq_bf16_argbuf: 1003 us`
      - combined square total: `2016 us`
- interpretation:
  - `out_proj` is not a safe or beneficial semantic split point for square-family source policy
  - even when correctness holds, this split destabilizes the broader decode mainline
  - square-family work should stay on a single row-major family and target weight-side address/read structure, not semantic role partitioning

## Rejected: Square `ushort4 x 2` Float-Input Read

The next square-family attempt widened the accepted pairwise BF16 read shape without changing launch, staging, or family boundaries.

- change:
  - square BF16 exact-shape family only
  - keep:
    - row-major layout
    - `rowsPerThreadgroup = 8`
    - float-staged input
    - `unrollFactor = 8`
  - change only the weight-read shape from:
    - `ushort2 x 4`
  - to:
    - `ushort4 x 2`
- rationale:
  - this was the narrowest remaining read-width experiment that preserved the accepted square-family loop structure
  - if the square family was still instruction-bound on weight reads, collapsing four pairwise reads into two contiguous reads should help
- result:
  - `ReferenceComparisonTests` full pass
  - known decode drift unchanged: `step 1 = Python 2944 / Metal 859`
  - diagnostics regressed:
    - hot exact-shape GEMV total: `2537 us`
    - `gemv_2048_sq_bf16_argbuf: 1280 us`
    - `gemv_2048_6144_bf16_argbuf: 1257 us`
- interpretation:
  - the square family does not want a broader `ushort4` read shape, even when loop form and staging remain fixed
  - square-family residual cost is still better explained by row-major address/reuse structure than by raw BF16 read width

## Rejected: `2048 -> 6144` `half4 + dot` Input Vectorization

The next `6144`-family attempt kept the accepted packed4 row-major weight path and changed only the staged-input read shape.

- change:
  - `gemv_2048_6144_bf16_argbuf` only
  - keep:
    - accepted `packed4ThreadgroupFixedPointerInput`
    - row-major layout
    - `rowsPerThreadgroup = 4`
    - `fixedSimdgroups = 4`
  - change only the inner-loop input side from:
    - four scalar `half` reads and scalar multiply-add
  - to:
    - one `half4` read and `dot(float4(weight), float4(input))`
- rationale:
  - after the accepted packed4 weight-path win, the remaining local overhead might have been on the threadgroup input-read side rather than the weight side
- result:
  - `ReferenceComparisonTests` full pass
  - known decode drift unchanged: `step 1 = Python 2944 / Metal 859`
  - diagnostics regressed slightly:
    - `gemv_2048_6144_bf16_argbuf: 1257 us -> 1268 us`
    - hot exact-shape GEMV total: `2544 us`
- interpretation:
  - the accepted packed4 `6144` path is not limited by scalar staged-input reads in a way that justifies vectorizing the input side
  - further `6144` work should continue to target weight-side address structure, not `dot`-style input vectorization

## Rejected: `fused_swiglu_projection_2048_bf16_argbuf` Packed4 Weight Read

The next secondary-hotspot attempt targeted the exact-shape fused SwiGLU projection family directly.

- change:
  - `fused_swiglu_projection_2048_bf16(_argbuf)` only
  - keep:
    - exact-shape `2048` family
    - accepted `unrollFactor = 8`
    - buffer-precision staging
    - the same launch policy
  - change only the inner read structure from:
    - scalar BF16 weight reads for `gateWeight` and `upWeight`
  - to:
    - `ushort4 x 2` packed BF16 reads for both weight streams
    - `dot(...)` accumulation against two staged-input vectors
- rationale:
  - `fused_swiglu_projection_2048_bf16_argbuf` remained the only meaningful secondary hotspot after exact-shape GEMV
  - because the kernel reads two independent BF16 weight streams (`gate` and `up`) over the same input tile, a wider packed read could reduce weight-read overhead more effectively than in the single-stream GEMV case
- result:
  - `ReferenceComparisonTests` full pass
  - known decode drift unchanged: `step 1 = Python 2944 / Metal 859`
  - diagnostics improved locally:
    - per-step decode total: `6702 us`
    - `fused_swiglu_projection_2048_bf16_argbuf: 1031 us -> 988 us`
  - but clean throughput regressed:
    - optimizer comparison: `none 126.5 / standard 130.8 / aggressive 135.0 tok/s`
    - standard decode: `131.1 tok/s`
    - aggressive decode: `135.0 tok/s`
    - end-to-end decode: `126.7 tok/s`
- interpretation:
  - lowering the secondary hotspot in isolation was not enough; the packed4 SwiGLU read path interacts poorly with the broader decode mainline
  - this branch should not be kept
  - the next useful work should stay on the primary exact-shape GEMV bottlenecks, or revisit secondary hotspots only with stricter throughput acceptance

## Rejected: Square Threadgroup-Fixed Row-Major Pairwise Base

The next square-family attempt copied the accepted `6144` idea more literally: keep the same BF16 pairwise loop, but replace `row * 2048` addressing with a threadgroup-fixed row base.

- change:
  - `gemv_2048_sq_bf16_argbuf` only
  - keep:
    - row-major layout
    - float staging
    - `unrollFactor = 8`
    - pairwise BF16 reads
  - change only the weight-lane base from:
    - `(args.weight + row * 2048u) + tiisg * 4`
  - to:
    - `gid * 8192u + sgitg * 1024u + tiisg * 4`
- rationale:
  - the accepted `2048 -> 6144` win came from removing per-row address formation inside the hot loop while preserving the same read width
  - the square family looked like the closest remaining place where the same address-generation reduction might apply
- result:
  - `ReferenceComparisonTests` full pass
  - known decode drift unchanged: `step 1 = Python 2944 / Metal 859`
  - diagnostics regressed:
    - per-step decode total: `6851 us`
    - `gemv_2048_sq_bf16_argbuf: 1416 us`
    - `gemv_2048_6144_bf16_argbuf: 1264 us`
    - hot exact-shape GEMV total: `2680 us`
  - clean benchmark after revert returned to the accepted band:
    - optimizer comparison: `none 127.1 / standard 130.6 / aggressive 136.8 tok/s`
    - standard decode: `131.0 tok/s`
    - aggressive decode: `134.5 tok/s`
    - end-to-end decode: `125.2 tok/s`
- interpretation:
  - the square family does not benefit from the same threadgroup-fixed row-major base that helped `2048 -> 6144`
  - its remaining cost is not the same address-generation shape as the `6144` family
  - further square-family work should avoid this branch and continue to focus on row-major reuse structure rather than copied `6144` addressing tricks

## Diagnostic: Square Role Timing Is Uniform Enough That Semantic Splits Are Low Value

The next diagnostic pass measured the three semantic roles that share the exact same square-family kernel body: `out_proj`, `q_proj`, and `o_proj`.

- measurement:
  - `BenchmarkDiagnosticsTests.squareExactShapeGEMVRoleMicrobench`
  - baseline row-major mainline only
  - role is derived from the square projection order in `dumpDispatchEntries(...)` and matched against the profiled square-family steps
- result:
  - square family total: `1261 us`
  - `out_proj`: `10 steps`, `576 us`, `57.6 us/step`, `45.7%`
  - `q_proj`: `6 steps`, `344 us`, `57.4 us/step`, `27.3%`
  - `o_proj`: `6 steps`, `341 us`, `56.9 us/step`, `27.0%`
- interpretation:
  - `out_proj` dominates only because it appears more often
  - per-step cost is effectively flat across all three semantic roles
  - this makes the next direction clearer:
    - semantic square splits are unlikely to pay
    - the remaining win, if it exists, has to come from a family-wide kernel-body/read-structure change, not a role-specific policy

## Rejected: Square Fixed-Iteration Pairwise Pointer Loop

The next square-family attempt removed the loop induction variable from the accepted BF16 pointer-float pairwise path.

- change:
  - `gemv_2048_sq_bf16_argbuf` only
  - keep:
    - row-major layout
    - float staging
    - `unrollFactor = 8`
    - pairwise BF16 reads
  - change only the inner loop from:
    - `for (uint j = tiisg * 8; j < 2048u; j += SIMD_WIDTH * 8)`
  - to:
    - a fixed-iteration loop with the same pointer increments
- rationale:
  - after ruling out read-width changes and semantic splits, the remaining square-family overhead might have been loop-control / induction overhead inside the pairwise pointer path
- result:
  - `ReferenceComparisonTests` full pass
  - known decode drift unchanged: `step 1 = Python 2944 / Metal 859`
  - diagnostics regressed heavily:
    - per-step decode total: `7010 us`
    - `gemv_2048_sq_bf16_argbuf: 1498 us`
    - `gemv_2048_6144_bf16_argbuf: 1399 us`
    - hot exact-shape GEMV total: `2898 us`
    - square role microbench total: `1416 us`
- interpretation:
  - square-family residual cost is not in the `j` induction structure
  - forcing fixed iteration makes the full decode mainline worse even though the arithmetic is unchanged
  - this branch should stay reverted

## Rejected: Square Split-Accumulator Pairwise Pointer Loop

The next square-family attempt targeted instruction-level parallelism directly by splitting the accepted pointer-float pairwise accumulation into two partial sums.

- change:
  - `gemv_2048_sq_bf16_argbuf` only
  - keep:
    - row-major layout
    - float staging
    - `unrollFactor = 8`
    - pairwise BF16 reads
  - change only the accumulation shape from:
    - one running `sum`
  - to:
    - two alternating partial sums `sum0` / `sum1`, merged once per loop iteration
- rationale:
  - after ruling out read-width and loop-control hypotheses, the remaining square-family cost might have been memory latency that a second accumulator could hide
- result:
  - correctness regressed:
    - `ReferenceComparisonTests` failed at `Decode step 1 final norm kernel`
    - `maxErr = 21.625`
  - diagnostics also regressed badly:
    - per-step decode total: `7736 us`
    - `gemv_2048_sq_bf16_argbuf: 1910 us`
    - hot exact-shape GEMV total: `3451 us`
- interpretation:
  - the accepted square-family path is sensitive to accumulator structure
  - this is not a safe ILP win; it both hurts performance and destabilizes decode
  - the branch should remain reverted

## Rejected: Square Register-Prefetched Input Lane

The next square-family attempt kept the accepted pairwise BF16 weight path but moved the staged input lane into scalar registers once per loop iteration.

- change:
  - `gemv_2048_sq_bf16_argbuf` only
  - keep:
    - row-major layout
    - float staging
    - `unrollFactor = 8`
    - the same pairwise BF16 weight reads
  - change only the input-use shape from:
    - repeated `inputLane[0...7]` reads inside the pairwise accumulation
  - to:
    - `const float in0...in7` loaded once per loop iteration before the BF16 conversions
- rationale:
  - after rejecting wider reads, split accumulators, and fixed-iteration control, the remaining square-family cost might have been repeated threadgroup address / load pressure on the staged input lane
- result:
  - correctness remained acceptable through the early reference passes
  - diagnostics regressed clearly:
    - per-step decode total: `~8.0 ms`
    - `gemv_2048_sq_bf16_argbuf: 1783 us`
    - hot exact-shape GEMV total: `3285 us`
    - square role microbench total: `1449 us`
- interpretation:
  - square-family residual cost is not improved by scalar register-prefetch of the staged input lane
  - this branch should remain reverted

## Rejected: Square Weight-Prefetched Pairwise BF16 Read

The next square-family attempt kept the accepted row-major float-staged path, but moved each `ushort2` BF16 weight read into an explicit scalar register before conversion.

- change:
  - `gemv_2048_sq_bf16_argbuf` only
  - keep:
    - row-major layout
    - float staging
    - `unrollFactor = 8`
    - the accepted `pairwisePointerFloatInput` addressing
  - change only the weight-use shape from:
    - `float2 w = bf16x2_to_float2(weightLane[pair])`
  - to:
    - `ushort2 raw = weightLane[pair]`
    - `float2 w = bf16x2_to_float2(raw)`
- rationale:
  - after ruling out input-side tricks, loop-control changes, and accumulator splits, the remaining square-family residual might have been in the weight-load/use chain itself
  - explicitly materializing the raw BF16 pair in a register before conversion might let the compiler schedule the load/convert more effectively
- result:
  - `ReferenceComparisonTests` full pass
  - known decode drift unchanged: `step 1 = Python 2944 / Metal 859`
  - diagnostics still regressed slightly:
    - per-step decode total: `3106 us`
    - `gemv_2048_sq_bf16_argbuf: 1287 us`
    - `gemv_2048_6144_bf16_argbuf: 1255 us`
    - hot exact-shape GEMV total: `2556 us`
    - square role microbench total: `1283 us`
- interpretation:
  - simply materializing `ushort2` into a temporary register does not improve the square-family BF16 read path
  - the remaining square-family opportunity is still in the deeper row-major weight-side read/reuse structure, not in this local load/convert reshaping
  - this branch should remain reverted

## Rejected: Square Weight-Only Software Pipeline

The next square-family attempt kept the accepted row-major float-input pairwise path, but turned the BF16 weight reads into a manual software pipeline.

- change:
  - `gemv_2048_sq_bf16_argbuf` only
  - keep:
    - row-major layout
    - float staging
    - `unrollFactor = 8`
    - `pairwisePointerFloatInput`
  - change only the weight-side schedule from:
    - convert directly from `weightLane[pair]` inside each loop iteration
  - to:
    - pre-load `ushort2 rawW0...rawW3`
    - convert from those temporaries
    - reload the next iteration's raw weights after the pointer increment
- rationale:
  - after rejecting address copying from the `6144` family and several input-side transformations, the remaining square-family opportunity might have been in the weight-load/use schedule itself
  - a manual software pipeline could in principle hide some BF16 load latency without changing read width or arithmetic
- result:
  - `ReferenceComparisonTests` full pass
  - known decode drift unchanged: `step 1 = Python 2944 / Metal 859`
  - diagnostics regressed clearly:
    - per-step decode total: `3224 us`
    - `gemv_2048_sq_bf16_argbuf: 1405 us`
    - `gemv_2048_6144_bf16_argbuf: 1256 us`
    - hot exact-shape GEMV total: `2668 us`
    - square role microbench total: `1405 us`
- interpretation:
  - the square-family residual is not improved by manually pipelining the BF16 `ushort2` weight reads
  - the added control/register pressure is worse than the baseline direct convert/use path
  - this branch should remain reverted

## Rejected: Square Interleaved Lane-to-Weight Mapping

The next square-family attempt changed the lane mapping itself instead of the local instruction shape.

- change:
  - `gemv_2048_sq_bf16_argbuf` only
  - keep:
    - row-major layout
    - float staging
    - `unrollFactor = 8`
    - BF16 pairwise reads
  - change the work distribution from:
    - each lane consuming 8 contiguous input elements and 4 contiguous `ushort2` weight pairs
  - to:
    - each lane consuming 4 interleaved `ushort2` weight pairs at `0, 32, 64, 96`
    - and matching interleaved float-input positions at `0/1`, `64/65`, `128/129`, `192/193`
- rationale:
  - after rejecting copied `6144` address tricks, software pipelining, and other input-side changes, the remaining square-family residual might have been in the lane-to-weight mapping itself
  - an interleaved mapping could have improved simdgroup-local read behavior without changing layout or arithmetic
- result:
  - `ReferenceComparisonTests` full pass
  - known decode drift unchanged: `step 1 = Python 2944 / Metal 859`
  - diagnostics regressed badly:
    - per-step decode total: `6942 us`
    - `gemv_2048_sq_bf16_argbuf: 1966 us`
    - `gemv_2048_6144_bf16_argbuf: 1564 us`
    - hot exact-shape GEMV total: `2652 us`
    - square role microbench total: `2203 us`
- interpretation:
  - the square-family bottleneck is not improved by interleaving the lane-to-weight mapping
  - this reshaping destabilizes the broader decode profile even though early correctness still holds
  - the branch should remain reverted
