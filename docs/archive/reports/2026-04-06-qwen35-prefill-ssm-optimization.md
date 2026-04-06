# Qwen3.5-0.8B VLM Prefill: SSM Optimizations

**Date**: 2026-04-06
**Module**: `MetalCompiler` (swift-lm)
**Hardware**: Apple M4 Max (36GB Unified Memory, 546 GB/s bandwidth)
**Model**: Qwen3.5-0.8B (24 layers: 18 DeltaNet + 6 Full Attention)

---

## Background

Qwen3.5-0.8B VLM の E2E prefill が 79 tokens で 28.9s (0.37s/token) だった。LFM2.5 の同等トークン数が ~0.25s であることから、約 117x 遅い。

アーキテクチャ: hidden_size=1024, intermediate_size=3584, 16 heads, dk=128, dv=128, convKernelSize=4, convDimension=6144。24 layers のうち 18 が DeltaNet (SSM recurrence)、6 が Full Attention。

VLM の multimodal prefill はシーケンスをセグメント分割して処理する。各セグメントは 5〜64 tokens 程度の小さなバッチとして独立に dispatch される。

---

## Changes

### 1. SSM Recurrence Conv-Silu Threadgroup Memory Cache

DeltaNet の SSM recurrence kernel は、各 position の処理で conv1d の出力に silu を適用した値 (`conv_silu`) を使う。変更前のカーネルは、各 head が独立に device memory から conv_silu を読み出していた。

```
Before:
  for each head (0..<16):
    for each dk iteration (0..<128):
      read conv_silu[qBase + i] from device memory     ← Q
      read conv_silu[kBase + i] from device memory     ← K
    for each dv iteration (0..<128):
      read conv_silu[vBase + i] from device memory     ← V

  Total reads per position: 16 heads × (128 + 128 + 128) = 6,144
  But conv_silu has only 6,144 unique values → each value read 16 times
  Redundant reads: 6,144 × 15 = 92,160 (from device memory)
```

変更後のカーネルは 3 フェーズ構成に変更。全スレッドが協調して conv_silu を threadgroup memory にプリコンピュートし、head ごとの recurrence は threadgroup memory から読む。

```
After:
  Phase 1: Shift conv state (all threads parallel)
    threadgroup_barrier(mem_device)

  Phase 2: Precompute conv_silu into threadgroup float convSiluCache[6144]
    for (uint c = tid; c < 6144; c += threads):
      convSiluCache[c] = silu(conv_output[c])
    threadgroup_barrier(mem_threadgroup)

  Phase 3: Per-head recurrence reading from convSiluCache
    for each head (tid..<16, stride=threads):
      for dk: read convSiluCache[qBase + i]   ← threadgroup memory
      for dk: read convSiluCache[kBase + i]   ← threadgroup memory
      for dv: read convSiluCache[vBase + i]   ← threadgroup memory
```

Decode kernel (`generateSSMRecurrence`) と prefill kernel (`generateSSMRecurrenceSequence`) の両方に適用。Prefill kernel は position ループ内で毎 position この 3 フェーズを実行する。

**変更ファイル**:

| File | Change |
|------|--------|
| `Sources/MetalCompiler/Fragments/MetalSourceGenerator+ConvAndState.swift` | `generateSSMRecurrence` と `generateSSMRecurrenceSequence` を 3-phase 構成に書き換え。`convDimension` パラメータ追加 |
| `Sources/MetalCompiler/Fragments/Primitives/SSMRecurrenceFragment.swift` | `convDimension` computed property 追加: `2 * groupCount * keyHeadDimension + headCount * valueHeadDimension` |
| `Sources/MetalCompiler/MetalKernelSourceCatalog.swift` | SSM sequence kernel 生成で `ssmFragment.convDimension` を渡すよう変更 |
| `Sources/MetalCompiler/Fragments/MetalSourceGenerator+Library.swift` | Dead-code library 生成にデフォルト `convDimension` を追加 |

### 2. MPP GEMM Registration for Fused Dispatch Entries

Prefill GEMM の MPP (Metal Performance Primitives) matmul2d が有効にならない問題を修正。

`MetalKernelSourceCatalog` の kernel source 生成フェーズで、`.projection` エントリのみが `mppGEMMNames` に GEMM kernel 名を登録していた。しかし `StandardOptimizer` が projection を `.fusedSwiGLUProjection` や `.batchedProjection` に fusion した場合、これらの fused エントリには MPP 登録コードがなく、`mppKernelNames` が空のまま `MetalPipelineCompiler` に渡されていた。

結果: MPP matmul2d kernel のソースは生成されるが、pipeline compile 時に mppKernelNames が空のため base library からのみ compile され、MPP library のビルドがスキップされる。全 GEMM が naive GEMM にフォールバック。

```
Before:
  .projection → mppGEMMNames.insert(gemmName)     ✓
  .fusedSwiGLUProjection → (no registration)       ✗
  .batchedProjection → (no registration)           ✗

After:
  .projection → mppGEMMNames.insert(gemmName)     ✓
  .fusedSwiGLUProjection → mppGEMMNames.insert()  ✓
  .batchedProjection → mppGEMMNames.insert()      ✓
```

**変更ファイル**:

| File | Change |
|------|--------|
| `Sources/MetalCompiler/MetalKernelSourceCatalog.swift` | `.fusedSwiGLUProjection` と `.batchedProjection` の `bufferPrecision == .float32` ブランチに `mppGEMMNames.insert(gemmName)` と `mppGEMMWeightFormat = weightFormat` を追加 |

### 3. MPP Grid Height Tiling Policy

MPP GEMM 有効化後に regression (11.1s → 12.3s) が発生。原因: matmul2d の grid height が compile 時の `maximumSequenceLength` (512) を 64 で割った値 (8 tiles) で固定されていた。5 tokens のセグメントでも 8 tiles 分の dispatch が走り、空 tile が無駄になる。

`sequenceLengthPolicy: .bind(index: 5)` は seqLen を kernel にバインドするだけで、grid height は調整しない。

新しいポリシー `bindAndAdjustGridHeightTiled` を追加。runtime の seqLen で `ceil(seqLen / tileHeight)` を計算し、grid height を動的に調整する。

```
Before:
  grid.height = ceil(maximumSequenceLength / 64) = 8  (fixed at compile time)
  5-token segment → 8 tiles dispatched, 7 tiles empty

After:
  grid.height = ceil(actualSeqLen / 64) at runtime
  5-token segment → ceil(5/64) = 1 tile dispatched
```

**変更ファイル**:

| File | Change |
|------|--------|
| `Sources/MetalCompiler/MetalPrefillPlan.swift` | `PrefillSequenceLengthPolicy` に `.bindAndAdjustGridHeightTiled(index:tileHeight:)` case 追加。`resolvedGridSize` で tiled height 計算を実装 |
| `Sources/MetalCompiler/MetalPrefillStepBuilder.swift` | MPP GEMM ステップの policy を `.bindAndAdjustGridHeightTiled(index: 5, tileHeight: 64)` に変更 |

### 4. SSM Recurrence Multi-Thread-Per-Head Parallelism

Phase 3 (per-head recurrence) のスレッド利用率が 6.25% (16/256) だった問題を修正。

変更前: 1 thread が 1 head の全 dk×dv 計算を担当。numHeads=16, threadgroupSize=256 で 240 スレッドがアイドル。

```
Before:
  for (uint headIndex = tid; headIndex < numHeads; headIndex += tgSize) {
      // 1 thread handles entire dk×dv = 16,384 multiply-accumulate operations
      for (uint idx = 0; idx < dk * dv; ++idx) state[idx] *= decay;
      for (uint d = 0; d < dv; ++d) { /* state update + output */ }
  }
  Thread utilization: 16/256 = 6.25%
```

変更後: dv 次元を threadsPerHead = tgSize/numHeads = 16 スレッドで分割。各スレッドは dv/16 = 8 要素を担当。出力の RMS norm は threadgroup memory を介した 2-pass reduction で計算。

```
After:
  Phase 3a: Parallel recurrence (all 256 threads active)
    threadsPerHead = tgSize / numHeads = 16
    dChunk = dv / threadsPerHead = 8
    for each head (tid / threadsPerHead):
      for d in [dStart, dEnd):  // 8 elements per thread
        state decay, update, output computation
      normPartials[tid] = localNormSq
    threadgroup_barrier(mem_threadgroup)

  Phase 3b: Norm reduction + z-gate (all 256 threads active)
    totalNormSq = sum(normPartials[head * threadsPerHead ..< (head+1) * threadsPerHead])
    rmsScale = rsqrt(totalNormSq / dv + eps)
    for d in [dStart, dEnd):
      output[d] = normed * z * sigmoid(z)

  Thread utilization: 256/256 = 100%
```

Decode kernel (`generateSSMRecurrence`) と prefill kernel (`generateSSMRecurrenceSequence`) の両方に適用。

**変更ファイル**:

| File | Change |
|------|--------|
| `Sources/MetalCompiler/Fragments/MetalSourceGenerator+ConvAndState.swift` | `generateSSMRecurrence` と `generateSSMRecurrenceSequence` の Phase 3 を multi-thread-per-head に書き換え。threadgroup `normPartials[256]` 追加 |

---

## Benchmark Results

### Test Configuration

- **Hardware**: Apple M4 Max (36GB Unified Memory, 546 GB/s bandwidth)
- **Test**: `QwenVisionRealBundleImageTests/realBundleImagePrompt` (1px PNG image, "Describe" prompt)
- **Model**: Qwen3.5-0.8B (HF cache, pre-converted STAF)
- **Profiling**: `SWIFTLM_PROFILE_MULTIMODAL=1`
- **Measurements**: 2 runs, values from 2nd run (1st includes derivedData build)

### Qwen3.5-0.8B VLM E2E Pipeline (Current)

| Phase | Latency | Detail |
|-------|---------|--------|
| prepare (chat template + image preprocess) | 0.053s | |
| executable (vision encoder CPU + Metal compile) | 15.048s | One-time cost per model load |
| **prefill segment 1: text (5 tokens)** | **0.126s** | 0.025s/token |
| **prefill segment 2: image (64 tokens)** | **0.681s** | 0.011s/token |
| **prefill segment 3: text (10 tokens)** | **0.144s** | 0.014s/token |
| **prefill total (79 tokens)** | **0.955s** | **0.012s/token** |

### Optimization Progression (E2E prefill, 79 tokens)

| State | Latency | Per-token | vs Baseline |
|-------|---------|-----------|-------------|
| Baseline (before all changes) | 28.9s | 0.37s/token | — |
| After SSM conv_silu cache | 11.2s | 0.14s/token | **2.6x** |
| After MPP GEMM fix (no tiling) | 12.3s | 0.16s/token | regression |
| After MPP grid tiling fix | 11.2s | 0.14s/token | **2.6x** |
| After conv_silu cache + MPP fixes | 10.933s | 0.138s/token | **2.6x** |
| **After multi-thread-per-head** | **0.955s** | **0.012s/token** | **30.3x** |

### Reference: LFM2.5-1.2B (text-only)

| Metric | Value |
|--------|-------|
| Prefill 16 tokens | 443.2 tok/s (36.1ms) |
| Prefill 32 tokens | 899.5 tok/s (35.6ms) |
| Prefill 64 tokens | 1799.2 tok/s (35.6ms) |
| Decode 50 tokens | 128.0 tok/s (7.82 ms/tok) |
| Bandwidth estimate | 581.2 GB/s |

**Comparison**: LFM2.5 prefills 64 tokens in 35.6ms vs Qwen3.5 image segment (64 tokens) in 0.681s — **19x slower** (was 246x slower before optimization).

### Per-Segment Analysis

| Segment | Tokens | Before | After | Per-token (After) | Per-token-per-layer (After) |
|---------|--------|--------|-------|--------------------|-----------------------------|
| text (pre-image) | 5 | 0.765s | 0.126s | 25.2ms | 1.05ms |
| image | 64 | 8.753s | 0.681s | 10.6ms | 0.44ms |
| text (post-image) | 10 | 1.410s | 0.144s | 14.4ms | 0.60ms |

Per-token-per-layer cost decreased from ~6ms to ~0.5-1ms. Text segments remain slightly slower per-token due to fixed overhead per segment (command buffer creation, CPU-GPU sync).

---

## Analysis

### SSM Conv-Silu Cache が 2.6x 改善をもたらした理由

Qwen3.5-0.8B は 24 layers 中 18 が DeltaNet。各 DeltaNet layer の SSM recurrence は position ごとに `convDimension=6144` 個の conv_silu 値を使う。16 heads が同じ値を device memory から独立に読むため、1 position あたり 92,160 回の冗長読み出しが発生していた。

Threadgroup memory (64KB/threadgroup on Apple Silicon) は device memory より桁違いに低レイテンシ。6,144 × 4B = 24.6KB で threadgroup memory budget 内に収まる。

18 layers × 79 positions × 92,160 冗長読み出し ≈ 1.3 億回の device memory read を threadgroup read に置換。

### Multi-Thread-Per-Head が 11.4x 改善をもたらした理由

Conv-silu cache 後の主要ボトルネックは Phase 3 のスレッド利用率だった。1 head あたり dk×dv = 16,384 回の multiply-accumulate を 1 スレッドが逐次実行していた。16 heads × 1 thread = 16 active threads / 256 total = 6.25% utilization。

dv=128 を threadsPerHead=16 で分割すると:
- 各スレッドの担当: dk×(dv/16) = 128×8 = 1,024 MAC (元の 1/16)
- State decay: 16,384 → 1,024 per thread
- State update (delta computation): 128 → 8 dv iterations per thread
- Output dot product: 128 → 8 per thread
- 全 256 スレッドがアクティブ

RMS norm の計算は threadgroup memory を介した 2-pass reduction を導入:
1. Phase 3a: 各スレッドが localNormSq を normPartials[tid] に書き込み
2. Phase 3b: 各スレッドが自 head の全 partials を読み取り rmsScale を計算、z-gate を適用

追加のコスト:
- threadgroup_barrier 1 回追加 (Phase 3a → 3b 間)
- normPartials[256] の threadgroup memory 1KB

性能改善: 10.933s → 0.955s (11.4x), ベースラインからは 28.9s → 0.955s (30.3x)。

### MPP GEMM が小セグメントで効果を示さない理由

MPP matmul2d は 64×32 タイルで AMX を使う。セグメントサイズが 5-64 tokens の場合、タイル数が 1-1 で AMX のパイプライン効率が上がらない。Naive GEMM も同程度の帯域利用率になる。

MPP の効果が出るのはセグメントサイズが数百 tokens 以上で、タイル並列度が AMX パイプラインを満たす場合。

### 残存ボトルネック

残りの ~0.95s のうち、SSM recurrence の serial dependency (position ごとの逐次処理) と GEMM projection が主要因。各 DeltaNet layer で position 間の並列化ができないため、18 layers × positions の逐次処理が下限を決定する。

Text segments の per-token コストが image segments より高い (25ms vs 11ms) のは、command buffer 作成と CPU-GPU 同期の固定オーバーヘッドが少ないトークン数で分散されるため。

---

## Baseline for Future Optimization

以下を現在のベースラインとする:

| Metric | Value |
|--------|-------|
| Model | Qwen3.5-0.8B VLM (24L: 18 DeltaNet + 6 Full Attention) |
| Input | 1px PNG image + "Describe" prompt |
| Prefill tokens | 79 (5 text + 64 image + 10 text) |
| Prefill total | **0.955s** |
| Per-token | **0.012s/token** |
| Per-token-per-layer | **~0.5ms** |
| Dispatch plan | 184 decode dispatches, 406 prefill steps |
| Hardware | Apple M4 Max (36GB, 546 GB/s) |
| vs original baseline | **30.3x faster** (28.9s → 0.955s) |

今後の最適化候補:

| Candidate | Expected Impact | Rationale |
|-----------|-----------------|-----------|
| Prefill command buffer fusion | Medium | Multimodal prefill が segment ごとに `waitUntilCompleted:true` で同期。fusion で CPU-GPU 同期削減。Text segments の per-token overhead が high (25ms vs 11ms) |
| Prefill barrier optimization | Medium | Decode 側で実施済みの `optimizeDecodeBarrierPolicies` と同等のアプローチを prefill に適用 |
| Segment coalescing | Low-Medium | 隣接テキストセグメントを結合し、GEMM バッチサイズを拡大。固定オーバーヘッド削減 |
| SSM position chunking | Unknown | Position 間の serial dependency を chunk 化して並列度向上。DeltaNet の recurrence は厳密に sequential だが、chunk 単位の並列化は理論的に可能 |
| Metal 4 cooperative tensor | Unknown | 大バッチセグメントで AMX 効率向上 |

---

## Files

| File | Change |
|------|--------|
| `Sources/MetalCompiler/Fragments/MetalSourceGenerator+ConvAndState.swift` | Modified: 3-phase SSM recurrence → multi-thread-per-head Phase 3 (decode + prefill) |
| `Sources/MetalCompiler/Fragments/Primitives/SSMRecurrenceFragment.swift` | Modified: `convDimension` property |
| `Sources/MetalCompiler/MetalKernelSourceCatalog.swift` | Modified: SSM convDimension passthrough + MPP GEMM registration for fused entries |
| `Sources/MetalCompiler/Fragments/MetalSourceGenerator+Library.swift` | Modified: default convDimension |
| `Sources/MetalCompiler/MetalPrefillPlan.swift` | Modified: `bindAndAdjustGridHeightTiled` policy |
| `Sources/MetalCompiler/MetalPrefillStepBuilder.swift` | Modified: MPP step policy |
