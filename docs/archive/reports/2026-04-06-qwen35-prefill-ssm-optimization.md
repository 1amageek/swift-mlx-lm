# Qwen3.5-0.8B VLM Prefill: SSM Optimizations

**Date**: 2026-04-06
**Module**: `MetalCompiler` (swift-lm)
**Hardware**: Apple M4 Max (36GB Unified Memory, 546 GB/s bandwidth)
**Model**: Qwen3.5-0.8B (24 layers: 18 DeltaNet + 6 Full Attention)

---

## Archive status update (2026-04-29)

This report is historical. The SSM sequence-kernel work described below is
still useful implementation context, but the reported 0.321s VLM prefill
number must not be read as the current correctness-approved prompt-ingestion
baseline.

The current runtime explicitly falls back to decode-equivalent sequential
prompt ingestion when a prefill plan contains Qwen-style SSM sequence kernels
(`ssm_recurrence_seq_*`). That fallback exists because Qwen's SSM sequence path
has not yet matched the token trace produced by token-by-token decode ingestion.
In other words, the broad "state buffer exists" gate has been narrowed:
BF16 `conv1d_causal_seq` is covered by an LFM short-trace equivalence test, but
Qwen SSM sequence prefill is still disabled until kernel equivalence is fixed
and covered by reference tests.

For current Qwen3.5 text-path throughput, prefer
`docs/benchmarks/qwen3_5-0.8b-mlx-vs-swiftlm.md`. The next valid speed milestone
is not another benchmark number; it is first-token and short-trace equivalence
between Qwen SSM sequence prefill and decode-equivalent ingestion.

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

Phase 3 (per-head recurrence) のスレッド利用率を改善。3 段階で最適化。

#### 4a. dv 次元の並列分割 (256 threads)

変更前: 1 thread が 1 head の全 dk×dv 計算を担当。numHeads=16, threadgroupSize=256 で 240 スレッドがアイドル (6.25%)。

変更後: dv 次元を threadsPerHead = tgSize/numHeads = 16 スレッドで分割。各スレッドは dv/16 = 8 要素を担当。出力の RMS norm は threadgroup memory を介した 2-pass reduction で計算。

```
  threadsPerHead = tgSize / numHeads = 16
  dChunk = dv / threadsPerHead = 8
  Phase 3a: decay + kvmem + update + output (per-thread dChunk)
  Phase 3b: norm reduction + z-gate (threadgroup normPartials)
  Thread utilization: 256/256 = 100%
```

#### 4b. State メモリアクセスパスの融合

変更前: state に対して 4 回の独立パスが存在 (decay, kvmem, update, output dot)。

変更後: d ごとに完全融合。decay+kvmem → delta → update+output を 2 つの dk ループに集約。同じ state cache line を 2 回の dk ループで共有し、L1 キャッシュ局所性を最大化。

```
Before (4 passes through state per position):
  pass 1: state[j*dv+d] *= decay                    // read + write
  pass 2: kvmem += state[j*dv+d] * k                // read
  pass 3: state[j*dv+d] += k * delta                // read + write
  pass 4: dot += state[j*dv+d] * q                  // read
  Total: 4 reads + 2 writes per element

After (2 dk-loops per d, sharing cache lines):
  loop 1: s = state[j*dv+d] * decay; state = s; kvmem += s*k  // read + write
  delta = beta * (v - kvmem)
  loop 2: state[j*dv+d] += k*delta; dot += state*q            // read + write
  Total: 2 reads + 2 writes per element (33% reduction)
```

#### 4c. kInv/qInv の内側ループ外への因数分解

Phase 3 の内側 dk ループ内で kInv と qInv が各反復で冗長に乗算されていた。これをループ外に因数分解。

```
Before (per d value):
  loop 1: kvmem += state * k * kInv   // dk muls of kInv
  loop 2: state += k * kInv * delta   // dk muls of kInv
           dot += state * q * qInv    // dk muls of qInv
  Total: 3 * dk = 384 redundant muls per d

After:
  loop 1: kvmemRaw += state * k       // no kInv
  kvmem = kvmemRaw * kInv             // 1 mul
  kInvDelta = kInv * delta            // 1 mul
  loop 2: state += k * kInvDelta      // 1 combined mul
           dotRaw += state * q         // no qInv
  dot = dotRaw * qInv                 // 1 mul
  Savings: (3 * dk - 3) = 381 muls per d, 48,768 per head, 780K per position
```

#### 4d. Threadgroup サイズ拡大 (256 → 1024)

Prefill kernel の threadgroup サイズを 256 → 1024 に変更。threadsPerHead = 64, dChunk = 2。

Decode kernel の dispatchDimension を `.reduction(dimension: 1)` → `.reduction(dimension: convDimension)` に変更。dimension=1 では 32 スレッド (1 SIMD group) しか割り当てられず、threadsPerHead = 2 だった。convDimension=6144 にすることで 1024 スレッドが割り当てられ、threadsPerHead = 64 に改善。

Threadgroup memory: convSiluCache[6144] (24.6KB) + normPartials[1024] (4KB) = 28.6KB < 32KB limit。

**変更ファイル**:

| File | Change |
|------|--------|
| `Sources/MetalCompiler/Fragments/MetalSourceGenerator+ConvAndState.swift` | Phase 3 を fully-fused per-d 構造に。normPartials[1024] に拡張 |
| `Sources/MetalCompiler/Fragments/Primitives/SSMRecurrenceFragment.swift` | prefill threadgroup size 256→1024, decode dispatchDimension .reduction(1)→.reduction(convDimension) |

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
| **prefill segment 1: text (5 tokens)** | **0.086s** | 0.017s/token |
| **prefill segment 2: image (64 tokens)** | **0.168s** | 0.003s/token |
| **prefill segment 3: text (10 tokens)** | **0.063s** | 0.006s/token |
| **prefill total (79 tokens)** | **0.321s** | **0.004s/token** |

### Optimization Progression (E2E prefill, 79 tokens)

| State | Latency | Per-token | vs Baseline |
|-------|---------|-----------|-------------|
| Baseline (before all changes) | 28.9s | 0.37s/token | — |
| After SSM conv_silu cache | 11.2s | 0.14s/token | **2.6x** |
| After MPP GEMM fix (no tiling) | 12.3s | 0.16s/token | regression |
| After MPP grid tiling fix | 11.2s | 0.14s/token | **2.6x** |
| After conv_silu cache + MPP fixes | 10.933s | 0.138s/token | **2.6x** |
| After multi-thread-per-head (256t) | 0.955s | 0.012s/token | **30.3x** |
| After fused state passes (256t) | 0.708s | 0.009s/token | **40.8x** |
| **After fused + 1024 threads** | **0.321s** | **0.004s/token** | **90.0x** |

### Reference: LFM2.5-1.2B (text-only)

| Metric | Value |
|--------|-------|
| Prefill 16 tokens | 462.3 tok/s (34.6ms) |
| Prefill 32 tokens | 935.2 tok/s (34.2ms) |
| Prefill 64 tokens | 1852.4 tok/s (34.5ms) |
| Decode 50 tokens | 134.5 tok/s (7.43 ms/tok) |

**Comparison**: LFM2.5 prefills 64 tokens in 34.5ms vs Qwen3.5 image segment (64 tokens) in 0.168s — **4.9x slower** (was 246x slower before optimization).

### Per-Segment Analysis

| Segment | Tokens | Original | Multi-thread (256t) | Fused + 1024t | Per-tok-per-layer |
|---------|--------|----------|---------------------|---------------|-------------------|
| text (pre-image) | 5 | 0.765s | 0.126s | **0.086s** | 0.72ms |
| image | 64 | 8.753s | 0.681s | **0.168s** | 0.11ms |
| text (post-image) | 10 | 1.410s | 0.144s | **0.063s** | 0.26ms |

Per-token-per-layer cost for image segment: 0.11ms (was 5.7ms at baseline). Text segments have higher per-token cost due to fixed overhead (command buffer, CPU-GPU sync) amortized over fewer tokens.

---

## Analysis

### SSM Conv-Silu Cache が 2.6x 改善をもたらした理由

Qwen3.5-0.8B は 24 layers 中 18 が DeltaNet。各 DeltaNet layer の SSM recurrence は position ごとに `convDimension=6144` 個の conv_silu 値を使う。16 heads が同じ値を device memory から独立に読むため、1 position あたり 92,160 回の冗長読み出しが発生していた。

Threadgroup memory (64KB/threadgroup on Apple Silicon) は device memory より桁違いに低レイテンシ。6,144 × 4B = 24.6KB で threadgroup memory budget 内に収まる。

18 layers × 79 positions × 92,160 冗長読み出し ≈ 1.3 億回の device memory read を threadgroup read に置換。

### Multi-Thread-Per-Head + Fused Passes + Threadgroup 拡大が 90x 改善をもたらした理由

3 段階の最適化が累積的に効果を発揮:

**段階1: dv 分割 (256 threads)** → 10.933s → 0.955s (11.4x)

Conv-silu cache 後の主要ボトルネックは Phase 3 のスレッド利用率 (6.25%)。dv=128 を 16 スレッドで分割し、100% 利用率を達成。

**段階2: State パス融合** → 0.955s → 0.708s (1.35x)

4 つの独立パス (decay, kvmem, update, output) を 2 つの dk ループに融合。同じ d に対する 2 つのループが同じ state cache line を共有し、L1 ヒット率が向上。State element あたりのメモリアクセスが 6 (4R+2W) → 4 (2R+2W) に削減。

**段階3: Threadgroup 1024** → 0.708s → 0.321s (2.21x)

threadsPerHead を 16 → 64 に拡大、dChunk = 128/64 = 2。各スレッドの計算量が 4x 減少。

Decode kernel は `.reduction(dimension: 1)` で 32 スレッド (1 SIMD group) に制限されていた。dimension を convDimension=6144 に変更し 1024 スレッドを割り当て。decode でも threadsPerHead が 2 → 64 に改善。

### MPP GEMM が小セグメントで効果を示さない理由

MPP matmul2d は 64×32 タイルで AMX を使う。セグメントサイズが 5-64 tokens の場合、タイル数が 1-1 で AMX のパイプライン効率が上がらない。Naive GEMM も同程度の帯域利用率になる。

MPP の効果が出るのはセグメントサイズが数百 tokens 以上で、タイル並列度が AMX パイプラインを満たす場合。

### 残存ボトルネック

残りの ~0.32s の内訳推定:

- 64-token image segment: 0.168s → per-token-per-layer 0.11ms × 64 × 24 = 0.169s (ほぼ全てが計算)
- 5-token text segment: 0.086s → per-token-per-layer ~0.72ms。理想値 (0.11ms × 5 × 24 = 13ms) との差 ~73ms は command buffer overhead + CPU-GPU sync
- 10-token text segment: 0.063s → overhead ~37ms

Text segments の固定オーバーヘッド (~40-70ms/segment) が支配的。Command buffer の作成・コミット・waitUntilCompleted の合計コスト。GEMM projection (Q/K/V, gate/up, down, o_proj × 24 layers) は小行列 (5-64 rows) で bandwidth-bound。

---

## Baseline for Future Optimization

以下を現在のベースラインとする:

| Metric | Value |
|--------|-------|
| Model | Qwen3.5-0.8B VLM (24L: 18 DeltaNet + 6 Full Attention) |
| Input | 1px PNG image + "Describe" prompt |
| Prefill tokens | 79 (5 text + 64 image + 10 text) |
| Prefill total | **0.321s** |
| Per-token | **0.004s/token** |
| Per-token-per-layer | **~0.11ms** (image segment) |
| Dispatch plan | 184 decode dispatches, 406 prefill steps |
| Hardware | Apple M4 Max (36GB, 546 GB/s) |
| vs original baseline | **90x faster** (28.9s → 0.321s) |

### 5. Conv-Shift + Conv-SiLU Fusion (Phase 1+2 Merge)

SSM kernel の Phase 1 (conv state shift) と Phase 2 (conv_silu 計算) を 1 パスに融合。

```
Before (2 phases + 1 device barrier):
  Phase 1: for each channel:
    read convState[base+1..3], shift to [base+0..2], write new to [base+3]
  threadgroup_barrier(mem_device)  ← eliminated
  Phase 2: for each channel:
    re-read convState[base+0..3] from device memory  ← eliminated
    compute sum = Σ convState[k] * weight[k]
    convSiluCache[channel] = sum * sigmoid(sum)
  threadgroup_barrier(mem_threadgroup)

After (1 fused phase):
  for each channel:
    read convState[base+k+1] → register val
    write shifted val to convState[base+k]
    accumulate val * weight[k] in register sum
    write new input to convState[base+3]
    accumulate new * weight[3]
    convSiluCache[channel] = sum * sigmoid(sum)
  threadgroup_barrier(mem_threadgroup)
```

Each thread processes the same channels in both phases, so device memory writes are visible to the same thread's subsequent reads (MSL program order guarantee). The intermediate device barrier is unnecessary.

Savings per position:
- 1 device memory barrier (expensive GPU synchronization point)
- convDimension × convKernelSize = 2560 × 4 = 10,240 device memory reads

For 18 DeltaNet layers × 16 positions: 288 device barriers + 2.95M device reads eliminated.

Decode kernel (`generateSSMRecurrence`) と prefill kernel (`generateSSMRecurrenceSequence`) の両方に適用。

### 6. VLM Embedding Dispatch Skip

VLM multimodal prefill で、全 token に hidden override がある場合 (image segment)、`prefill.embedding` command buffer をスキップ。

```
Before: image segment (16 tokens)
  1. GPU: embedding lookup → hidden buffer     ← wasted (immediately overwritten)
  2. waitUntilCompleted: true                  ← unnecessary sync
  3. CPU: overwrite all hidden rows with vision embeddings

After:
  1. CPU: write vision embeddings to hidden buffer directly
```

64-token image segment (4 chunks of 16): 4 × (GPU dispatch + command buffer round-trip) 削減。

### 7. SSM Prefill Final Position Barrier Skip

Prefill SSM kernel の position loop 最終反復で `threadgroup_barrier(mem_device)` をスキップ。最終 position 後は次の反復がないため、barrier は不要。Command encoder の implicit barrier が cross-dispatch visibility を保証。

18 DeltaNet layers × chunks/segment 分の device barriers 削減。

### 8. Multimodal Chunk Size Increase

VLM multimodal prefill のデフォルト chunk size を増加:
- 64+ tokens: 16 → 32
- 24+ tokens: 8 → 16
- <24 tokens: 4 → 8

Larger chunks amortize per-chunk fixed overhead (command buffer creation, CPU hidden write, GPU sync) without increasing total GPU work. SSM serial loop と GEMM は chunk size に関係なく memory-bandwidth bound。

**変更ファイル (5-8)**:

| File | Change |
|------|--------|
| `Sources/MetalCompiler/Fragments/MetalSourceGenerator+ConvAndState.swift` | Phase 1+2 fusion (both kernels), final position barrier skip |
| `Sources/MetalCompiler/MetalPrefillExecutor.swift` | Skip embedding dispatch when allTokensOverridden |
| `Sources/MetalCompiler/MetalInferenceModel.swift` | Chunk size defaults doubled |

---

今後の最適化候補:

| Candidate | Expected Impact | Rationale |
|-----------|-----------------|-----------|
| Prefill command buffer fusion | Medium | Multimodal prefill が segment ごとに `waitUntilCompleted:true` で同期。fusion で CPU-GPU 同期削減。Text segments の per-token overhead が high (25ms vs 11ms) |
| ~~Prefill barrier optimization~~ | ~~Medium~~ | **実施済み**: offset-aware buffer region tracking で prefill barriers を最適化。DeltaNet layer あたり 4 barriers 削減 (norm + z + b + a projections) |
| Segment coalescing | Low-Medium | 隣接テキストセグメントを結合し、GEMM バッチサイズを拡大。固定オーバーヘッド削減 |
| in_proj_b + in_proj_a 統合 | Low-Medium | 1024→16 の tiny GEMM が 18 DeltaNet 層で 2 回ずつ dispatch。1024→32 に結合すれば dispatch 数半減 |
| in_proj_b + in_proj_a inline in SSM | Medium | SSM kernel 内で beta/alpha projection を直接計算。2 dispatches/layer 削減 × 18 layers。ただし fragment 抽象化の変更が必要 |
| ~~Conv-shift + conv_silu fusion~~ | ~~Low-Medium~~ | **実施済み**: device barrier + redundant reads 削減 |
| Metal 4 cooperative tensor | Unknown | 大バッチセグメントで AMX 効率向上 |

---

## Files

| File | Change |
|------|--------|
| `Sources/MetalCompiler/Fragments/MetalSourceGenerator+ConvAndState.swift` | Modified: 3-phase SSM → fused Phase 1+2, multi-thread Phase 3, kInv/qInv factored, final barrier skip |
| `Sources/MetalCompiler/Fragments/Primitives/SSMRecurrenceFragment.swift` | Modified: `convDimension` property, threadgroup 1024, dispatch dimension fix |
| `Sources/MetalCompiler/MetalPrefillStepBuilder.swift` | Modified: offset-aware prefill barrier optimization |
| `Sources/MetalCompiler/MetalPrefillExecutor.swift` | Modified: skip embedding dispatch for fully-overridden segments |
| `Sources/MetalCompiler/MetalInferenceModel.swift` | Modified: doubled multimodal chunk size defaults |
| `Sources/MetalCompiler/MetalKernelSourceCatalog.swift` | Modified: SSM convDimension passthrough + MPP GEMM registration for fused entries |
| `Sources/MetalCompiler/Fragments/MetalSourceGenerator+Library.swift` | Modified: default convDimension |
| `Sources/MetalCompiler/MetalPrefillPlan.swift` | Modified: `bindAndAdjustGridHeightTiled` policy |
