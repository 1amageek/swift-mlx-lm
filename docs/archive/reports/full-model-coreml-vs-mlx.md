# Full Model CoreML vs MLX — 全層 1 回実行の公平比較

**Date**: 2026-03-14
**Hardware**: Apple M4 Max (36GB)
**Test**: `FullModelComparisonTests.swift`

---

## 1. 目的

全層を 1 回の実行にまとめた条件で CoreML と MLX を比較する。先行ベンチマークは 1 層単位の比較（per-layer prediction vs per-layer eval）であり、両バックエンドの multi-layer グラフ最適化を正しく反映していなかった。

## 2. 先行ベンチマークの問題

| ベンチマーク | 方法 | 問題 |
|:---|:---|:---|
| v1 (stateless) | 1L CoreML vs 1L MLX (手動 ops) | MLX 側に MLXFast 未使用のハンディキャップ |
| v2 (stateless, 修正) | 1L CoreML vs 1L MLX (MLXFast 使用) | per-layer 比較は multi-layer 最適化を見ない |
| v3 (stateful 1L) | 1L CoreML stateful vs 1L MLX | KV cache overhead で差が消滅 |
| **本ベンチマーク** | **24L CoreML stateful vs 24L MLX** | **正しい比較** |

### 過去レポートからの教訓

`qwen35-ane-comprehensive-analysis.md` (Experiment 3):
```
Metal-only (24L lazy eval): 16.6ms → per-layer 0.69ms
ANE+Metal hybrid:           27.9ms → per-layer 1.16ms (0.60x LOSE)
```

**MLX の lazy eval は multi-layer でグラフを最適化し、per-layer コストを 44% 削減する。** 1 層単位のベンチマークではこの効果が見えない。CoreML の MPSGraph も同様に multi-layer 最適化を行うが、1 層単位の prediction では効果が発揮されない。

→ **正しい比較は全層を 1 回の実行にまとめること。**

## 3. 実験設計

### MLX 側
- `BenchTransformer` DSL → `ModelGraph` → `MLXInferenceCompiler` → `MLXLoweredInferenceModel`
- 全最適化有効: fused RMSNorm, fused SDPA, QKV packing, flat decode plan
- `decode()` → `executeFlatSteps()` → 全層を lazy に実行 → `eval(logits)` で 1 回の GPU dispatch

### CoreML 側
- 全層を 1 つの MIL program に含む `.mlpackage`
- Stateful KV cache (`StateTensorSpec` × 2L states)
- fused SDPA (`scaled_dot_product_attention` MIL op)
- RoPE (cos/sin テーブル + offset 入力)
- Attention mask (one_hot blend + range_1d dynamic mask)
- `prediction(from:using:)` 1 回で全層を実行

### 構成

| 変数 | 値 |
|------|-----|
| D | 896 (Qwen3.5-0.6B scale) |
| H/KVH | 14/2 (GQA) |
| hd | 64 |
| I | 4864 |
| L | 4, 24 |
| max_seq | 256 |
| vocab | 32000 |
| T | 1 (decode) |

## 4. 結果

### 4.1 全層 decode latency

| Model | L | MLX 1-eval | CoreML .all | CoreML GPU | CoreML ANE |
|:---|---:|---:|---:|---:|---:|
| 0.6B | 4 | 1.774 | 1.788 | **1.466** | 1.799 |
| 0.6B | 24 | 8.036 | 4.987 | **4.971** | 7.620 |

### 4.2 Per-layer effective cost

| L | MLX per-layer | CoreML GPU per-layer | CoreML / MLX |
|---:|---:|---:|---:|
| 4 | 0.444 ms | 0.366 ms | **1.21x CoreML** |
| 24 | 0.335 ms | 0.207 ms | **1.62x CoreML** |

### 4.3 Multi-layer optimization 効果

| | 4L → 24L per-layer 改善 | 改善率 |
|---|---:|---:|
| MLX | 0.444 → 0.335 ms | **25% 削減** |
| CoreML GPU | 0.366 → 0.207 ms | **43% 削減** |

**CoreML の multi-layer 最適化が MLX より強力。** 層数が増えるほど CoreML の優位が拡大する。

## 5. 分析

### 5.1 なぜ CoreML が 1.62x 速いか

```
MLX (24L, 1 eval):
  eval() → 24L × ~14 Metal kernel = ~336 kernel を command buffer に encode
  GPU パイプライン内で最適化 (MLX の lazy eval)
  → 各 kernel は独立した Metal dispatch
  → 中間テンソルは Global Memory 経由

CoreML GPU (24L, 1 prediction):
  MPSGraph が 24L 全体を 1 つの実行計画にコンパイル
  → kernel fusion (隣接する element-wise ops を 1 kernel に)
  → メモリプランニング (中間バッファの再利用・aliasing)
  → 全体のスケジューリング最適化
```

### 5.2 CoreML の multi-layer 最適化がなぜ MLX より強いか

**MLX**: lazy eval は op-level fusion のみ。matmul 境界を跨ぐ fusion はしない。24L でも各 matmul は独立した Metal kernel。改善は GPU command buffer のバッチ submit による dispatch overhead 削減が主。

**CoreML/MPSGraph**: graph compiler が全体の data flow を分析。同じ shape の中間バッファを複数層で再利用（memory aliasing）。隣接する element-wise ops（RMSNorm の mul/rsqrt/mul、SiLU の sigmoid/mul）を fuse。結果として、層数が増えるほど最適化の余地が増え、per-layer コストが下がる。

### 5.3 CoreML ANE が 24L で遅い理由

CoreML .all (4.987ms) ≈ CoreML GPU (4.971ms) → 24L では CoreML は事実上 GPU のみを使用。

CoreML ANE (7.620ms) が遅いのは:
- 24L × weight が ANE SRAM (~32MB) に収まらない
- DRAM spill が発生し、weight streaming のオーバーヘッドが蓄積
- ANE は小モデル（≤1B, D≤1024）でのみ有効

## 6. 先行レポートとの統合

| レポート | 条件 | 結果 | 正しさ |
|:---|:---|:---|:---|
| `analysis-phase1-2.md` | ANE fused FFN (isolated) | ANE 2-12x | 正しい（isolated 条件） |
| `qwen35-ane-comprehensive-analysis.md` | ANE+Metal hybrid 24L | 0.60x LOSE | 正しい（graph flush が原因） |
| `final-ane-feasibility-report.md` | ANE private API 全体 | 4B+ で ANE 不可 | 正しい（private API 限定） |
| `metal-fusion-boundary-benchmark.md` | eval() 粒度 × D | eval 回数が支配的 | 正しい（MLX 内部最適化） |
| `coreml-vs-mlx-benchmark.md` v2 | 1L stateless | CoreML 1.1-3.9x | **バイアスあり**（per-layer 比較） |
| `coreml-stateful-decoder-phase1.md` | 1L stateful | 同等 | **不完全**（per-layer 比較） |
| **本レポート** | **24L stateful, 全層 1 実行** | **CoreML 1.62x** | **正しい比較** |

### 教訓

**per-layer ベンチマークは model-level の性能を予測できない。** これは `qwen35-ane-comprehensive-analysis.md` の Finding #1 と同じ教訓:

> "Per-op / per-layer benchmarks はモデルレベルの性能を予測できない — Metal lazy eval が 44% の効率化をもたらし、これは per-layer 計測では見えない"

CoreML でも同様。per-layer CoreML prediction は MPSGraph の multi-layer 最適化を反映しない。

## 7. 結論

### 7.1 CoreML は全層 1 実行で MLX を 1.62x 上回る

KV cache 付き stateful model で、全 24 層を 1 回の prediction で実行した場合、CoreML GPU が MLX の lazy eval を 1.62x 上回る。これは MPSGraph のグラフコンパイラが MLX の op-at-a-time dispatch より効率的にメモリと kernel をスケジューリングするため。

### 7.2 層数が増えるほど CoreML の優位が拡大

4L: 1.21x → 24L: 1.62x。CoreML の multi-layer 最適化は per-layer コストを 43% 削減するのに対し、MLX は 25% 削減にとどまる。実用モデル（24-64L）では CoreML の優位がさらに拡大する可能性。

### 7.3 CoreML バックエンドの実装は価値がある

Phase 1 のレポートで「KV cache 付きでは同等」と結論したのは 1 層単位の比較だったため。全層を 1 つの model に含めることで CoreML の真の性能が発揮される。

### 7.4 次のステップ

1. **D=2048 以上の検証** — 24L D=2048 の full model を生成・ベンチマーク
2. **Prefill (T=128) の比較** — decode だけでなく prefill でも全層 1 実行で比較
3. **10-token decode sequence** — 連続 decode でのスループット (tok/s) 比較
4. **Phase 2: IR → MIL コンパイラ** — 任意の ModelGraph から full stateful model を自動生成

---

*M4 Max, macOS 26.3. CoreML: coremltools MIL Builder + stateful KV cache. MLX: MLXInferenceCompiler with all optimizations. Both execute all 24 layers in a single dispatch (eval/prediction).*
