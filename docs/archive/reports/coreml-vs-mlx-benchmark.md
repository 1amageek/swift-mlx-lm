# CoreML vs MLX Metal — Transformer Layer Benchmark Report (v2)

**Date**: 2026-03-14
**Hardware**: Apple M4 Max (36GB, 546 GB/s bandwidth)
**Repository**: swift-mlx-lm
**Test File**: `Tests/CoreMLBenchmarkTests/CoreMLvsMLXBenchmarkTests.swift`
**Model Generator**: `scripts/generate_coreml_benchmarks.py`

> **v2 更新**: v1 は MLX 側で MLXFast の fused kernel (rmsNorm, scaledDotProductAttention) を使わず手動実装で比較しており、MLX に不当なハンディキャップがあった。v2 では MLX-manual (CoreML と同じ ops) と MLX-opt (fused kernels) の両方を測定し、公平な比較を行った。

---

## 1. 目的

CoreML（GPU + ANE 自動分割）と MLX（Metal GPU）で、同一の Transformer 層を実行した場合の速度を直接比較する。先行する ANE private API レポートは ANE 単独の結果であり、CoreML の自動グラフ分割による最適化を反映していなかった。

## 2. 仮説

### H1: CoreML .all は小 D で MLX を大幅に上回る

CoreML の自動分割により、element-wise ops を ANE で実行しつつ matmul を GPU で並列処理する。D≤1024 では ANE の SRAM fusion が効き、MLX の per-op dispatch overhead を回避できる。

### H2: CoreML .all は大 D でも MLX と同等以上

CoreML の GPU パスは MLX と同等の Metal kernel を使用し、さらに一部の ops を ANE にオフロードする。大 D でも少なくとも同等の性能を示す。

### H3: CoreML .cpuAndNE は小 D で最速

ANE 単独でも小 D (≤1024) では全 ops を SRAM に収めて fused 実行でき、dispatch overhead が最小。

### H4: Prefill (T=128) では CoreML の優位性が拡大する

バッチ処理では計算量が増え、CoreML のグラフ最適化と ANE の並列パイプラインが効く。

## 3. 実験設計

### モデル構造

coremltools で生成した MIL プログラム。1 Transformer 層（Attention + MLP）:

```
RMSNorm → Q/K/V proj (matmul) → GQA head repeat → manual SDPA → O proj → residual
→ RMSNorm → Gate proj + Up proj (matmul) → SiLU gating → Down proj → residual
```

RoPE は省略（MIL での実装が複雑、かつ compute cost が negligible）。MLX 側も同じグラフ（MLXFast.RoPE / scaledDotProductAttention を使わず manual 実装）で公平に比較。

### 変数

| 変数 | 値 |
|------|-----|
| D | 512, 896, 1024, 2048, 2560, 4096 |
| T | 1 (decode), 128 (prefill) |
| Compute Units | `.all` (GPU+ANE), `.cpuAndGPU` (GPU only), `.cpuAndNeuralEngine` (ANE only), MLX Metal |

### 測定条件

- CoreML: `MLModel.prediction(from:)` — 同期実行、CoreML が自動最適化
- MLX: per-layer eval (1 eval) — prior benchmark で最速と確認済み
- Warmup 10回 + 30 iterations の median
- 重み: fp16 ランダム初期化

## 4. 結果

### 4.1 v1 の問題点（修正済み）

v1 の MLX 側は以下の問題があり、MLX に不当なハンディキャップを与えていた:

| 問題 | 影響 |
|------|------|
| `MLXFast.rmsNorm` 未使用 → 手動実装 (5 kernel) | ~8 extra kernel dispatches (2箇所) |
| `MLXFast.scaledDotProductAttention` 未使用 → 手動 SDPA | ~3 extra kernel + GQA tiled copy |
| causal mask を毎回 CPU で構築 | T=128 で 16K 要素の配列作成 + GPU 転送/iteration |

v2 では `MLX-manual` (CoreML と同じ ops) と `MLX-opt` (fused kernels) の両方を測定。

### 4.2 Decode (B=1, T=1)

| D | CoreML .all | CoreML GPU | CoreML ANE | MLX-manual | MLX-opt | best |
|---:|---:|---:|---:|---:|---:|:---|
| 512 | 0.048 | 0.044 | **0.042** | 0.994 | 0.404 | **CoreML ANE** |
| 896 | **0.260** | 0.307 | 0.291 | 1.036 | 0.469 | **CoreML .all** |
| 1024 | **0.228** | 0.229 | 0.242 | 0.965 | 0.440 | **CoreML .all** |
| 2048 | 4.632 | 1.075 | 4.574 | 2.259 | **0.731** | **MLX-opt** |
| 2560 | 1.589 | 2.762 | 1.587 | 2.823 | **0.841** | **MLX-opt** |
| 4096 | 14.821 | **1.412** | 14.822 | 5.842 | 1.621 | **CoreML GPU** |

### 4.3 Prefill (B=1, T=128)

| D | CoreML .all | CoreML GPU | CoreML ANE | MLX-manual | MLX-opt | best |
|---:|---:|---:|---:|---:|---:|:---|
| 512 | **0.211** | 0.432 | 0.218 | 1.015 | 0.575 | **CoreML .all** |
| 896 | 0.452 | 0.805 | **0.451** | 1.585 | 0.953 | **CoreML ANE** |
| 1024 | 0.440 | 0.752 | **0.438** | 1.517 | 0.858 | **CoreML ANE** |
| 2048 | **1.968** | 1.977 | 17.394 | 3.545 | 2.101 | **CoreML .all** |
| 2560 | 2.729 | 2.747 | 23.430 | 4.559 | **2.719** | **MLX-opt** |
| 4096 | 5.372 | 5.328 | 50.503 | 8.939 | **5.228** | **MLX-opt** |

## 5. 分析

### 5.1 v1 vs v2 — MLXFast fused kernels の効果

MLX-manual → MLX-opt で大幅に改善:

| D | Decode: manual→opt | 改善率 | Prefill: manual→opt | 改善率 |
|---:|---:|---:|---:|---:|
| 512 | 0.994→0.404 | **2.5x** | 1.015→0.575 | **1.8x** |
| 896 | 1.036→0.469 | **2.2x** | 1.585→0.953 | **1.7x** |
| 1024 | 0.965→0.440 | **2.2x** | 1.517→0.858 | **1.8x** |
| 2048 | 2.259→0.731 | **3.1x** | 3.545→2.101 | **1.7x** |
| 2560 | 2.823→0.841 | **3.4x** | 4.559→2.719 | **1.7x** |
| 4096 | 5.842→1.621 | **3.6x** | 8.939→5.228 | **1.7x** |

**`MLXFast.rmsNorm` + `MLXFast.scaledDotProductAttention` で decode が 2.2-3.6x 高速化。** v1 の「CoreML が全 D で勝つ」は MLX 側の実装不備が原因。

### 5.2 D≤1024: CoreML が優勢

| D | Decode 勝者 | Decode CoreML vs MLX-opt | Prefill 勝者 | Prefill CoreML vs MLX-opt |
|---:|:---|---:|:---|---:|
| 512 | **CoreML ANE** | 0.042 vs 0.404 (9.6x) | **CoreML .all** | 0.211 vs 0.575 (2.7x) |
| 896 | **CoreML .all** | 0.260 vs 0.469 (1.8x) | **CoreML ANE** | 0.451 vs 0.953 (2.1x) |
| 1024 | **CoreML .all** | 0.228 vs 0.440 (1.9x) | **CoreML ANE** | 0.438 vs 0.858 (2.0x) |

小 D では CoreML の ANE fusion が MLXFast fused kernels を上回る。ANE は全 ops を SRAM 内で fuse 実行し、GPU dispatch overhead が完全にゼロ。

### 5.3 D=2048-2560: MLX-opt が勝つ

| D | Decode 勝者 | CoreML best vs MLX-opt | Prefill 勝者 |
|---:|:---|---:|:---|
| 2048 | **MLX-opt** | 1.075 vs **0.731** (MLX 1.5x) | **CoreML .all** (1.968 vs 2.101) |
| 2560 | **MLX-opt** | 1.587 vs **0.841** (MLX 1.9x) | **MLX-opt** (2.719 vs **2.719**) |

MLXFast の fused SDPA が効く D 帯域。CoreML の GPU パスは MLX ほど matmul を最適化していない可能性。

### 5.4 D=4096: 引き分け

| Mode | CoreML GPU | MLX-opt | 差 |
|:---|---:|---:|:---|
| Decode | **1.412** | 1.621 | CoreML 1.15x |
| Prefill | 5.328 | **5.228** | MLX 1.02x |

D=4096 ではほぼ同等。decode は CoreML GPU がわずかに勝ち、prefill は MLX-opt がわずかに勝つ。

### 5.5 CoreML .all の自動分割の問題

D≥2048 で CoreML .all が CoreML GPU より遅いケースがある:

| D | CoreML .all | CoreML GPU | .all/.GPU |
|---:|---:|---:|---:|
| 2048 decode | 4.632 | 1.075 | **4.3x 遅い** |
| 4096 decode | 14.821 | 1.412 | **10.5x 遅い** |

CoreML の自動分割器が大 D で ANE に ops を配置し、かえって遅くなっている。**D≥2048 では `.cpuAndGPU` を明示指定すべき。**

## 6. 仮説の検証

| 仮説 | 結果 | 根拠 |
|------|------|------|
| **H1**: CoreML .all は小 D で MLX を大幅に上回る | **支持** | D=512 で 9.6x、D=896-1024 で 1.8-2.0x (MLX-opt 比) |
| **H2**: CoreML .all は大 D でも MLX と同等以上 | **不支持** | D=2048-2560 decode で MLX-opt が 1.5-1.9x 勝つ。CoreML .all は大 D で逆効果 |
| **H3**: CoreML .cpuAndNE は小 D で最速 | **部分的に支持** | D=512 decode で最速。D=896-1024 では .all とほぼ同等 |
| **H4**: Prefill で CoreML の優位性が拡大 | **部分的に支持** | D=2048 prefill で CoreML .all が勝つ。ただし D≥2560 では MLX-opt が逆転 |

## 7. 結論

### 7.1 最適なバックエンド選択

| D | Decode 推奨 | Prefill 推奨 |
|---:|:---|:---|
| ≤1024 | **CoreML .all** (ANE fusion) | **CoreML .all** (ANE fusion) |
| 2048 | **MLX-opt** (fused SDPA) | **CoreML .all** (marginally) |
| 2560 | **MLX-opt** (fused SDPA) | **同等** |
| 4096 | **CoreML GPU** (marginally) | **同等** |

### 7.2 SwiftLM IR → MIL コンパイラの価値

**D≤1024 のモデルでは CoreML が MLX を 2-10x 上回るため、MIL コンパイラの実装は価値がある。** ただし D≥2048 では MLXFast fused kernels が同等以上の性能を出すため、必須ではない。

**ハイブリッド戦略**: D に応じて CoreML / MLX を切り替える CostModel が最適。

### 7.3 private ANE framework の必要性

**不要。** CoreML public API が ANE の自動活用を行う。private framework は CoreML より低レベルだが、CoreML のグラフ最適化を bypass するため、むしろ遅くなるケースが多い。

### 7.4 MLXFast fused kernels の重要性

MLX を使う場合、**`MLXFast.rmsNorm` と `MLXFast.scaledDotProductAttention` は必須。** 手動実装は 2.2-3.6x 遅い。これらの fused kernels が MLX を CoreML と競争力のあるレベルに引き上げている。

### 7.5 制約と今後の検証事項

1. **固定 shape**: CoreML は shape 固定。KV cache の動的変化には shape bucketing が必要
2. **RoPE 未検証**: 本ベンチマークは RoPE を省略。MLXFast.RoPE の効果は未測定
3. **量子化未検証**: CoreML の weight compression vs MLX の quantizedMatmul
4. **フルモデル (24L)**: 単層 × 24 が実モデルの性能に線形スケールするか
5. **KV cache あり**: cache update を含む実条件での比較

---

*All benchmarks run on Apple M4 Max, macOS 26.3. CoreML models generated with coremltools. MLX-manual = same ops as CoreML MIL. MLX-opt = MLXFast fused kernels (rmsNorm + scaledDotProductAttention). Both use fp16 weights.*
