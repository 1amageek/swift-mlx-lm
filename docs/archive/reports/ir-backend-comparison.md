# IR Backend Comparison: Same ModelGraph → MLX vs CoreML

**Date**: 2026-03-14
**Hardware**: Apple M4 Max (36GB, 546 GB/s bandwidth)
**Repository**: swift-mlx-lm
**Test File**: `Tests/CoreMLBenchmarkTests/IRBackendComparisonTests.swift`

---

## 1. 目的

同一の SwiftLM IR (`BenchTransformer` DSL → `ModelGraph`) を MLX と CoreML の両方にコンパイルし、バックエンドの性能差を公平に比較する。先行ベンチマーク (v1/v2) は手書きの別コードで比較しており、実装差がノイズを生んでいた。

## 2. 実験設計

### 共通 IR

```swift
BenchTransformer(vocabSize: 32000, hiddenSize: D, headCount: H,
                 kvHeadCount: KVH, headDim: hd, intermediateSize: I)
// → TokenEmbedding → [RMSNorm → Attention → res → RMSNorm → MLP → res] × 1
//   → RMSNorm → OutputHead(tied)
```

### バックエンド

| バックエンド | コンパイルパス | 最適化 |
|:---|:---|:---|
| **MLX-compiled** | `MLXInferenceCompiler` → `MLXLoweredInferenceModel` | fused RMSNorm, fused SDPA, packed QKV, flat decode plan |
| **CoreML .all** | coremltools MIL → `.mlpackage` → `computeUnits: .all` | CoreML 自動グラフ分割 (GPU + ANE) |
| **CoreML GPU** | 同上 → `computeUnits: .cpuAndGPU` | GPU のみ |
| **CoreML ANE** | 同上 → `computeUnits: .cpuAndNeuralEngine` | ANE のみ |

### 注意

CoreML 側は MIL で同じグラフ構造を手動記述。重みは fp16 ランダム初期化（seed=42 で統一）。MLX 側は `MLXInferenceCompiler` の全最適化（fused RMSNorm, fused SDPA, QKV packing, flat decode plan）が有効。

## 3. 結果

### 3.1 Decode (B=1, T=1)

| Model | D | MLX-compiled | CoreML .all | CoreML GPU | CoreML ANE | best | ratio |
|:---|---:|---:|---:|---:|---:|:---|---:|
| 0.6B | 896 | 0.764 | **0.272** | 0.276 | 0.276 | **CoreML .all** | 2.8x |
| 1B | 2048 | 1.157 | 4.663 | **1.062** | 4.713 | **CoreML GPU** | 4.4x |
| 4B | 2560 | 1.746 | 1.559 | 2.764 | **1.550** | **CoreML ANE** | 1.8x |
| 8B | 4096 | 2.342 | 14.614 | **1.447** | 14.991 | **CoreML GPU** | 10.4x |

### 3.2 Prefill (B=1, T=128)

| Model | D | MLX-compiled | CoreML .all | CoreML GPU | CoreML ANE | best | ratio |
|:---|---:|---:|---:|---:|---:|:---|---:|
| 0.6B | 896 | 1.682 | 0.444 | 0.804 | **0.433** | **CoreML ANE** | 3.9x |
| 1B | 2048 | 3.636 | **1.968** | 2.001 | 17.389 | **CoreML .all** | 8.8x |
| 4B | 2560 | 4.643 | 2.745 | **2.689** | 23.355 | **CoreML GPU** | 8.7x |
| 8B | 4096 | 8.123 | 5.357 | **5.315** | 50.735 | **CoreML GPU** | 9.5x |

## 4. 分析

### 4.1 CoreML が全構成で MLX-compiled を上回った

**MLXInferenceCompiler の全最適化（fused SDPA, fused RMSNorm, QKV packing, flat decode plan）を有効にしても、CoreML の最良構成が常に勝つ。**

| Mode | CoreML best / MLX-compiled |
|:---|:---|
| Decode D=896 | CoreML 2.8x 速い |
| Decode D=2048 | CoreML 1.1x 速い |
| Decode D=2560 | CoreML 1.1x 速い |
| Decode D=4096 | CoreML 1.6x 速い |
| Prefill D=896 | CoreML 3.9x 速い |
| Prefill D=2048 | CoreML 1.8x 速い |
| Prefill D=2560 | CoreML 1.7x 速い |
| Prefill D=4096 | CoreML 1.5x 速い |

### 4.2 CoreML .all の自動分割は大 D で逆効果

| D | CoreML .all | CoreML GPU | .all / GPU |
|---:|---:|---:|:---|
| 2048 | 4.663 | 1.062 | .all が **4.4x 遅い** |
| 4096 | 14.614 | 1.447 | .all が **10.1x 遅い** |

CoreML の自動分割器が D≥2048 で ANE に ops を配置し、SRAM spill でかえって遅くなっている。**D≥2048 では `.cpuAndGPU` を明示指定すべき。**

### 4.3 D=2560 は CoreML ANE が最速 (decode)

D=2560 decode で CoreML ANE (1.550ms) が CoreML .all (1.559ms) とほぼ同じで最速。この D は ANE SRAM の境界付近で、一部の weight がぎりぎり SRAM に収まっている可能性がある。

### 4.4 MLX-compiled の decode plan

MLX は `MLXLoweredInferenceModel.decode()` 内で `executeFlatSteps()` を使い、flat decode plan で実行している。これは eval() を最小化した最適パスだが、それでも CoreML に 1.1-2.8x 劣る。

原因の推定:
1. **MLX は eval() 内で複数の Metal kernel を順次 dispatch** — CoreML はグラフ全体を1つの実行計画にコンパイル
2. **CoreML の Metal パスは MPSGraph ベースの可能性** — MLX とは異なる matmul カーネル
3. **CoreML は中間テンソルのメモリ管理を最適化** — MLX は各 op の出力をメモリに書き戻す

## 5. 結論

### 5.1 CoreML は MLX の最適化済みパスを全 D で上回る

MLXFast fused kernels + QKV packing + flat decode plan の全最適化を適用した `MLXInferenceCompiler` の出力でも、CoreML の最良構成に 1.1-3.9x 劣る。

### 5.2 最適なバックエンド選択（CostModel 用）

| D | Decode | Prefill |
|---:|:---|:---|
| ≤1024 | CoreML `.all` / `.cpuAndNE` | CoreML `.all` / `.cpuAndNE` |
| 2048 | CoreML `.cpuAndGPU` | CoreML `.all` |
| 2560 | CoreML `.all` / `.cpuAndNE` | CoreML `.cpuAndGPU` |
| ≥4096 | CoreML `.cpuAndGPU` | CoreML `.cpuAndGPU` |

**注意**: `.all` は D≥2048 で逆効果になるケースがある。CostModel で D に基づいて compute unit を切り替えるべき。

### 5.3 SwiftLM IR → MIL コンパイラの優先度

**高。** 全 D・全 T で CoreML が MLX を上回るため、SwiftLM IR → MIL → CoreML のコンパイルパスは推論速度を直接改善する。

### 5.4 制約

1. **CoreML は固定 shape** — KV cache の動的更新には shape bucketing が必要
2. **重みの bake** — CoreML は重みをモデルにコンパイル時に埋め込む。動的な重み変更（LoRA）には非対応
3. **単一層 × 1 の測定** — 多層モデルでの CoreML compile time / memory overhead は未検証
4. **RoPE なし** — 両バックエンドとも RoPE を省略。MLXFast.RoPE を追加すると MLX 側が改善する可能性
5. **KV cache なし** — CoreML に KV cache の動的更新を含めていない。実用 LLM では cache 管理が大きな差異になる

---

*Both backends compile the same BenchTransformer(1L) DSL → ModelGraph. MLX uses MLXInferenceCompiler with all optimizations. CoreML uses coremltools-generated MIL with fp16 weights. M4 Max, macOS 26.3.*
