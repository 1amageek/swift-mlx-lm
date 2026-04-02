# Phase 1 Report: CoreML Stateful Decoder vs MLX

**Date**: 2026-03-14
**Hardware**: Apple M4 Max (36GB)
**Test**: `CoreMLStatefulTests.swift`

---

## 1. 目的

KV cache 付きの stateful CoreML model で Transformer decoder を実行し、MLX の `MLXLoweredInferenceModel`（全最適化）と比較する。先行ベンチマーク（stateless）では CoreML が MLX を 1.1-3.9x 上回ったが、実用 LLM に必須の KV cache を含む条件での比較は初。

## 2. 実装

### CoreML 側
- `StateTensorSpec` で KV cache を state として定義
- `read_state` → `one_hot` mask blend → `coreml_update_state` で cache 更新
- `range_1d` + `less` で動的 attention mask 生成
- `scaled_dot_product_attention` (fused SDPA) + additive mask
- `canonicalize_inplace_pattern` パスを除外（coremltools バグ回避）

### MLX 側
- `MLXInferenceCompiler` → `MLXLoweredInferenceModel`
- fused RMSNorm (`MLXFast.rmsNorm`)
- fused SDPA (`MLXFast.scaledDotProductAttention`)
- QKV packing, flat decode plan
- `LoweredKVCache` で KV cache 管理

## 3. 結果

### D=896 (Qwen3.5-0.6B scale), 1 layer

| Metric | MLX | CoreML .all | CoreML GPU | CoreML ANE |
|:---|---:|---:|---:|---:|
| Single decode (ms) | 0.780 | 0.771 | 0.766 | **0.694** |
| 10-token sequence (ms) | **7.513** | — | — | 8.388 (.all) |

### D=2048 (Llama-3.2-1B scale)

**CoreML compile error**: `Failed to build the model execution plan (error code: -14)`

## 4. 分析

### 4.1 Stateful model では CoreML と MLX がほぼ同等

**先行ベンチマーク (stateless) との比較:**

| 条件 | D=896 CoreML best / MLX |
|:---|:---|
| Stateless (KV cache なし) | **CoreML 2.8x 勝ち** |
| Stateful (KV cache あり) | **ほぼ同等 (CoreML ANE 1.12x)** |

CoreML の優位性が大幅に縮小した理由:

1. **one_hot mask blend のオーバーヘッド**: cache 更新に `one_hot` → `cast` → `mul` × 2 → `add` が必要。MLX の `LoweredKVCache` は直接スライス代入で済む
2. **動的 attention mask**: `range_1d` + `less` + `cast` で毎回 mask を生成。MLX は `.causal` / `.none` のフラグで済む
3. **全 cache に対する SDPA**: CoreML は `max_seq_len` 全体の cache + mask で SDPA を実行。MLX は `cache[:offset]` だけをスライスして小さいテンソルで SDPA を実行
4. **state の read/write overhead**: CoreML の state は prediction ごとに read → update → write のサイクル

### 4.2 10-token sequence で MLX がわずかに勝つ

MLX 7.51ms vs CoreML 8.39ms — MLX が 11% 速い。これは CoreML の prediction 呼び出しオーバーヘッドが累積するため。MLX は eval() 1回で全 op を lazy 実行できる。

### 4.3 D=2048 で CoreML がコンパイル失敗

error code -14 は CoreML の MPSGraph コンパイラが大きなグラフを処理できないことを示唆。原因候補:
- `one_hot(one_hot_vector_size=512)` × 2 (K, V) のサイズ
- 動的 `range_1d` + `less` の組み合わせ
- 全体のグラフ複雑度

## 5. 結論

### Stateless vs Stateful で結果が根本的に異なる

| | Stateless | Stateful (実用条件) |
|---|---|---|
| CoreML 優位性 | **1.1-3.9x** | **~1.0x (同等)** |
| 原因 | グラフ fusion が効く | KV cache 管理のオーバーヘッドが fusion 効果を打ち消す |
| D=2048+ | CoreML が勝つ | CoreML がコンパイル失敗 |

### 先行レポートの結論を修正

> ~~CoreML が全 D で MLX を上回る~~ → **KV cache なしの条件限定**

**実用的な LLM decode（KV cache + 動的 mask）では CoreML と MLX は同等。** CoreML の graph fusion 優位性は、KV cache 管理の追加コストで相殺される。

### 次のアクション

1. **D=2048+ のコンパイルエラー解決** — `one_hot` vector size を減らすか、cache 更新方式を変更
2. **PyTorch 経由の変換テスト** — MIL Builder 直接生成ではなく、PyTorch `register_buffer` + `torch.jit.trace` 経由で stateful model を生成する方が CoreML の最適化パスが効く可能性
3. **Apple の Llama 3.1 実装方式との比較** — Apple は独自の cache 更新パターンを使っている可能性（公式実装のソースコード確認が必要）

---

*M4 Max, macOS 26.3. CoreML model: coremltools MIL Builder with stateful KV cache. MLX: MLXInferenceCompiler with all optimizations.*
