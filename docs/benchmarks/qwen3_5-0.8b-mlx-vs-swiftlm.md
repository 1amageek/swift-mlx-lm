# Qwen 3.5 0.8B BF16: swift-lm vs mlx-swift-lm throughput

Apple Silicon 上の **同一アーキテクチャ** Qwen 3.5 0.8B BF16 モデルを、swift-lm (独自 Metal コンパイラ)
と mlx-swift (MLX graph compilation) で同一手順で計測した結果。Gemma 4 vs Gemma 3n のような
アーキテクチャのミスマッチはない — 両者ともオリジナル `Qwen/Qwen3.5-0.8B` から派生した BF16 weights。

Date: 2026-04-17
Host: macOS (Apple Silicon, same machine, same session)

## 方法論

両ランナーとも共通:
- **Raw token IDs 入力** (chat template なし) — プロンプトは固定長シーケンス
- **argmax sampling** (temperature = 0, 決定論的)
- **Fresh KV cache per run** — 計測ごとに状態をリセット
- 1 warmup + **3 measured runs, median reported** (σ / σ/μ 併記)
- **Prefill** = 入力プロンプト投入 → 最初のトークン確定 (GPU sync 込み) までの壁時計時間
- **Decode** = 短い prefill + 3-decode warmup の後、`.next()` / `decodeSync` を N 回ループ
  (各ステップで GPU sync を強制)

実装:
- swift-lm: `Tests/MetalCompilerTests/Models/Qwen35/Qwen35BenchmarkTests.swift` (`mlxAlignedBenchmark`)
- mlx-swift: `mlx-swift-lm/IntegrationTesting/IntegrationTestingTests/Qwen35BenchmarkTests.swift`
  (`qwen35Benchmark`)

## モデル

| 項目 | swift-lm | mlx-swift |
|---|---|---|
| Source | `Qwen/Qwen3.5-0.8B` (local HF cache) | `mlx-community/Qwen3.5-0.8B-MLX-bf16` |
| Format | BF16 safetensors (+ STAF cache) | BF16 safetensors (MLX-converted) |
| model_type | `qwen3_5` | `qwen3_5` (Qwen35Model = text-only wrapper of Qwen35TextModel) |
| text_config.hidden_size | 1024 | 1024 |
| text_config.intermediate_size | 3584 | 3584 |
| num_hidden_layers | 24 | 24 |
| layer_types | linear_attention × 18 + full_attention × 6 (3:1 pattern) | 同上 |
| num_attention_heads | 8 | 8 |
| num_key_value_heads | 2 (GQA 4:1) | 2 |
| linear attention (DeltaNet): num_key_heads/num_value_heads | 16 / 16 | 16 / 16 |
| vocab_size | 248320 | 248320 |
| head_dim | 256 | 256 |
| partial_rotary_factor | 0.25 (MRoPE, section [11,11,10]) | 同上 |
| tie_word_embeddings | true | true |
| File size | ~1.7 GB | ~1.7 GB |

**アーキテクチャは両者で完全一致** — Gemma 4 vs Gemma 3n の時のような構造の違いはない。
VLM 由来の vision_tower / model.visual weights は両方とも drop 済み (text path のみ実行)。

## 結果

### Prefill throughput (tok/s — time-to-first-token)

| Prompt len | swift-lm (tok/s) | MLX (tok/s) | MLX / swift-lm | swift-lm ms | MLX ms |
|---:|---:|---:|---:|---:|---:|
|   16 |  335.7 ±0.2% |  390.0 ±2.5% | **1.16×** |  47.66 |  41.03 |
|   32 |  402.7 ±0.2% |  768.6 ±0.3% | **1.91×** |  79.47 |  41.63 |
|   64 |  449.5 ±0.3% | 1550.0 ±2.8% | **3.45×** | 142.37 |  41.29 |
|  128 |  465.2 ±0.1% | 3000.5 ±1.7% | **6.45×** | 275.16 |  42.66 |

- **MLX prefill は length が伸びても ms がほぼ一定 (41–43 ms)** — graph compilation で全 token を単一
  fused dispatch で処理しており、固定オーバーヘッドが支配的。
- **swift-lm prefill は length に対して線形** — per-token ~2.2 ms の per-dispatch コストが支配的。
- 短いプロンプト (len=16) では MLX の固定オーバーヘッドが効いて差は小さい (1.16×) が、
  長いプロンプトほど MLX の graph 単一 dispatch 戦略が圧倒的に強い。

### Decode throughput (tok/s — steady state)

| Steps | swift-lm | MLX | swift-lm / MLX |
|---:|---:|---:|---:|
|  100 | **83.7** tok/s (11.95 ms/tok) ±1.3% | 52.2 tok/s (19.15 ms/tok) ±0.9% | **1.60×** |

- Decode は bandwidth-bound regime であり、swift-lm が勝つ。
- swift-lm の explicit barrier + resource-scoped barrier (writeBufferIndices) が効いていると考えられる。

### 有効メモリ帯域 (decode)

Decode は GEMV 主体で bandwidth-bound。`bytes_per_token = 2 × weight_bytes` の簡易モデル
(weight read + write の 2 passes):

| Runner | weight GB | tok/s | 有効帯域 (GB/s) |
|---|---:|---:|---:|
| swift-lm | 1.7 | 83.7 | **285** |
| MLX      | 1.7 | 52.2 | 178 |

Apple Silicon DRAM ピーク帯域 (~800 GB/s) 比で swift-lm ~35%, MLX ~22%。
Gemma 4 (swift-lm 77% / MLX 32%) と比べると両者とも低い — 0.8B は weight が小さく
dispatch/barrier overhead の比率が上がるため、モデル全体での帯域効率は悪化する。

## 計測のばらつき

- swift-lm: 全メトリクスで σ/μ **< 1.3%** (非常に安定)
- MLX: decode は σ/μ < 1% で安定、prefill は 1–3% (graph cache の hit 状況に依存)
- 初回実行時の MLX は Metal JIT 分散で σ/μ 40% を超える瞬間があるが、2 回目以降 cache warm で安定

## Gemma 4 E2B との違い

[gemma-e2b-mlx-vs-swiftlm.md](gemma-e2b-mlx-vs-swiftlm.md) の結果 (但しアーキテクチャミスマッチあり)
と比較:

| モデル | params | layers | swift-lm decode | MLX decode | swift-lm / MLX |
|---|---:|---:|---:|---:|---:|
| Gemma 4 E2B (BF16, 5B effective) | 5B | 35 | 30.3 tok/s | 14.5 tok/s (Gemma 3n) | **2.09×** |
| **Qwen 3.5 0.8B (BF16)** | 0.9B | 24 | 83.7 tok/s | 52.2 tok/s | **1.60×** |

| モデル | swift-lm prefill (len=128) | MLX prefill (len=128) | swift-lm / MLX |
|---|---:|---:|---:|
| Gemma 4 E2B | 1557 tok/s | 861 tok/s (Gemma 3n) | **1.81×** (swift-lm 勝ち) |
| **Qwen 3.5 0.8B** | 465 tok/s | 3000 tok/s | **0.16×** (MLX 勝ち) |

### Why the prefill flip?

- MLX は length 16→128 で ms がほぼ固定 (41 → 43 ms) — 1 回の graph compile + 全 token 同時 dispatch。
- swift-lm は length に対してほぼ線形 (47 → 275 ms, per-token ~2.2 ms) — per-token の
  dispatch/routing コストが支配的。
- Gemma 4 では layer 数が多い (35 vs 24) + hidden=1536 と大きく、swift-lm の kernel fusion が
  per-token コストを相殺できていた。
- Qwen 3.5 は小さい (layer=24, hidden=1024) ため、swift-lm の fusion のメリットが相対的に小さく、
  per-dispatch 固定コストの比率が上がる。

### Decode は一貫して swift-lm が勝つ

- swift-lm の decode 優位は 1.60–2.09× で一貫している。
- bandwidth-bound regime では swift-lm の resource-scoped barrier + hazardTrackingModeUntracked
  + Q4/BF16 kernel の最適化が効いている。

## 再現手順

### swift-lm
```
cd /Users/1amageek/Desktop/swift-lm
xcodebuild build-for-testing -scheme swift-lm-Package -destination 'platform=macOS' -quiet
xcodebuild test-without-building \
  -scheme swift-lm-Package \
  -destination 'platform=macOS' \
  -only-testing:'MetalCompilerTests/Qwen35BenchmarkTests/mlxAlignedBenchmark()' \
  -parallel-testing-enabled NO
```

Bundle は自動で `~/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots/<hash>/` を探す。
`SWIFTLM_QWEN35_BUNDLE` 環境変数でパスを上書きできる。

### mlx-swift
```
cd /Users/1amageek/Desktop/mlx-swift-lm
xcodebuild build-for-testing \
  -project IntegrationTesting/IntegrationTesting.xcodeproj \
  -scheme IntegrationTesting -destination 'platform=macOS' -quiet
xcodebuild test-without-building \
  -project IntegrationTesting/IntegrationTesting.xcodeproj \
  -scheme IntegrationTesting -destination 'platform=macOS' \
  -only-testing:'IntegrationTestingTests/Qwen35BenchmarkTests/qwen35Benchmark()' \
  -parallel-testing-enabled NO
```

初回は Metal JIT で prefill の σ/μ が大きくなる — 2 回目実行で Metal cache が warm な状態を採用。

## Raw logs

- swift-lm (run 2): `/tmp/bench-qwen35/swiftlm-run2.txt`
- MLX (run 3)    : `/tmp/bench-qwen35/mlx-run3.txt`

## 要約

同一 Qwen 3.5 0.8B BF16 アーキテクチャで、swift-lm と mlx-swift を比較:

- **Decode: swift-lm 1.60×** (83.7 vs 52.2 tok/s) — bandwidth-bound regime で swift-lm 優位
- **Prefill: MLX 1.16×–6.45×** (length 依存) — length が伸びるほど MLX の graph compilation
  (全 token 単一 dispatch) が圧倒的

swift-lm の decode 優位は Gemma 4 (2.09×) → Qwen 3.5 (1.60×) でモデルサイズが小さいほど
縮小するが依然として勝つ。Prefill は逆に MLX が小さいモデルで圧勝 — swift-lm の per-dispatch
コストが amortize しきれないため。

**swift-lm の今後の勝負所**: prefill の dispatch 数削減 (graph 単位 fusion / Metal 4 concurrent
dispatch) で MLX に追随する必要がある。decode path は既に MLX を 1.6× 上回っている。
