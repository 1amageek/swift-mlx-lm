# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Goal

swift-mlx-lm は Apple Silicon 上での LLM 推論パッケージ。[AnyFoundationModels](https://github.com/1amageek) の `MLXFoundationModels` バックエンドとして消費される。3つの層で構成される:

1. **SwiftLM** — モデルアーキテクチャを宣言的に記述する DSL と IR。フォーマットにもランタイムにも依存しない
2. **MLXCompiler** — SwiftLM IR を MLX/Metal 上で最適化された推論エンジンにコンパイルする
3. **MLXLM** — 重みの読み込み（GGUF, safetensors 等）、トークナイザ、生成パイプラインを提供する

## Build & Test

```bash
# Build
swift build

# Run all tests (always use xcodebuild — swift test crashes on Metal-dependent tests)
xcodebuild test -scheme swift-mlx-lm-Package -destination 'platform=macOS'

# Run specific module
xcodebuild test -scheme swift-mlx-lm-Package -destination 'platform=macOS' -only-testing:MLXLMTests
xcodebuild test -scheme swift-mlx-lm-Package -destination 'platform=macOS' -only-testing:GGUFParserTests
xcodebuild test -scheme swift-mlx-lm-Package -destination 'platform=macOS' -only-testing:GGUFTokenizerTests
```

**Important**: `swift test` は使わない — Metal metallib が見つからずクラッシュするため。テスト実行は常に `xcodebuild test` を使用すること。

Swift tools version: 6.2. Platforms: macOS 15, iOS 18, visionOS 2.

## Goal: Compiled Inference Path

standard path（`loadContext`）と compiled path（`loadCompiledContext`）の2つのローディングパスがある。compiled path のゴールは以下の通り。

### なぜ compiled path が必要か

standard path は MLXNN Module ツリーを構築してから `model.update(parameters:)` で重みを注入する。これは柔軟だが、Module ツリーの構築コスト・メモリオーバーヘッド・動的ディスパッチが推論時に不要な負担になる。compiled path は SwiftLM IR（`ModelGraph`）からコンパイル時にカーネル選択を行い、MLXNN Module を経由しない軽量な推論エンジン（`MLXLoweredInferenceModel`）を直接生成する。

### ゴール: 振る舞いの同一性

compiled path は standard path の**完全な代替**でなければならない。つまり、`loadCompiledContext()` が返す `ModelContext` は、`loadContext()` が返す `ModelContext` と**下流パイプラインから区別できない**こと。

具体的に、以下の全てが成り立つ必要がある:

1. **`ModelContainer` 互換** — `ModelContainer(context:)` に渡して `generate()` / `prepare()` / `perform()` が standard path と同じ振る舞いをする
2. **`TokenIterator` 互換** — `TokenIterator` が `LanguageModel` / `KVCache` プロトコル経由で正常に動作する
3. **`PrefixCachePool` 互換** — `KVCache.isTrimmable` / `trim()` によるプレフィックスキャッシュの再利用が機能する
4. **`PromptCacheSnapshot` 互換** — `capturePromptCache()` でスナップショットを取得し、`materializePromptCache()` で復元できる
5. **Weight sanitize 同等性** — standard path で `model.sanitize(weights:)` が行う変換（`rotary_emb.inv_freq` の除去、`conv1d.weight` の reshape 等）が compiled path でも等価に適用される

### 現在のギャップ

| # | 問題 | 影響 | 場所 |
|---|------|------|------|
| P1 | compiled path で `sanitize(weights:)` が呼ばれない | Qwen35Model の `conv1d.weight` reshape が未適用。現状は `LoweredDeltaNet` 内で実行時に ndim チェックして暗黙 reshape しているが、他の sanitize 変換が追加された場合に壊れる | `GGUFModelLoader.compileModel()` |
| P2 | `CompiledKVCache.state` が空配列 `[]` を返す | `capturePromptCache()` が空スナップショットを保存する。`materializePromptCache()` に `CompiledKVCache` 分岐がなく復元もできない | `CompiledLanguageModel.swift`, `PromptCacheSnapshot.swift` |
| P3 | `CompiledKVCache.isTrimmable` が `false` | `PrefixCachePool` がトリム不可と判断してキャッシュ再利用を諦め、常に新規キャッシュを生成する | `CompiledLanguageModel.swift` |

### 完了条件

- P1〜P3 が全て解消されている
- compiled path で生成したモデルが `PrefixCachePool` / `PromptCacheSnapshot` を含む全下流パイプラインで standard path と同じ振る舞いをすることがテストで検証されている
- `GGUFCompilableModel` に準拠するモデル型（TransformerModel, Qwen35Model）が compiled path でロードしてトークン生成できることがテストで検証されている

## Architecture

6モジュール構成: GGUFParser → GGUFTokenizer → SwiftLM (IR/DSL) → Models (DSL declarations) → MLXCompiler (IR→推論エンジン) → MLXLM (ロード・生成パイプライン)。外部依存は `mlx-swift` と `swift-jinja` のみ。詳細は `/skeleton` で確認すること。

AnyFoundationModels が `MLXFoundationModels` 経由で消費する。公開インターフェースは `ModelContainer`。

### 設計ルール

- GGUF 単一ファイルで自己完結 — 外部の tokenizer.json / config.json に依存しない（VLM は mmproj GGUF を別途使用）
- `chat_template` 評価が正規のプロンプトフォーマッタ — 手書きのモデル別フォーマッタは作らない
- 設定値は GGUF メタデータから取得する — トークン ID、次元数、正規化パラメータ等をハードコードしない
- 計算グラフが異なる場合のみモデル固有実装を許可（VLM vision encoder 等）。設定値の違いは汎用型でフラグ駆動する
- 全 public 型は `Sendable`
- **量子化された重みを F16 にデクォンタイズしない** — 量子化の利点（メモリ圧縮・高速 matmul）が完全に失われるため。全ての GGUF 量子化型は MLX ネイティブ量子化形式にダイレクトパッキングすること

### 禁止: 量子化 → F16 デクォンタイズ

GGUF の量子化テンソルを F16 に展開する仕組みは**設計上の誤り**であり、存在してはならない。

**なぜ禁止か:**
- 量子化モデル（Q4_K_M 等）を読んでも重みが F16 に膨張し、メモリ使用量が 2〜4 倍になる
- `quantizedMM` を使えず、通常の `matmul` フォールバックになり推論速度が低下する
- 量子化モデルを使う意味そのものが消失する

**正しいアプローチ:**
- 全ての GGUF 量子化型に対して `pack*()` 関数を実装し、MLX の `quantizedMM` が受け付けるネイティブ形式（UInt32 packed weight + F16 scales/biases）に変換する
- MLX が直接サポートしないビット幅（Q3_K → 3-bit 等）は、`quantizedMM` がサポートする最寄りのビット幅にパッキングする
- IQ 系（非線形量子化）は LUT でデコードした後、MLX affine 量子化形式に再パッキングする
- `convertToFloat16()` と `dequantize*()` 関数群は削除し、全パスを `convertDirect()` → `pack*()` に統一する

**例外（F16 が許容されるケース）:**
- 1D テンソル（norm, bias）— 量子化の維持が不要
- F32/BF16 の非量子化テンソル — ダウンキャストは妥当
- LoRA 合成時の一時的なデクォンタイズ — ランク分解に dense weight が必要
- 量子化 KV キャッシュのアテンション計算時 — 計算上の必然

## Weight Loading Flow

GGUF ファイルから重みがロードされ推論カーネルに到達するまでの全体フロー。

### GGUF → MLX ネイティブ量子化パッキング

```
GGUF ファイル
     │
     ▼
GGUFTensorBridge.convertDirect()
     │
     ├── norm/bias tensor (.weight 以外) → F16
     │
     ├── weight matrix + preserveDenseWeights → F16 (quantization: .disabled)
     │
     └── weight matrix + !preserveDenseWeights
              │
              ├── Tier 0: 既存7型 (Q4_0,Q4_1,Q4_K,Q5_K,Q6_K,Q8_0,Q8_1)
              │     → Direct pack (ロスなし)
              │
              ├── Tier 1: 追加6型 (Q5_0,Q5_1,Q8_K,Q2_K,Q3_K,TQ2_0)
              │     → Direct pack (ロスなし)
              │
              ├── Tier 2: LUT型 (IQ4_NL,IQ4_XS)
              │     → LUT decode → 4-bit affine re-quantize
              │
              ├── Tier 3: Grid型 (IQ2_XXS,IQ2_XS,IQ2_S,IQ3_XXS,IQ3_S,IQ1_S,IQ1_M)
              │     → Grid decode → 4-bit affine re-quantize
              │
              └── Tier 4: Ternary型 (TQ1_0)
                    → Ternary decode → 2-bit affine re-quantize
```

### `quantization: .disabled` ゲート

`GGUFModelLoader.loadWeightsWithLoRA()` は `quantization` パラメータで制御される:

```
quantization パラメータ
     │
     ├── .disabled → preserveDenseWeights = true
     │                → convert() → F16 (デバッグ/比較用ベースライン)
     │
     └── .enabled / nil → preserveDenseWeights = false
                          → convertDirect() → ネイティブ量子化パック
```

`preserveDenseWeights` フラグ (`GGUFModelLoader.swift:254`) が `convertDirect()` 呼び出しをゲートする。`.disabled` は「全重みを F16 にする」デバッグパスとして機能する。

### カーネルディスパッチ: groupSize による分岐

MLX `quantizedMM` Metal kernel は `groupSize >= 32` のみサポート。Q2_K/Q3_K/Q6_K は `groupSize = 16` を生成するため、両パスで分岐が必要:

```
                    ConvertedTensor(.quantized)
                             │
                ┌────────────┴────────────┐
                │                         │
         Standard path              Compiled path
                │                         │
                ▼                         ▼
         DirectQuantizedLinear     LoweredProjection.init(storage:)
                │                         │
         ┌──────┴──────┐           ┌──────┴──────┐
         │             │           │             │
    gs >= 32      gs < 32     gs >= 32      gs < 32
         │             │           │             │
         ▼             ▼           ▼             ▼
    quantizedMM   dequantized  .affineQuantized  .dequantizeMatmul
                  + matmul     → quantizedMM     → dequantized + matmul
```

- **Standard path**: `DirectQuantizedLinear.callAsFunction()` が `groupSize` を実行時にチェック
- **Compiled path**: `LoweredProjection.init(storage:)` がコンパイル時に `ProjectionKernel` バリアント（`.affineQuantized` / `.dequantizeMatmul`）を決定。`apply()` は分岐なしでディスパッチ
- **`.dequantizeMatmul`**: 量子化ストレージをメモリ上に保持しつつ、forward pass で一時的にデクォンタイズして `matmul` を実行。メモリ圧縮の利点は維持される
