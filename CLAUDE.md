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
```

**Important**: `swift test` は使わない — Metal metallib が見つからずクラッシュするため。テスト実行は常に `xcodebuild test` を使用すること。

Swift tools version: 6.2. Platforms: macOS 15, iOS 18, visionOS 2.

### テスト実行の原則

**複数モジュールの同時実行はハングする** — Metal/GPU リソースの競合が原因。テストは必ずモジュール単位またはスイート単位で分割して実行すること。

### 実行手順

1. **GPU を使わないモジュールから先に実行する**（高速、ハングリスクなし）
2. **GPU 依存テストはスイート単位で小分けに実行する**
3. **各バッチの完了を確認してから次に進む** — 並列起動しない
4. **必ず `-maximum-test-execution-time-allowance 60` を付ける** — ハング時の無限待ちを防止

### モジュール別実行順序

```bash
# Step 1: GPU 非依存モジュール（高速）
xcodebuild test -scheme swift-mlx-lm-Package -destination 'platform=macOS' \
  -only-testing:GGUFParserTests -maximum-test-execution-time-allowance 60

xcodebuild test -scheme swift-mlx-lm-Package -destination 'platform=macOS' \
  -only-testing:GGUFTokenizerTests -maximum-test-execution-time-allowance 60

xcodebuild test -scheme swift-mlx-lm-Package -destination 'platform=macOS' \
  -only-testing:SwiftLMTests -maximum-test-execution-time-allowance 60

xcodebuild test -scheme swift-mlx-lm-Package -destination 'platform=macOS' \
  -only-testing:ModelsTests -maximum-test-execution-time-allowance 60

# Step 2: GPU 依存モジュール（スイート単位で分割）
xcodebuild test -scheme swift-mlx-lm-Package -destination 'platform=macOS' \
  -only-testing:MLXCompilerTests -maximum-test-execution-time-allowance 60

# Step 3: MLXLMTests — スイート群ごとに分割実行
# 3a: Core（GPU 不要）
xcodebuild test -scheme swift-mlx-lm-Package -destination 'platform=macOS' \
  -only-testing:MLXLMTests/LMInputTests \
  -only-testing:MLXLMTests/ChatMessageTests \
  -only-testing:MLXLMTests/UserInputTests \
  -only-testing:MLXLMTests/GenerateParametersTests \
  -only-testing:MLXLMTests/ModelConfigurationTests \
  -only-testing:MLXLMTests/GenerationTests \
  -only-testing:MLXLMTests/StringOrNumberTests \
  -maximum-test-execution-time-allowance 60

# 3b: GGUFGraphBuilder
xcodebuild test -scheme swift-mlx-lm-Package -destination 'platform=macOS' \
  -only-testing:MLXLMTests/ArchitectureDetectorTests \
  -only-testing:MLXLMTests/ConfigExtractorTests \
  -only-testing:MLXLMTests/IRAssemblerTransformerTests \
  -only-testing:MLXLMTests/IRAssemblerVariantTests \
  -only-testing:MLXLMTests/SlotEnumerationTests \
  -only-testing:MLXLMTests/GGUFGraphBuilderIntegrationTests \
  -maximum-test-execution-time-allowance 60

# 3c: GGUFBridge batch 1
xcodebuild test -scheme swift-mlx-lm-Package -destination 'platform=macOS' \
  -only-testing:MLXLMTests/LlamaTensorNameMapperTests \
  -only-testing:MLXLMTests/TransformerConfigurationGGUFTests \
  -only-testing:MLXLMTests/GGUFTensorBridgeTests \
  -only-testing:MLXLMTests/DirectQuantizedPackingTests \
  -only-testing:MLXLMTests/TransformerModelTests \
  -only-testing:MLXLMTests/ChatTemplateRendererTests \
  -maximum-test-execution-time-allowance 60

# 3d: GGUFBridge batch 2
xcodebuild test -scheme swift-mlx-lm-Package -destination 'platform=macOS' \
  -only-testing:MLXLMTests/GGUFUserInputProcessorTests \
  -only-testing:MLXLMTests/KVCacheTests \
  -only-testing:MLXLMTests/TransformerConfigurationBiasTests \
  -only-testing:MLXLMTests/PromptCacheSnapshotTests \
  -only-testing:MLXLMTests/E2EIntegrationTests \
  -only-testing:MLXLMTests/LoadReportTests \
  -maximum-test-execution-time-allowance 60

# 3e: Compiled path
xcodebuild test -scheme swift-mlx-lm-Package -destination 'platform=macOS' \
  -only-testing:MLXLMTests/CompiledKVCacheTests \
  -only-testing:MLXLMTests/CompiledLanguageModelTests \
  -only-testing:MLXLMTests/CompiledPathSanitizeTests \
  -only-testing:MLXLMTests/CompiledKVCacheSnapshotTests \
  -only-testing:MLXLMTests/CompiledPathBinderTests \
  -only-testing:MLXLMTests/CompiledPipelineIntegrationTests \
  -maximum-test-execution-time-allowance 60

# 3f: LoRA + PrefixReuse
xcodebuild test -scheme swift-mlx-lm-Package -destination 'platform=macOS' \
  -only-testing:MLXLMTests/PrefixReuseTests \
  -maximum-test-execution-time-allowance 60

# 3g: GPU Pack + Benchmark
xcodebuild test -scheme swift-mlx-lm-Package -destination 'platform=macOS' \
  -only-testing:MLXLMTests/GPUPackCorrectnessTests \
  -only-testing:MLXLMTests/PackBenchmarkTests \
  -maximum-test-execution-time-allowance 60

# Step 4: Diagnostic（実モデル使用、最も重い）
xcodebuild test -scheme swift-mlx-lm-Package -destination 'platform=macOS' \
  -only-testing:MLXLMDiagnosticTests -maximum-test-execution-time-allowance 60
```

### ハング検知

- 全テストスイート（`.tags(.unit)` 以外）に `swift-testing-heartbeat` の `.heartbeat` trait を付与済み
- ハング判定は `/tmp/test-heartbeat/` ディレクトリの JSON ファイルで監視可能
- ハングの典型原因: AsyncStream の `continuation.finish()` 漏れ、Metal GPU デッドロック、複数テストの GPU リソース競合

## Goal: GGUF-Driven Inference

### ゴール

**GGUF ファイルだけで推論が完結する**こと。モデル固有の Swift 型（`TransformerModel`, `Qwen35Model` 等）は不要。消費者（AnyFoundationModels）が指定するのは**モデル名だけ**。

```
GGUF ファイル
     │
     ▼
GGUFGraphBuilder (MLXLM)
  ├── テンソル名パターンからアーキテクチャを検出
  │   (blk.N.attn_q.weight → Attention, blk.N.ssm_beta.weight → DeltaNet, etc.)
  ├── GGUF メタデータから設定値を抽出
  │   (hidden_size, num_heads, rope_theta, etc.)
  └── ModelGraph (IR) を直接構築
           │
           ▼
     MLXCompiler
           │
           ▼
     推論エンジン (MLXLoweredInferenceModel)
           │
           ▼
     ModelContainer → TokenIterator → generate()
```

### なぜ GGUF 駆動か

- **モデル固有型が不要** — GGUF テンソル名とメタデータがアーキテクチャの完全な記述。新アーキテクチャ対応は `GGUFGraphBuilder` のパターン追加だけで完結する
- **消費者は何も知らなくてよい** — モデル名（= GGUF ファイルパス）を渡すだけ。`CompiledModelEntry` や `ModelComponent` の指定は不要
- **コンパイル時カーネル選択** — 重みのストレージ型を見て `quantizedMatmul` or `matmul` を静的に確定。実行時の型判定が不要
- **IR と実行の分離** — モデル構造（ModelGraph IR）とランタイム（MLXCompiler）が独立。将来のバックエンド追加や最適化がモデル定義に影響しない

### SwiftLM の役割

SwiftLM は2つの目的を持つ:

1. **IR スキーマ** — `ModelGraph`, `OperationKind`, `Attributes` 等の型定義。GGUF → IR 変換と MLXCompiler の両方が依存する共通語彙
2. **DSL（ModelComponent）** — 人間が新アーキテクチャを設計・トレーニングする際の宣言的記述。推論ロードパスでは使わない

```
SwiftLM
├── IR スキーマ (ModelGraph, OperationKind, Attributes)
│   ├── ← GGUFGraphBuilder が IR を構築
│   └── ← MLXCompiler が IR を消費
│
└── DSL (ModelComponent, @ModelComponentBuilder)
    └── ← Models/ の宣言で使用（トレーニング、設計用途）
```

### Family と Model の境界

新しいモデルが論文とともに導入する新規性は、まず **paper-level architecture family** として捉えること。`Qwen35` や `Cohere` のような商品名・モデル名を DSL や GGUF bridge の抽象名に使ってはいけない。

- **Family**: 今後他モデルでも再利用されうる計算グラフ上の単位
  - 例: `DeltaNet`, `parallel attention + MLP`, `MoE`, `windowed vision transformer`, `full-attention vision transformer`
- **Model**: 具体的な family の組み合わせ・層配置・設定を持つ製品/論文単位
  - 例: `Qwen35`, `Cohere`

ルール:

- `SwiftLM/Declaration` と `MLXLM/Bridge` には **family-level の名前だけ** を置く
- 固有名 (`Qwen*`, `Cohere`, `Llama`, `Gemma`, `Mixtral` 等) を持ってよいのは `Sources/Models/` と `Sources/MLXLM/Models/` のみ
- 新しいモデル対応で追加するのは、まず **新しい family component / mapper / lowering rule** であり、商品名ではない
- `DeltaNet` のような family は `ModelComponent` として明示し、raw string の variant 判定や固有モデル名で隠蔽しない

これは「未知のモデル」ではなく「未知の計算グラフ family」が将来も出る、という前提に基づく設計ルールである。

### 新しいアーキテクチャが出たときにやること

新しい論文・モデルが出た場合は、まず商品名ではなく **論文が導入した計算グラフ上の新規性** を抽出すること。

手順:

1. **論文・公式実装を読む**
   - 新しい層の種類、ブロック構成、キャッシュ形状、位置埋め込み、ルーティング、merge 規則を確認する
2. **既存 family で表現できるか判定する**
   - 既存 `ModelComponent` の組み合わせで表せるなら、新しい product-specific component は作らない
3. **新しい family が必要なら family 単位で追加する**
   - `ModelComponent`
   - GGUF tensor mapper
   - IR lowering / compiler support
   - 必要なら specialized kernel
4. **GGUF/mmproj metadata の必須項目を列挙する**
   - 欠けている値をコードで補完しない
   - exporter 側に追加すべき metadata key を明確化する
5. **bridge は family 名で検出・構築する**
   - `GGUFGraphBuilder` に商品名を持ち込まない
6. **最後に model 宣言へ落とす**
   - `Qwen35` のような固有モデルは family の組み合わせと layer schedule を与えるだけにする

禁止事項:

- 論文にない product-specific wrapper を bridge 層に増やす
- metadata 欠落を convenience default で埋める
- 新しい family を既存モデル名で代表させる

### モジュール依存関係

```
GGUFParser ─────────────────────────────┐
GGUFTokenizer ──────────────────────────┤
SwiftLM (IR + DSL) ────────────────┐    │
                                   │    │
MLXLMComponents (depends: SwiftLM) │    │
  └── DSL ビルディングブロック      │    │
      (Attention, MLP, Norm, etc.) │    │
                                   │    │
Models (depends: MLXLMComponents)  │    │
  └── DSL モデル宣言               │    │
      (Qwen35, LFM2, etc.)        │    │
      ※ トレーニング・設計用途     │    │
                                   │    │
MLXCompiler (depends: SwiftLM) ────┤    │
  └── IR → 推論エンジン            │    │
                                   │    │
MLXLM (depends: SwiftLM, MLXCompiler, GGUFParser, GGUFTokenizer)
  └── GGUFGraphBuilder: GGUF → IR → compile → 推論
      ※ Models / MLXLMComponents には依存しない
```

### 下流パイプライン互換性

GGUF-driven inference が生成する `ModelContext` は全下流パイプラインで動作すること:

1. **`ModelContainer` 互換** — `generate()` / `prepare()` / `perform()` が正常動作
2. **`TokenIterator` 互換** — `LanguageModel` / `KVCache` プロトコル経由で正常動作
3. **`PrefixCachePool` 互換** — `KVCache.isTrimmable` / `trim()` によるキャッシュ再利用が機能
4. **`PromptCacheSnapshot` 互換** — `capturePromptCache()` / `materializePromptCache()` が機能
5. **Weight sanitize 同等性** — `conv1d.weight` reshape、`rotary_emb.inv_freq` 除去等が適用

## Architecture

8モジュール構成。外部依存は `mlx-swift` と `swift-jinja` のみ。詳細は `/skeleton` で確認すること。

| モジュール | 役割 | 依存先 |
|---|---|---|
| GGUFParser | GGUF v2/v3 パーサー | なし |
| GGUFTokenizer | BPE/SPM トークナイザ | GGUFParser |
| SwiftLM | IR スキーマ + DSL | なし |
| MLXLMComponents | DSL ビルディングブロック | SwiftLM |
| Models | DSL モデル宣言（設計・トレーニング用） | MLXLMComponents |
| MLXCompiler | IR → 推論エンジン | SwiftLM |
| MLXLM | GGUF ローダー・生成パイプライン | SwiftLM, MLXCompiler, GGUFParser, GGUFTokenizer |

**重要**: MLXLM は Models / MLXLMComponents に依存しない。GGUF → IR は `GGUFGraphBuilder` が直接構築する。

AnyFoundationModels が `MLXFoundationModels` 経由で消費する。公開インターフェースは `ModelContainer`。消費者が指定するのはモデル名のみ。

### 設計ルール

- GGUF 単一ファイルで自己完結 — 外部の tokenizer.json / config.json に依存しない（VLM は mmproj GGUF を別途使用）
- `chat_template` 評価が正規のプロンプトフォーマッタ — 手書きのモデル別フォーマッタは作らない
- 設定値は GGUF メタデータから取得する — トークン ID、次元数、正規化パラメータ等をハードコードしない
- **必須 metadata が欠けている場合は補完せずエラーにする** — `0.25` のような既定値をコードで埋めてはならない。`ModelComponent` は要求仕様を表現し、欠落値の補完装置ではない
- 計算グラフが異なる場合のみモデル固有実装を許可（VLM vision encoder 等）。設定値の違いは汎用型でフラグ駆動する
- family-level に抽象化できるものは model-specific 型にしない。新規実装はまず `Family → IR → lowering/kernel` の単位で設計する
- 新アーキテクチャ対応では、論文から `Family → required metadata → mapper → lowering/kernel → model declaration` の順で設計する
- 全 public 型は `Sendable`
- **量子化された重みを F16 にデクォンタイズしない** — 量子化の利点（メモリ圧縮・高速 matmul）が完全に失われるため。全ての GGUF 量子化型は MLX ネイティブ量子化形式にダイレクトパッキングすること

### 禁止: Vision Encoder を IR に含める

**現状の問題:**

現在の VLM 実装は vision encoder を text decoder と同一の `ModelGraph` に含めている:

```
parallel(merge: .visionMerge) {
    branch 0: tokenEmbedding → textEmbeddings
    branch 1: visionEncoder → visionFeatures
}
→ decoder blocks → outputHead
```

これにより以下の問題が発生している:

1. **VLM がコンパイルパスを使えない** — `MLXInferenceCompiler` は `visionEncoder` ノードを lowered step に変換できず、VLM は `MLXExecutor`（インタプリタ）に強制される。text-only モデルが得ている最適化（packed QKV projection、fused sub-layer 等）が VLM では全て無効
2. **IR にランタイム概念が混入** — `visionEncoder` はゼロオペランドのソースノードで、実行時にピクセルデータを外部から読む。これは IR の宣言的性質に反する
3. **不要な並列分岐** — `parallel(merge: .visionMerge)` は text embedding と vision encoding を並列実行するが、実際にはシーケンシャルに処理すれば十分

**正しいアプローチ（llama.cpp / Ollama 方式）:**

Vision encoder と text decoder は**独立したモデル**として扱う。GGUF も実際に別ファイル（text GGUF + mmproj GGUF）であり、この設計を IR に反映する:

```
[Vision Encoder]          [Text Decoder IR]
mmproj GGUF               text GGUF
     │                         │
     ▼                         ▼
MLXNN Module              ModelGraph (text-only と同一)
(別モデル)                tokenEmbedding → decoder → outputHead
     │                         │
     ▼                         ▼
vision embeddings         MLXLoweredInferenceModel
     │                    (コンパイルパス使用)
     └──────┐                  │
            ▼                  ▼
      CompiledLanguageModel.prepare()
      → sequential chunk processing
```

**Sequential chunk processing:**

llama.cpp の `llama_batch` と同じ方式。text decoder に対して token IDs または pre-computed embeddings をチャンク単位で渡す:

1. テキストチャンク → `prefill(tokenIDs:)` → embedding lookup + decode
2. ビジョンチャンク → `prefill(embeddings:)` → skip embedding lookup, decode directly
3. テキストチャンク → `prefill(tokenIDs:)` → embedding lookup + decode

各チャンクで KV cache が更新される。text decoder は vision の存在を知らない。

**IR への影響:**

- `OperationKind.visionEncoder` を削除 — IR に属さない
- `ParallelMergeStrategy.visionMerge` を削除 — テキストデコーダに vision 分岐は不要
- `VisionEncoderAttributes`, `VisionMergeConfig` を削除
- VLM の text decoder IR は text-only と完全に同一になる

**実行エンジンへの影響:**

- `MLXLoweredInferenceModel` に `embeddings` 入力を追加（token embedding skip）
- `MLXLoweredInferenceModel` に `positionIds` 入力を追加（M-RoPE 対応）
- `CompiledVisionLanguageModel` + `MLXExecutor` による VLM パスを削除
- `CompiledLanguageModel` が text-only と VLM の両方を処理

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
              ├── Tier 0: 既存7型 (Q4_0,Q4_1,Q4_K,Q5_K,Q8_0,Q8_1,Q8_K)
              │     → Direct pack (ロスなし, groupSize >= 32)
              │
              ├── Tier 0.5: 再量子化3型 (Q6_K,Q2_K,Q3_K)
              │     → Decode → requantizeGroup32 (groupSize=16→32, 微小精度損失)
              │
              ├── Tier 1: 追加3型 (Q5_0,Q5_1,TQ2_0)
              │     → Direct pack (ロスなし, groupSize >= 32)
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

### カーネルディスパッチ: 全量子化型が quantizedMM を使用

全ての GGUF 量子化型は `groupSize >= 32` でパッキングされる。Q2_K/Q3_K/Q6_K は元々 groupSize=16 だが、パッキング時に隣接する2グループ（16要素×2）をデコードし、32要素単位で再量子化する（`requantizeGroup32`）。これにより全型が `quantizedMM` Metal カーネルを使用でき、`dequantize+matmul` フォールバックは不要:

```
                    ConvertedTensor(.quantized)
                             │
                             ▼
                    groupSize >= 32 (全型)
                             │
                             ▼
                      LoweredProjection
                             │
                             ▼
                      .affineQuantized
                       → quantizedMM
```

- **Compiled path**: `LoweredProjection.init(storage:)` がコンパイル時に `.affineQuantized` カーネルを選択。`apply()` は分岐なしでディスパッチ
- **`.dequantizeMatmul`**: `LoweredProjection` に残っているが、全型が groupSize >= 32 を生成するため実質的に未使用。将来の非標準量子化型への安全ネットとして保持
