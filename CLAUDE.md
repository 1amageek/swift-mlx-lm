# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Goal

swift-mlx-lm は Apple Silicon 上での LLM 推論パッケージ。[AnyFoundationModels](https://github.com/1amageek) の `MLXFoundationModels` バックエンドとして消費される。3つの層で構成される:

1. **SwiftLM** — モデルアーキテクチャを宣言的に記述する DSL と IR。フォーマットにもランタイムにも依存しない
2. **MLXCompiler** — SwiftLM IR を MLX/Metal 上で最適化された推論エンジンにコンパイルする
3. **MLXLM** — HF ディレクトリ（config.json + safetensors + tokenizer.json）からの重み読み込み、トークナイザ、生成パイプラインを提供する

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
  -only-testing:MLXLMTests/ChatTemplateRendererTests \
  -maximum-test-execution-time-allowance 60

# 3b: ModelBundleLoader + IR Assembly
xcodebuild test -scheme swift-mlx-lm-Package -destination 'platform=macOS' \
  -only-testing:MLXLMTests/ModelBundleLoaderTests \
  -maximum-test-execution-time-allowance 60

# 3c: Compiled path
xcodebuild test -scheme swift-mlx-lm-Package -destination 'platform=macOS' \
  -only-testing:MLXLMTests/CompiledKVCacheTests \
  -only-testing:MLXLMTests/CompiledLanguageModelTests \
  -only-testing:MLXLMTests/CompiledPathSanitizeTests \
  -only-testing:MLXLMTests/CompiledKVCacheSnapshotTests \
  -only-testing:MLXLMTests/CompiledPathBinderTests \
  -only-testing:MLXLMTests/CompiledPipelineIntegrationTests \
  -maximum-test-execution-time-allowance 60

# 3d: KVCache + PrefixReuse
xcodebuild test -scheme swift-mlx-lm-Package -destination 'platform=macOS' \
  -only-testing:MLXLMTests/KVCacheTests \
  -only-testing:MLXLMTests/PrefixReuseTests \
  -maximum-test-execution-time-allowance 60

# Step 4: Diagnostic（実モデル使用、最も重い）
xcodebuild test -scheme swift-mlx-lm-Package -destination 'platform=macOS' \
  -only-testing:MLXLMDiagnosticTests -maximum-test-execution-time-allowance 60
```

### ハング検知

- 全テストスイート（`.tags(.unit)` 以外）に `swift-testing-heartbeat` の `.heartbeat` trait を付与済み
- ハング判定は `/tmp/test-heartbeat/` ディレクトリの JSON ファイルで監視可能
- ハングの典型原因: AsyncStream の `continuation.finish()` 漏れ、Metal GPU デッドロック、複数テストの GPU リソース競合

## Goal: HF-First Inference

### ゴール

**HF ディレクトリ（config.json + safetensors + tokenizer.json）だけで推論が完結する**こと。モデル固有の Swift 型は不要。消費者（AnyFoundationModels）が指定するのは**モデル名（HuggingFace repo ID）だけ**。

```
HF ディレクトリ (config.json + *.safetensors + tokenizer.json)
     │
     ▼
ModelBundleLoader (MLXLM)
  ├── HFConfigDecoder: config.json → ModelConfig
  ├── HFArchitectureDetector: model_type → DetectedArchitecture
  ├── IRGraphAssembler: (ModelConfig, DetectedArchitecture) → ModelGraph (IR)
  ├── HFDirectoryBundle: safetensors → WeightManifest → RawWeights
  ├── MLXWeightPathBinder: RawWeights → BoundWeights
  └── MLXInferenceCompiler: (ModelGraph, BoundWeights) → MLXLoweredInferenceModel
           │
           ▼
     CompiledLanguageModel
           │
           ▼
     ModelContainer → TokenIterator → generate()
```

### なぜ HF ディレクトリ駆動か

- **config.json が正規ソース** — 完全なメタデータ。GGUF metadata のような converter 依存の不完全コピーではない
- **モデル固有型が不要** — config.json の `model_type` と構造情報がアーキテクチャの完全な記述。新アーキテクチャ対応は `HFConfigDecoder` + `IRGraphAssembler` のパターン追加で完結する
- **消費者は何も知らなくてよい** — HuggingFace repo ID を渡すだけ。`CompiledModelEntry` や `ModelComponent` の指定は不要
- **mlx-community の事前量子化モデル** — safetensors に MLX ネイティブ形式で格納済み。GGUF 量子化パッキングコード（270KB+）が不要
- **コンパイル時カーネル選択** — 重みのストレージ型を見て `quantizedMatmul` or `matmul` を静的に確定。実行時の型判定が不要
- **IR と実行の分離** — モデル構造（ModelGraph IR）とランタイム（MLXCompiler）が独立。将来のバックエンド追加や最適化がモデル定義に影響しない

### SwiftLM の役割

SwiftLM は2つの目的を持つ:

1. **IR スキーマ** — `ModelGraph`, `OperationKind`, `Attributes` 等の型定義。config.json → IR 変換と MLXCompiler の両方が依存する共通語彙
2. **DSL（ModelComponent）** — 人間が新アーキテクチャを設計・トレーニングする際の宣言的記述。推論ロードパスでは使わない

```
SwiftLM
├── IR スキーマ (ModelGraph, OperationKind, Attributes)
│   ├── ← IRGraphAssembler が IR を構築
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

- `SwiftLM/Declaration` と `MLXLM/Bridge`, `MLXLM/IR` には **family-level の名前だけ** を置く
- 固有名 (`Qwen*`, `Cohere`, `Llama`, `Gemma`, `Mixtral` 等) を持ってよいのは `Sources/Models/` のみ
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
   - HF config decoder support
   - IR lowering / compiler support
   - 必要なら specialized kernel
4. **config.json の必須項目を列挙する**
   - 欠けている値をコードで補完しない
   - HF config の必須キーを明確化する
5. **bridge は family 名で検出・構築する**
   - `IRGraphAssembler` に商品名を持ち込まない
6. **最後に model 宣言へ落とす**
   - `Qwen35` のような固有モデルは family の組み合わせと layer schedule を与えるだけにする

禁止事項:

- 論文にない product-specific wrapper を bridge 層に増やす
- metadata 欠落を convenience default で埋める
- 新しい family を既存モデル名で代表させる

### Component を作るときのフロー

新しい family は、最初に動いた 1 モデルをそのまま一般化して作ってはいけない。`ModelComponent` を書く前に **family spec** を完成させること。

必須フロー:

1. component を書く前に family spec を書く
2. family の可変軸を全部列挙する
3. required metadata と derived value を分離する
4. cache / recurrent state / routing state の shape を先に確定する
5. 可能なら 2 variant 以上で spec を照合する
6. その後に `ModelComponent` / mapper / lowering / kernel を実装する

family spec に必ず含めるもの:

- 必須 metadata key 一覧
- 導出式と invariant
- tensor packing の前提
- head / group / expert / window / state の関係
- cache/state tensor shape
- layer schedule 規則
- gate / norm / positional encoding の意味論

ルール:

- 論文や公式実装に自由度があるなら、最初のモデルでその値が固定に見えても型に載せる
- あるモデルでたまたま一致している 2 つの概念を、family レベルで 1 つに潰さない
- runtime-critical な値を runtime path で推定しない。推定は tooling に閉じ込める
- 「ロードできる」「少し生成できる」を完成条件にしない

新しい family の最低検証項目:

- metadata extraction test
- tensor shape / mapper test
- missing metadata failure test
- variant 間 differential test
- layer-wise activation / logit trace
- compiled path と standard path の parity check（両方ある場合）

避けるべき失敗パターン:

- 小さい variant を family spec の代わりにしてしまう
- product-specific な偶然を family abstraction に埋め込む
- 大きい variant で初めて必要軸が露呈する

variant 間で解釈差が出たら、「どちらかを例外扱いする」のではなく「family spec が未完成」と見なして、足りない軸を type / IR / configuration に追加すること。

### モジュール依存関係

```
SwiftLM (IR + DSL) ────────────────┐
                                   │
ModelDeclarations (depends: SwiftLM)│
  └── DSL モデル宣言               │
      (Qwen35, LFM2, etc.)        │
      ※ トレーニング・設計用途     │
                                   │
MLXCompiler (depends: SwiftLM) ────┤
  └── IR → 推論エンジン            │
                                   │
MLXLM (depends: SwiftLM, MLXCompiler)
  └── ModelBundleLoader: HF config → IR → compile → 推論
      ※ ModelDeclarations には依存しない

GGUFParser ─────────────────────────┐
GGUFTokenizer ──────────────────────┤ (独立ツールモジュール)
GGUFValidation (depends: GGUFParser, MLXLM)
  └── GGUF ファイル検証ツール用
```

### 下流パイプライン互換性

`ModelBundleLoader` が生成する `ModelContext` は全下流パイプラインで動作すること:

1. **`ModelContainer` 互換** — `generate()` / `prepare()` / `perform()` が正常動作
2. **`TokenIterator` 互換** — `LanguageModel` / `KVCache` プロトコル経由で正常動作
3. **`PrefixCachePool` 互換** — `KVCache.isTrimmable` / `trim()` によるキャッシュ再利用が機能
4. **`PromptCacheSnapshot` 互換** — `capturePromptCache()` / `materializePromptCache()` が機能
5. **Weight sanitize** — `rotary_emb.inv_freq` 除去（`WeightSanitizer.filterRotaryEmbeddings`）

## Architecture

コアは3モジュール（SwiftLM + MLXCompiler + MLXLM）。外部依存は `mlx-swift` と `swift-jinja` のみ。詳細は `/skeleton` で確認すること。

| モジュール | 役割 | 依存先 |
|---|---|---|
| SwiftLM | IR スキーマ + DSL | なし |
| ModelDeclarations | DSL モデル宣言（設計・トレーニング用） | SwiftLM |
| MLXCompiler | IR → 推論エンジン | SwiftLM |
| MLXLM | HF ローダー・生成パイプライン | SwiftLM, MLXCompiler |
| GGUFParser | GGUF v2/v3 パーサー（ツール用） | なし |
| GGUFTokenizer | BPE/SPM トークナイザ（ツール用） | GGUFParser |
| GGUFValidation | GGUF ファイル検証（ツール用） | GGUFParser, MLXLM |

**重要**: MLXLM は ModelDeclarations / GGUFParser / GGUFTokenizer に依存しない。config.json → IR は `IRGraphAssembler` が直接構築する。

AnyFoundationModels が `MLXFoundationModels` 経由で消費する。公開インターフェースは `ModelContainer`。消費者が指定するのは HuggingFace repo ID のみ。

### 設計ルール

- HF ディレクトリ（config.json + safetensors + tokenizer.json）が正規ソース — config.json が完全なメタデータを提供する
- `chat_template` 評価が正規のプロンプトフォーマッタ（`ChatTemplateInputProcessor`）— 手書きのモデル別フォーマッタは作らない
- 設定値は config.json から `HFConfigDecoder` 経由で取得する — トークン ID、次元数、正規化パラメータ等をハードコードしない
- **必須項目が config.json に欠けている場合は補完せずエラーにする** — `0.25` のような既定値をコードで埋めてはならない
- 計算グラフが異なる場合のみモデル固有実装を許可（VLM vision encoder 等）。設定値の違いは汎用型でフラグ駆動する
- family-level に抽象化できるものは model-specific 型にしない。新規実装はまず `Family → IR → lowering/kernel` の単位で設計する
- 新アーキテクチャ対応では、論文から `Family → required config keys → IR assembler → lowering/kernel → model declaration` の順で設計する
- 全 public 型は `Sendable`
- mlx-community の事前量子化モデルを使用する — safetensors に MLX ネイティブ量子化形式で格納済み。`quantizedMatmul` が直接使用可能

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

Vision encoder と text decoder は**独立したモデル**として扱う:

```
[Vision Encoder]          [Text Decoder IR]
vision weights            text config.json + safetensors
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

## Weight Loading Flow

HF safetensors から重みがロードされ推論カーネルに到達するまでの全体フロー。

### HF safetensors → MLX 推論エンジン

```
HF ディレクトリ (*.safetensors)
     │
     ▼
HFDirectoryBundle.loadWeights()
     │
     ▼
WeightManifest (MLXArray + quantization info)
     │
     ▼
ModelBundleLoader.convertToRawWeights()
     │
     ├── weight + scales + biases あり → AffineQuantizedTensor
     │     → MLXTensorStorage.affineQuantized
     │
     └── それ以外 → MLXTensorStorage.dense
              │
              ▼
RawWeights → WeightSanitizer.filterRotaryEmbeddings
     │
     ▼
MLXWeightPathBinder.bind() → BoundWeights
     │
     ▼
MLXInferenceCompiler.compile() → MLXLoweredInferenceModel
```

### カーネルディスパッチ

mlx-community の事前量子化モデルは safetensors に MLX ネイティブ形式（weight + scales + biases）で格納されている。`LoweredProjection` がコンパイル時にカーネルを選択:

- **量子化 weight（scales + biases あり）** → `.affineQuantized` → `quantizedMatmul`
- **Dense weight** → `.dense` → `matmul`

## MLXCompiler 実装規則

### 禁止: `MLXFast.rmsNorm` に `1 + weight` を渡す

mlx-swift の `MLXFast.rmsNorm(x, weight: w, eps:)` は `x_normalized * w` を計算する（weight を直接乗算）。Python MLX の `(1 + weight)` 慣例とは異なる。

- **mlx-swift**: `RMSNorm` は weight を **ones** で初期化。`rmsNorm` は weight を**そのまま乗算** → `x_norm * 1.0 = x_norm` (identity)
- **mlx Python**: `RMSNorm` は weight を **zeros** で初期化。`rms_norm` は **`(1 + weight)` を乗算** → `x_norm * (1 + 0) = x_norm` (identity)

GGUF の norm weight は HuggingFace/PyTorch 由来で 1.0 付近の値。`1 + weight` とすると scale が約2倍になり、全層の出力が破壊される。

```swift
// ✗ WRONG — doubles the scale
MLXFast.rmsNorm(x, weight: 1 + weight, eps: eps)

// ✓ CORRECT — mlx-swift uses weight directly
MLXFast.rmsNorm(x, weight: weight, eps: eps)
```

### 禁止: per-head interleaved テンソルの flat split

Q projection が gate を含む場合（`sigmoidPackedInQProj`）、出力 `[B, L, H * 2D]` は per-head interleaved: `[q0, g0, q1, g1, ...]`。flat tensor を半分で分割すると head 境界を跨いで q と gate が混在する。

```swift
// ✗ WRONG — flat split crosses head boundaries
let qDim = queries.dim(-1) / 2
gateValues = queries[0..., 0..., qDim...]

// ✓ CORRECT — reshape per-head then split within each head
let perHead = queries.reshaped(B, L, headCount, 2 * headDim)
gateValues = perHead[0..., 0..., 0..., headDim...].reshaped(B, L, -1)
queries = perHead[0..., 0..., 0..., 0..<headDim].reshaped(B, L, -1)
```

### 参照実装との一致検証

コンパイルパス（`LoweredAttention`, `LoweredDeltaNet` 等）は HuggingFace の参照実装（Python）と**数値的に一致**しなければならない。新しい lowered op を実装する際は:

1. 参照実装の forward pass を読み、各操作の入出力テンソル形状・dtype を確認する
2. テンソルの次元レイアウト（per-head interleaved vs contiguous）を特定する
3. API の慣例（`MLXFast.rmsNorm` の weight 処理等）をソースレベルで検証する — ドキュメントやコメントではなく C++ 実装を確認する
