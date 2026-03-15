# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Goal

swift-lm は Apple Silicon 上での高速 LLM 推論パッケージ。[AnyFoundationModels](https://github.com/1amageek) の `MLXFoundationModels` バックエンドとして消費される。3つの層で構成される:

1. **SwiftLM** — モデルアーキテクチャを宣言的に記述する DSL と IR。フォーマットにもランタイムにも依存しない
2. **LMCompiler** — SwiftLM IR を MLX/MPSGraph 上で最適化された推論エンジンにコンパイルする
3. **LMInference** — HF ディレクトリからの重み読み込み、トークナイザ、生成パイプライン、MPSGraph / MLX バックエンド選択を提供する

### 推論バックエンド

MPSGraph と MLX は**独立したバックエンド**。それぞれ個別に選択可能。

```
ModelBundleLoader(backend:)
  ├── .mpsgraph: MPSGraphInferenceCompiler → MPSGraphInferenceModel
  │   → MPSGraph グラフコンパイラ。全層を fused 実行計画に
  └── .mlx: MLXInferenceCompiler → MLXInferenceModel
      → fused RMSNorm, fused SDPA, QKV packing, flat decode plan。全最適化済み
```

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
xcodebuild test -scheme swift-lm-Package -destination 'platform=macOS' \
  -only-testing:SwiftLMTests -maximum-test-execution-time-allowance 60

xcodebuild test -scheme swift-lm-Package -destination 'platform=macOS' \
  -only-testing:ModelsTests -maximum-test-execution-time-allowance 60

# Step 2: GPU 依存モジュール（スイート単位で分割）
xcodebuild test -scheme swift-lm-Package -destination 'platform=macOS' \
  -only-testing:LMCompilerTests -maximum-test-execution-time-allowance 60

# Step 3: LMInferenceTests — スイート群ごとに分割実行
# 3a: Core（GPU 不要）
xcodebuild test -scheme swift-lm-Package -destination 'platform=macOS' \
  -only-testing:LMInferenceTests/LMInputTests \
  -only-testing:LMInferenceTests/ChatMessageTests \
  -only-testing:LMInferenceTests/UserInputTests \
  -only-testing:LMInferenceTests/GenerateParametersTests \
  -only-testing:LMInferenceTests/ModelConfigurationTests \
  -only-testing:LMInferenceTests/GenerationTests \
  -only-testing:LMInferenceTests/StringOrNumberTests \
  -only-testing:LMInferenceTests/ChatTemplateRendererTests \
  -maximum-test-execution-time-allowance 60

# 3b: ModelBundleLoader + IR Assembly
xcodebuild test -scheme swift-lm-Package -destination 'platform=macOS' \
  -only-testing:LMInferenceTests/ModelBundleLoaderTests \
  -maximum-test-execution-time-allowance 60

# 3c: Compiled path
xcodebuild test -scheme swift-lm-Package -destination 'platform=macOS' \
  -only-testing:LMInferenceTests/CompiledKVCacheTests \
  -only-testing:LMInferenceTests/CompiledLanguageModelTests \
  -only-testing:LMInferenceTests/CompiledPathSanitizeTests \
  -only-testing:LMInferenceTests/CompiledKVCacheSnapshotTests \
  -only-testing:LMInferenceTests/CompiledPathBinderTests \
  -only-testing:LMInferenceTests/CompiledPipelineIntegrationTests \
  -maximum-test-execution-time-allowance 60

# 3d: KVCache + PrefixReuse
xcodebuild test -scheme swift-lm-Package -destination 'platform=macOS' \
  -only-testing:LMInferenceTests/KVCacheTests \
  -only-testing:LMInferenceTests/PrefixReuseTests \
  -maximum-test-execution-time-allowance 60

# Step 4: Diagnostic（実モデル使用、最も重い）
xcodebuild test -scheme swift-lm-Package -destination 'platform=macOS' \
  -only-testing:LMInferenceDiagnosticTests -maximum-test-execution-time-allowance 60
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
     ├── model_type 抽出
     │
     ▼
ModelRegistry.resolve(modelType, config)
     │
     ├── Known ("qwen3_5") → Qwen35(config).makeModelGraph()    ← 検証済み
     ├── Known ("lfm2")    → LFM2(config).makeModelGraph()      ← 検証済み
     ├── Known ("llama")   → Transformer(config).makeModelGraph()
     │
     └── Unknown           → AnyModel(config).makeModelGraph()   ← ベストエフォート
     │
     ▼
ModelGraph (IR) + WeightNameMapper
     │
     ├── [MPSGraph path]
     │   └── MPSGraphInferenceCompiler → MPSGraphInferenceModel → MPSGraphLanguageModel
     │
     └── [MLX path]
         ├── WeightNameMapper.manifest(graph) → [SlotManifestEntry]
         ├── MLXWeightPathBinder(manifest).bind(rawWeights) → BoundWeights
         └── MLXInferenceCompiler → MLXInferenceModel → MLXLanguageModel
              │
              ▼
        ModelContainer → TokenIterator → generate()
```

### config.json と safetensors の役割

config.json と safetensors はそれぞれ異なる役割を持ち、両方からモデルを構築する:

- **config.json → モデルの構造** — hidden_size、layer_types、num_attention_heads 等の構造パラメータ。IR（ModelGraph）を構築するために必須
- **safetensors → weight の実体** — どのテンソルが存在するかの正規ソース。weight binding に使う

config.json だけではモデルの完全な記述にならない。例: LFM2 の config.json に `use_qk_norm` フラグがないが、safetensors には `q_layernorm.weight` が存在する。**safetensors のテンソルキーが weight 構造の正規ソース。**

設計ルール:
- IR の weight slot はオプショナルに宣言する（QK norm 等）
- safetensors にテンソルがあればバインド、なければスキップ
- 必須 weight が safetensors に無い場合はエラー
- config.json のフラグで weight slot の有無を決めない — safetensors が正規ソース

### なぜ HF ディレクトリ駆動か

- **消費者は何も知らなくてよい** — HuggingFace repo ID を渡すだけ
- **mlx-community の事前量子化モデル** — safetensors に MLX ネイティブ形式で格納済み。`quantizedMatmul` が直接使用可能
- **コンパイル時カーネル選択** — 重みのストレージ型を見て `quantizedMatmul` or `matmul` を静的に確定。実行時の型判定が不要
- **IR と実行の分離** — モデル構造（ModelGraph IR）とランタイム（LMCompiler）が独立。バックエンド追加がモデル定義に影響しない

### SwiftLM の役割

SwiftLM は3つの目的を持つ:

1. **IR スキーマ** — `ModelGraph`, `OperationKind`, `Attributes` 等の型定義。ModelComponent と LMCompiler の両方が依存する共通語彙
2. **DSL（ModelComponent）** — モデルアーキテクチャの宣言的記述。推論パスで `makeModelGraph()` により IR を構築する
3. **ModelConfig** — config.json の構造パラメータを保持する汎用型。全モジュールが共有

```
SwiftLM
├── IR スキーマ (ModelGraph, OperationKind, Attributes)
│   └── ← LMCompiler が IR を消費
├── DSL (ModelComponent, @ModelComponentBuilder)
│   └── ← Models/ の宣言 + AnyModel で使用。推論パスで IR を構築
└── ModelConfig
    ├── ← HFConfigDecoder が config.json から構築
    └── ← Models/ が直接受け取り ModelComponent を構成
```

### Family と Model の境界

新しいモデルが論文とともに導入する新規性は、まず **paper-level architecture family** として捉えること。`Qwen35` や `Cohere` のような商品名・モデル名を DSL や bridge の抽象名に使ってはいけない。

- **Family**: 今後他モデルでも再利用されうる計算グラフ上の単位
  - 例: `DeltaNet`, `parallel attention + MLP`, `MoE`, `windowed vision transformer`, `full-attention vision transformer`
- **Model**: 具体的な family の組み合わせ・層配置・設定を持つ製品/論文単位
  - 例: `Qwen35`, `Cohere`

ルール:

- `SwiftLM/Declaration` と `LMInference/Bridge`, `LMInference/IR` には **family-level の名前だけ** を置く
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
LMCompiler (depends: SwiftLM) ─────┤
  ├── MLXInferenceCompiler         │
  └── MPSGraphInferenceCompiler    │
                                   │
LMInference (depends: SwiftLM, LMCompiler)
  ├── ModelBundleLoader: HF config → IR → backend
  │     ├── .mpsgraph: MPSGraphInferenceCompiler → MPSGraphLanguageModel
  │     └── .mlx: MLXInferenceCompiler → MLXLanguageModel
  └── ※ ModelDeclarations には依存しない
```

### 下流パイプライン互換性

`ModelBundleLoader` が生成する `ModelContext` は全下流パイプラインで動作すること:

1. **`ModelContainer` 互換** — `generate()` / `prepare()` / `perform()` が正常動作
2. **`TokenIterator` 互換** — `LanguageModel` / `KVCache` プロトコル経由で正常動作
3. **`PrefixCachePool` 互換** — `KVCache.isTrimmable` / `trim()` によるキャッシュ再利用が機能
4. **`PromptCacheSnapshot` 互換** — `capturePromptCache()` / `materializePromptCache()` が機能
5. **Weight sanitize** — `rotary_emb.inv_freq` 除去（`WeightSanitizer.filterRotaryEmbeddings`）

## Architecture

コアは3モジュール（SwiftLM + LMCompiler + LMInference）。外部依存は `mlx-swift` と `swift-jinja` のみ。詳細は `/skeleton` で確認すること。

| モジュール | 役割 | 依存先 |
|---|---|---|
| SwiftLM | IR スキーマ + DSL | なし |
| ModelDeclarations | DSL モデル宣言（設計・トレーニング用） | SwiftLM |
| LMCompiler | IR → 推論エンジン（MLX + MPSGraph） | SwiftLM |
| LMInference | HF ローダー・生成パイプライン・バックエンド選択 | SwiftLM, LMCompiler |

**重要**: LMInference は ModelDeclarations に依存する。config.json → ModelConfig → ModelComponent.makeModelGraph() で IR を構築する。

### 新しい Model を追加するとき

**必ず `OperationKind` を確認してから実装する。** DSL コンポーネント（`Sources/SwiftLM/Declaration/Components/`）と IR の `OperationKind`（`Sources/SwiftLM/IR/ModelGraph.swift`）は 1:1 で対応する。対応するコンポーネントが無い `OperationKind` を使う場合はコンパイラが未対応でエラーになる。

手順:

1. **`OperationKind` を確認** — 使いたい計算ノードに対応する case があるか確認する（`attention`, `mlp`, `moe`, `stateSpace`, `shortConv` 等）
2. **DSL コンポーネントを確認** — 対応する `ModelComponent`（`Attention`, `MLP`, `MoE`, `StateSpace`, `ShortConv` 等）が `Sources/SwiftLM/Declaration/Components/` に存在するか確認する
3. **無ければ作る** — `PrimitiveComponent` に準拠した新コンポーネントを作成し、`operationKind` で正しい `OperationKind` case を返す
4. **`StateSpace` の variant に新しい文字列を入れて別の operation を代用しない** — `stateSpace(variant: "short_conv")` のように別の operation kind を誤用すると、コンパイラが未知の variant としてエラーを出す

### 推論バックエンドの使い分け

MPSGraph と MLX は独立したバックエンド。消費者が明示的に選択する。

| バックエンド | 利点 | 制約 |
|---|---|---|
| **MPSGraph** | MPSGraph fused execution, compiled graph plan | Pure Swift、Python 不要 |
| **MLX** | 動的 shape、LoRA、量子化、custom kernels | op-at-a-time dispatch |

```swift
// 使い方
let container = try await ModelBundleLoader().load(repo: "...", backend: .mpsgraph)
let container = try await ModelBundleLoader().load(repo: "...", backend: .mlx)
```

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
vision embeddings         MLXInferenceModel
     │                    (コンパイルパス使用)
     └──────┐                  │
            ▼                  ▼
      MLXLanguageModel.prepare()
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

- `MLXInferenceModel` に `embeddings` 入力を追加（token embedding skip）
- `MLXInferenceModel` に `positionIds` 入力を追加（M-RoPE 対応）
- `CompiledVisionLanguageModel` + `MLXExecutor` による VLM パスを削除
- `MLXLanguageModel` が text-only と VLM の両方を処理

## Weight Loading Flow

### MPSGraph パス

```
HF ディレクトリ (config.json + *.safetensors)
     │
     ▼
ModelBundleLoader → IR assembly → MPSGraphInferenceCompiler
     │
     ▼
MPSGraphInferenceModel → MPSGraphLanguageModel
```

### MLX パス

```
HF ディレクトリ (*.safetensors)
     │
     ▼
HFDirectoryBundle.loadWeights() → WeightManifest
     │
     ▼
ModelBundleLoader.convertToRawWeights()
     │
     ├── weight + scales + biases → AffineQuantizedTensor → quantizedMatmul
     └── それ以外 → dense → matmul
     │
     ▼
WeightSanitizer.filterRotaryEmbeddings → MLXWeightPathBinder.bind()
     │
     ▼
MLXInferenceCompiler.compile() → MLXInferenceModel
```

MLX パスは全最適化済み: fused RMSNorm (`MLXFast.rmsNorm`), fused SDPA (`MLXFast.scaledDotProductAttention`), QKV packing, Gate+Up packing, flat decode plan, 全層 lazy eval (1 eval)。

## LMCompiler 実装規則

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

### Lowered Op 実装の原則

新しい `Lowered*` 型を実装する際は、必ず [mlx-swift-lm](https://github.com/ml-explore/mlx-swift-lm) の対応実装を参照すること。独自最適化を入れる前にまず参照実装と同一のロジックで動作させる。

#### mlx-swift-lm を参照する理由

- mlx-swift-lm は MLX Swift の公式サンプル。MLX の API（`conv1d`, `split`, `concatenated` 等）の正しい使い方を示している
- ゼロコピー操作（`split`, スライス）と MLX の lazy evaluation を前提とした設計になっている
- 独自最適化（dot product 展開、layout 変換等）は MLX の内部最適化と競合し、逆に遅くなることがある

#### 実装手順

1. **mlx-swift-lm の対応モジュールを読む** — forward pass のロジック、テンソル layout、cache 管理を確認する
2. **同一のロジックで実装する** — layout 変換や独自最適化を入れない。MLX の lazy eval が最適化を行う
3. **MLX API を正しく使う**:
   - `split(parts:axis:)` でゼロコピー分割（スライスを 3 回呼ぶより効率的）
   - `concatenated(axis: -2)` で cache 結合
   - `conv1d` は channels-last `[N, L, C]` layout。weight は `[C_out, K, C_in/groups]`
   - weight の layout 変換は `init` 時に 1 回だけ行い、`apply` 内では行わない
4. **decode パスの独自最適化を避ける** — T=1 の特殊パス（dot product 展開等）は MLX の graph fusion と競合する。conv1d を直接呼ぶ方が MLX が内部で最適化できる
5. **中間テンソルの `eval()` を呼ばない** — 全層の計算グラフが構築されてから 1 回の `eval()` で実行されるべき
6. **`dim()` 呼び出しを最小化する** — dynamic shape の `dim()` は eval を trigger する可能性がある

#### ShortConv の正しい実装パターン（mlx-swift-lm 準拠）

```swift
// init: weight layout を 1 回だけ変換
convWeight = rawWeight.transposed(0, 2, 1)  // [D, 1, K] → [D, K, 1]

// apply: ゼロコピー分割 + concat + conv1d
let parts = inProj.apply(x).split(parts: 3, axis: -1)
var bx = parts[0] * parts[2]

// cache concat（state は [B, K-1, D]）
bx = concatenated([state, bx], axis: -2)
cache = bx[0..., (bx.dim(-2) - (K - 1))..., 0...]  // 次回用 cache
let convOut = conv1d(bx, convWeight, stride: 1, padding: 0, groups: D)

let y = parts[1] * convOut
return outProj.apply(y)
```
