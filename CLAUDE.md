# CLAUDE.md

## Project Goal

swift-lm は Apple Silicon 上での最速 LLM 推論パッケージ。[AnyFoundationModels](https://github.com/1amageek) の `MLXFoundationModels` バックエンドとして消費される。

**核心命題: IR グラフを解析し、最適化された Metal kernel を出力するコンパイラを構築する。** MetalCompilable が IR のデータを問い合わせ、Fragment を部品として使い、最適化済みの dispatch plan を返す。Component 内の最適化は MetalCompilable の責務、Component 跨ぎの最適化（自動 kernel fusion）は Compiler の責務。

入力は **config.json + safetensors + tokenizer.json** のみ。モデル固有の Swift 型は不要。消費者が指定するのは **HuggingFace repo ID だけ**。

## Architecture

```
LMIR (IR モジュール — backend 非依存)
    │  ModelGraph, OperationKind, ParameterBinding, OperationAttributes
    │  純粋なデータ型。Metal を知らない。TPU を知らない。
    │
    ├── LMArchitecture (DSL + Validation — LMIR を re-export)
    │   ├── ModelComponent, @ModelComponentBuilder
    │   ├── SemanticNormalizer, GraphValidator, DimensionValidator
    │   └── @_exported import LMIR
    │
    ├── ModelDeclarations (depends: LMArchitecture)
    │   └── Transformer, Qwen35, LFM2, Cohere
    │
    ├── MetalCompiler (depends: LMIR only — no MLX)
    │   ├── MetalCompilable protocol (IR → Metal bridge: query interface)
    │   ├── PrimitiveMetalKernelFragment (Metal kernel の再利用可能な部品)
    │   ├── InferencePolicy (deployment intent: KV cache quantization, max sequence length)
    │   ├── FusionContract / FusionSynthesizer / KernelScaffold (自動 kernel fusion)
    │   ├── MetalEntryCollector (IR walk → fragment collection → automatic fusion)
    │   ├── MetalInferenceCompiler (IR walk → MetalCompilable query → dispatch plan)
    │   ├── MetalInferenceModel (dispatch plan 実行)
    │   ├── STAF (STAFConverter, STAFLoader, QuantizationFormat, ParameterResolver)
    │   └── KVCacheSpecification (compiler internal: resolved K/V cache layout)
    │
    └── SwiftLM (consumer API — depends: LMArchitecture, MetalCompiler, ModelDeclarations)
        ├── ModelBundleLoader (HF download → STAF → compile(inferencePolicy:) → LanguageModelContainer / TextEmbeddingContainer)
        ├── LanguageModelContainer / LanguageModelContext
        ├── TextEmbeddingContainer / TextEmbeddingContext
        ├── ModelInput, PreparedPrompt, ExecutablePrompt, PromptSnapshot, GenerationParameters, GenerationEvent
        └── InputMessage, InputImage, InputVideo, TextEmbeddingInput, ModelConfiguration
```

## Metal Backend 設計原則

### IR と Backend の分離

**LMIR は接続と構造だけを記述する。計算の中身は opaque。**

```swift
// LMIR — 接続のみ
struct Operation {
    let operands: [Operand]
    let results: [OperationResult]
    let attributes: any OperationAttributes   // opaque — IR は中身を知らない
}

// 構造操作 — IR が知る唯一のもの
enum OperationKind {
    case primitive(any OperationAttributes)     // 計算ノード (opaque)
    case residual(strategy, body: Region)       // copy → body → add
    case parallel(merge, branches: [Region])    // 並列分岐
    case repeating(count, body: Region)         // N 回繰り返し
    case conditional(condition, then, else)     // compile-time 分岐
}
```

### IR Graph の構造

IR は Region と Operation のネスト構造。ModelComponent DSL が宣言的に構築し、SemanticNormalizer が正規化する。

```
ModelGraph.rootRegion:
  Region [
    Operation(.primitive(TokenEmbeddingAttributes))
    Operation(.repeating(count: 16, body:
      Region [
        Operation(.residual(body:
          Region [
            Operation(.primitive(RMSNormAttributes))        // operator_norm
            Operation(.primitive(AttentionAttributes))      // or ShortConvAttributes
          ]))
        Operation(.residual(body:
          Region [
            Operation(.primitive(RMSNormAttributes))        // ffn_norm
            Operation(.primitive(MLPAttributes))
          ]))
      ]))
    Operation(.primitive(RMSNormAttributes))                // final norm
    Operation(.primitive(OutputHeadAttributes))
  ]
```

構造操作 (residual, repeating, conditional) は compiler が walk 時に処理する。primitive 操作の attributes は `MetalCompilable` protocol を通じて Metal fragment tree に変換される。

### MetalCompilable — IR への問い合わせ interface

`MetalCompilable` は IR と Metal 実装を繋ぐ問い合わせ protocol。IR のデータを読み、Fragment を部品として使い、**最適化済みの結果を返す**。

IR は Metal を知らない。Metal 側が IR を知っている。Swift の retroactive conformance（別モジュールで protocol conformance を追加する機能）で接続する。

```swift
// LMIR モジュール — Metal を知らない
struct AttentionAttributes: OperationAttributes {
    let hiddenSize: Int
    let headCount: Int
    let kvHeadCount: Int
    let headDimension: Int
    // ... パラメータだけ。実装方法は知らない
}

// MetalCompiler モジュール — IR のデータを問い合わせて最適化済みの結果を返す
extension AttentionAttributes: MetalCompilable {
    func fragment(context: KernelContext) -> some MetalKernelFragment {
        // self.headCount, self.hiddenSize を問い合わせて最適な Metal 実装を返す
        BatchedProjection(entries: [q_proj, k_proj, v_proj])  // 3→1 dispatch
        BatchedFragment([QKNormFragment(q), QKNormFragment(k)])  // 2→1 dispatch
        FlashAttentionFragment(...)
        LinearFragment(field: "o_proj", ...)
    }
}
```

Compiler は `as? any MetalCompilable` で capability query する:

```swift
guard let compilable = attributes as? any MetalCompilable else {
    fatalError("... does not conform to MetalCompilable")
}
let fragment = compilable.fragment(context: kernelContext)
```

#### MetalCompilable の責務

- IR のパラメータ（hiddenSize, headCount 等）を読む
- Fragment を部品として使い、Metal 実装を組み立てる
- **Component 内の最適化はここの責務**: Q/K/V batch 化、SwiGLU fusion 等
- Compilable ファイル（`AttentionFragment.swift` 等）に 1 OperationAttributes : 1 ファイルで定義

MetalCompilable は「naive な tree を返して optimizer に任せる」のではなく、**最初から最適化済みの結果を返す**。最適化の判断は Compilable ファイルが行う。

#### Multi-backend 拡張

他の backend は同じ retroactive conformance パターンで独自の protocol を定義する:

```swift
// 仮に MLX backend があれば
protocol MLXCompilable {
    func mlxOperation(context: MLXContext) -> MLXOp
}
extension AttentionAttributes: MLXCompilable { ... }
```

LMIR は backend を知らない。各 backend が自分の protocol を定義し、OperationAttributes への conformance を提供する。Fragment は Metal 固有の概念であり、MLX 等の他 backend には関係しない。

### Fragment — Metal kernel の再利用可能な部品

Fragment は Metal kernel の MSL source を生成する再利用可能な部品。MetalCompilable と Compiler が使う**道具**であり、最適化対象の宣言ではない。

同じ Fragment（LinearFragment 等）を Attention, MLP, StateSpace, OutputHead が再利用する。Buffer precision（F16/F32）や weight format（F16/BF16/Q4）の差異を吸収する。

```
PrimitiveMetalKernelFragment (部品)
├── Reduction                        // RMSNorm, LayerNorm
├── LinearFragment                   // GEMV/GEMM projection
├── ElementwiseFragment              // SwiGLU activation
├── FlashAttentionFragment           // QKV attention + KV cache
├── Conv1dFragment                   // temporal conv + conv_state
├── RoPEFragment                     // rotary position encoding
├── QKNormFragment                   // per-head RMS norm
├── GatherFragment                   // embedding lookup
├── ArgmaxFragment                   // argmax for token selection
├── ScalarMultiplyFragment           // per-element scalar multiply
├── SSMRecurrenceFragment            // state-space model recurrence
└── ...

MetalCompilable conformances (IR → 最適化済み Metal 実装)
├── AttentionAttributes              // → BatchedProjection(Q,K,V) + FlashAttn + o_proj
├── MLPAttributes                    // → BatchedProjection(gate,up) + SwiGLU + down_proj
├── StateSpaceAttributes             // → SSM projections + recurrence
├── ShortConvAttributes              // → conv1d + activation
├── TokenEmbeddingAttributes         // → gather
├── OutputHeadAttributes             // → norm + projection + argmax
├── RMSNormAttributes                // → Reduction
├── LinearAttributes                 // → LinearFragment
├── PerLayerInputAttributes          // → modulation
├── LayerScaleAttributes             // → scalar multiply
└── MoEAttributes                    // → expert routing + MLP
```

### Fragment Design Principles

**Fragment is self-describing. Compiler is generic.**

Fragment は自分自身を完全に記述する。Compiler は fragment の具体的な型を知らずに、fragment が宣言する contract のみに基づいて kernel 合成・buffer routing・dispatch 構成を行う。新しい fragment を追加しても compiler のコードは 1 行も変わらない。

#### Open-Closed Principle

- **Fragment を追加するために compiler (MetalSourceGenerator, MetalKernelSourceCatalog, MetalPrefillStepBuilder) を変更してはならない**
- compiler が fragment を名前や型で switch/case するコードを書いてはならない — fragment の protocol properties のみで判断する
- fusion パターンを追加するために `DispatchKind` に case を追加してはならない — compiler が `FusionContract` から機械的に fusion を判定する

#### Fragment が提供する 3 つの宣言

1. **`kernelBody()`** — 合成可能な MSL コード片。標準化された変数名 (dispatch dimension ごとに規定) を使い、kernel signature・buffer 宣言・dispatch routing を含まない。compiler がこれを素材として fused kernel を合成する。
2. **`FusionContract`** — 合成可能性の宣言的仕様。入出力ポート (名前・型・アクセスパターン)、parallelism pattern、threadgroup memory 要件、barrier 要件を記述。compiler はこの contract のみを見て隣接 fragment の fusion 可否を機械的に判定する。
3. **`kernelSource()`** — 非合成 fragment のフォールバック。FlashAttention のような複雑な kernel は `kernelBody()` から nil を返し、`kernelSource()` で完成品 MSL を提供する。全てを合成可能にする必要はない。

#### Fusable / Non-fusable の境界

| Fragment | Fusable | Reason |
|---|---|---|
| Reduction (RMSNorm) | Yes | threadgroup reduction, standard pattern |
| ElementwiseFragment (SwiGLU) | Yes | pure per-element, no dependency |
| ScalarMultiplyFragment | Yes | pure per-element |
| QKNormFragment | Yes | per-head reduction |
| RoPEFragment | Yes | per-head elementwise |
| ResidualAddFragment / CopyFragment | Yes | per-element |
| GatherFragment | No | irregular access pattern |
| LinearFragment (GEMV/GEMM) | No | fundamentally different parallelism (matrix multiply) |
| FlashAttentionFragment | No | complex multi-pass, KV cache management |
| SSMRecurrenceFragment | No | sequential recurrence |
| Conv1dFragment | No | temporal dependency |

#### Prohibition: hand-written fused kernels

- `MetalSourceGenerator` に `generateFusedXxx()` 系の手書き fused kernel generator を追加してはならない
- fused kernel は compiler が `kernelBody()` の合成で自動出力する
- 旧 `generateFusedCopyRMSNorm`, `generateFusedResidualAddCopyRMSNorm` 等は削除済み — 自動 fusion に移行完了

#### Compiler の end-to-end パイプライン

```
MetalEntryCollector.collect()
  │
  ├── Phase 1: IR walk (walkRegion)
  │     IR Graph を再帰的に walk → [DispatchEntry] (unfused)
  │     structural operations (residual, repeating, conditional) は
  │     CopyFragment / ResidualAddFragment に展開
  │     primitive operations は MetalCompilable.fragment(context:) で fragment tree を取得し flatten
  │
  ├── Phase 2: Automatic fusion (fuseCrossComponent)
  │     greedy while-loop で隣接 fusable entries を SynthesizedFragment に合成
  │     FusionContract の 3 条件 (parallelism, port, TG memory) のみで判定
  │     → [DispatchEntry] (fused) — dispatch 数が削減される
  │
  └── 返却: (walkContext, unfusedCount, fusedEntries)

SynthesizedFragment (lazy kernel synthesis)
  │
  ├── kernelSource() 呼び出し時に FusionSynthesizer.synthesize() を実行
  │     1. LoopGroup 分割: register vs threadgroup intermediate で境界決定
  │     2. Variable renaming: producer output → consumer input の接続
  │     3. Body concatenation: LoopGroup 内で kernelBody() を結合
  │     4. Parallelism adaptation: perElement body を perRow loop でラップ
  │     5. Contract merging: 外部ポートのみ残す、内部 junction を除去
  │
  ├── KernelScaffold.generate()
  │     merged contract + concatenated body → 完全な MSL source
  │     parallelism に応じた scaffold template (perRow/perElement/perHead)
  │     sequence mode (prefill F32) / decode mode (F16) の自動切替
  │
  └── Metal pipeline compilation → GPU dispatch

Buffer routing (MetalPrefillStepBuilder / decode bindings)
  SynthesizedFragment の merged contract の外部ポートに対して
  BufferRoutingState でバッファ割り当て
```

### Fragment Correctness Contract

`swift-lm` は fragment と compiler routing の合成で成立する。局所的な最適化や emit 順序より、まず graph semantics を守ること。

必須ルール:

- 並列な sibling projection は、明示的に依存関係がない限り同じ入力 source を読む。
- dispatch entry の並び順をそのままデータ依存とみなしてはいけない。compiler が元の IR 意味論を保持する。
- `isOutput` や最終出力マークは「最後に emit された projection」ではなく「意味的に hidden へ戻る projection」に対して付与する。
- prefill と decode の routing state は別物として検証する。片方が正しくてももう片方は壊れうる。
- fragment 追加時は `kernelBody()` と `FusionContract` で計算ロジック・合成可能性を宣言する。buffer routing は compiler に委ねる。

破損時の調査順序 — Probe First:

**出力が壊れたら、まず Probe で壊れた場所を特定する。コードの静的解析や機能の無効化で原因を推測してはならない。**

1. **Layer Probe で障害層を特定する**: `debugPrefillLastTokenHiddenSnapshots` で全層の hidden を検査し、最初にゼロ/異常値になる層を見つける
2. **障害層の step を精査する**: `debugPrefillStepSummaries` と `debugDescribePrefillSteps` で障害層内の各 dispatch step の kernel 名・buffer routing を確認する
3. **HuggingFace Python の中間値と比較する**: Python で同一入力の各層・各 step の中間値を取得し、swift-lm の値と並べて乖離箇所を特定する。swift-lm 内部の「正常層」との比較は参考にしない — 全層が壊れている可能性がある
4. **中間値を Probe する**: 障害層内の各 step 後の hidden/residual/scratch 値を snapshot して、どの step で値が壊れるか絞り込む
5. **壊れた step の kernel source を検査する**: 特定された kernel の MSL source を `kernelSource()` / `kernelBody()` で確認し、HuggingFace の該当 forward() コードとの差異を探す

禁止事項:

- **機能を無効化して切り分けてはならない** — fusion の無効化、optimizer のバイパス等は禁止。正常な動作パスを壊す変更は診断ではない
- **コードの静的解析だけで原因を推測してはならない** — 実行時の値を観測してから仮説を立てる
- **一度に複数の仮説を検証してはならない** — Probe で 1 つの仮説を確認し、結果を見てから次に進む
- **swift-lm 内部の比較だけで正しいと判断してはならない** — HuggingFace Python の参照値と一致して初めて正しいとする

### Test Execution Stability

- `xcodebuild` の単発失敗だけで実装バグと断定しない。`unexpected exit` や suite restart はテストプロセスの不安定さで起きることがある。
- 実 bundle を読む suite は `build-for-testing` 後に `test-without-building` で 1 suite ずつ回す。
- 不安定な suite は [`scripts/xcodebuild/test-timeout.sh`](/Users/1amageek/Desktop/swift-lm/scripts/xcodebuild/test-timeout.sh) または [`scripts/xcodebuild/test-hang-guard.sh`](/Users/1amageek/Desktop/swift-lm/scripts/xcodebuild/test-hang-guard.sh) で再現性を確認する。
- suite-level filter で切り分けられるように、重い smoke tests は output / prompt-state / capability などの関心ごとごとに分割する。
- **ベンチマーク反復や `/loop` の `ScheduleWakeup` で 5 分 (300s) 以上の間隔を空けない** — Anthropic prompt cache の TTL が 5 分のため、それ以上空けると次回起動時にコンテキストを uncached で読み直すことになりコスト・応答速度が悪化する。cool-state を取る場合も 270s 未満で刻む。

### Compiler — Optimizing Kernel Compiler

Compiler の責務は IR グラフから**最適化された GPU dispatch plan を自動生成**すること。

**Compiler は fragment の具象型を知らない。** `FusionContract` と `kernelBody()` という protocol interface のみで全ての判断を行う。新しい計算パターンを追加するとき、compiler のコード変更はゼロ。fragment が自分自身を宣言すれば、compiler は自動的にそれを検出・合成・最適化する。

Compiler の動作は上述の「Compiler の end-to-end パイプライン」と「Phase 2: Automatic kernel fusion」に詳述。以下は dispatch plan 生成の各フェーズの補足。

### Compiler の IR walk → dispatch plan 生成

#### Phase 1: IR walk (walkRegion)

compiler は IR Graph を再帰的に walk し、dispatch entries を生成する。

```
walkRegion(region):
  for operation in region.operations:
    switch operation.kind:

    case .primitive(attributes):
      compilable = attributes as? MetalCompilable  // capability query
      fragment = compilable.fragment(context)       // witness table dispatch
      primitives = collectPrimitives(fragment)      // tree を flatten
      for entry in primitives: emit(entry)
      markLastProjectionAsOutput()                // o_proj, down_proj → isOutput

    case .residual(body):
      emit(CopyFragment)         // hidden → residual buffer
      walkRegion(body)            // body の entries を再帰生成
      emit(ResidualAddFragment)   // hidden += residual

    case .repeating(count, body):
      for i in 0..<count:
        walkRegion(body, layerIndex: i)

    case .conditional(condition, then, else):
      walkRegion(selectedBody)  // compile-time 分岐
```

#### Phase 2: Automatic kernel fusion

上述の「Phase 2: Automatic kernel fusion (fuseCrossComponent)」セクションを参照。

#### Phase 3: Prefill dispatch plan 生成 (buildPrefillSteps)

dispatch entries → MetalPrefillStep (実行可能な GPU dispatch 列) に変換。DispatchEntry は統一的に `fragment: any PrimitiveMetalKernelFragment` を持ち、kind による分岐は存在しない。

```
buildPrefillSteps(entry):
  fragment = entry.fragment

  if fragment is LinearFragment:
    // GEMV/GEMM projection
    input  = lastOutputIsHidden ? hidden : scratch[currentInputOffset]
    output = isOutput ? hidden : scratch[projectionIndex + 1]
    projectionIndex += 1
    // ⚠ isOutput でない projection は lastOutputIsHidden を変更しない
    //   → 並列射影 (gate+up, Q+K+V) が同じ入力を読めるようにする

  if fragment is BatchedProjection:
    // 内部の LinearFragment 列を展開して個別に dispatch

  if fragment is Reduction && shouldCaptureResidualInput:
    // RMSNorm が residual block 先頭にある場合:
    // hidden → residual copy + norm → hidden の 2 step を生成
    copy hidden → residual
    norm(residual) → hidden
    projectionIndex = 0
    lastOutputIsHidden = true

  else:
    // CopyFragment, ResidualAddFragment, SynthesizedFragment 等
    steps = fragment.prefillSteps(context)
    lastOutputIsHidden = result.outputIsHidden
    if result.resetsProjectionIndex: projectionIndex = 0
```

Note: CopyFragment と ResidualAddFragment は Phase 2 の自動 fusion で隣接する Reduction と合成される。合成された SynthesizedFragment は `prefillSteps(context:)` で統一的に処理される。

#### Buffer routing state

Compiler は routing state を通じて各 step のバッファ割り当てを決定:

```
struct BufferRoutingState {
    lastOutputIsHidden: Bool     // true: 次の入力は hidden, false: scratch
    projectionIndex: Int         // scratch slot の番号 (slot = projectionIndex + 1)
}
```

**重要**: 非出力 projection (gate, up, Q, K, V) は `lastOutputIsHidden` を変更しない。これにより並列射影が同一入力 (RMSNorm 出力) を読める。

#### Scratch buffer slot layout

```
scratch buffer:
  slot 0: RMSNorm 出力 → conv1d 出力 → SwiGLU 出力 → attention 出力
  slot 1: 1st projection 出力 (in_proj / gate_proj / q_proj)
  slot 2: 2nd projection 出力 (up_proj / k_proj)
  slot 3: 3rd projection 出力 (v_proj)

  slot stride = slotDimension × maxSequenceLength × sizeof(float)
  slotDimension = max(hiddenSize, intermediateSize, maxProjectionOutputDimension)
```

#### Dispatch mode (prefill)

| Mode | 動作 | 用途 |
|------|------|------|
| `.batch` | 全 position を 1 dispatch で処理。grid.height = seqLen | GEMM, RMSNorm, SwiGLU, conv1d |
| `.perPosition` | position ごとに dispatch + barrier | flash_attn (KV cache 依存) |
| `.lastToken` | 最終 position のみ dispatch | output head GEMM, argmax |

### Prefill と Decode の precision 分離

- **Prefill**: hidden/residual/scratch 全て **Float32**。16+ layers の蓄積誤差を防ぐ
- **Decode**: hidden/residual/scratch 全て **Float16**。1 token ずつの計算で蓄積なし
- **F32↔F16 変換**: prefill→decode の転送時に **1 箇所で独立して** 行う。計算 kernel 内で混在させない
- **KV cache**: F16 (prefill/decode 共通)。flash_attn kernel 内で F32→F16 変換して書き込む
- **conv_state**: F16 (decode format)。extract kernel 内で F32→F16 変換して書き込む

### STAF (SafeTensor Accelerated Format)

Weight は safetensors からの直接ロードではなく、STAF 実行キャッシュ経由で GPU にロードする。

```
*.safetensors (source of truth)
    ↓ STAFConverter (1回だけ、オフライン)
*.staf (GPU-ready executable cache)
    ↓ mmap → bytesNoCopy → MTLBuffer (ゼロコピー)
STAFWeightStore
    ↓ quant_scheme_id → QuantizationFormat → kernel 選択
GEMV kernel がブロックを直接読んで計算
```

**量子化ブロック**: weight + scale + zero を interleave して 1 block に。cache line 内で全情報が取れる。

```
┌──────────┬──────────┬──────────────────────────┐
│scale (2B)│ zero (2B)│ packed quants (32-64B)    │
└──────────┴──────────┴──────────────────────────┘
```

**QuantizationFormat protocol**: 量子化形式ごとに struct を追加。kernel は format.gemvKernelName で選択。

### Buffer 管理

Apple Silicon unified memory 前提。`private` バッファは GPU ロスレス圧縮が有効化される。

#### Decode buffers

| Buffer | Mode | 理由 |
|---|---|---|
| Weight (STAF) | `shared` + `bytesNoCopy` | mmap ゼロコピー |
| KV cache | `private` + `hazardTrackingModeUntracked` | GPU のみ |
| hidden / scratch / residual / logits | `private` + `hazardTrackingModeUntracked` | GPU のみ |
| tokenIn / tokenOut / position | `shared` | CPU read/write 必要 |

#### Prefill buffers

| Buffer | Mode | 理由 |
|---|---|---|
| hidden | `shared` | Vision model が CPU から hidden override を注入 |
| scratch / residual / logits | `private` | GPU のみ |
| KV cache (prefill-only) | `private` | GPU のみ |
| tokenIDs / positions / tokenOut | `shared` | CPU read/write 必要 |

#### F32→F16 hidden conversion

Prefill (F32) → Decode (F16) の hidden 転送は `hidden_copy_from_float` GPU kernel で変換。CPU 側のループ変換は行わない（`didModifyRange` も不要 — Apple Silicon unified memory では shared バッファの CPU↔GPU 同期は暗黙的）。

### InferencePolicy — Deployment Intent

デプロイメント判断（KV cache 量子化、最大シーケンス長、レイアウトモード）は `InferencePolicy` で宣言的に指定する。IR（構造）でもなく compiler 内部（実装）でもない第三の関心事。

```
Consumer (ModelBundleLoader)
  │ InferencePolicy
  │   ├── maximumSequenceLength: Int        // KV cache + prefill buffer sizing
  │   └── kvCache: KVCachePolicy
  │         ├── keyScheme: .automatic | .fixed(scheme)
  │         ├── valueScheme: .automatic | .fixed(scheme)
  │         └── layoutMode: .sequenceMajor | .headMajor
  │
  ├── compile(inferencePolicy:)       → MetalCompiledModel (decode plan + KV cache)
  └── compilePrefill(inferencePolicy:) → MetalPrefillPlan (shared KV cache from decode)
```

| レイヤー | 責務 | 例 |
|---|---|---|
| IR (LMIR) | モデル構造 (WHAT) | Attention, MLP, layer count |
| InferencePolicy | デプロイメント意図 | KV cache Q4, max 8192 tokens |
| Compiler (MetalCompiler) | 実装方法 (HOW) | kernel 選択, buffer 確保, KVCacheSpecification |

- `InferencePolicy.default` = FP16 KV cache, 4096 max tokens, sequenceMajor
- `SchemeSelection.automatic` は weight format から KV cache scheme を導出（BF16 weights → BF16 cache, else → FP16）
- K/V は独立量子化可能 — K は dot product 用（攻撃的量子化可）、V は weighted sum 用（保守的が必要）
- `KVCacheSpecification` は compiler 内部の実装詳細。InferencePolicy（意図）→ KVCacheSpecification（実体）の変換は `MetalBufferAllocator` が担う
### RotorQuant — Clifford Rotor KV Cache Quantization

Clifford Cl(3,0) rotor (3D rotation) で KV cache を回転してから量子化する。PolarQuant の原理: ランダム回転が外れ値を次元間に分散させ、同ビット幅でも量子化品質が向上する。

#### Scheme identifiers

| Scheme | ID | Base | Memory ratio vs FP16 |
|---|---|---|---|
| `rotorQ8Group32ScaleF16` | 0x70 | Q8 group32 | 62.5% |
| `rotorQ4Group64ScaleF16` | 0x71 | Q4 group64 | 37.5% |

#### Rotor representation

Each rotor is a unit quaternion `[s, b₁₂, b₁₃, b₂₃]` in Float16, satisfying `s² + b₁₂² + b₁₃² + b₂₃² = 1`. Groups of 3 dimensions are rotated via sandwich product `RvR̃`. Buffer layout: `[layer × kvHeadCount × numRotorGroups × 4]` where `numRotorGroups = ceil(headDim / 3)`.

#### Initialization

Deterministic LCG hash chain (Knuth multiplier `6364136223846793005`). 4 sequential hashes per rotor → map to [-1,1] in Float32 → normalize to unit quaternion → store as Float16. Same model parameters always produce the same rotors. No calibration data needed.

#### Kernel pipeline

```
Write path (K/V):  data → rotor_apply_forward → quantize → store
Read path (K):     pre-rotate Q via rotor_apply_forward → Q'·K' = Q·K (orthogonality)
Read path (V):     dequantize → weighted sum → rotor_apply_inverse → output
```

K and V share the same rotor buffer per (layer, head, group). `kRotor`/`vRotor` flags enable/disable rotation per scheme.

#### QJL correction (optional)

Johnson-Lindenstrauss projected residual for unbiased inner product estimation. Rademacher matrix Φ (±1/√m) projects quantization residual. Controlled by `KVCachePolicy.qjlDimension`.

#### Performance characteristics (Gemma4-E2B, 35 attention layers)

**Throughput vs context length (tok/s):**

| Context | FP16 | RotorQ8 | RotorQ4 | RotorQ8/FP16 | RotorQ4/FP16 |
|---|---|---|---|---|---|
| 64 | 37.9 | 37.6 | 39.1 | 0.99x | 1.03x |
| 512 | 21.1 | 22.5 | 21.3 | 1.07x | 1.01x |
| 1024 | 13.8 | 12.1 | 12.8 | 0.87x | 0.92x |
| 2048 | 8.6 | 7.8 | 7.0 | 0.90x | 0.81x |

At these context lengths, rotor computation overhead exceeds KV cache bandwidth savings. At fill=2048, KV read is only 3.5% of total decode bandwidth (~70 MB vs ~2 GB weight read). Crossover requires context lengths where KV bandwidth becomes a significant fraction of total bandwidth — estimated at ~55K tokens for this model size.

**Benchmark methodology note:** Multiple `hazardTrackingModeUntracked` models must NOT be simultaneously alive during measurement. GPU cache interference causes anomalous speedups (up to 4.6x) on the last-measured model. Build and measure one model at a time.

**Token quality (FP16 agreement, 100 tokens × 3 prompts):**

| Policy | Agreement | vs Q8 non-rotor |
|---|---|---|
| Q8 (non-rotor) | 39.9% | baseline |
| RotorQ8 | 42.9% | +3.0pp |
| RotorQ4 | 38.6% | -1.3pp |

Random rotation provides measurable quality improvement for Q8. Q4 information loss exceeds rotation benefit.

### 新しい計算の追加手順

1. LMIR に新 Attributes 型を追加 (`OperationAttributes` 準拠、backend 非依存)
2. IR の変更は不要 — `any OperationAttributes` として opaque に格納される
3. MetalCompiler で extension: `NewAttributes: MetalCompilable` — `fragment(context:)` で primitive fragment tree を構築
4. 必要なら新しい `PrimitiveMetalKernelFragment` を追加:
   - `kernelBody()` で合成可能な MSL コード片を返す (fusable な場合)
   - `FusionContract` で入出力ポート・parallelism・threadgroup memory 要件を宣言する
   - `kernelSource()` で完成品 MSL を返す (non-fusable な場合のフォールバック)
   - `decodeBindings()` で `writeBufferIndices` を必ず宣言する — 省略は conservative barrier fallback を引き起こす
5. **compiler 本体 (MetalSourceGenerator, MetalKernelSourceCatalog, MetalPrefillStepBuilder) の変更は不要** — `MetalCompilable` conformance で自動取得、`FusionContract` で自動 fusion 判定、`kernelBody()` で自動合成
6. MetalSourceGenerator に `generateXxx()` や `generateFusedXxx()` を追加してはならない — fragment が `kernelBody()` で自分自身を記述する
7. `MetalCompilable` に対応しない OperationAttributes は fatalError — silent fallback 禁止

## Module Dependencies

```
LMIR (IR — 依存なし)
    │
    ├── LMArchitecture (DSL + Validation — depends: LMIR)
    │
    ├── ModelDeclarations (depends: LMArchitecture)
    │
    ├── MetalCompiler (depends: LMIR only)
    │
    └── SwiftLM (consumer API — depends: LMArchitecture, MetalCompiler, ModelDeclarations)
```

## Build & Test

```bash
swift build
```

`swift test` は使わない（Metal metallib が見つからずクラッシュ）。`xcodebuild test` を使用。複数モジュールの同時実行はハングする（Metal/GPU リソース競合）。モジュール単位で分割実行。

### Correctness-First Test Procedure

性能測定の前に、次の順で検証する:

1. fragment / planner の contract test
2. **Layer Probe で全層の hidden state を検査** — 障害層を特定
3. 障害層内の step-by-step probe で壊れた kernel を特定
4. focused real-model output test (token IDs, HuggingFace Python 参照値との比較)
5. benchmark

追加ルール:

- 出力が壊れている状態で benchmark を進めない。
- **Probe で壊れた層・step を特定してからコードを読む** — コードの静的解析から始めない。
- sampling 経路と argmax 経路は別に確認する。`temperature` が変わるだけで壊れる場合、sampling path の所有権や CPU 読み出し前提を疑う。
- `storageModePrivate` のバッファを host sampling に渡してはいけない。CPU readable な logits source を明示的に使う。
- end-to-end の弱い文字列一致だけで合格にしない。可能なら token IDs、先頭トークン列、HuggingFace Python 実装との一致を取る。

### Crash-Resistant Real-Model Test Procedure

- `build-for-testing` を 1 回だけ実行し、その後は `test-without-building` で 1 case ずつ流す。
- 1 プロセスに複数の重い bundle test を詰め込まない。
- 同期 helper で大きな bundle や container を繰り返し作る場合は `autoreleasepool` で寿命を切る。
- クラッシュ時に最初に疑うのは Metal compiler ではなく、GPU resource retention、unfinished stream、state restore、複数 model の同時生存。
- output 調査用の probe は `ENABLE_METAL_PROBES` で制御し、常時 `DEBUG` 出力にはしない。

## Design Rules

### HuggingFace が唯一の正 (Single Source of Truth)

- **各モデルの計算ロジックは HuggingFace `modeling_*.py` の `forward()` を正とする** — swift-lm の既存実装・テストを正しいと仮定してはならない
- **モデルごとに独立して HuggingFace 実装を検証する** — 別のモデルの実装を参考にしてはならない（例: Gemma2 の RMSNorm 式を Gemma4 に適用しない。Qwen3.5 のパラメータを Gemma4 に流用しない）
- **新しいモデルの実装手順**: (1) HuggingFace `modeling_*.py` の forward() を読む → (2) Python で参照値を取得する → (3) swift-lm で実装する → (4) Python の参照値と比較して検証する
- **バグ調査手順**: (1) Python で正しい中間値を取得する → (2) swift-lm の中間値と比較する → (3) 乖離箇所を特定する → (4) HuggingFace の該当コードと swift-lm の該当コードを比較する
- **swift-lm のコードから「正しいはず」と推測してはならない** — HuggingFace と比較して初めて正しいと判断する

### その他

- HF ディレクトリ (config.json + safetensors + tokenizer.json) が正規ソース
- config.json に必須項目が欠けていたら補完せずエラー
- IR はランタイム非依存。Metal/MLX/TPU 固有の型を IR に持ち込まない
- `MetalCompilable` / MetalKernelFragment / PrimitiveMetalKernelFragment は MetalCompiler モジュール内。SwiftLM/LMIR に属さない
- OperationAttributes → fragment の bridge は `MetalCompilable` protocol conformance で行う。compiler が具象型を switch/case してはならない
- 全 public 型は `Sendable`
- デプロイメント判断（KV cache 量子化、最大シーケンス長）は `InferencePolicy` で外部化。compiler 内部にハードコードしない
- `KVCacheSpecification` の `maximumSequenceLength` にデフォルト値を持たせない — silent fallback 防止
- **全てのモデル計算は ModelComponent → IR → MetalCompiler → Metal GPU で実装する** — CPU 純 Swift での計算実装は禁止

## Vision Encoder

Vision encoder と text decoder は独立したモデル。それぞれ独立した ModelGraph を持つ。

### アーキテクチャ概要

Vision encoder は text decoder と同様に ModelComponent → IR → MetalCompiler → Metal GPU で実行する。

### Gemma4 Vision Encoder

Gemma4 の vision encoder は `vision_config` セクションで構成される。

```
Image → PatchEmbedding → PositionEmbedding(additive, learnable table)
      → N × VisionLayer(sandwich norm + MHA + SwiGLU MLP)
      → AveragePooling(kernel_size × kernel_size)
      → RMSNorm → Linear(vision_hidden → text_hidden)
      → [embedding vectors]
```

#### config.json → 実装のマッピング

| config.json field | 用途 |
|---|---|
| `vision_config.hidden_size` | Vision encoder の hidden dimension |
| `vision_config.intermediate_size` | MLP intermediate dimension |
| `vision_config.num_attention_heads` | MHA head count (= num_key_value_heads) |
| `vision_config.num_hidden_layers` | Transformer layer count |
| `vision_config.patch_size` | Image patch size (pixels) |
| `vision_config.pooling_kernel_size` | Average pooling kernel |
| `vision_config.position_embedding_size` | Max grid positions per axis |
| `vision_config.rope_parameters.rope_theta` | Vision RoPE base frequency (Gemma4 E2B: 100.0) |
| `vision_config.standardize` | Post-pooling standardization |
| `text_config.hidden_size` | Projection output dimension (text embedding space) |

#### 設計上の注意

- Vision-to-text 射影の出力次元は `text_config.hidden_size` から取得する。`vision_config` に `out_hidden_size` / `output_proj_dims` は存在しない
- `processor_class` は `preprocessor_config.json`、`processor_config.json`、`tokenizer_config.json` のいずれかから読む
- RoPE theta は config から読む。ハードコードしない — Gemma4 vision は 100.0 (text の 10000.0 と異なる)
- `use_clipped_linears` は training stability 機能であり、推論時は無視する

### QwenVision Encoder

Qwen3.5 の vision encoder は画像・動画の両方をサポートし、temporal/spatial grid で管理する。deepstack visual indexes による multi-scale feature 抽出を含む。詳細は `Sources/SwiftLM/Qwen35/` を参照。

## Metal 4 (Primary Target)

**Metal 4 を優先して使用する。** Xcode 26 + Metal 4 が利用可能な環境を前提とし、新規実装は Metal 4 API を最初に採用する。Metal 3 fallback は互換性が必要な場合のみ。

### Metal 4 API 方針

- **MTL4CommandBuffer + MTL4CommandAllocator**: command buffer を長寿命オブジェクトとして reuse する。fire-and-forget パターンは廃止
- **MTL4ComputeCommandEncoder**: デフォルトで concurrent dispatch。明示的 barrier で順序制御
- **Barrier model**: `barrierAfterEncoderStages(_:beforeEncoderStages:visibilityOptions:)` で stage-to-stage 同期。`MTL4VisibilityOptionNone`（実行順序のみ）と `MTL4VisibilityOptionDevice`（+ キャッシュ flush）を使い分ける
- **Argument tables**: per-dispatch のバッファ binding を argument table に事前構築し、encode コストを削減
- **Cooperative tensors / matmul2d**: Metal 4 の cooperative tensor API。Prefill GEMM で利用可能だが、ボトルネックは kernel 計算速度ではなく dispatch 数であるため優先度は低い

### References

- [Metal Shading Language Specification v4](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf) — cooperative tensor, tensor_inline, matmul2d_descriptor, Metal Performance Primitives (MPP)
- [Metal 4 matmul example](https://github.com/liuliu/example_matmul_metal4) — tensor_inline で既存 MTLBuffer をラップし matmul2d を compute shader から呼ぶ実装例
- [Discover Metal 4 (WWDC25)](https://developer.apple.com/videos/play/wwdc2025/205/) — MTL4CommandBuffer reuse, command allocator, argument tables, barrier model
- [Understanding the Metal 4 core API](https://developer.apple.com/documentation/Metal/understanding-the-metal-4-core-api)
- [MTL4ComputeCommandEncoder](https://developer.apple.com/documentation/metal/mtl4computecommandencoder) — concurrent dispatch, stage-to-stage barriers
- [Synchronizing passes with producer barriers](https://developer.apple.com/documentation/Metal/synchronizing-passes-with-producer-barriers)
- [Synchronizing passes with consumer barriers](https://developer.apple.com/documentation/Metal/synchronizing-passes-with-consumer-barriers)
- [Combine Metal 4 ML and graphics (WWDC25)](https://developer.apple.com/videos/play/wwdc2025/262/) — MTL4MachineLearningCommandEncoder
- [Explore LLMs on Apple Silicon with MLX (WWDC25)](https://developer.apple.com/videos/play/wwdc2025/298/) — Apple Silicon 上の LLM 推論最適化手法
- [Apple GPU TBDR Architecture](https://developer.apple.com/documentation/metal/tailor-your-apps-for-apple-gpus-and-tile-based-deferred-rendering)
- [Resource Storage Modes for Apple GPUs](https://developer.apple.com/documentation/metal/choosing-a-resource-storage-mode-for-apple-gpus)

### Performance Findings

- Decode GEMV は memory bandwidth bound (~516 GB/s)。kernel 最適化 (threadgroup cache, vectorized load) は Apple Silicon の L2 cache により逆効果
- **個別 kernel の計算速度を上げても効果がない** — ボトルネックは kernel 内の計算ではなく dispatch 間の barrier 同期。matmul2d/AMX 等の kernel 最適化は優先度が低い
- Weight quantization (Q4/Q8) が decode 高速化の最大レバー (帯域 2-4x 削減)
- **Dispatch overhead が支配的**: decode で GPU 時間の ~85% が barrier 同期 (528 × ~30μs)、prefill で ~500 dispatches × ~45μs。個別 kernel の計算時間ではなく dispatch 数がボトルネック
- **最適化の焦点は dispatch 数の削減**: compiler による自動 kernel fusion が唯一の根本解決策。fusable な隣接 fragment を単一 kernel に合成し、barrier 数を直接削減する
- **MLX graph compilation との比較**: EmbeddingGemma prefill で swift-lm BF16 66.2 emb/s vs MLX 62.0 emb/s — 自動 kernel fusion により MLX を逆転 (旧: swift-lm 44.3 emb/s, +49.4% 改善)
- **Prefill は compute-bound**: Q4 (54.0 emb/s) が BF16 (66.2 emb/s) より遅い。GEMM は計算密度が高く weight 帯域がボトルネックにならない。Q4 の dequant dispatch overhead (projection あたり +1 dispatch) が純損失。Q4 の帯域削減が活きるのは decode (GEMV, bandwidth-bound) のみ

### Barrier Optimization — Decode Path

`hazardTrackingModeUntracked` バッファは明示的な barrier が必要。バリアは GPU を同期させるため、不要バリアは深刻なパフォーマンス劣化を起こす。

**Gemma4-E2B 実測値 (564 decode steps, AggressiveOptimizer + RoPE fusion):**
- GPU kernel 計算時間: ~2.9ms
- バリア同期 overhead: ~17ms（~30μs × 528 barriers）
- 1 バリアあたり ~30μs — kernel 1 つの平均実行時間 5μs より遥かに大きい

#### Metal 3 最適化レベル (実施済み)

| 手法 | Steps | Barriers | Encode(μs) | Single CB GPU |
|---|---|---|---|---|
| conservative (全 read+write) | 705 | 669 (95%) | ~2000 | 20.93ms |
| BufferRegion + writeBufferIndices | 705 | 564 (80%) | ~1900 | 20.46ms |
| + AggressiveOptimizer (Q/K/V batch) | 599 | 563 (94%) | ~1900 | 20.60ms |
| + `memoryBarrier(resources:)` | 599 | 563 (94%) | **~580** | **20.00ms** |
| + RoPE + flash_attn fusion | 564 | 528 (94%) | ~540 | ~20.5ms |

#### Metal 4 barrier への移行方針

Metal 3 の `memoryBarrier(resources:)` を Metal 4 の stage-to-stage barrier に置換する:
- `MTL4VisibilityOptionNone`: private バッファの実行順序保証（キャッシュ flush 不要な場合）
- `MTL4VisibilityOptionDevice`: shared バッファまたはキャッシュ coherency が必要な場合
- Decode path の intermediate バッファ (hidden, scratch, residual) は全て `private` + GPU-only → `VisibilityOptionNone` で十分な可能性が高い

#### resource-scoped barrier の原則

`memoryBarrier(scope: .buffers)` は全バッファを同期する。大半のステップは 1 バッファのみに conflict があるため、`memoryBarrier(resources: [conflicting])` で対象を限定。CPU encode コストが **70% 削減** (1,891μs → 580μs)。

#### BufferRegion 追跡の原則

scratch buffer は単一 MTLBuffer に複数 slot を offset で配置する。`BufferRegion(buffer, offset)` で (buffer identity, offset) をペアで管理し、独立した scratch slot 間の false dependency を防ぐ。`conflictingResources(from:)` で conflict する MTLBuffer を特定し、resource-scoped barrier を生成。

#### Fragment の writeBufferIndices 宣言

各 `PrimitiveMetalKernelFragment` は `decodeBindings()` で `writeBufferIndices: Set<Int>` を宣言し、どのバッファ binding が write されるかを明示する。宣言がない場合は conservative fallback（全 binding を read+write）になり、barrier が急増する。

**新しい fragment を追加するときは必ず `writeBufferIndices` を宣言する。** 省略は conservative fallback を引き起こし、decode 性能が大幅に劣化する。

さらに:

- `writeBufferIndices` が正しくても source routing が壊れていれば出力は壊れる。barrier 最適化テストだけで fragment correctness を証明したことにはならない。
- equality や diagnostics に使う policy 型は resource identity を保持する。resource count のみで同値とみなしてはいけない。
- residency は ownership とセットで設計する。long-lived buffer と ephemeral snapshot buffer を同じ経路で扱わない。

#### 残存する barrier overhead と自動 fusion による削減

fusion 前: 528 barriers × ~30μs = ~15.8ms が GPU 時間の ~85% を占める。これらは全て genuine な RAW 依存（線形計算チェーン）。

**barrier overhead は個別の barrier を最適化しても解決しない。** dispatch 数そのものを削減する必要がある。これが compiler による自動 kernel fusion の動機である。

自動 fusion による削減 (実装済み — 「Phase 2: Automatic kernel fusion」参照):
- 隣接する fusable fragment (ResidualAdd, Copy, Reduction 等) を FusionContract に基づいて自動合成
- Transformer 1 layer あたり ~4 dispatch 削減 (28.6%)
- Gemma4-E2B (35 layers) 外挿: ~528 → ~388 barriers, ~15.8ms → ~11.6ms (~4.2ms 削減)
- Metal 4 concurrent dispatch との組み合わせで更なる削減が可能

#### EmbeddingGemma Prefill Benchmark

EmbeddingGemma-300M (24 layers, hiddenSize=768, sandwich norms) の embedding throughput:

| Variant | Steps | Throughput | vs MLX |
|---|---|---|---|
| BF16 (community-bf16) | 314 | **66.2 emb/s** | **+6.8%** (MLX: 62.0 emb/s) |
| Q4 (community-4bit) | 242 | **54.0 emb/s** | — |
| swift-lm (旧, fusion 前) | ~500 | 44.3 emb/s | -28.5% |

自動 kernel fusion により MLX を逆転。sandwich norm パターン (RMSNorm+ResidualAdd+Copy+RMSNorm) の 4-way fusion で dispatch 数を大幅に削減。

Q4 が BF16 より遅い理由: prefill は GEMM (compute-bound) であり、Q4 の weight 帯域削減が活きない。かつ Q4 は projection ごとに dequant dispatch (+1) が必要で純粋な overhead になる。Q4 の帯域削減が有効なのは decode (GEMV, bandwidth-bound) のみ。

## Qwen3.5 = VLM（Vision Language Model）

**Qwen3.5 はマルチモーダルモデルである。テキスト専用モデルではない。**

- `model_type: "qwen3_5"` は Vision Encoder を含む VLM を指す
- config.json に `vision_config`, `image_token_id`, `video_token_id` が存在する
- `preprocessor_config.json` の `processor_class: "Qwen3VLProcessor"` で画像・動画処理を宣言
- テキスト専用で使う場合は vLLM の `--language-model-only` 相当のフラグが必要
- テキストバックボーンは DeltaNet + Full Attention hybrid（`layer_types` で混在指定）
- 「Qwen3-VL」という別のモデル系列は存在しない — Qwen3.5 自体が VLM

### Qwen3.5 config.json の構造

```
{
  "model_type": "qwen3_5",
  "image_token_id": 248056,
  "video_token_id": 248057,
  "vision_start_token_id": 248053,
  "vision_end_token_id": 248054,
  "text_config": { ... },          // DeltaNet + Attention hybrid
  "vision_config": {
    "depth": 12,
    "hidden_size": 768,
    "patch_size": 16,
    "spatial_merge_size": 2,
    "temporal_patch_size": 2,
    "deepstack_visual_indexes": []  // 0.8B では空
  }
}
```

## Family と Model の境界

- Family: 計算グラフ上の再利用可能な単位 (DeltaNet, MoE, parallel attention+MLP)
- Model: family の組み合わせ + 設定 (Qwen35, LFM2, Cohere)
- `SwiftLM/Declaration` と `LMInference/Bridge` には family 名のみ
- 固有名は `Sources/Models/` のみ
