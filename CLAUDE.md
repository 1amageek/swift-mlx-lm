# CLAUDE.md

## Project Goal

swift-lm は Apple Silicon 上での最速 LLM 推論パッケージ。[AnyFoundationModels](https://github.com/1amageek) の `MLXFoundationModels` バックエンドとして消費される。

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
    │   ├── MetalComponent protocol (dispatchDeclarations)
    │   ├── MetalComputeOperation protocol (kernelName, isFusable, dispatchDimension)
    │   ├── MetalKernelSource (compiler 所有の全 kernel MSL source)
    │   ├── MetalInferenceCompiler (IR walk → fusion → dispatch plan)
    │   ├── MetalInferenceModel (dispatch plan 実行)
    │   ├── STAF (STAFConverter, STAFLoader, QuantizationFormat, ParameterResolver)
    │   └── KVCacheSpecification (K/V 独立量子化, layout mode)
    │
    └── SwiftLM (consumer API — depends: LMArchitecture, MetalCompiler, ModelDeclarations)
        ├── ModelBundleLoader (HF download → STAF → compile → ModelContainer)
        ├── ModelContainer (generate, encode, decode)
        ├── ModelInput, PreparedInput, ExecutablePrompt, Generation, GenerateParameters
        └── InputMessage, InputImage, InputVideo, ModelConfiguration
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

構造操作 (residual, repeating, conditional) は compiler が walk 時に処理する。primitive 操作の attributes は MetalKernelFragment として Metal backend に橋渡しされる。

### MetalKernelFragment — SwiftUI 風 fragment tree

**OperationAttributes → MetalKernelFragment** : IR の各計算ノードを Metal kernel の組み合わせに展開する。

```swift
// MetalCompiler モジュール内で extension
extension AttentionAttributes: MetalKernelFragment {
    @MetalKernelFragmentBuilder
    func fragment(context: KernelContext) -> some MetalKernelFragment {
        LinearFragment(field: "q_proj", ...)     // Q 射影
        LinearFragment(field: "k_proj", ...)     // K 射影
        LinearFragment(field: "v_proj", ...)     // V 射影
        if qkNorm != nil {
            QKNormFragment(headCount: headCount, ...)
            QKNormFragment(headCount: kvHeadCount, ...)
        }
        if let rope = rope {
            RoPEFragment(...)
        }
        FlashAttentionFragment(...)              // attention 本体
        LinearFragment(field: "o_proj", ...)     // 出力射影
    }
}
```

Fragment tree のレイヤー:

```
MetalKernelFragment (protocol)
├── PrimitiveMetalKernelFragment    // 1 kernel に対応する leaf
│   ├── Reduction                   // RMSNorm, LayerNorm
│   ├── LinearFragment              // GEMV/GEMM 射影
│   ├── ElementwiseFragment         // SwiGLU, sigmoid gate
│   ├── FlashAttentionFragment      // QKV attention + KV cache
│   ├── Conv1dFragment              // temporal conv + conv_state
│   ├── RoPEFragment                // rotary position encoding
│   ├── QKNormFragment              // per-head RMS norm
│   ├── GatherFragment              // embedding lookup
│   └── ArgmaxFragment              // argmax for token selection
│
├── TupleFragment                   // 複数 fragment の直列結合
├── OptionalFragment                // 条件付き fragment
└── ConditionalFragment             // if-else 分岐
```

### Compiler の IR walk → dispatch plan 生成

#### Phase 1: IR walk (walkRegion)

compiler は IR Graph を再帰的に walk し、dispatch entries を生成する。

```
walkRegion(region):
  for operation in region.operations:
    switch operation.kind:

    case .primitive(attributes):
      fragment = attributes as? MetalKernelFragment
      primitives = collectPrimitives(fragment)    // tree を flatten
      optimized = optimizer.optimizeFragment(primitives)
      for entry in optimized: emit(entry)
      markLastProjectionAsOutput()                // o_proj, down_proj → isOutput

    case .residual(body):
      emit(.structuralCopy)     // hidden → residual buffer
      walkRegion(body)          // body の entries を再帰生成
      emit(.structuralAdd)      // hidden += residual

    case .repeating(count, body):
      for i in 0..<count:
        walkRegion(body, layerIndex: i)

    case .conditional(condition, then, else):
      walkRegion(selectedBody)  // compile-time 分岐
```

#### Phase 2: Graph-level optimization

StandardOptimizer が隣接 entries をパターンマッチで fusion:

```
structuralAdd + structuralCopy + Reduction → fusedResidualAddCopyNorm
structuralCopy + Reduction                → fusedCopyNorm
```

#### Phase 3: Prefill dispatch plan 生成 (buildPrefillSteps)

dispatch entries → MetalPrefillStep (実行可能な GPU dispatch 列) に変換。

```
buildPrefillSteps(entry):
  switch entry.kind:

  case .projection(proj, isOutput):
    input  = lastOutputIsHidden ? hidden : scratch
    output = isOutput ? hidden : scratch[projectionIndex + 1]
    projectionIndex += 1
    // ⚠ isOutput でない projection は lastOutputIsHidden を変更しない
    //   → 並列射影 (gate+up, Q+K+V) が同じ入力を読めるようにする

  case .fragment(frag):
    steps = frag.prefillSteps(context)
    lastOutputIsHidden = result.outputIsHidden
    if result.resetsProjectionIndex: projectionIndex = 0

  case .structuralCopy:
    copy hidden → residual
    projectionIndex = 0

  case .structuralAdd:
    hidden = hidden + residual
    lastOutputIsHidden = true

  case .fusedResidualAddCopyNorm:
    decompose → [structuralAdd, structuralCopy, Reduction]
    再帰的に buildPrefillSteps
```

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

用途別に分離。storage mode を最適化:

| Buffer | Mode | 理由 |
|---|---|---|
| Weight (STAF) | `shared` + `bytesNoCopy` | mmap ゼロコピー。memcpy なし |
| KV cache | `shared` | GPU write + read。unified memory で overhead なし |
| hidden / scratch / residual / logits | `private` + `hazardTrackingModeUntracked` | GPU のみ |
| tokenIn / tokenOut / position | `shared` | CPU read/write 必要 |

### 新しい計算の追加手順

1. LMIR に新 Attributes 型を追加 (`OperationAttributes` 準拠、backend 非依存)
2. IR の変更は不要 — `any OperationAttributes` として opaque に格納される
3. MetalCompiler で extension: `NewAttributes: MetalComponent` — `computeOps` で計算種別を宣言
4. もし新しい kernel が必要なら compiler の kernel source に追加
5. compiler 本体の変更は不要 (graph walk + `as? MetalComponent` で自動検出)

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

## Design Rules

- HF ディレクトリ (config.json + safetensors + tokenizer.json) が正規ソース
- config.json に必須項目が欠けていたら補完せずエラー
- IR はランタイム非依存。Metal/MLX/TPU 固有の型を IR に持ち込まない
- MetalComponent は MetalCompiler モジュール内。SwiftLM/LMIR に属さない
- 全 public 型は `Sendable`

## Vision Encoder

Vision encoder と text decoder は独立したモデル。IR に vision encoder を含めない。

## Metal 4 / Apple GPU References

- [Metal Shading Language Specification v4](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf) — cooperative tensor, tensor_inline, matmul2d_descriptor, Metal Performance Primitives (MPP)
- [Metal 4 matmul example](https://github.com/liuliu/example_matmul_metal4) — tensor_inline で既存 MTLBuffer をラップし matmul2d を compute shader から呼ぶ実装例。Metal 3 API でも動作
- [Discover Metal 4 (WWDC25)](https://developer.apple.com/videos/play/wwdc2025/205/) — MTL4CommandBuffer reuse, command allocator, argument tables, barrier model
- [Combine Metal 4 ML and graphics (WWDC25)](https://developer.apple.com/videos/play/wwdc2025/262/) — MTL4MachineLearningCommandEncoder, CoreML model を GPU timeline で直接実行
- [Explore LLMs on Apple Silicon with MLX (WWDC25)](https://developer.apple.com/videos/play/wwdc2025/298/) — Apple Silicon 上の LLM 推論最適化手法
- [Apple GPU TBDR Architecture](https://developer.apple.com/documentation/metal/tailor-your-apps-for-apple-gpus-and-tile-based-deferred-rendering) — tile memory, imageblocks, threadgroup memory の特性
- [Resource Storage Modes for Apple GPUs](https://developer.apple.com/documentation/metal/choosing-a-resource-storage-mode-for-apple-gpus) — shared/private/memoryless の使い分け

### Performance Findings

- Decode GEMV は memory bandwidth bound (~516 GB/s)。kernel 最適化 (threadgroup cache, vectorized load) は Apple Silicon の L2 cache により逆効果
- Prefill GEMM は Metal 4 `matmul2d` への置換で AMX 最適化パスが期待できる
- Weight quantization (Q4/Q8) が decode 高速化の最大レバー (帯域 2-4x 削減)

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
