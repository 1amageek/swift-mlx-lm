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
    │   ├── InferencePolicy (deployment intent: KV cache quantization, max sequence length)
    │   ├── MetalComponent protocol (dispatchDeclarations)
    │   ├── MetalComputeOperation protocol (kernelName, isFusable, dispatchDimension)
    │   ├── MetalKernelSource (compiler 所有の全 kernel MSL source)
    │   ├── MetalInferenceCompiler (IR walk → fusion → dispatch plan)
    │   ├── MetalInferenceModel (dispatch plan 実行)
    │   ├── STAF (STAFConverter, STAFLoader, QuantizationFormat, ParameterResolver)
    │   └── KVCacheSpecification (compiler internal: resolved K/V cache layout)
    │
    └── SwiftLM (consumer API — depends: LMArchitecture, MetalCompiler, ModelDeclarations)
        ├── ModelBundleLoader (HF download → STAF → compile(inferencePolicy:) → ModelContainer)
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

### Fragment Correctness Contract

`swift-lm` は fragment と compiler routing の合成で成立する。局所的な最適化や emit 順序より、まず graph semantics を守ること。

必須ルール:

- 並列な sibling projection は、明示的に依存関係がない限り同じ入力 source を読む。
- dispatch entry の並び順をそのままデータ依存とみなしてはいけない。compiler が元の IR 意味論を保持する。
- `isOutput` や最終出力マークは「最後に emit された projection」ではなく「意味的に hidden へ戻る projection」に対して付与する。
- prefill と decode の routing state は別物として検証する。片方が正しくてももう片方は壊れうる。
- fragment 追加時は kernel 本体だけでなく、入力 source、出力 destination、write set、barrier 要件まで contract として定義する。

破損時の調査順序:

1. declaration が正しい graph を作っているか
2. fragment expansion が期待どおりか
3. dispatch entries が意味的依存を保っているか
4. prefill/decode step builder の buffer routing が一致しているか
5. optimizer の有無で出力が変わらないか

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

### Correctness-First Test Procedure

性能測定の前に、次の順で検証する:

1. fragment / planner の contract test
2. focused real-model output test
3. optimizer-mode regression test
4. benchmark

追加ルール:

- 出力が壊れている状態で benchmark を進めない。
- sampling 経路と argmax 経路は別に確認する。`temperature` が変わるだけで壊れる場合、sampling path の所有権や CPU 読み出し前提を疑う。
- `storageModePrivate` のバッファを host sampling に渡してはいけない。CPU readable な logits source を明示的に使う。
- 比較時は `none` / `standard` / `aggressive` の実効 optimizer を確認する。ラベルではなく実際に有効な plan を基準にする。
- end-to-end の弱い文字列一致だけで合格にしない。可能なら token IDs、先頭トークン列、reference 実装との一致を取る。

### Crash-Resistant Real-Model Test Procedure

- `build-for-testing` を 1 回だけ実行し、その後は `test-without-building` で 1 case ずつ流す。
- 1 プロセスに複数の重い bundle test を詰め込まない。
- 同期 helper で大きな bundle や container を繰り返し作る場合は `autoreleasepool` で寿命を切る。
- クラッシュ時に最初に疑うのは Metal compiler ではなく、GPU resource retention、unfinished stream、state restore、複数 model の同時生存。
- output 調査用の probe は `ENABLE_METAL_PROBES` で制御し、常時 `DEBUG` 出力にはしない。

## Design Rules

- HF ディレクトリ (config.json + safetensors + tokenizer.json) が正規ソース
- config.json に必須項目が欠けていたら補完せずエラー
- IR はランタイム非依存。Metal/MLX/TPU 固有の型を IR に持ち込まない
- MetalComponent は MetalCompiler モジュール内。SwiftLM/LMIR に属さない
- 全 public 型は `Sendable`
- デプロイメント判断（KV cache 量子化、最大シーケンス長）は `InferencePolicy` で外部化。compiler 内部にハードコードしない
- `KVCacheSpecification` の `maximumSequenceLength` にデフォルト値を持たせない — silent fallback 防止

## Vision Encoder

Vision encoder と text decoder は独立したモデル。IR に vision encoder を含めない。

## Metal 4 (Primary Target)

**Metal 4 を優先して使用する。** Xcode 26 + Metal 4 が利用可能な環境を前提とし、新規実装は Metal 4 API を最初に採用する。Metal 3 fallback は互換性が必要な場合のみ。

### Metal 4 API 方針

- **MTL4CommandBuffer + MTL4CommandAllocator**: command buffer を長寿命オブジェクトとして reuse する。fire-and-forget パターンは廃止
- **MTL4ComputeCommandEncoder**: デフォルトで concurrent dispatch。明示的 barrier で順序制御
- **Barrier model**: `barrierAfterEncoderStages(_:beforeEncoderStages:visibilityOptions:)` で stage-to-stage 同期。`MTL4VisibilityOptionNone`（実行順序のみ）と `MTL4VisibilityOptionDevice`（+ キャッシュ flush）を使い分ける
- **Argument tables**: per-dispatch のバッファ binding を argument table に事前構築し、encode コストを削減
- **Cooperative tensors / matmul2d**: Prefill GEMM を Metal 4 `matmul2d` に置換し AMX 最適化パスを活用

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
- Prefill GEMM は Metal 4 `matmul2d` への置換で AMX 最適化パスが期待できる
- Weight quantization (Q4/Q8) が decode 高速化の最大レバー (帯域 2-4x 削減)

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

#### 残存する barrier overhead

528 barriers × ~30μs = ~15.8ms が GPU 時間の ~85% を占める。これらは全て genuine な RAW 依存（線形計算チェーン）。Metal 4 の `MTL4VisibilityOptionNone` barrier でコスト削減、または kernel fusion で step 数を削減する。

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
