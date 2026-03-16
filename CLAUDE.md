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
        ├── LMInput, Generation, GenerateParameters
        └── UserInput, ChatMessage, ModelConfiguration
```

## Metal Backend 設計原則

### IR と Backend の分離

**LMIR は接続だけを記述する。計算の中身は opaque。**

```swift
// LMIR — 接続のみ
struct Operation {
    let operands: [Operand]
    let results: [OperationResult]
    let attributes: any OperationAttributes   // opaque — IR は中身を知らない
}

// 構造操作 — IR が知る唯一のもの
enum StructuralKind {
    case residual(strategy, body: Region)
    case parallel(merge, branches: [Region])
    case repeating(count, body: Region)
    case conditional(condition, then: Region, else: Region)
}
```

Backend (MetalCompiler, 将来の TPUCompiler) が attributes の型を判定して実行方法を決める:

```
LMIR (接続)           MetalCompiler              TPUCompiler
Operation             MetalComponent protocol    TPUComponent protocol
  .attrs (opaque) ──▶  attrs as? AttentionAttrs   attrs as? AttentionAttrs
                       → flash_attn kernel         → tpu_attn_op
                       attrs as? MLPAttrs          attrs as? MLPAttrs
                       → gemv + swiglu              → tpu_mlp_op

StructuralKind        compiler が walk
  .residual   ───────▶  copy → body → add
  .parallel   ───────▶  barrier なし (concurrent)
  .repeating  ───────▶  unroll
  .conditional ──────▶  compile-time 分岐
```

IR は OperationKind enum を持たない。`any OperationAttributes` として opaque に保持する。新しい計算を追加しても IR の変更は不要。

### MetalComponent Protocol — 責務の分離

MetalComponent は **Metal backend 側のプロトコル** (MetalCompiler モジュール内)。IR (LMIR) には属さない。LMIR の Attributes 型を MetalCompiler 内で extension して準拠させる。

**核心原則: MetalComponent は「何の計算が必要か」を宣言する。Compiler は「どう実行するか」を決定する。**

この2つを混同しない:
- **計算の宣言** (MetalComponent の責務): 「GEMV が必要」「reduction(dim: D) が必要」「flashAttention(heads: H) が必要」
- **実行の決定** (Compiler の責務): kernel source、dispatch config、fusion、buffer routing

```swift
// MetalCompiler モジュール内
protocol MetalComponent {
    var projections: [MetalProjection] { get }       // "GEMV が必要" の宣言
    var computeOps: [MetalComputeOp] { get }         // "この非 GEMV 計算が必要" の宣言
    var weightSlots: [MetalWeightSlot] { get }       // weight 名 (variant)
    var cacheSlots: [MetalCacheSlot] { get }         // cache 宣言
}

// 計算種別の宣言 — MetalComponent が返す
// compiler はこれを読んで kernel 選択・dispatch config 計算を行う
enum MetalComputeOp {
    case reduction(dim: Int)                          // norm, argmax
    case elementwise(count: Int)                      // activation, add, copy
    case flashAttention(numHeads: Int, headDim: Int)  // SDPA + KV cache
    case gather(count: Int)                           // embedding lookup
    case conv1d(dim: Int, kernelSize: Int)            // depthwise conv
    case recurrence(numHeads: Int, dk: Int, dv: Int)  // SSM state update
}

// Attributes 型への準拠は MetalCompiler 内の extension で行う
extension AttentionAttributes: MetalComponent { ... }
extension MLPAttributes: MetalComponent { ... }
```

#### 禁止: Compiler 内部に計算種別を持つ

```swift
// ✗ WRONG — compiler が計算種別を決め打ちしている
// 新しい MetalComponent を追加するたびに compiler の enum も変更が必要
enum KernelKind {  // compiler 内部
    case reduction(dim: Int)
    case gemv(...)
}
func primitiveKernelKind(for entry: KernelEntry) -> KernelKind { ... }

// ✓ CORRECT — compiler は MetalComponent から計算種別を読み取る
// 新しい MetalComponent を追加しても compiler の変更は不要
let ops = metalComponent.computeOps
for op in ops {
    let config = computeDispatchConfig(op: op, pipeline: pipeline)
    ...
}
```

### Compiler の役割

Compiler は MetalComponent の宣言を読んで実行方法を決定する:

```
IR Graph walk:

  operation を見る → metalComponent を取得
    → projections から GEMV dispatch を生成
    → computeOps から非 GEMV dispatch を生成
    → graph の接続 (sequential/parallel/residual) に従って繋ぐ

  graph 構造だけで dispatch 列が決まる:

    residual {                      → copy(hidden → residual_buf)
      rmsNorm                       → norm(hidden → scratch)
      parallel {                    → barrier なし (独立)
        linear(q_proj)              → gemv(scratch → scratch_q)
        linear(k_proj)              → gemv(scratch → scratch_k)
        linear(v_proj)              → gemv(scratch → scratch_v)
      }
      flash_attn                    → flash_attn(Q,K,V,cache → scratch)
      linear(o_proj)                → gemv(scratch → hidden)
    }                               → add(hidden, residual_buf → hidden)

  新しい計算を追加しても compiler は変更不要。
  MetalComponent 準拠を追加するだけ。
```

### Kernel Source — Compiler が所有

**kernel source (.metal / MSL) は compiler が 1 箇所に集約して持つ。** MetalComponent には kernel source を持たせない。llama.cpp の ggml-metal.metal と同じ配置方針。

理由:
- GEMV は全 operation で共有 — Attention も MLP も Linear も同じ kernel
- Fusion で kernel が変わる — compiler が fused variant を生成する
- MetalComponent に持たせると fusion 時に 2 つの MetalComponent の source を統合する必要がある

Grid/threadgroup は compiler が `pipeline.maxTotalThreadsPerThreadgroup` と `threadExecutionWidth` から計算。固定値を焼き込まない。

### Op Fusion — llama.cpp パターンマッチ方式

**fusion の本質は dispatch 数削減ではなく、中間バッファの device memory read/write を消すこと。**

llama.cpp と同じパターンマッチ方式で fusion を検出する:

#### Fusion 判定条件 (llama.cpp `ggml_can_fuse_ext` 準拠)

```
1. 中間ノードの use count == 1 (他の consumer がいない)
2. 次ノードの入力が前ノードの出力
3. 前後ノードの shape が同一
4. 型が contiguous
```

**use count == 1 が fusion の本質的条件。** 出力が 1 箇所にしか消費されないなら、中間 buffer への書き出しを省略して register に保持できる。

#### Fusion 対象パターン

```
1. rmsNorm + mul (weight)           → kernel_rms_norm_mul
2. rmsNorm + mul + add (residual)   → kernel_rms_norm_mul_add
3. silu(gate) * up                  → kernel_swiglu (既に 1 kernel)
4. residualAdd + rmsNorm            → kernel_residual_norm
```

GEMV / SDPA / conv1d / recurrence は fuse しない (最適化済み dedicated kernel)。

#### Compiler の fusion pass

```
Phase 1: IR walk → MetalComponent.computeOps を読んで dispatch 列を生成
Phase 2: Fusion pass — 隣接する dispatch を pattern match で fuse
Phase 3: Fused dispatch の kernel source を選択 (compiler が持つ fused variant)
Phase 4: Pipeline 生成 + dispatch config 計算
```

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

## Family と Model の境界

- Family: 計算グラフ上の再利用可能な単位 (DeltaNet, MoE, parallel attention+MLP)
- Model: family の組み合わせ + 設定 (Qwen35, LFM2, Cohere)
- `SwiftLM/Declaration` と `LMInference/Bridge` には family 名のみ
- 固有名は `Sources/Models/` のみ
