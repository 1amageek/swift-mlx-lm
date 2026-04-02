# Apple Silicon でローカル LLM を動かすとき、GPU の中で何が起きているか

## ローカル LLM の現在地

2025年以降、10B 以下のモデルが実用水準に達してきた。Qwen3.5-1.2B、Gemma-3-4B、Llama-3.2-3B、LFM2.5-1.2B など、MacBook の 8-36GB Unified Memory に収まるサイズで、コード生成・要約・翻訳・ツール呼び出しが動く。

クラウド API に比べたローカル推論の利点は明確だ：

- ネットワーク遅延ゼロ
- プライバシー（データが端末外に出ない）
- コスト（API 課金なし）
- オフライン動作

問題は速度だ。ローカル推論のボトルネックはどこにあり、何を最適化すれば速くなるのか。この記事では Apple Silicon の GPU (Metal) でLLM 推論を実行するとき、内部で何が起きているかを解説する。

## Apple Silicon のアーキテクチャ

### Unified Memory Architecture (UMA)

従来の PC では CPU と GPU が別々のメモリを持ち、データをコピーして渡す必要があった。Apple Silicon は CPU・GPU・Neural Engine が**同一の物理メモリ（Unified Memory）**を共有する。

```
┌─────────────────────────────────────────────────────┐
│                  Unified Memory (LPDDR5)             │
│                   8GB - 192GB                        │
│                                                      │
│    ┌─────────┐  ┌──────────┐  ┌──────────────────┐  │
│    │   CPU   │  │   GPU    │  │  Neural Engine   │  │
│    │ (高効率 │  │ (Metal)  │  │     (ANE)        │  │
│    │  +高性能)│  │          │  │                  │  │
│    └────┬────┘  └────┬─────┘  └───────┬──────────┘  │
│         │            │                │              │
│         └────────────┴────────────────┘              │
│                      │                               │
│              Memory Controller                       │
│              (帯域: 100-800 GB/s)                    │
└─────────────────────────────────────────────────────┘
```

LLM 推論にとって UMA は重要だ。モデルの重み（数 GB）を CPU ↔ GPU 間でコピーする必要がない。メモリに一度ロードすれば、CPU でも GPU でもゼロコピーでアクセスできる。

### 3つの計算ユニット

| ユニット | 得意な処理 | LLM での用途 |
|---------|-----------|-------------|
| **CPU** | 逐次処理、分岐の多い制御 | トークナイズ、サンプリング、制御フロー |
| **GPU (Metal)** | 大規模並列演算 | 行列積、attention、正規化 — **推論の主役** |
| **ANE (Neural Engine)** | 固定パターンの NN 推論 | CoreML 経由の推論（制約が多い） |

### ANE の制約

ANE は iPhone/iPad のオンデバイス ML のために設計されたプロセッサだ。固定的なニューラルネットワーク演算（Conv2D、全結合層など）を非常に低電力で実行できるが、LLM 推論には向かない：

- **動的な shape に弱い** — KV キャッシュのシーケンス長が毎トークンで変わる
- **対応演算が限定的** — RoPE、GQA、MoE ルーティングなど LLM 特有の演算が非対応
- **CoreML 変換が必要** — モデルを CoreML 形式にコンパイルする必要があり、柔軟性が低い

結果として、現在のローカル LLM 推論は **Metal (GPU)** が主戦場になっている。

### メモリ帯域 — 推論速度の決定要因

Apple Silicon のメモリ帯域はチップによって大きく異なる：

| チップ | メモリ帯域 | メモリ容量 |
|--------|-----------|-----------|
| M1 | 68 GB/s | 8-16 GB |
| M1 Pro/Max | 200-400 GB/s | 16-64 GB |
| M2 | 100 GB/s | 8-24 GB |
| M2 Pro/Max | 200-400 GB/s | 16-96 GB |
| M3 | 100 GB/s | 8-24 GB |
| M3 Pro/Max | 150-400 GB/s | 18-128 GB |
| M4 | 120 GB/s | 16-32 GB |
| M4 Pro/Max | 273-546 GB/s | 24-128 GB |

**なぜ帯域が速度を決めるのか。** LLM のデコード（1トークン生成）では、モデルの全重みを1回読む必要がある。4-bit 量子化された 3B モデルの重みは約 1.8 GB。1トークン生成するたびに 1.8 GB を GPU メモリから読む。

```
理論上限 tok/s ≈ メモリ帯域 / モデルサイズ

M4 Max (546 GB/s) + 3B 4-bit (1.8 GB):
  546 / 1.8 ≈ 303 tok/s (理論上限)

M3 (100 GB/s) + 3B 4-bit (1.8 GB):
  100 / 1.8 ≈ 56 tok/s (理論上限)
```

実際にはカーネル起動のオーバーヘッドやキャッシュミスで理論値の 50-70% 程度になるが、**メモリ帯域が律速**であることは変わらない。これが bandwidth-bound と呼ばれる状態だ。

### HBM と LPDDR の違い

NVIDIA GPU は HBM (High Bandwidth Memory) を使う。HBM3 の帯域は 2-3 TB/s で、Apple Silicon の LPDDR5 (100-546 GB/s) より数倍速い。

しかし Apple Silicon には別の利点がある：

- **UMA** — CPU ↔ GPU コピー不要
- **大容量メモリが安価** — 36GB の MacBook で 7B モデルが余裕で動く。HBM は容量あたりのコストが高い
- **電力効率** — ノート PC で推論できる

帯域では NVIDIA に劣るが、「手元で動く」ことの価値が LLM の用途では大きい。

## Metal — Apple の GPU プログラミング API

Metal は Apple の GPU 向け低レベル API だ。LLM 推論では、行列積 (matmul)、正規化 (RMSNorm)、attention などの演算を **Metal kernel** として GPU で実行する。

### Metal kernel とは

Metal kernel は GPU で実行される関数だ。C++ に似た Metal Shading Language で書く。

```
[CPU]                              [GPU]
  │                                  │
  ├── kernel 1 を dispatch ─────────→├── 数千スレッドで並列実行
  │                                  │   Global Memory → 計算 → Global Memory
  │                                  │
  ├── kernel 2 を dispatch ─────────→├── 数千スレッドで並列実行
  │                                  │   Global Memory → 計算 → Global Memory
  │                                  │
  └── ...                            └── ...
```

重要な点：

1. **1つの kernel = 1回の Global Memory 往復** — kernel が起動すると、入力をメモリから読み、計算し、結果をメモリに書き戻す
2. **kernel 間はメモリ経由** — kernel A の出力を kernel B の入力にするには、一旦 Global Memory に書き戻す必要がある
3. **kernel dispatch にはオーバーヘッドがある** — CPU → GPU への命令発行自体にコストがかかる

### MLX の遅延評価

[MLX](https://github.com/ml-explore/mlx) は Apple Silicon 向けの ML フレームワークだ。MLX は**遅延評価 (lazy evaluation)** を採用している。

```swift
let y = rmsNorm(x, weight: w, eps: 1e-5)  // まだ計算しない — グラフに追加
let z = matmul(y, weight.T)               // まだ計算しない — グラフに追加
let out = silu(z)                          // まだ計算しない — グラフに追加
eval(out)                                   // ここで一気に実行
```

`eval()` が呼ばれるまで、MLX は演算グラフを構築するだけだ。`eval()` 時に、隣接する element-wise 演算（加算、乗算、活性化関数など）を自動的に 1 つの Metal kernel にまとめる（fuse する）ことがある。ただし、matmul のような大きな演算は fuse されず、独立した kernel として dispatch される。

## モデル全体の実行フロー — Transformer 1層で起きていること

ここが本題だ。Transformer 1層（Attention + MLP）が実行されるとき、GPU 内部で何が起きているかをステップごとに追う。

### 前提

- モデル: Llama 系 (GQA + SiLU-gated MLP)
- hidden_size: D
- デコード（1トークン生成）を想定

### 実行フロー

```
入力 x [B, 1, D]
│
│ ──── Attention ブロック ────────────────────────────
│
├─① RMSNorm(x)
│    Metal kernel 1: read x(D) + weight(D), write norm(D)
│    → メモリ往復 1回
│
├─② matmul(norm, W_qkv)                   ← packed QKV projection
│    Metal kernel 2: read norm(D) + W_qkv(D × 3Dh), write qkv(3Dh)
│    → メモリ往復 1回
│    ※ Q,K,V を個別に matmul すると 3回。Packing で 1回に削減
│
├─③ split(qkv) → Q, K, V
│    Metal kernel 3: read qkv(3Dh), write Q(Dq), K(Dk), V(Dv)
│    → メモリ往復 1回
│
├─④ RoPE(Q), RoPE(K)
│    Metal kernel 4-5: 位置情報を回転行列として埋め込む
│    → メモリ往復 2回
│
├─⑤ KV Cache 更新
│    Metal kernel 6: K, V をキャッシュに追記
│    → メモリ往復 1回
│
├─⑥ Scaled Dot-Product Attention
│    Metal kernel 7: Q × K^T → scores → softmax → × V → attn_out
│    → メモリ往復 1回 (MLXFast が内部で fuse 済み)
│
├─⑦ matmul(attn_out, W_o)
│    Metal kernel 8: read attn(Dv) + W_o(Dv × D), write proj(D)
│    → メモリ往復 1回
│
├─⑧ x + proj (residual add)
│    Metal kernel 9: read x(D) + proj(D), write x2(D)
│    → メモリ往復 1回
│
│ ──── MLP ブロック ─────────────────────────────────
│
├─⑨ RMSNorm(x2)
│    Metal kernel 10: read x2(D) + weight(D), write norm2(D)
│    → メモリ往復 1回
│
├─⑩ matmul(norm2, W_gate_up)              ← packed Gate+Up projection
│    Metal kernel 11: read norm2(D) + W_gu(D × 2I), write gu(2I)
│    → メモリ往復 1回
│    ※ gate と up を個別に matmul すると 2回。Packing で 1回に削減
│
├─⑪ gate, up = split(gu)
│    silu(gate) * up → activated
│    Metal kernel 12: MLX の遅延評価で split + silu + mul が自動 fuse
│    → メモリ往復 1回
│
├─⑫ matmul(activated, W_down)
│    Metal kernel 13: read activated(I) + W_down(I × D), write mlp(D)
│    → メモリ往復 1回
│
└─⑬ x2 + mlp (residual add)
     Metal kernel 14: read x2(D) + mlp(D), write x3(D)
     → メモリ往復 1回

出力 x3 [B, 1, D]
```

**1層あたり約 14 回の Metal kernel dispatch、14 回のメモリ往復が発生する。**

24 層のモデルなら、1トークンの生成で **約 336 回の kernel dispatch** だ。

## 最適化手法

### Projection Packing — matmul の回数を減らす

Transformer の Attention では Q, K, V の3つの projection がある。それぞれが独立した matmul だと 3 回の kernel dispatch が必要だ：

```
通常:
  Q = matmul(x, W_q)    ← Metal kernel 1: read x + W_q, write Q
  K = matmul(x, W_k)    ← Metal kernel 2: read x + W_k, write K
  V = matmul(x, W_v)    ← Metal kernel 3: read x + W_v, write V

  入力 x を3回読む。W_q, W_k, W_v を各1回読む。
```

Packing では、3つの重み行列を事前に結合して1回の matmul にする：

```
Packed:
  W_qkv = concat([W_q, W_k, W_v], axis=0)   ← コンパイル時に結合
  QKV = matmul(x, W_qkv)                     ← Metal kernel 1回
  Q, K, V = split(QKV)                       ← Metal kernel 1回（軽量）

  入力 x を1回だけ読む。W_qkv を1回読む。
```

同様に MLP の Gate + Up projection もパッキングできる。

ただし packing にはカーネル互換性が条件になる。量子化モデルでは、Q, K, V の重みが同じ量子化パラメータ（ビット数、グループサイズ）を持つ必要がある。異なるパラメータが混在すると packing は不可能で、個別の matmul にフォールバックする。

コンパイラはこの判定をコンパイル時に行い、結果を `CompilationStats` に記録する：

```
packedAttentionCount:   24  ← QKV packing 成功
unpackedAttentionCount:  0  ← フォールバックなし
packedMLPCount:         24  ← Gate+Up packing 成功
unpackedMLPCount:        0  ← フォールバックなし
```

### Fused SubLayer — dispatch オーバーヘッドの削減

Transformer の各サブ層は同じパターンを繰り返す：

```
residual_save → norm → operation → residual_add
```

通常の実行では、この構造を再帰的に歩く（switch 文で分岐し、関数呼び出しのネストが深くなる）。Fused SubLayer はこのパターンを検出し、1つの関数呼び出しにまとめる：

```swift
// Before: 4つのステップ
case .saveResidual:
    residualStack.append(h)
case .op(.norm(let norm)):
    h = norm.apply(h)
case .op(.attention(let attn)):
    h = attn.apply(h, ...)
case .addResidual:
    h = residualStack.removeLast() + h

// After: 1つの fused ステップ
case .fusedSubLayer(.attention(let norm, let attn)):
    h = h + attn.apply(norm.apply(h), ...)
```

これは **Swift のディスパッチレベルの最適化**であり、Metal kernel の数は変わらない。しかし、decode（1トークンずつ生成する高頻度ループ）では kernel 間の CPU 側オーバーヘッドが蓄積するため、効果がある。

### Compile-time Kernel Selection — 実行時の分岐を排除

量子化モデルでは、重みが dense（Float16）か quantized（4-bit packed）かによって使うカーネルが変わる：

- **Dense** → `matmul(x, W.T)`
- **Quantized** → `quantizedMatmul(x, W_packed, scales, biases, ...)`

通常のフレームワークでは実行時に型を判定する。コンパイル時カーネル選択では、モデルロード時に重みの型を確認し、適切なカーネルを静的に確定する：

```swift
// コンパイル時に確定
public enum ProjectionKernel {
    case dense(weight: MLXArray)                    // → matmul
    case affineQuantized(AffineQuantizedTensor)     // → quantizedMatmul
    case dequantizeMatmul(AffineQuantizedTensor)    // → dequantize + matmul
}
```

実行時に `if quantized { ... } else { ... }` の分岐が不要になる。

## Metal Kernel Fusion — 本当のボトルネックに踏み込む

ここまでの最適化（packing、fused sublayer、compile-time kernel selection）は重要だが、本質的なボトルネックには触れていない。**本質的なボトルネックはメモリ帯域だ。**

### 問題の可視化

RMSNorm → matmul という2つの連続した演算を考える：

```
現状: 2つの独立した Metal kernel

  Global Memory                     GPU Compute
  ┌──────────┐
  │  x (D)   │──── read ──────────→ RMSNorm 計算
  │  w (D)   │──── read ──────────→
  └──────────┘                      │
  ┌──────────┐                      │
  │ norm (D) │←─── write ──────────┘  ← ここで Global Memory に書き戻す
  └──────────┘
  ┌──────────┐
  │ norm (D) │──── read ──────────→ matmul 計算    ← もう一度読み直す
  │  W (D×D) │──── read ──────────→
  └──────────┘                      │
  ┌──────────┐                      │
  │ out (D)  │←─── write ──────────┘
  └──────────┘

  norm(D) を「書く→読む」で 2回メモリアクセスが発生。
  D=2048 × Float16 = 4KB。帯域の無駄。
```

### Metal Kernel Fusion とは

2つの kernel を1つにまとめ、中間結果をオンチップの SRAM (Threadgroup Memory / Register) に留めることで、Global Memory への書き戻しを省く：

```
理想: 1つの fused Metal kernel

  Global Memory                     GPU Compute (SRAM)
  ┌──────────┐
  │  x (D)   │──── read ──────────→ RMSNorm 計算
  │  w (D)   │──── read ──────────→    │
  └──────────┘                         │ norm は SRAM に留まる
                                       ↓
  ┌──────────┐                      matmul 計算
  │  W (D×D) │──── read ──────────→    │
  └──────────┘                         │
  ┌──────────┐                         │
  │ out (D)  │←─── write ──────────────┘
  └──────────┘

  norm(D) の書き戻し + 読み直しが消える。
```

削減されるのは 4KB のメモリアクセスだ。1回の削減量は小さいが、24層 × 1トークンあたり複数箇所 × 毎トークン繰り返す。帯域が律速のデコードでは効果が蓄積する。

### 理想的な fusion の全体像

```
現状: 1層あたり ~14 kernel dispatch

  norm→matmul→split→RoPE→RoPE→cache→attention→matmul→add→norm→matmul→silu*up→matmul→add
  [1]   [2]    [3]   [4]  [5]  [6]    [7]      [8]   [9] [10]  [11]   [12]    [13]  [14]

理想: 1層あたり ~5 kernel dispatch

  ┌─── fused_norm_qkv_rope ───┐  ┌── attention ──┐  ┌── fused_proj_res_norm ──┐
  │ norm→matmul→split→RoPE×2  │  │  sdpa+cache   │  │ matmul→add→norm         │
  └────────────────────────────┘  └───────────────┘  └─────────────────────────┘
              [1]                        [2]                    [3]

  ┌── fused_gated_mlp ──┐  ┌── add ──┐
  │ matmul→silu*up→matmul│  │ residual│
  └──────────────────────┘  └─────────┘
           [4]                  [5]
```

### 実現の難しさ

Metal Kernel Fusion は「やれば速くなる」ことはわかっているが、実装コストが高い：

1. **手書き Metal kernel が必要** — matmul を含む fusion は、MLX の自動 fuse では実現できない。Metal Shading Language で専用 kernel を書く必要がある
2. **タイルサイズの最適化** — SRAM のサイズ（Threadgroup Memory は最大 32KB）に収まるようにタイル分割戦略を設計する必要がある
3. **量子化との組み合わせ** — Dense と Quantized で kernel が別になるため、組み合わせが爆発する
4. **機種ごとの調整** — M1 と M4 では GPU コア数や SRAM サイズが異なり、最適なタイルサイズも異なる

MLX 自体がすでに一部の fusion を提供している：

- `MLXFast.rmsNorm` — RMSNorm の内部演算を fuse
- `MLXFast.RoPE` — RoPE の回転演算を fuse
- `MLXFast.scaledDotProductAttention` — Attention の QK^T → softmax → ×V を fuse

これらは matmul の内部や単一演算の fuse であり、**matmul を跨いだ fusion（norm→matmul や matmul→activation→matmul）は提供されていない。**

## 実験: swift-mlx-lm で測定したこと

以下は [swift-mlx-lm](https://github.com/1amageek/swift-mlx-lm) プロジェクトの MLXCompiler で実際に測定した結果だ。環境は Apple M4 Max (36GB)。

### 実験 1: Fused SubLayer の効果

Dispatch-level fusion（fused sublayer）の効果を、同一モデルの fused / unfused バージョンで比較した。

| モデル規模 | 層数 | Fused decode | Unfused decode | 改善率 |
|-----------|------|-------------|---------------|--------|
| D=896 (Qwen3.5-0.6B相当) | 24L | 13.1 ms | 36.5 ms | **+64%** |
| D=2048 (Llama-3.2-1B相当) | 16L | 19.0 ms | 26.0 ms | **+27%** |

小さいモデル (D=896) で改善率が大きいのは、matmul 自体のコストが相対的に小さく、dispatch overhead の比率が高いため。大きいモデルでは matmul のコストが支配的になり、dispatch overhead の相対的な影響が小さくなる。

Fused と unfused で実行される Metal kernel の数は同じだ。変わるのは Swift 側の処理ステップ数：

```
D=896, 24L:
  Fused:    51 steps
  Unfused: 195 steps
  (residual の save/add ステップが消え、switch 分岐が減少)

D=2048, 16L:
  Fused:    35 steps
  Unfused: 131 steps
```

### 実験 2: Projection Packing の効果

D=2048 のモデルで、QKV packing と Gate+Up packing の効果を個別に測定した。

| Packing 対象 | Packed | Unpacked | 改善率 |
|-------------|--------|----------|--------|
| QKV projection | 1 dispatch | 3 dispatch | **+28-31%** |
| Gate+Up projection | 1 dispatch | 2 dispatch | **+13-15%** |

QKV packing の効果が大きいのは、3 dispatch → 1 dispatch の削減幅が大きく、入力テンソル x を3回読むところを1回に減らせるため。

### 実験 3: レイヤースケーリング

層数を変えたときの decode latency の変化を測定した（D=896、dense weights）。

| 層数 | Decode latency | 理想的な線形増加 |
|------|---------------|----------------|
| 4L | 11.3 ms | 基準 |
| 8L | 14.7 ms | 22.6 ms |
| 16L | 25.4 ms | 45.2 ms |
| 24L | 29.6 ms | 67.8 ms |

24L/4L の実測比は 2.6x。理想的な線形スケーリング（6.0x）より大幅に小さい。これは：

- **GPU パイプライニング** — MLX のストリーム機構が複数 kernel を重複実行
- **メモリ帯域の飽和** — 小さいモデルでは計算よりも kernel 起動の固定コストが支配的
- **キャッシュ効果** — 小さい重みは GPU のキャッシュに載る

### 実験 4: CompilationStats による最適化の可視化

コンパイラが各アーキテクチャで適用した最適化を `CompilationStats` として記録している。

**Transformer (Llama 4L, D=4, dense weights):**

```
packedAttentionCount:   4   ← 全層で QKV packing 成功
unpackedAttentionCount: 0
packedMLPCount:         4   ← 全層で Gate+Up packing 成功
unpackedMLPCount:       0
fusedSubLayerCount:     8   ← 4層 × 2 (attention + MLP) = 8 fused
unfusedResidualCount:   0
```

全最適化が適用されている。

### まとめ

| 最適化 | レベル | 何が減るか | 効果 |
|--------|-------|----------|------|
| **Projection Packing** | Metal kernel 数 | matmul dispatch 回数 | 中〜大 |
| **Fused SubLayer** | Swift dispatch | CPU 側のループ・分岐 | 小〜中 |
| **Compile-time Kernel Selection** | 実行時分岐 | if-else 判定 | 小 |
| **Metal Kernel Fusion** (未実装) | メモリ帯域 | Global Memory 往復 | **大** |

現在のコンパイラは上3つを実装している。最も効果の大きい Metal Kernel Fusion は未実装だ。

LLM 推論の高速化は「計算を速くする」問題ではなく「メモリの読み書きを減らす」問題だ。Apple Silicon の Unified Memory はデータコピーを排除するが、帯域の壁は残る。Metal Kernel Fusion はその壁を直接攻める技術であり、ローカル LLM 推論の次のフロンティアになる。
