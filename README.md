# swift-mlx-lm

Declarative LLM inference on Apple Silicon via MLX/Metal.

## Overview

A Swift package for LLM inference on Apple Silicon. Models are declared with the SwiftLM DSL, compiled to optimized MLX inference engines, and loaded from GGUF or safetensors.

```swift
import MLXLM

let loader = GGUFModelLoader()

// Standard path — MLXNN Module tree
let container = try await loader.load(repo: "Qwen/Qwen2.5-0.5B-Instruct-GGUF")

// Compiled path — SwiftLM IR → MLXInferenceCompiler (compile-time kernel selection)
let compiled = try await loader.loadCompiled(repo: "Qwen/Qwen2.5-0.5B-Instruct-GGUF")

// Both paths produce identical ModelContainer API
let input = UserInput(chat: [
    .system("You are a helpful assistant."),
    .user("What is 1+1?"),
])
let lmInput = try await container.prepare(input: input)

let stream = try await container.generate(input: lmInput, parameters: GenerateParameters())
for await generation in stream {
    switch generation {
    case .chunk(let text):
        print(text, terminator: "")
    case .info(let info):
        print("\n\(info.tokensPerSecond) tok/s")
    default:
        break
    }
}
```

## Modules

| Module | Depends on | Description |
|--------|------------|-------------|
| **GGUFParser** | — | Binary parser, metadata extraction, tensor directory. |
| **GGUFTokenizer** | GGUFParser | BPE tokenizers restored from GGUF metadata. Merges-based and scores-based. |
| **SwiftLM** | — | Declarative model description framework. DSL → IR → validation → canonicalization. |
| **Models** | SwiftLM | Pure architecture declarations. No MLX dependency. |
| **MLXCompiler** | SwiftLM, MLX | Compiles ModelGraph into optimized inference engines with compile-time kernel selection. |
| **MLXLM** | GGUFParser, GGUFTokenizer, MLXCompiler, MLX | Weight loading (GGUF/safetensors), chat template rendering, streaming generation. |

```
GGUFParser          SwiftLM ─── Models
    │                  │
GGUFTokenizer      MLXCompiler
    │                  │
    └──── MLXLM ──────┘
```

## Supported Architectures

### GGUF Runtime (MLXLM)

| Architecture | GGUF ID | Model |
|-------------|---------|-------|
| Llama | `llama` | `TransformerModel` |
| Qwen 2/2.5 | `qwen2` | `TransformerModel` |
| Qwen 3 | `qwen3` | `TransformerModel` |
| Mistral | `mistral` | `TransformerModel` |
| Gemma 2 | `gemma2` | `TransformerModel` |
| Phi-3 | `phi3` | `TransformerModel` |
| StarCoder 2 | `starcoder2` | `TransformerModel` |
| Mixtral (MoE) | `llama` + experts | `TransformerModel` |
| Qwen 3.5 (DeltaNet) | `qwen35` | `Qwen35Model` |
| Command-R | `command-r` | `CohereModel` |

### Declarative Models (Models)

| Model | Architecture | VLM |
|-------|-------------|-----|
| `Transformer` | Standard pre-norm decoder (Llama, Qwen 2, Mistral, Phi, Mixtral MoE) | — |
| `Qwen35` | Hybrid Gated DeltaNet + Full Attention with VisionEncoder + M-RoPE | Yes |
| `Cohere` | LayerNorm + QK normalization (Command-R) | — |
| `LFM2` | Hybrid ShortConv + GQA Attention (LiquidAI LFM2/2.5) | — |

#### LFM2 Presets

| Preset | Layers | MoE | Pattern |
|--------|--------|-----|---------|
| `lfm2_350M` | 16 | — | (conv×2+attn)×3, (conv+attn)×3, conv×1 |
| `lfm25_1_2B` | 16 | — | (conv×2+attn)×3, (conv+attn)×3, conv×1 |
| `lfm2_2_6B` | 30 | — | (conv×2+attn)×2, (conv×3+attn)×4, (conv×2+attn)×2, conv×2 |
| `lfm2_8B_A1B` | 24 | 32 experts, 4 active | (conv×2+attn)×1, (conv×3+attn)×4, (conv×2+attn)×1, conv×2 |
| `lfm2_24B_A2B` | 40 | 64 experts, 4 active | (conv×2+attn)×1, (conv×3+attn)×9, conv×1 |

## Supported Quantizations

GGUF native: F32, F16, BF16, Q2_K, Q3_K, Q4_0, Q4_K, Q5_K, Q6_K, Q8_0

MLX runtime re-quantization: 2-bit, 4-bit, 8-bit (auto-detected from GGUF).

## Features

- **Declarative architecture** — models described with SwiftLM DSL, compiled to optimized MLX inference
- **Two loading paths** — standard path (`loadContext`) via MLXNN Module, compiled path (`loadCompiledContext`) via MLXInferenceCompiler with compile-time kernel selection
- **Single-file loading** — one GGUF file contains weights, tokenizer, and chat template
- **Chat template** — Jinja2 evaluation via [swift-jinja](https://github.com/huggingface/swift-jinja), no hand-written formatters
- **Streaming generation** — `AsyncStream<Generation>` with token-by-token output
- **Tool calling** — JSON and XML tool call format detection
- **LoRA/DoRA** — auto-detect from embedded GGUF tensors or load external adapters
- **Prompt caching** — `PrefixCachePool` for live cache reuse, `PromptCacheSnapshot` for serialized prefix restore
- **Hybrid caching** — KV cache for attention layers, recurrent state for DeltaNet layers
- **HuggingFace Hub** — download models directly with `GGUFModelLoader.load(repo:)`

## SwiftLM

SwiftLM is a declarative model description framework for LLM architectures. It separates **what a model is** (structure and weights) from **how it runs** (backend, devices, caches).

### Declaring a Model

```swift
import SwiftLM
import Models

let llama = Transformer(config: .init(
    hiddenSize: 4096,
    hiddenLayers: 32,
    intermediateSize: 11008,
    attentionHeads: 32,
    kvHeads: 8,
    vocabularySize: 32000
))

let graph = try llama.makeModelGraph()
```

### Components

Models are composed from semantic building blocks:

```swift
struct MyModel: ModelComponent {
    @ModelComponentBuilder
    var body: some ModelComponent {
        TokenEmbedding(vocabSize: 32000, embeddingSize: 4096)
        Repeat(count: 32, label: "layers") {
            Residual {
                RMSNorm(dimension: 4096)
                Attention(
                    hiddenSize: 4096,
                    headCount: 32,
                    kvHeadCount: 8,
                    rope: RoPEAttributes(dimension: 128)
                )
            }
            Residual {
                RMSNorm(dimension: 4096)
                MLP(inputSize: 4096, intermediateSize: 11008)
            }
        }
        RMSNorm(dimension: 4096)
        OutputHead(inputSize: 4096, vocabSize: 32000)
    }
}

// Weights are attached externally:
let weighted = MyModel().weights(.gguf(location: "model.gguf"))
```

Available components:

| Component | Description |
|-----------|-------------|
| `TokenEmbedding` | Token ID → dense vector |
| `Attention` | Multi-head attention with GQA, RoPE, M-RoPE, QK norm, sliding window |
| `MLP` | Feed-forward with configurable activation and gating (SwiGLU, GELU, etc.) |
| `MoE` | Mixture-of-Experts routing |
| `RMSNorm` / `LayerNorm` | Normalization layers |
| `StateSpace` | SSM variants (Mamba, DeltaNet, ShortConv) |
| `VisionEncoder` | Vision Transformer for VLM (patch embedding, spatial merge, temporal patch) |
| `OutputHead` | Hidden → vocabulary logits (optional weight tying) |
| `Residual` | Skip connection: `output = input + f(input)` |
| `Parallel` | Multi-branch with merge strategy (add, concat, visionMerge) |
| `Repeat` | Stack identical blocks N times |
| `ForEach` | Data-driven iteration with type-safe content |
| `Group` | Labeled grouping for metadata |
| `Linear` | Raw linear projection |
| `RoPE` | Standalone rotary position embedding |
| `Custom` | Escape hatch for experimental ops |

### VLM Support

Vision-language models use `Parallel(merge: .visionMerge(...))` to merge vision and text branches:

```swift
struct MyVLM: ModelComponent {
    @ModelComponentBuilder
    var body: some ModelComponent {
        Parallel(merge: .visionMerge(VisionMergeConfig(
            imageTokenId: 248056,
            videoTokenId: 248057
        ))) {
            TokenEmbedding(vocabSize: 248320, embeddingSize: 1024)
            VisionEncoder(
                hiddenSize: 768,
                outputSize: 1024,
                depth: 12,
                headCount: 12,
                patchSize: 16,
                intermediateSize: 3072,
                temporalPatchSize: 2
            )
        }
        // ... decoder layers with M-RoPE ...
    }
}
```

### Pipeline

```
ModelComponent (DSL)
      │  .makeModelGraph()
      ▼
NormalizedModel
  ├── ModelGraph (semantic IR)
  └── ModelGraphMetadata (labels, diagnostics)
      │
      │  GraphValidator.validate()      — SSA dominance, arity contracts
      │  LLMProfileValidator.validate() — single-result root, operand arity
      │
      ▼
ModelGraph (validated)
      │
      ├── canonicalize() → canonical form (for equivalence comparison)
      │
      └── MLXInferenceCompiler.compile() → MLXLoweredInferenceModel
              │                              (prefill/decode with FlatStep execution)
              │
              └── CompiledLanguageModel adapter → LanguageModel protocol
                      → ModelContainer → generate()
```

### Semantic IR

`ModelGraph` is a hierarchical SSA IR inspired by MLIR:

- **Regions** — scoped blocks with explicit parameters (inputs) and results (outputs)
- **Operations** — consume operands, produce results (SSA: defined once, used many times)
- **Structural operations** (`residual`, `parallel`, `repeating`) contain nested Regions with scope isolation
- **Metadata sidecar** — labels and diagnostics stored separately, never affect canonical equivalence

### Canonical Equivalence

Two models with the same structure produce identical canonical graphs, regardless of how they were declared:

```swift
let graph1 = try modelA.makeModelGraph()
let graph2 = try modelB.makeModelGraph()

canonicalize(graph1) == canonicalize(graph2)  // structural equivalence
```

### Weights Declaration

Weights are attached externally via the `.weights()` modifier — they are NOT part of the model declaration:

```swift
let model = Transformer(config: .init(
    hiddenSize: 4096, hiddenLayers: 32, intermediateSize: 11008,
    attentionHeads: 32, kvHeads: 8, vocabularySize: 32000
))

// Attach weights:
let weighted = model.weights(.gguf(location: "base.gguf"))

// Or compose multiple weight sources:
let weighted = model.weights {
    WeightsDeclaration.override(
        base: .gguf(location: "base.gguf"),
        with: .safetensors(directory: "adapter/", indexFile: nil)
    )
}
```

`WeightedModel` carries both structure and weight source, but is intentionally NOT a `ModelComponent` — it represents a different concept (structure + weights bundle).

The weight pipeline resolves declarations into concrete tensors independently from the structural graph:

```
WeightsDeclaration → WeightsResolver → RawWeights
                                           │
ModelGraph + RawWeights → WeightBinder → BoundWeights → Compiler
```

`ParameterSlot` (a `StructuralPath` + `ParameterRole` pair) provides stable addressing that survives normalization and canonicalization.

## Installation

Add to your `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/1amageek/swift-mlx-lm.git", from: "0.1.0"),
],
targets: [
    .target(
        name: "YourTarget",
        dependencies: [
            .product(name: "MLXLM", package: "swift-mlx-lm"),
        ]
    ),
]
```

Requires macOS 15+, iOS 18+, or visionOS 2+. Swift 6.2.

## Dependencies

- [mlx-swift](https://github.com/ml-explore/mlx-swift) (0.30.6+) — MLX framework
- [swift-jinja](https://github.com/huggingface/swift-jinja) (2.3.2+) — Chat template evaluation

## License

MIT
