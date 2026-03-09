# swift-mlx-lm

GGUF-first language model inference on Apple Silicon via MLX/Metal.

## Overview

A Swift package that loads a single GGUF file and runs LLM inference entirely on Metal — no Python, no external tokenizer files, no config.json.

```swift
import MLXLM

let loader = GGUFModelLoader()
let container = try await loader.load(repo: "Qwen/Qwen2.5-0.5B-Instruct-GGUF")

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
| **MLXLM** | GGUFParser, GGUFTokenizer, MLX | GGUF→MLX bridge, chat template rendering, streaming generation. |

```
GGUFParser          SwiftLM
    │                  │
GGUFTokenizer      Models
    │                  │
    └──── MLXLM ───────┘  (future integration)
```

## Supported Architectures

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

## Supported Quantizations

GGUF native: F32, F16, BF16, Q2_K, Q3_K, Q4_0, Q4_K, Q5_K, Q6_K, Q8_0

MLX runtime re-quantization: 2-bit, 4-bit, 8-bit (auto-detected from GGUF).

## Features

- **Single-file loading** — one GGUF file contains weights, tokenizer, and chat template
- **Chat template** — Jinja2 evaluation via [swift-jinja](https://github.com/huggingface/swift-jinja), no hand-written formatters
- **Streaming generation** — `AsyncStream<Generation>` with token-by-token output
- **Tool calling** — JSON and XML tool call format detection
- **LoRA/DoRA** — auto-detect from embedded GGUF tensors or load external adapters
- **Prompt caching** — `PromptCacheSnapshot` for prefix reuse across turns
- **Hybrid caching** — KV cache for attention layers, recurrent state for DeltaNet layers
- **HuggingFace Hub** — download models directly with `GGUFModelLoader.load(repo:)`

## SwiftLM

SwiftLM is a declarative model description framework for LLM architectures. It separates **what a model is** (structure and weights) from **how it runs** (backend, devices, caches).

### Design

SwiftLM follows the same paradigm as SwiftUI — models are declared, not constructed:

| SwiftUI | SwiftLM |
|---------|---------|
| `View` protocol | `LanguageModel` protocol |
| `body: some View` | `body: some ModelComponent` |
| `@ViewBuilder` | `@ModelComponentBuilder` |
| View tree → render | `ModelDeclaration` → `ModelGraph` |

A `LanguageModel` is a pure declaration. It is not a file loader, not a stateful runtime module, not a forward-pass executor.

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
```

The `Models` module provides concrete architecture declarations using SwiftLM components:

| Model | Architecture |
|-------|-------------|
| `Transformer` | Standard pre-norm decoder (Llama, Qwen 2, Mistral, Phi, Mixtral MoE) |
| `Qwen35` | Hybrid Gated DeltaNet + Full Attention |
| `Cohere` | LayerNorm + QK normalization (Command-R) |

### Components

Models are composed from semantic building blocks:

```swift
struct MyModel: LanguageModel {
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
| `Attention` | Multi-head attention with GQA, RoPE, QK norm, sliding window |
| `MLP` | Feed-forward with configurable activation and gating (SwiGLU, GELU, etc.) |
| `MoE` | Mixture-of-Experts routing |
| `RMSNorm` / `LayerNorm` | Normalization layers |
| `StateSpace` | SSM variants (Mamba, DeltaNet) |
| `OutputHead` | Hidden → vocabulary logits (optional weight tying) |
| `Residual` | Skip connection: `output = input + f(input)` |
| `Parallel` | Multi-branch with merge strategy (add, concat, stack) |
| `Repeat` | Stack identical blocks N times |
| `Linear` | Raw linear projection |
| `Custom` | Escape hatch for experimental ops |

### Pipeline

```
LanguageModel (DSL)
      │  .makeModelDeclaration()
      ▼
ModelDeclaration (open tree)
      │  normalize()
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
      │  canonicalize()
      ▼
ModelGraph (canonical form — for equivalence comparison)
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

Weights are attached externally via the `.weights()` modifier — they are NOT part of `LanguageModel`:

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

`WeightedModel` carries both structure and weight source, but is intentionally NOT a `LanguageModel` — it represents a different concept (structure + weights bundle).

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
