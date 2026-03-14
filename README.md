# swift-lm

High-performance LLM inference on Apple Silicon.

## Overview

A Swift package for LLM inference on Apple Silicon. Models are loaded directly from HuggingFace directories (config.json + safetensors + tokenizer.json) — no model-specific Swift types required.

```swift
import LMInference

// Just specify a HuggingFace repo ID — that's all
let container = try await ModelBundleLoader().load(repo: "mlx-community/Qwen2.5-0.5B-Instruct-4bit")

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

## Architecture

Three core modules with clear separation of concerns:

| Module | Depends on | Description |
|--------|------------|-------------|
| **SwiftLM** | — | IR schema + declarative DSL for model architectures |
| **ModelDeclarations** | SwiftLM | DSL model declarations (design & training) |
| **LMCompiler** | SwiftLM | Compiles IR to optimized inference engines (MLX + MPSGraph) |
| **LMInference** | SwiftLM, LMCompiler | HF loader, tokenizer, generation pipeline, backend selection |

```
SwiftLM ─── ModelDeclarations
    │
LMCompiler
    │
LMInference
```

**Important**: LMInference does not depend on ModelDeclarations. `IRGraphAssembler` builds IR directly from config.json.

### Inference Backends

MPSGraph is the default backend. MLX is used as fallback for hybrid architectures.

| Backend | Target Architectures | Advantage |
|---|---|---|
| **MPSGraph** (default) | transformer, parallelAttentionMLP, MoE | MPSGraph fused execution (1.6x faster) |
| **MLX** (fallback) | hybridDeltaNet, hybridConvAttention, all | Dynamic shapes, LoRA, quantization |

```swift
// Auto-selects backend (default: MPSGraph)
let container = try await ModelBundleLoader().load(repo: "...", backend: .auto)

// Force MLX backend
let container = try await ModelBundleLoader().load(repo: "...", backend: .mlx)
```

### HF-First Pipeline

```
HF Directory (config.json + *.safetensors + tokenizer.json)
     │
     ▼
ModelBundleLoader
  ├── HFConfigDecoder: config.json → ModelConfig
  ├── HFArchitectureDetector: model_type → DetectedArchitecture
  ├── IRGraphAssembler: (ModelConfig, DetectedArchitecture) → ModelGraph (IR)
  │
  ├── [MPSGraph path — default]
  │   └── MPSGraphInferenceCompiler → MPSGraphInferenceModel → MPSGraphLanguageModel
  │
  └── [MLX path — fallback]
      ├── HFDirectoryBundle: safetensors → WeightManifest → RawWeights
      ├── MLXWeightPathBinder: RawWeights → BoundWeights
      └── MLXInferenceCompiler → MLXInferenceModel → MLXLanguageModel
           │
           ▼
     ModelContainer → TokenIterator → generate()
```

## Supported Architectures

### Declarative Models (ModelDeclarations)

| Model | Architecture | VLM |
|-------|-------------|-----|
| `Transformer` | Standard pre-norm decoder (Llama, Qwen 2, Mistral, Phi, Gemma, Mixtral MoE) | — |
| `Qwen35` | Hybrid Gated DeltaNet + Full Attention | — |
| `Cohere` | LayerNorm + QK normalization (Command-R) | — |
| `LFM2` | Hybrid ShortConv + GQA Attention (LiquidAI LFM2/2.5) | — |

## Features

- **HF-first loading** — config.json + safetensors + tokenizer.json is all you need
- **Dual backends** — MPSGraph (default, 1.6x faster) with MLX fallback
- **Chat template** — Jinja2 evaluation via [swift-jinja](https://github.com/huggingface/swift-jinja), no hand-written formatters
- **Streaming generation** — `AsyncStream<Generation>` with token-by-token output
- **Tool calling** — JSON and XML tool call format detection
- **LoRA/DoRA** — adapter loading support
- **Prompt caching** — `PrefixCachePool` for live cache reuse, `PromptCacheSnapshot` for serialized prefix restore
- **Hybrid caching** — KV cache for attention layers, recurrent state for DeltaNet layers
- **Pre-quantized models** — mlx-community safetensors with native quantization format

## SwiftLM DSL

SwiftLM is a declarative model description framework. It separates **what a model is** (structure) from **how it runs** (backend).

```swift
import SwiftLM
import ModelDeclarations

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

| Component | Description |
|-----------|-------------|
| `TokenEmbedding` | Token ID → dense vector |
| `Attention` | Multi-head attention with GQA, RoPE, M-RoPE, QK norm, sliding window |
| `MLP` | Feed-forward with configurable activation and gating (SwiGLU, GELU, etc.) |
| `MoE` | Mixture-of-Experts routing |
| `RMSNorm` / `LayerNorm` | Normalization layers |
| `StateSpace` | SSM variants (Mamba, DeltaNet, ShortConv) |
| `OutputHead` | Hidden → vocabulary logits (optional weight tying) |
| `Residual` | Skip connection |
| `Parallel` | Multi-branch with merge strategy |
| `Repeat` | Stack identical blocks N times |

## Installation

Add to your `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/1amageek/swift-lm.git", from: "0.1.0"),
],
targets: [
    .target(
        name: "YourTarget",
        dependencies: [
            .product(name: "LMInference", package: "swift-lm"),
        ]
    ),
]
```

Requires macOS 15+, iOS 18+, or visionOS 2+. Swift 6.2.

## Dependencies

- [mlx-swift](https://github.com/ml-explore/mlx-swift) (0.30.6+) — MLX framework
- [swift-jinja](https://github.com/huggingface/swift-jinja) (2.3.2+) — Chat template evaluation
- [swift-transformers](https://github.com/huggingface/swift-transformers) (1.1.9+) — Tokenizers and HuggingFace Hub

## License

MIT
