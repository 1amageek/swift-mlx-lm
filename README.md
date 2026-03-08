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

| Module | Description |
|--------|-------------|
| **GGUFParser** | Binary parser, metadata extraction, tensor directory. No external deps. |
| **GGUFTokenizer** | BPE tokenizers restored from GGUF metadata. Merges-based (GPT-2/Llama 3/Qwen) and scores-based (SentencePiece). |
| **MLXLM** | Model architectures, GGUF→MLX bridge, chat template rendering, streaming generation. |

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
