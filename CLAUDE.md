# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

swift-mlx-lm is a Swift package providing GGUF-first language model inference on Apple Silicon via MLX/Metal. It is consumed by [AnyFoundationModels](https://github.com/1amageek) as the `MLXFoundationModels` backend.

**Core design principle**: GGUF is the standard input, `chat_template` is the canonical prompt formatter, MLX/Metal handles optimized execution. No dependency on swift-transformers — tokenizer, chat template evaluation, and model configuration are all restored from GGUF metadata.

## Build & Test

```bash
# Build
swift build

# Run all tests (always use xcodebuild — swift test crashes on Metal-dependent tests)
xcodebuild test -scheme swift-mlx-lm-Package -destination 'platform=macOS'

# Run specific module
xcodebuild test -scheme swift-mlx-lm-Package -destination 'platform=macOS' -only-testing:MLXLMTests
xcodebuild test -scheme swift-mlx-lm-Package -destination 'platform=macOS' -only-testing:GGUFParserTests
xcodebuild test -scheme swift-mlx-lm-Package -destination 'platform=macOS' -only-testing:GGUFTokenizerTests
```

**Important**: `swift test` は使わない — Metal metallib が見つからずクラッシュするため。テスト実行は常に `xcodebuild test` を使用すること。

Swift tools version: 6.2. Platforms: macOS 15, iOS 18, visionOS 2.

## Architecture

### Design Philosophy (vs mlx-swift-lm)

This project differs fundamentally from mlx-swift-lm:

| | mlx-swift-lm | swift-mlx-lm |
|---|---|---|
| Input format | safetensors + config.json + tokenizer.json | GGUF single file |
| Tokenizer | swift-transformers dependency | Self-contained from GGUF metadata |
| Chat template | swift-transformers Jinja evaluator | swift-jinja (huggingface/swift-jinja) |
| Quantization | MLX post-load quantization | GGUF native (Q4_K, Q8_0, etc.) direct on Metal |
| Dependencies | mlx-swift + swift-transformers + HubApi | mlx-swift only |

### Integration Point

AnyFoundationModels (`/Users/1amageek/Desktop/AnyFoundationModels`) consumes this package through `MLXFoundationModels` module. The key interface is `ModelContainer`, which provides:
- `prepare(input:)` — tokenization and prompt processing
- `perform(values:operation:)` — generation with exclusive model access
- `generate(input:parameters:)` — streaming generation via `AsyncStream<Generation>`
- `decode(tokens:)` — reverse tokenization

### Module Structure

- **GGUFParser** — Binary parser, metadata extraction, tensor directory, mmap/lazy load. No external deps.
- **GGUFTokenizer** — BPE tokenizers from GGUF metadata. Merges-based (GPT-2/Llama3/Qwen2) and scores-based (SentencePiece/Llama1/2). Depends on GGUFParser.
- **MLXLM** — Model architectures, GGUF→MLX bridge, generation engine. Depends on GGUFParser, GGUFTokenizer, swift-jinja, mlx-swift.

External dependencies: `mlx-swift` (0.30.6+), `swift-jinja` (2.3.2+, library name `Jinja`).

### MLXLM Data Flow

```
GGUF URL → GGUFModelLoader.load(url:)
  ├─ GGUFFile.parse()          → metadata + tensor directory
  ├─ GGUFConfigExtractor       → LlamaConfiguration
  ├─ LlamaModel(config)        → empty model
  ├─ GGUFTensorBridge          → dequantize tensors → MLXArray
  ├─ LlamaTensorNameMapper     → GGUF names → MLX weight paths
  ├─ model.update(parameters:) → loaded model
  ├─ GGUFTokenizerFactory      → Tokenizer
  ├─ ChatTemplateRenderer      → Jinja prompt formatter
  └─ ModelContainer(context:)  → ready for generation
```

### MLXLM Key Types

- `ModelContainer` (actor) — Serializes all model access
- `ModelContext` — Bundles model + tokenizer + processor + config
- `GGUFModelLoader` — End-to-end GGUF → ModelContainer
- `GGUFTensorBridge` — Dequantizes F16/F32/BF16/Q4_0/Q8_0/Q2_K-Q6_K
- `LlamaModel` — Llama/Mistral/Qwen2 architecture
- `TokenIterator` — Autoregressive token generation loop
- `generate()` — Top-level AsyncStream generation function

### Key Constraints

- GGUF single-file self-contained — no external tokenizer.json or config.json
- No hand-written per-model formatters — `chat_template` evaluation is canonical
- No float16 fallback for unsupported quantization — all major GGUF quant types must run natively
- All public types must be `Sendable`
