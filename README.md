# swift-lm

High-performance LLM inference on Apple Silicon using direct Metal compute.

## Overview

A Swift package for LLM inference on Apple Silicon. Models are loaded directly from HuggingFace directories (config.json + safetensors + tokenizer.json) — no model-specific Swift types required. The consumer specifies only a HuggingFace repo ID.

```swift
import SwiftLM

let container = try await ModelBundleLoader().load(repo: "LiquidAI/LFM2.5-1.2B-Instruct")

for await generation in container.generate(input: try container.prepare(input: UserInput("Hello"))) {
    if let text = generation.chunk { print(text, terminator: "") }
}
```

## Architecture

```
LMIR (IR — no dependencies)
    │  ModelGraph, OperationKind, ParameterBinding, OperationAttributes
    │
    ├── LMArchitecture (DSL + Validation — re-exports LMIR)
    │   ├── ModelComponent protocol + @ModelComponentBuilder
    │   ├── Components: Attention, MLP, RMSNorm, ShortConv, ...
    │   └── SemanticNormalizer, GraphValidator, DimensionValidator
    │
    ├── ModelDeclarations (depends: LMArchitecture)
    │   └── Transformer, Qwen35, LFM2, Cohere
    │
    ├── MetalCompiler (depends: LMIR only)
    │   ├── MetalKernelFragment protocol + fragment tree
    │   ├── MetalSourceGenerator (on-demand MSL generation)
    │   ├── MetalInferenceCompiler (IR walk → fusion → dispatch plan)
    │   ├── MetalInferenceModel (decode/prefill execution)
    │   └── STAF (weight format, parameter resolution)
    │
    └── SwiftLM (consumer API)
        ├── ModelBundleLoader (HF download → STAF → compile)
        ├── ModelContainer (generate, encode, decode)
        └── UserInput, ChatMessage, GenerateParameters
```

## Metal Backend

All inference runs on direct Metal compute — no MLX, no MPSGraph.

### Fragment-Driven Kernel Generation

Each `ModelComponent` has a corresponding `MetalKernelFragment`, declaring its Metal execution as a compositional tree:

```
LMArchitecture/Components/          MetalCompiler/Fragments/
├── Attention.swift            ↔    ├── AttentionFragment.swift
├── MLP.swift                  ↔    ├── MLPFragment.swift
├── Norm.swift                 ↔    ├── NormFragment.swift
├── ShortConv.swift            ↔    ├── ShortConvFragment.swift
├── TokenEmbedding.swift       ↔    ├── TokenEmbeddingFragment.swift
├── OutputHead.swift           ↔    ├── OutputHeadFragment.swift
└── ...                        ↔    └── ...
```

The compiler walks the fragment tree to:
1. Emit dispatch entries from primitive fragments (Reduction, Linear, FlashAttention, ...)
2. Fuse adjacent fragments (residualAdd + copy + RMSNorm → single kernel)
3. Generate MSL kernel source on-demand from fragment parameters + weight format + buffer precision
4. Compile into MTLLibrary → dispatch plan

No hardcoded kernel variants. dtype/precision is determined by the compiler from STAF weight format and execution phase.

### Precision

- **Prefill**: Float32 hidden/residual/scratch — prevents accumulation error across 16+ layers
- **Decode**: Float16 — single token per step, no accumulation
- **Weights**: BFloat16 natively supported — `bf16_to_float()` conversion in kernel
- **KV cache**: Float16

### STAF (SafeTensor Accelerated Format)

Weights are converted from safetensors to STAF for zero-copy GPU loading:

```
*.safetensors → STAFConverter (once) → *.staf → mmap + bytesNoCopy → MTLBuffer
```

## Supported Models

| Model | Type | Architecture |
|-------|------|-------------|
| Transformer | Dense / MoE | Llama, Qwen 2/3, Mistral, Gemma, Phi, Mixtral, DeepSeek |
| Qwen 3.5 | Hybrid | Gated DeltaNet + Full Attention |
| LFM2 / LFM2.5 | Hybrid | ShortConv + GQA Attention (dense and MoE) |
| Cohere | Dense | LayerNorm + QK normalization (Command-R) |

## Build & Test

```bash
swift build
```

Tests require Metal GPU — use `xcodebuild test`, not `swift test`:

```bash
xcodebuild test -scheme swift-lm-Package -destination 'platform=macOS' \
    -only-testing 'ModelsTests' -parallel-testing-enabled NO
```

### Reference Comparison Tests

Verify Metal output against Python HuggingFace reference:

```bash
# Generate reference tensors
python3 scripts/dump_lfm2_reference.py

# Run comparison
xcodebuild test -scheme swift-lm-Package -destination 'platform=macOS' \
    -only-testing 'MetalCompilerTests/ReferenceComparisonTests'
```

## Requirements

- macOS 15+ / iOS 18+ / visionOS 2+
- Swift 6.2+
- Apple Silicon (Metal GPU)

## Dependencies

- [swift-jinja](https://github.com/huggingface/swift-jinja) — Chat template evaluation
- [swift-transformers](https://github.com/huggingface/swift-transformers) — Tokenizers and HuggingFace Hub

## License

MIT
