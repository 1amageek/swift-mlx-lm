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

## Release Status

`0.1.0` is the first public release candidate for application developers who want direct Metal inference from Swift.

Supported in `0.1.0`:

- Apple Silicon devices with Metal
- HuggingFace snapshot directories containing `config.json`, `tokenizer.json`, and `.safetensors`
- direct Metal decode/prefill execution through `SwiftLM`
- text prompts and chat prompts through `UserInput`
- the currently documented model families in this README and in `docs/using-swift-lm.md`

Not part of `0.1.0`:

- multimodal image or video input
- tool calling or structured function-calling APIs
- non-Metal backends
- training or fine-tuning workflows
- non-Apple-Silicon deployment targets

## Developer Quick Start

Application developers should start with [`docs/using-swift-lm.md`](docs/using-swift-lm.md). It covers:

- SwiftPM integration
- required model bundle files
- loading from HuggingFace or a local directory
- text and chat generation
- `PromptState` reuse for shared prefixes
- cache reset and tokenizer helpers
- current public API limits and troubleshooting

DocC sources now live in [`Sources/SwiftLM/SwiftLM.docc`](/Users/1amageek/Desktop/swift-lm/Sources/SwiftLM/SwiftLM.docc).

Package requirements currently declared by `Package.swift`:

- Swift 6.2+
- macOS 26+, iOS 26+, visionOS 26+
- a Metal-capable device

SwiftPM dependency example:

```swift
dependencies: [
    .package(url: "https://github.com/1amageek/swift-lm.git", from: "0.1.0")
]
```

## Architecture

```
LMIR (IR — no dependencies)
    │  ModelGraph, OperationAttributes, ParameterBinding
    │  Pure data types. No Metal. No backend awareness.
    │
    ├── LMArchitecture (DSL + Validation — re-exports LMIR)
    │   ├── ModelComponent protocol + @ModelComponentBuilder
    │   ├── Components: Attention, MLP, RMSNorm, ShortConv, ...
    │   └── SemanticNormalizer, GraphValidator, DimensionValidator
    │
    ├── ModelDeclarations (depends: LMArchitecture)
    │   └── Transformer, Qwen35, LFM2, Cohere
    │
    ├── MetalCompiler (depends: LMIR only — not LMArchitecture)
    │   ├── PrimitiveMetalKernelFragment protocol
    │   ├── DispatchOptimizer protocol (pluggable graph optimization)
    │   ├── MetalInferenceCompiler (IR walk → optimize → compiled model)
    │   ├── MetalCompiledModel (opaque decode/prefill runtime artifact)
    │   ├── MetalInferenceModel (decode/prefill execution façade)
    │   └── STAF (weight format, parameter resolution)
    │
    └── SwiftLM (consumer API)
        ├── ModelBundleLoader (HF download → STAF → compile)
        ├── ModelContainer (generate, encode, decode)
        └── UserInput, ChatMessage, GenerateParameters
```

### Module Dependency Direction

```
LMIR  ←──  LMArchitecture  ←──  ModelDeclarations
  │                                     │
  └──────  MetalCompiler                │
                │                       │
                └───────  SwiftLM  ─────┘
```

`LMArchitecture` and `MetalCompiler` never depend on each other. Both depend on `LMIR`. This separation allows multiple backends to coexist without modification.

## Design Principles

### IR and Backend Separation

LMIR describes **connections only**. Computation is opaque — stored as `any OperationAttributes`. Backends extend these types to add execution behavior:

```swift
// LMIR — backend-independent
public struct AttentionAttributes: OperationAttributes {
    let hiddenSize: Int, headCount: Int, kvHeadCount: Int, ...
}

// MetalCompiler — extends IR type with Metal execution
extension AttentionAttributes: MetalKernelFragment {
    func fragment(context: KernelContext) -> some MetalKernelFragment { ... }
}
```

New backends extend the same IR types independently:

```swift
// Future TPUCompiler — no changes to LMIR or MetalCompiler
extension AttentionAttributes: TPUKernelFragment { ... }
```

### Fragment Declares, Compiler Dispatches

Each fragment owns its complete execution specification through protocol methods. The compiler never checks concrete fragment types — it dispatches through the protocol:

| Responsibility | Protocol Method | Owner |
|---|---|---|
| Kernel name | `kernelName(context:)` | Fragment |
| MSL source | `kernelSource(name:bufferPrecision:weightFormat:)` | Fragment |
| Decode buffer layout | `decodeBindings(context:)` | Fragment |
| Prefill step building | `prefillSteps(context:)` | Fragment |
| Dispatch optimization | `DispatchOptimizer.optimizeFragment()` | Optimizer |

The compiler is a thin dispatcher that reads protocol properties and calls protocol methods. Adding a new fragment never requires modifying the compiler.

### Pluggable Graph Optimization

The `DispatchOptimizer` protocol separates optimization strategy from compilation:

```swift
let compiler = MetalInferenceCompiler(optimizer: AggressiveOptimizer())
```

| Optimizer | Dispatch Count | Strategy |
|---|---|---|
| `NoOptimizer` | 242 | No optimization (baseline) |
| `StandardOptimizer` | 179 | Exact-shape MLP front-half fusion + norm fusion |
| `AggressiveOptimizer` | 144 | Standard optimization + projection batching + per-head batching |

Optimization runs during the IR walk (collect → optimize → emit), not as a post-hoc pass on a flat list. This preserves structural information from the IR.

### Context-Aware Fragment Tree

Fragment tree evaluation receives `KernelContext` (buffer precision + weight format), enabling fragments to make context-dependent decisions:

```swift
func fragment(context: KernelContext) -> some MetalKernelFragment {
    // Can vary tree structure based on context
    LinearFragment(field: "q_proj", ...)
    LinearFragment(field: "k_proj", ...)
    // ...
}
```

## Adding a New Component

### Step 1: IR Attributes (LMIR)

Define the operation's parameters as a backend-independent data type:

```swift
public struct NewOpAttributes: OperationAttributes, Sendable {
    public let dimension: Int
    public let epsilon: Float
}
```

### Step 2: Model Component (LMArchitecture)

Create the DSL component that builds IR:

```swift
public struct NewOp: ModelComponent {
    let dimension: Int
    let epsilon: Float

    public var body: some ModelComponent {
        Primitive(NewOpAttributes(dimension: dimension, epsilon: epsilon))
    }
}
```

### Step 3: Metal Fragment (MetalCompiler/Fragments/)

Extend the IR attributes with Metal execution — this is the bridge:

```swift
// Fragments/NewOpFragment.swift
extension NewOpAttributes: MetalKernelFragment, _FragmentBodyAccessor {
    @MetalKernelFragmentBuilder
    public func fragment(context: KernelContext) -> some MetalKernelFragment {
        NewOpPrimitiveFragment(dimension: dimension, epsilon: epsilon)
    }
    public var isFusable: Bool { false }
    public func _visitBody(context: KernelContext, _ visitor: (any MetalKernelFragment) -> Void) {
        visitor(fragment(context: context))
    }
}
```

### Step 4: Primitive Fragment (MetalCompiler/Fragments/Primitives/)

Define the leaf dispatch unit with all 4 protocol methods:

```swift
// Fragments/Primitives/NewOpPrimitiveFragment.swift
public struct NewOpPrimitiveFragment: PrimitiveMetalKernelFragment {
    public let dimension: Int
    public let epsilon: Float

    public var dispatchDimension: MetalDispatchDimension { .reduction(dimension: dimension) }
    public var isFusable: Bool { true }

    public func kernelName(context: KernelContext) -> String {
        context.bufferPrecision == .float32 ? "new_op_f32" : "new_op"
    }

    public func kernelSource(name: String, bufferPrecision: BufferPrecision, weightFormat: WeightFormat) -> String {
        MetalSourceGenerator.generateNewOp(name: name, ...)
    }

    public func decodeBindings(context: BufferBindingContext) -> FragmentBindings {
        FragmentBindings(
            buffers: [(0, context.bufferSet.hidden, 0), ...],
            bytes: [uint32Binding(1, UInt32(dimension))],
            outputIsHidden: true)
    }

    public func prefillSteps(context: PrefillBindingContext) throws -> FragmentPrefillSteps {
        // Build MetalPrefillStep with grid/threadgroup for sequence processing
    }
}
```

### What You Don't Need to Modify

- **MetalInferenceCompiler** — dispatches through protocol, no type checks
- **DispatchOptimizer** — detects optimization opportunities from properties
- **Other fragments** — completely independent

## Metal Backend

All inference runs on direct Metal compute — no MLX, no MPSGraph.

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

- macOS 26+ / iOS 26+ / visionOS 26+
- Swift 6.2+
- Apple Silicon (Metal GPU)

## Dependencies

- [swift-jinja](https://github.com/huggingface/swift-jinja) — Chat template evaluation
- [swift-transformers](https://github.com/huggingface/swift-transformers) — Tokenizers and HuggingFace Hub

## Documentation

- `docs/using-swift-lm.md` — developer integration guide
- `docs/releases/0.1.0.md` — release notes and support boundary for `0.1.0`
- `Sources/SwiftLM/SwiftLM.docc` — DocC catalog for the `SwiftLM` module
- `AGENTS.md` — repository architecture and contribution guidance
- `DESIGN-Metal4.md` — forward-looking Metal 4 design work
- `docs/README.md` — documentation map and archive layout

## License

MIT
