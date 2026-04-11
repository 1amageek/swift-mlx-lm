# swift-lm

High-performance LLM inference on Apple Silicon using direct Metal compute.

## Overview

A Swift package for LLM inference on Apple Silicon. Models are loaded directly from HuggingFace directories (config.json + safetensors + tokenizer.json) — no model-specific Swift types required. The consumer specifies only a HuggingFace repo ID.

```swift
import SwiftLM

let session = try await ModelBundleLoader().load(repo: "LiquidAI/LFM2.5-1.2B-Instruct")

let prepared = try await session.prepare(ModelInput(prompt: "Hello"))
let executable = try session.makeExecutablePrompt(from: prepared)

for await generation in try session.generate(from: executable) {
    if let text = generation.text { print(text, terminator: "") }
}

if session.configuration.inputCapabilities.supportsImages {
    print("This bundle declares image input support.")
}

if session.configuration.executionCapabilities.supportsImagePromptPreparation {
    print("The current runtime can prepare Qwen-style image prompts.")
}

if session.configuration.executionCapabilities.supportsVideoPromptPreparation {
    print("The current runtime can prepare Qwen-style video prompts.")
}

if let vision = session.configuration.vision {
    print("image_token_id =", vision.imageTokenID as Any)
}

if session.configuration.executionCapabilities.supportsImagePromptPreparation {
    let visualInput = ModelInput(chat: [
        .user([
            .text("Describe this image."),
            .image(InputImage(fileURL: URL(fileURLWithPath: "/path/to/image.jpg")))
        ])
    ])

    let prepared = try await session.prepare(visualInput)
    let executable = try session.makeExecutablePrompt(from: prepared)

    for await generation in try session.generate(from: executable) {
        if let chunk = generation.text { print(chunk, terminator: "") }
    }
}
```

## Release Status

The current branch is targeting `0.3.0`.

Supported in the `0.3.0` line:

- Apple Silicon devices with Metal
- HuggingFace snapshot directories containing `config.json`, `tokenizer.json`, and `.safetensors`
- direct Metal decode/prefill execution through `SwiftLM`
- text prompts and chat prompts through `ModelInput`
- model-declared input capabilities through `ModelConfiguration.inputCapabilities`
- runtime execution capabilities through `ModelConfiguration.executionCapabilities`
- Qwen3-VL style vision metadata inspection and prompt preparation for image-bearing and video-bearing chat input
- Qwen3-VL style image and video execution through the bundled vision encoder path
- the currently documented model families in this README and in `docs/using-swift-lm.md`

Not part of the `0.3.0` line:
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
- `PromptSnapshot` reuse for shared prefixes
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
    .package(url: "https://github.com/1amageek/swift-lm.git", from: "0.3.0")
]
```

## Test Matrix

The Qwen3.5+ multimodal path is exercised in four layers:

- unit suites for capability decode, prompt preparation, execution layout, and vision encoding
- component suites for `PreparedPrompt -> ExecutablePrompt -> generate`
- synthetic integration suites for `load -> prepare -> generate`
- optional local real-bundle suites for `Qwen3-VL` snapshots when they exist on the machine

The suites added for this path live under [`Tests/SwiftLMTests`](/Users/1amageek/Desktop/swift-lm/Tests/SwiftLMTests):

- [`QwenVisionCapabilityTests.swift`](/Users/1amageek/Desktop/swift-lm/Tests/SwiftLMTests/QwenVisionCapabilityTests.swift)
- [`QwenVisionPromptProcessorTests.swift`](/Users/1amageek/Desktop/swift-lm/Tests/SwiftLMTests/QwenVisionPromptProcessorTests.swift)
- [`QwenVisionExecutionLayoutTests.swift`](/Users/1amageek/Desktop/swift-lm/Tests/SwiftLMTests/QwenVisionExecutionLayoutTests.swift)
- [`QwenVisionEncoderTests.swift`](/Users/1amageek/Desktop/swift-lm/Tests/SwiftLMTests/QwenVisionEncoderTests.swift)
- [`QwenVisionExecutionTests.swift`](/Users/1amageek/Desktop/swift-lm/Tests/SwiftLMTests/QwenVisionExecutionTests.swift)
- [`QwenVisionIntegrationTests.swift`](/Users/1amageek/Desktop/swift-lm/Tests/SwiftLMTests/QwenVisionIntegrationTests.swift)
- [`QwenVisionRealBundleImageTests.swift`](/Users/1amageek/Desktop/swift-lm/Tests/SwiftLMTests/QwenVisionRealBundleImageTests.swift)
- [`QwenVisionRealBundleVideoTests.swift`](/Users/1amageek/Desktop/swift-lm/Tests/SwiftLMTests/QwenVisionRealBundleVideoTests.swift)
- [`QwenVisionRealBundleMixedTests.swift`](/Users/1amageek/Desktop/swift-lm/Tests/SwiftLMTests/QwenVisionRealBundleMixedTests.swift)
- [`QwenVisionRealBundlePromptStateTests.swift`](/Users/1amageek/Desktop/swift-lm/Tests/SwiftLMTests/QwenVisionRealBundlePromptStateTests.swift)

Release-facing validation should include:

```bash
scripts/run-qwen35-vision-tests.sh
```

For low-memory machines, do not run the full Qwen3.5+ multimodal matrix in one `xcodebuild test` process. Use `build-for-testing` once and then `test-without-building` suite-by-suite through [`run-qwen35-vision-tests.sh`](/Users/1amageek/Desktop/swift-lm/scripts/run-qwen35-vision-tests.sh). Add `--include-real` only when you want the optional local real-bundle suite.

To run only one suite while keeping the same low-memory flow:

```bash
scripts/run-qwen35-vision-tests.sh --suite SwiftLMTests/QwenVisionCapabilityTests
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
        ├── InferenceSession (generate, encode, decode)
        └── ModelInput, InputMessage, InputImage, GenerationParameters
```

## Declarative Model DSL

Model families in `swift-lm` are not hard-coded as backend-specific execution graphs. They are written as declarative `ModelComponent` trees, then normalized into `LMIR.ModelGraph`, and finally compiled for Metal.

A standard decoder transformer looks like this:

```swift
public struct Transformer: ModelComponent {
    public let config: ModelConfig

    @ModelComponentBuilder
    public var body: some ModelComponent {
        TokenEmbedding(vocabSize: config.vocabSize, embeddingSize: config.hiddenSize)

        Repeat(count: config.layerCount, label: "layers") {
            Residual {
                RMSNorm(dimension: config.hiddenSize, epsilon: config.normEps)
                Attention(
                    hiddenSize: config.hiddenSize,
                    headCount: config.attentionHeads,
                    kvHeadCount: config.kvHeads,
                    headDimension: config.headDim
                )
            }
            Residual {
                RMSNorm(dimension: config.hiddenSize, epsilon: config.normEps)
                MLP(inputSize: config.hiddenSize, intermediateSize: config.intermediateSize)
            }
        }

        RMSNorm(dimension: config.hiddenSize, epsilon: config.normEps)
        OutputHead(
            inputSize: config.hiddenSize,
            vocabSize: config.vocabSize,
            tiedToEmbedding: config.tiedEmbeddings
        )
    }
}
```

A hybrid family such as LFM2 can switch layer structure declaratively:

```swift
public struct LFM2: ModelComponent {
    public let config: ModelConfig
    private let convLayerIndices: Set<Int>

    @ModelComponentBuilder
    public var body: some ModelComponent {
        TokenEmbedding(vocabSize: config.vocabSize, embeddingSize: config.hiddenSize)

        LayerStack(0..<config.layerCount) { layerIndex in
            if convLayerIndices.contains(layerIndex) {
                Residual {
                    RMSNorm(dimension: config.hiddenSize, epsilon: config.normEps)
                    ShortConv(hiddenSize: config.hiddenSize, kernelSize: config.convLCache ?? 3)
                }
            } else {
                Residual {
                    RMSNorm(dimension: config.hiddenSize, epsilon: config.normEps)
                    Attention(
                        hiddenSize: config.hiddenSize,
                        headCount: config.attentionHeads,
                        kvHeadCount: config.kvHeads,
                        headDimension: config.hiddenSize / config.attentionHeads
                    )
                }
            }

            Residual {
                RMSNorm(dimension: config.hiddenSize, epsilon: config.normEps)
                MLP(inputSize: config.hiddenSize, intermediateSize: config.intermediateSize)
            }
        }

        RMSNorm(dimension: config.hiddenSize, epsilon: config.normEps)
        OutputHead(inputSize: config.hiddenSize, vocabSize: config.vocabSize, tiedToEmbedding: config.tiedEmbeddings)
    }
}
```

This is the intended mental model for contributors:

- `LMArchitecture` describes the model semantically with `TokenEmbedding`, `Residual`, `Attention`, `MLP`, `MoE`, `ShortConv`, and `OutputHead`
- `LMIR` holds the normalized backend-independent graph
- `MetalCompiler` decides how that graph turns into kernels, pipelines, buffers, and dispatch steps

For Qwen3-VL style bundles, `SwiftLM` also reads the official vision markers from `config.json` and `preprocessor_config.json`. `InferenceSession.prepare()` expands image and video placeholders into the token layout expected by the official Qwen processor, and `makeExecutablePrompt(from:)` / `generate(from:)` execute the resulting visual prompt when the loaded bundle provides a compatible Qwen vision encoder.

If you want to inspect the real declarations, start with `Sources/Models/Transformer.swift` and `Sources/Models/LFM2.swift`.

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
| Qwen 3.5 / Qwen3-VL text backbone | Hybrid | Gated DeltaNet + Full Attention |
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
- `docs/releases/0.3.0.md` — release notes and support boundary for `0.3.0`
- `docs/releases/0.1.0.md` — release notes and support boundary for `0.1.0`
- `Sources/SwiftLM/SwiftLM.docc` — DocC catalog for the `SwiftLM` module
- `AGENTS.md` — repository architecture and contribution guidance
- `DESIGN-Metal4.md` — forward-looking Metal 4 design work
- `docs/README.md` — documentation map and archive layout

## License

MIT
