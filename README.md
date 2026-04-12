# swift-lm

High-performance language model inference on Apple Silicon using direct Metal compute.

## Overview

`swift-lm` loads Hugging Face model bundles directly from `config.json`, `tokenizer.json`, `chat_template.jinja`, and `*.safetensors`, builds a backend-independent graph, compiles it to Metal, and exposes a small public API for prompt preparation and generation.

- Direct Metal prefill and decode
- Hugging Face snapshot directories as the source of truth
- STAF as a regenerable GPU execution cache
- Text generation plus multimodal prompt preparation where the loaded runtime supports it
- Prompt snapshots for shared-prefix reuse
- Request-level reasoning visibility control

The public API is centered on:

- `ModelBundleLoader`
- `LanguageModelContainer`
- `LanguageModelContext`
- `TextEmbeddingContainer`
- `TextEmbeddingContext`
- `ModelInput`
- `TextEmbeddingInput`
- `GenerationParameters`
- `PromptSnapshot`

## API Shape

`swift-lm` uses a consistent public API shape:

- container types own immutable loaded bundles
- context types own reusable mutable runtime state
- input value types carry request data

For generation, that means:

- `LanguageModelContainer`
- `LanguageModelContext`
- `ModelInput`

For embeddings, that means:

- `TextEmbeddingContainer`
- `TextEmbeddingContext`
- `TextEmbeddingInput`

This split keeps prompt/render configuration, generation policy, and runtime ownership separate.

## Quick Start

```swift
import SwiftLM

let container = try await ModelBundleLoader().load(
    repo: "LiquidAI/LFM2.5-1.2B-Instruct"
)

let input = ModelInput(
    chat: [
        .system("You are a concise assistant."),
        .user("Write a haiku about Metal shaders.")
    ],
    promptOptions: .init(isThinkingEnabled: true)
)

let stream = try await container.generate(
    input,
    parameters: GenerationParameters(
        maxTokens: 128,
        streamChunkTokenCount: 8,
        temperature: 0.6,
        topP: 0.9,
        reasoning: .separate
    )
)

for await event in stream {
    switch event {
    case .text(let text):
        print(text, terminator: "")
    case .reasoning(let reasoning):
        fputs(reasoning, stderr)
    case .completed(let info):
        print("\nGenerated \(info.tokenCount) tokens at \(info.tokensPerSecond) tok/s")
    }
}
```

`LanguageModelContainer` is the immutable loaded bundle and factory for execution state. `LanguageModelContext` owns mutable decode state such as KV position, prompt snapshots, and generation progress.

For most applications, start from `LanguageModelContainer.generate(_:, parameters:)`.
Use `LanguageModelContext`, `PreparedPrompt`, and `ExecutablePrompt` only when you need explicit prompt staging or prompt snapshot reuse.

## Public API

### Loading

`ModelBundleLoader` loads either a Hugging Face repo or a local directory and returns a `LanguageModelContainer`.

```swift
let container = try await ModelBundleLoader().load(
    directory: URL(fileURLWithPath: "/path/to/model")
)
```

`load(repo:)` and `load(directory:)` also accept `inferencePolicy:`. The loader may upgrade the default KV cache policy to RotorQuant for eligible attention-backed decode graphs. Pass an explicit `InferencePolicy` when you want to override that behavior.

### Text Embeddings

`ModelBundleLoader.loadTextEmbeddings(...)` loads sentence-transformers style text embedding bundles and returns a `TextEmbeddingContainer`.

```swift
let embeddings = try await ModelBundleLoader().loadTextEmbeddings(
    repo: "google/embeddinggemma-300m"
)

let vector = try embeddings.embed(
    TextEmbeddingInput(
        "swift metal inference",
        promptName: embeddings.defaultPromptName
    )
)
print(vector.count)
```

Recommended flow for most apps:

```swift
let embeddings = try await ModelBundleLoader().loadTextEmbeddings(
    repo: "google/embeddinggemma-300m"
)

let vector = try embeddings.embed(
    TextEmbeddingInput(
        "swift metal inference",
        promptName: embeddings.defaultPromptName
    )
)
```

Context-owned flow when you want explicit mutable state:

```swift
let context = try TextEmbeddingContext(embeddings)
let vector = try context.embed(
    TextEmbeddingInput(
        "swift metal inference",
        promptName: embeddings.defaultPromptName
    )
)
print(vector.count)
```

As with language generation, `TextEmbeddingContainer` is immutable and shareable, while `TextEmbeddingContext` owns isolated mutable runtime state for embedding execution.

For most applications, start from `TextEmbeddingContainer.embed(_:)`.
Use `TextEmbeddingContext` only when you want explicit ownership of reusable mutable embedding state.

`TextEmbeddingInput` is the primary public request value for embedding APIs.
The existing `embed(_ text:promptName:)` overloads remain as convenience entry points.

Internally, sentence-transformers embedding pipelines are modeled as:

- structural stages: pooling and dense layers
- postprocessors: output-only modifiers such as L2 normalization

This keeps embedding structure separate from output modifiers without exposing a fluent modifier API publicly.

### Prompt Preparation

`ModelInput` is the user-facing input type.

- `ModelInput(prompt:)` for plain text
- `ModelInput(chat:)` for chat prompts
- `PromptPreparationOptions` for prompt-template-time options

`PromptPreparationOptions` and `GenerationParameters` intentionally cover different concerns:

- `PromptPreparationOptions.isThinkingEnabled`
  - Controls template rendering for models that expose `enable_thinking`
- `PromptPreparationOptions.templateVariables`
  - Passes extra values only to template rendering
- `GenerationParameters.reasoning`
  - Controls whether reasoning content is hidden, inline, or emitted separately during generation

This separation matters because prompt-template variables affect rendered input, while reasoning visibility affects output presentation.

### Thinking and Reasoning

`swift-lm` intentionally separates prompt-time thinking control from output-time reasoning visibility.

- `PromptPreparationOptions.isThinkingEnabled`
  - Affects chat template rendering for bundles that expose `enable_thinking`
- `PromptPreparationOptions.templateVariables`
  - Provides additional template-only render inputs
- `GenerationParameters.reasoning`
  - Controls how reasoning is surfaced in generation output

Example:

```swift
let input = ModelInput(
    chat: [
        .user("Solve this carefully.")
    ],
    promptOptions: .init(isThinkingEnabled: true)
)

let parameters = GenerationParameters(
    maxTokens: 128,
    reasoning: .separate
)
```

When `reasoning` is `.separate`, the stream may emit `.reasoning(String)` events in addition to visible `.text(String)` events.

### Generation

`LanguageModelContext.generate(from:parameters:)` returns an `AsyncStream<GenerationEvent>`.

`GenerationEvent` can emit:

- `.text(String)`
- `.reasoning(String)`
- `.completed(CompletionInfo)`

`LanguageModelContainer` also exposes one-shot convenience generation methods that create a fresh context internally:

- `generate(_ input: ModelInput, parameters:)`
- `generate(from: ExecutablePrompt, parameters:)`

`LanguageModelContext` also exposes:

- `generate(_ input: ModelInput, parameters:)` for context-owned generation without manual staging
- `generate(from: ExecutablePrompt, parameters:)` for explicit prompt staging
- `generate(from: PromptSnapshot, parameters:)` for prefix reuse

Recommended flow for most apps:

```swift
let stream = try await container.generate(
    ModelInput(chat: [
        .user("Summarize the benefits of zero-copy model loading.")
    ]),
    parameters: GenerationParameters(maxTokens: 128, temperature: 0)
)
```

Context-owned flow when you want to keep mutable state explicitly:

```swift
let context = try LanguageModelContext(container)
let stream = try await context.generate(
    ModelInput(chat: [
        .user("Hello")
    ]),
    parameters: GenerationParameters(maxTokens: 64)
)
```

Advanced flow when you need explicit prompt staging:

```swift
let context = try LanguageModelContext(container)
let prepared = try await context.prepare(ModelInput(prompt: "Hello"))
let executable = try ExecutablePrompt(preparedPrompt: prepared, using: context)
let stream = try context.generate(from: executable)
```

`LanguageModelContext.generate(from:parameters:)` is the low-level execution entry point used after explicit staging.

### Prompt Reuse

Use `PromptSnapshot` when many requests share the same prefix.

```swift
let context = try LanguageModelContext(container)
let snapshot = try await PromptSnapshot(from: ModelInput(chat: [
    .system("You are a careful code reviewer."),
    .user("Review this patch.")
]), using: context)

for await event in try context.generate(
    from: snapshot,
    parameters: GenerationParameters(maxTokens: 64)
) {
    if case .text(let text) = event {
        print(text, terminator: "")
    }
}
```

### Tokenizer Helpers

Stateless encode/decode helpers live on the container:

```swift
let tokenIDs = container.encode("Hello")
let text = container.decode(tokenIDs)
```

Mutable cache control lives on the context:

```swift
context.resetState()
```

### Generation Parameters

`GenerationParameters` currently exposes:

- `maxTokens`
- `streamChunkTokenCount`
- `temperature`
- `topP`
- `topK`
- `minP`
- `repetitionPenalty`
- `presencePenalty`
- `repetitionContextSize`
- `reasoning`

For predictable behavior, set `maxTokens` explicitly. If `maxTokens` is `nil`, the runtime uses its default cap.

## Multimodal Support

The runtime exposes declared input capabilities separately from execution capabilities:

- `ModelConfiguration.inputCapabilities`
- `ModelConfiguration.executionCapabilities`
- `ModelConfiguration.vision`

Current support in the repository:

- Qwen vision families
  - Image prompt preparation
  - Video prompt preparation
  - Image execution
  - Video execution
- Gemma4
  - Image prompt preparation
  - Image execution
  - Video execution is not implemented

Example:

```swift
if container.configuration.executionCapabilities.supportsImagePromptPreparation {
    let input = ModelInput(chat: [
        .user([
            .text("Describe this image."),
            .image(InputImage(fileURL: URL(fileURLWithPath: "/path/to/image.jpg")))
        ])
    ])

    let prepared = try await context.prepare(input)
    let executable = try ExecutablePrompt(preparedPrompt: prepared, using: context)

    for await event in try context.generate(from: executable) {
        if case .text(let chunk) = event {
            print(chunk, terminator: "")
        }
    }
}
```

## Supported Model Families

The current loader resolves these families from `config.json["model_type"]`:

| Family | `model_type` examples |
|---|---|
| Transformer | `llama`, `qwen2`, `qwen3`, `mistral`, `gemma`, `gemma2`, `phi`, `phi3`, `starcoder2`, `gpt_neox`, `internlm2`, `deepseek`, `yi`, `baichuan`, `chatglm`, `mixtral`, `qwen2_moe`, `deepseek_v2`, `arctic`, `dbrx` |
| Gemma4 | `gemma4`, `gemma4_text` |
| Qwen 3.5 hybrid / Qwen vision text backbone | `qwen3_5`, `qwen3_vl`, `qwen2_5_vl`, `qwen2_vl` |
| LFM2 / LFM2.5 hybrid | `lfm2`, `lfm2_moe` |
| Cohere | `cohere`, `command-r` |

`nemotron_h` is explicitly rejected by the current loader.

## Requirements

- Swift 6.2+
- macOS 26+, iOS 26+, visionOS 26+
- Apple Silicon with Metal support
- A Hugging Face model bundle containing `config.json`, `tokenizer.json`, and one or more `.safetensors` files

Optional files used when present:

- `tokenizer_config.json`
- `special_tokens_map.json`
- `chat_template.jinja`
- `preprocessor_config.json`
- `processor_config.json`

`swift-lm` generates `model.staf` next to the source weights as an execution cache. `safetensors` remains the source of truth.

## Build, Test, Benchmark

Build:

```bash
swift build
```

Focused test target:

```bash
xcodebuild test -scheme swift-lm-Package -destination 'platform=macOS' -only-testing:SwiftLMTests
```

For Qwen3.5+ multimodal suites, prefer the low-memory runner:

```bash
scripts/run-qwen35-vision-tests.sh
```

For generation pipeline benchmarks, prefer the split benchmark runner:

```bash
scripts/run-generation-pipeline-benchmarks.sh
```

When debugging real-model correctness, do not batch many heavy suites into one long `xcodebuild test` process. Build once, then run focused `test-without-building` invocations one suite at a time.

## Architecture

The repository is split into five layers:

```text
LMIR
  Backend-independent graph and operation model.

LMArchitecture
  Declarative model DSL and validation.

ModelDeclarations
  Family-specific model declarations such as Transformer, Qwen35, LFM2, Gemma4, and Cohere.

MetalCompiler
  IR lowering, fragment planning, dispatch optimization, kernel generation,
  STAF loading, and direct Metal execution planning.

SwiftLM
  Public loading, prompt preparation, tokenization, and generation API.
```

Dependency direction:

```text
LMIR  <-  LMArchitecture  <-  ModelDeclarations
  |                               |
  +--------  MetalCompiler  ------+
                  |
                  +--------  SwiftLM
```

The key design constraints are:

- `LMIR` stays semantic and backend-independent
- Model declarations describe architecture families, not backend tricks
- `safetensors` is canonical and STAF is a regenerable execution cache
- Decode hot-path cost matters more than convenience abstractions

## Repository Notes

- `Sources/SwiftLM/SwiftLM.docc` contains the DocC sources
- [`AGENTS.md`](/Users/1amageek/Desktop/swift-lm/AGENTS.md) documents repository-specific testing and debugging procedure
- [`DESIGN-Metal4.md`](/Users/1amageek/Desktop/swift-lm/DESIGN-Metal4.md) contains backend design notes; treat it as design guidance, not as a guarantee that every forward-looking idea is already shipped
