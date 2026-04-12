# Getting Started

## Requirements

- Swift 6.2+
- macOS 26+, iOS 26+, or visionOS 26+ as declared by `Package.swift`
- An Apple Silicon device with Metal support for local inference
- A HuggingFace model bundle containing `config.json`, `tokenizer.json`, and one or more `.safetensors` files

## Add the Package

```swift
dependencies: [
    .package(url: "https://github.com/1amageek/swift-lm.git", from: "0.3.0")
],
targets: [
    .target(
        name: "MyApp",
        dependencies: [
            .product(name: "SwiftLM", package: "swift-lm")
        ]
    )
]
```

## Load a Model

Load from HuggingFace:

```swift
import SwiftLM

let container = try await ModelBundleLoader().load(
    repo: "LiquidAI/LFM2.5-1.2B-Instruct"
)
```

Load from a local model snapshot:

```swift
import Foundation
import SwiftLM

let directory = URL(fileURLWithPath: "/path/to/model-snapshot")
let container = try await ModelBundleLoader().load(directory: directory)
```

`SwiftLM` creates `model.staf` next to the source weights as a regenerable execution cache.

## Load a Text Embedding Model

```swift
import SwiftLM

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

``TextEmbeddingContainer`` is the immutable loaded bundle and factory for execution state. ``TextEmbeddingContext`` owns the isolated mutable prefill runtime used for embedding execution.

For most applications, start from ``TextEmbeddingContainer/embed(_:)`` and pass a ``TextEmbeddingInput`` value. Use ``TextEmbeddingContext`` only when you want explicit ownership of reusable mutable embedding state.

For more details, see <doc:TextEmbeddings>.

## Generate from Text

```swift
let stream = try await container.generate(
    ModelInput(prompt: "Write a haiku about Metal shaders."),
    parameters: GenerationParameters(
        maxTokens: 128,
        streamChunkTokenCount: 8,
        temperature: 0.6,
        topP: 0.9
    )
)

for await event in stream {
    if let chunk = event.text {
        print(chunk, terminator: "")
    }
    if let info = event.completion {
        print("\nGenerated \(info.tokenCount) tokens in \(info.totalTime)s")
    }
}
```

``LanguageModelContainer`` is the immutable loaded bundle and factory for execution state. ``LanguageModelContext`` owns mutable decode state such as KV position, prompt snapshots, and generation progress.

For most applications, start from ``LanguageModelContainer/generate(_:parameters:)``. Use ``LanguageModelContext/generate(_:parameters:)`` when you want explicit ownership of mutable generation state, and use ``ExecutablePrompt`` only when you need manual prompt staging.

`LanguageModelContext.generate(from:parameters:)` returns an `AsyncStream` of ``GenerationEvent`` values. ``LanguageModelContainer/generate(_:parameters:)`` and ``LanguageModelContainer/generate(from:parameters:)`` are one-shot convenience APIs that create a fresh context internally. The public generation entry points throw when the prompt cannot be executed.

`ModelInput` is the primary prompt type. It now supports Qwen3-VL style image-bearing and video-bearing chat prompts during preparation and execution when the loaded bundle includes compatible vision weights.

When a bundle declares multimodal markers, inspect ``ModelConfiguration/vision``, ``ModelConfiguration/inputCapabilities``, and ``ModelConfiguration/executionCapabilities`` before deciding whether to surface image/video UI affordances.

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
