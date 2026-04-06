# Getting Started

## Requirements

- Swift 6.2+
- macOS 26+, iOS 26+, or visionOS 26+ as declared by `Package.swift`
- An Apple Silicon device with Metal support for local inference
- A HuggingFace model bundle containing `config.json`, `tokenizer.json`, and one or more `.safetensors` files

## Add the Package

```swift
dependencies: [
    .package(url: "https://github.com/1amageek/swift-lm.git", from: "0.2.0")
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

## Generate from Text

```swift
let input = try await container.prepare(
    input: ModelInput(prompt: "Write a haiku about Metal shaders.")
)
let executable = try container.makeExecutablePrompt(from: input)

let stream = try container.generate(
    prompt: executable,
    parameters: GenerateParameters(
        maxTokens: 128,
        streamChunkTokenCount: 8,
        temperature: 0.6,
        topP: 0.9
    )
)

for await event in stream {
    if let chunk = event.chunk {
        print(chunk, terminator: "")
    }
    if let info = event.info {
        print("\nGenerated \(info.tokenCount) tokens in \(info.totalTime)s")
    }
}
```

`ModelContainer/generate(prompt:parameters:)` returns an `AsyncStream` of ``Generation`` values. `ModelContainer/generate(input:parameters:)` is the async convenience API that prepares and executes in one step. The public generation entry points throw when the prompt cannot be executed.

`ModelInput` is the primary prompt type. It now supports Qwen3-VL style image-bearing and video-bearing chat prompts during preparation and execution when the loaded bundle includes compatible vision weights.

When a bundle declares multimodal markers, inspect ``ModelConfiguration/vision``, ``ModelConfiguration/inputCapabilities``, and ``ModelConfiguration/executionCapabilities`` before deciding whether to surface image/video UI affordances.

## Supported Model Families

The current loader resolves these families from `config.json["model_type"]`:

| Family | `model_type` examples |
|---|---|
| Transformer | `llama`, `qwen2`, `qwen3`, `mistral`, `gemma`, `phi`, `mixtral`, `deepseek` |
| Qwen 3.5 hybrid / Qwen3-VL text backbone | `qwen3_5`, `qwen3_vl` |
| LFM2 / LFM2.5 hybrid | `lfm2`, `lfm2_moe` |
| Cohere | `cohere`, `command-r` |

`nemotron_h` is explicitly rejected by the current loader.
