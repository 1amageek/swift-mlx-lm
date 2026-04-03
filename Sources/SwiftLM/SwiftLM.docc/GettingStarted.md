# Getting Started

## Requirements

- Swift 6.2+
- macOS 26+, iOS 26+, or visionOS 26+ as declared by `Package.swift`
- An Apple Silicon device with Metal support for local inference
- A HuggingFace model bundle containing `config.json`, `tokenizer.json`, and one or more `.safetensors` files

## Add the Package

```swift
dependencies: [
    .package(url: "https://github.com/1amageek/swift-lm.git", from: "0.1.0")
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
let input = try container.prepare(
    input: UserInput(prompt: "Write a haiku about Metal shaders.")
)

let stream = container.generate(
    input: input,
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

`ModelContainer/generate(input:parameters:)` returns an `AsyncStream` of ``Generation`` values.

## Supported Model Families

The current loader resolves these families from `config.json["model_type"]`:

| Family | `model_type` examples |
|---|---|
| Transformer | `llama`, `qwen2`, `qwen3`, `mistral`, `gemma`, `phi`, `mixtral`, `deepseek` |
| Qwen 3.5 hybrid | `qwen3_5` |
| LFM2 / LFM2.5 hybrid | `lfm2`, `lfm2_moe` |
| Cohere | `cohere`, `command-r` |

`nemotron_h` is explicitly rejected by the current loader.
