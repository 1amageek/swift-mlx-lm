# Using SwiftLM

This guide is for application developers integrating `SwiftLM` as a library.

## Requirements

- Swift 6.2+
- Apple platforms declared by this package: macOS 26+, iOS 26+, visionOS 26+
- An Apple Silicon device with Metal support for local inference
- A HuggingFace model bundle with `config.json`, `tokenizer.json`, and one or more `.safetensors` files

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

## Model Bundle Layout

`ModelBundleLoader` works with a HuggingFace snapshot directory. At minimum, keep these files together:

- `config.json`
- `tokenizer.json`
- `*.safetensors`

These files are also used when present:

- `tokenizer_config.json`
- `special_tokens_map.json`
- `chat_template.jinja`

`swift-lm` generates `model.staf` next to the source weights as an execution cache. `safetensors` remains the source of truth, and `model.staf` can be deleted and regenerated.

## Load a Model

Load directly from HuggingFace:

```swift
import SwiftLM

let loader = ModelBundleLoader()
let container = try await loader.load(repo: "LiquidAI/LFM2.5-1.2B-Instruct")
```

Load from a local directory:

```swift
import Foundation
import SwiftLM

let directory = URL(fileURLWithPath: "/path/to/model-snapshot")
let container = try await ModelBundleLoader().load(directory: directory)
```

## Supported Model Families

The current loader resolves these model families from `config.json["model_type"]`:

| Family | `model_type` examples |
|---|---|
| Transformer | `llama`, `qwen2`, `qwen3`, `mistral`, `gemma`, `phi`, `mixtral`, `deepseek` |
| Qwen 3.5 hybrid | `qwen3_5` |
| LFM2 / LFM2.5 hybrid | `lfm2`, `lfm2_moe` |
| Cohere | `cohere`, `command-r` |

Unsupported or incomplete families currently fail during loading or graph construction. `nemotron_h` is explicitly rejected in the current loader.

## Generate from Text

```swift
import SwiftLM

let input = try container.prepare(input: UserInput(prompt: "Write a haiku about Metal shaders."))
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
        print("\n\nGenerated \(info.tokenCount) tokens at \(info.tokensPerSecond) tok/s")
    }
}
```

`generate(input:)` returns `AsyncStream<Generation>`. The stream yields:

- `.chunk(String)` for decoded text
- `.info(CompletionInfo)` once at the end

## Generate from Chat Messages

```swift
let input = try container.prepare(input: UserInput(chat: [
    .system("You are a concise assistant."),
    .user("Summarize the benefits of zero-copy model loading.")
]))

for await event in container.generate(input: input) {
    if let chunk = event.chunk {
        print(chunk, terminator: "")
    }
}
```

When `chat_template.jinja` or `tokenizer_config.json["chat_template"]` is available, `prepare(input:)` renders the model's template. Otherwise, `swift-lm` falls back to a simple role-prefixed transcript.

## Reuse a Prompt Prefix

If many requests share the same prompt prefix, build a `PromptState` once and reuse it.

```swift
let promptState = try container.makePromptState(input: UserInput(chat: [
    .system("You are a helpful code review assistant."),
    .user("Review this patch carefully.")
]))

for await event in container.generate(
    from: promptState,
    parameters: GenerateParameters(maxTokens: 64)
) {
    if let chunk = event.chunk {
        print(chunk, terminator: "")
    }
}
```

`PromptState` stores the post-prefill decode state and the first predicted token, so later calls can skip prompt prefill.

## Tokenizer and Cache Helpers

Use the container helpers when you need lower-level control:

```swift
let tokens = container.encode("Hello")
let text = container.decode(tokens: tokens)
container.resetCaches()
```

- `encode(_:)` converts text to token IDs
- `decode(tokens:)` converts token IDs back to text
- `resetCaches()` clears KV/cache state between unrelated conversations

## Generation Parameters

`GenerateParameters` currently exposes:

- `maxTokens`
- `streamChunkTokenCount`
- `temperature`
- `topP`
- `repetitionPenalty`
- `repetitionContextSize`

For predictable behavior, set `maxTokens` explicitly. If `maxTokens` is `nil`, the current runtime applies its default cap.

## Supported Inputs and Current Limits

- Public input is text-only or chat-only through `UserInput`
- Multimodal image or video input is not part of the current public API
- Tool calling and structured function-calling APIs are not part of the current public API
- Generation is stream-based; collect `.chunk` values yourself if you need a single final string

## Troubleshooting

`ModelBundleLoader` currently throws these public errors:

- `noMetalDevice`
- `noSafetensorsFiles(path)`
- `invalidConfig(message)`

Common causes:

- Missing `config.json` or required tokenizer files in the local model directory
- No `.safetensors` files in the bundle
- Running on a device without a usable Metal GPU
- Model metadata that does not include required architecture fields such as `hidden_size`, `num_hidden_layers`, or `vocab_size`

## Related Documents

- `../README.md` for project overview and architecture
- `../Sources/MetalCompiler/STAF/README.md` for STAF cache details
- `../DESIGN-Metal4.md` for forward-looking backend design notes
