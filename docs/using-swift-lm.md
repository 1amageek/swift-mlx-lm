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

## Model Bundle Layout

`ModelBundleLoader` works with a HuggingFace snapshot directory. At minimum, keep these files together:

- `config.json`
- `tokenizer.json`
- `*.safetensors`

These files are also used when present:

- `tokenizer_config.json`
- `special_tokens_map.json`
- `chat_template.jinja`
- `preprocessor_config.json`

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

if container.configuration.inputCapabilities.supportsImages {
    print("This model bundle declares image input support.")
}

if container.configuration.executionCapabilities.supportsImagePromptPreparation {
    print("This runtime can prepare Qwen-style image prompts.")
}

if let vision = container.configuration.vision {
    print("vision_start_token_id =", vision.visionStartTokenID as Any)
}
```

## Supported Model Families

The current loader resolves these model families from `config.json["model_type"]`:

| Family | `model_type` examples |
|---|---|
| Transformer | `llama`, `qwen2`, `qwen3`, `mistral`, `gemma`, `phi`, `mixtral`, `deepseek` |
| Qwen 3.5 hybrid / Qwen3-VL text backbone | `qwen3_5`, `qwen3_vl` |
| LFM2 / LFM2.5 hybrid | `lfm2`, `lfm2_moe` |
| Cohere | `cohere`, `command-r` |

Unsupported or incomplete families currently fail during loading or graph construction. `nemotron_h` is explicitly rejected in the current loader.

## Generate from Text

```swift
import SwiftLM

let context = try container.makeContext()
let prepared = try await context.prepare(ModelInput(prompt: "Write a haiku about Metal shaders."))
let executable = try context.makeExecutablePrompt(from: prepared)
let stream = try context.generate(
    from: executable,
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
        print("\n\nGenerated \(info.tokenCount) tokens at \(info.tokensPerSecond) tok/s")
    }
}
```

`LanguageModelContainer` is the immutable loaded bundle and factory for execution state. `LanguageModelContext` is the mutable runtime state for one conversation or generation flow.

Use `PreparedPrompt` as the result of prompt preparation and `ExecutablePrompt` as the runtime execution shape. `LanguageModelContainer.generate(_:parameters:)` is the async one-shot convenience entry point; `LanguageModelContext.generate(from:parameters:)` is the low-level execution API when you want explicit context ownership.

## Generate from Chat Messages

```swift
let context = try container.makeContext()
let prepared = try await context.prepare(ModelInput(chat: [
    .system("You are a concise assistant."),
    .user("Summarize the benefits of zero-copy model loading.")
]))
let executable = try context.makeExecutablePrompt(from: prepared)

for await event in try context.generate(from: executable) {
    if let chunk = event.text {
        print(chunk, terminator: "")
    }
}
```

When `chat_template.jinja` or `tokenizer_config.json["chat_template"]` is available, `prepare(_:)` renders the model's template. Otherwise, `swift-lm` falls back to a simple role-prefixed transcript.

## Prepare a Qwen3-VL Style Visual Prompt

`swift-lm` now understands the official Qwen3-VL marker tokens and processor metadata well enough to prepare and execute image-bearing and video-bearing prompts.

```swift
let input = ModelInput(chat: [
    .user([
        .text("Describe the attached image."),
        .image(InputImage(fileURL: URL(fileURLWithPath: "/path/to/image.jpg")))
    ])
])

let context = try container.makeContext()
let prepared = try await context.prepare(input)

if let multimodal = prepared.multimodalMetadata {
    print("image items =", multimodal.images.count)
    print("mm token types =", Array(multimodal.mmTokenTypeIDs.prefix(8)))
}

let executable = try context.makeExecutablePrompt(from: prepared)
for await event in try context.generate(from: executable) {
    if let chunk = event.text {
        print(chunk, terminator: "")
    }
}
```

For Qwen3-VL bundles, `prepare(_:)` expands the image and video placeholder counts from the bundle's `preprocessor_config.json` so the token stream matches the processor contract. `makeExecutablePrompt(from:)` builds the executable visual payload, and generation uses the bundled Qwen vision encoder plus the text backbone decode path.

## Reuse a Prompt Prefix

If many requests share the same prompt prefix, build a `PromptSnapshot` once and reuse it.

```swift
let context = try container.makeContext()
let promptSnapshot = try await context.makePromptSnapshot(from: ModelInput(chat: [
    .system("You are a helpful code review assistant."),
    .user("Review this patch carefully.")
]))

for await event in try context.generate(
    from: promptSnapshot,
    parameters: GenerationParameters(maxTokens: 64)
) {
    if let chunk = event.text {
        print(chunk, terminator: "")
    }
}
```

`PromptSnapshot` stores the post-prefill decode state and the first predicted token, so later calls can skip prompt prefill.

## Tokenizer and Cache Helpers

Use the container helpers when you need stateless tokenizer access, and the context when you need mutable cache control:

```swift
let tokens = container.encode("Hello")
let text = container.decode(tokens)
context.resetState()
```

- `encode(_:)` converts text to token IDs
- `decode(_:)` converts token IDs back to text
- `resetState()` clears KV/cache state between unrelated conversations

## Generation Parameters

`GenerationParameters` currently exposes:

- `maxTokens`
- `streamChunkTokenCount`
- `temperature`
- `topP`
- `repetitionPenalty`
- `repetitionContextSize`

For predictable behavior, set `maxTokens` explicitly. If `maxTokens` is `nil`, the current runtime applies its default cap.

## Supported Inputs and Current Limits

- Public input is represented by `ModelInput`
- `ModelConfiguration.inputCapabilities` tells you whether the model bundle declares image or video inputs
- `ModelConfiguration.executionCapabilities` tells you what the current runtime can actually execute or prepare
- `ModelConfiguration.vision` exposes image/video placeholder token IDs, processor names, and Qwen-style patch sizing metadata when the bundle provides them
- Image content can be expressed through `ModelInput`, and Qwen3-VL style prompt preparation now expands the correct placeholder count before tokenization
- `makeExecutablePrompt(from:)` converts prepared input into an executable prompt, while `makePromptSnapshot(from:)` on `LanguageModelContext` captures reusable decode state for the same context
- Other multimodal model families still throw `multimodalInputNotSupported` until a matching processor + vision runtime is implemented
- Tool calling and structured function-calling APIs are not part of the current public API
- Generation is stream-based; collect `GenerationEvent.text` values yourself if you need a single final string

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

## Validation Notes

The current `Qwen3.5+` multimodal path is covered by focused suites under [`Tests/SwiftLMTests`](/Users/1amageek/Desktop/swift-lm/Tests/SwiftLMTests). The release-facing set is:

- [`LoadTests.swift`](/Users/1amageek/Desktop/swift-lm/Tests/SwiftLMTests/LoadTests.swift)
- [`ReleaseSmokeTests.swift`](/Users/1amageek/Desktop/swift-lm/Tests/SwiftLMTests/ReleaseSmokeTests.swift)
- [`QwenVisionCapabilityTests.swift`](/Users/1amageek/Desktop/swift-lm/Tests/SwiftLMTests/QwenVisionCapabilityTests.swift)
- [`QwenVisionPromptProcessorTests.swift`](/Users/1amageek/Desktop/swift-lm/Tests/SwiftLMTests/QwenVisionPromptProcessorTests.swift)
- [`QwenVisionExecutionLayoutTests.swift`](/Users/1amageek/Desktop/swift-lm/Tests/SwiftLMTests/QwenVisionExecutionLayoutTests.swift)
- [`QwenVisionEncoderTests.swift`](/Users/1amageek/Desktop/swift-lm/Tests/SwiftLMTests/QwenVisionEncoderTests.swift)
- [`QwenVisionExecutionTests.swift`](/Users/1amageek/Desktop/swift-lm/Tests/SwiftLMTests/QwenVisionExecutionTests.swift)
- [`QwenVisionIntegrationTests.swift`](/Users/1amageek/Desktop/swift-lm/Tests/SwiftLMTests/QwenVisionIntegrationTests.swift)

The real-bundle suites are optional-local:
[`QwenVisionRealBundleImageTests.swift`](/Users/1amageek/Desktop/swift-lm/Tests/SwiftLMTests/QwenVisionRealBundleImageTests.swift),
[`QwenVisionRealBundleVideoTests.swift`](/Users/1amageek/Desktop/swift-lm/Tests/SwiftLMTests/QwenVisionRealBundleVideoTests.swift),
[`QwenVisionRealBundleMixedTests.swift`](/Users/1amageek/Desktop/swift-lm/Tests/SwiftLMTests/QwenVisionRealBundleMixedTests.swift),
[`QwenVisionRealBundlePromptStateTests.swift`](/Users/1amageek/Desktop/swift-lm/Tests/SwiftLMTests/QwenVisionRealBundlePromptStateTests.swift).
Each skips cleanly when no local `Qwen3.5/VL` snapshot is available.

When running the Qwen3.5+ multimodal matrix on a developer machine, prefer [`run-qwen35-vision-tests.sh`](/Users/1amageek/Desktop/swift-lm/scripts/run-qwen35-vision-tests.sh). It builds once and executes each suite with `test-without-building`, which reduces peak memory usage compared to one large `xcodebuild test` process.

Use `--suite` to narrow the run when you only need one area:

```bash
scripts/run-qwen35-vision-tests.sh --suite SwiftLMTests/QwenVisionCapabilityTests
```
