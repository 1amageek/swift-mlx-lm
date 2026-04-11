# Chat and Prompt Reuse

## Chat Input

Use ``ModelInput`` with ``InputMessage`` values to generate from a conversation:

```swift
let prepared = try await container.prepare(
    ModelInput(chat: [
        .system("You are a concise assistant."),
        .user("Summarize the benefits of zero-copy model loading.")
    ])
)
let executable = try container.makeExecutablePrompt(from: prepared)

for await event in try container.generate(from: executable) {
    if let chunk = event.text {
        print(chunk, terminator: "")
    }
}
```

When `chat_template.jinja` or `tokenizer_config.json["chat_template"]` is available, `SwiftLM` renders the model's chat template automatically. Otherwise it falls back to a simple role-prefixed transcript.

For Qwen3-VL style bundles, image-bearing and video-bearing chat messages are normalized during preparation so the placeholder token count matches the bundle's processor metadata. Prompt reuse also preserves the multimodal rope offset, so prepared visual prefixes can be snapshotted and restored through ``PromptSnapshot``.

## Reuse a Prompt Prefix

If many requests share the same prefix, build a ``PromptSnapshot`` once and restore it later:

```swift
let promptSnapshot = try await container.makePromptSnapshot(
    from: ModelInput(chat: [
        .system("You are a helpful code review assistant."),
        .user("Review this patch carefully.")
    ])
)

for await event in try container.generate(
    from: promptSnapshot,
    parameters: GenerationParameters(maxTokens: 64)
) {
    if let chunk = event.text {
        print(chunk, terminator: "")
    }
}
```

`PromptSnapshot` stores the post-prefill decode state and the first predicted token so later calls can skip prompt prefill.

## Cache and Tokenizer Helpers

`InferenceSession` also exposes lower-level helpers:

```swift
let tokens = container.encode("Hello")
let text = container.decode(tokens)
container.resetState()
```

Use `resetState()` between unrelated conversations to clear KV and decode state.
