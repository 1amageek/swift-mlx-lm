# Containers and Contexts

## Overview

`SwiftLM` uses a consistent public API shape:

- container types own immutable loaded bundles
- context types own reusable mutable runtime state
- input value types carry request data

This split is used for both generation and embeddings.

## Generation

For text generation:

- ``LanguageModelContainer`` is the immutable loaded bundle
- ``LanguageModelContext`` owns mutable decode state such as KV position and prompt snapshots
- ``ModelInput`` is the primary public request value

Start with the container API when each request is independent:

```swift
let stream = try await container.generate(
    ModelInput(prompt: "Explain rotary embeddings."),
    parameters: GenerationParameters(maxTokens: 64)
)
```

Use the context API when requests need shared mutable runtime state:

```swift
let context = try LanguageModelContext(container)

let stream = try await context.generate(
    ModelInput(chat: [
        .system("You are a concise assistant."),
        .user("Continue the previous answer.")
    ])
)
```

Use ``PreparedPrompt``, ``ExecutablePrompt``, and ``PromptSnapshot`` only when you need explicit staging or prefix reuse.

## Embeddings

For text embeddings:

- ``TextEmbeddingContainer`` is the immutable loaded bundle
- ``TextEmbeddingContext`` owns reusable mutable runtime state
- ``TextEmbeddingInput`` is the primary public request value

Start with the container API:

```swift
let vector = try embeddings.embed(
    TextEmbeddingInput("swift metal inference")
)
```

Use the context when runtime ownership matters:

```swift
let context = try TextEmbeddingContext(embeddings)
let vector = try context.embed(
    TextEmbeddingInput("swift metal inference")
)
```

## Why the Split Exists

The split keeps responsibilities clear:

- containers are cheap to pass around and describe loaded model capabilities
- contexts make mutable runtime ownership explicit
- input values keep request configuration together instead of scattering it across overloads

That is why generation options belong in ``GenerationParameters``, prompt-render options belong in ``PromptPreparationOptions``, and embedding prompt selection belongs in ``TextEmbeddingInput``.
