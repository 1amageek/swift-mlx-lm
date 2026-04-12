# Text Embeddings

## Overview

Use ``TextEmbeddingContainer`` when you want one-shot embedding generation from a loaded sentence-transformer style bundle.

Use ``TextEmbeddingContext`` when you want explicit ownership of reusable mutable embedding state.

The primary public request value is ``TextEmbeddingInput``:

```swift
let input = TextEmbeddingInput(
    "swift metal inference",
    promptName: embeddings.defaultPromptName
)
```

For most applications, call ``TextEmbeddingContainer/embed(_:)``. The context-owned APIs are for advanced flows where you need to control runtime ownership directly.

## Load an Embedding Bundle

```swift
import SwiftLM

let embeddings = try await ModelBundleLoader().loadTextEmbeddings(
    repo: "google/embeddinggemma-300m"
)
```

`SwiftLM` loads the tokenizer, sentence-transformer metadata, and the compiled Metal runtime for the embedding model.

## Embed Text

```swift
let vector = try embeddings.embed(
    TextEmbeddingInput(
        "swift metal inference",
        promptName: embeddings.defaultPromptName
    )
)

print(vector.count)
```

The returned vector is the final embedding output after the bundle's declared pooling, dense layers, and postprocessors have been applied.

## Prompt Selection

Some embedding bundles expose named prompts such as `"query"` and `"passage"`.

Use these helpers to inspect the bundle:

```swift
let names = embeddings.availablePromptNames
let defaultName = embeddings.defaultPromptName
```

Pass the selected name through ``TextEmbeddingInput``:

```swift
let query = try embeddings.embed(
    TextEmbeddingInput("How does Metal residency work?", promptName: "query")
)

let passage = try embeddings.embed(
    TextEmbeddingInput(
        "Metal residency controls which resources remain available to the GPU.",
        promptName: "passage"
    )
)
```

If a bundle does not declare named prompts, leave `promptName` as `nil`.

## Explicit Runtime Ownership

Use ``TextEmbeddingContext`` when you want to keep runtime state under explicit ownership:

```swift
let context = try TextEmbeddingContext(embeddings)

let vector = try context.embed(
    TextEmbeddingInput(
        "swift metal inference",
        promptName: embeddings.defaultPromptName
    )
)
```

This is the embedding equivalent of using ``LanguageModelContext`` for text generation.

## API Shape

The embedding API follows the same container/context split as generation:

- ``TextEmbeddingContainer`` is the immutable loaded bundle and the recommended entry point.
- ``TextEmbeddingContext`` owns reusable mutable runtime state.
- ``TextEmbeddingInput`` keeps the text and optional prompt selection together.

Use the string-based overloads only as convenience entry points when you do not need to carry prompt selection as a first-class value.
