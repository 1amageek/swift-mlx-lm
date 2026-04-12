# ``SwiftLM``

High-performance local language model inference on Apple Silicon using direct Metal compute.

## Overview

Use `SwiftLM` when you want to load a HuggingFace model bundle, compile it for Metal, and stream generated text from Swift.

The public entry points are:

- ``ModelBundleLoader`` to load and compile a model
- ``LanguageModelContainer`` as the primary public entry point for generation
- ``LanguageModelContext`` as the mutable generation context when you need explicit state ownership
- ``ModelInput`` and ``InputMessage`` for text, chat, and Qwen-style multimodal prompt preparation
- ``TextEmbeddingContainer`` as the primary public entry point for text embeddings
- ``TextEmbeddingContext`` when you need explicit mutable embedding state
- ``TextEmbeddingInput`` as the request value for embedding APIs
- ``PreparedPrompt`` and ``ExecutablePrompt`` for the explicit prepare/execute boundary when you need staged generation
- ``GenerationParameters`` to control decoding behavior
- ``PromptSnapshot`` to reuse prompt prefixes efficiently

## Topics

### Essentials

- <doc:GettingStarted>
- <doc:ContainersAndContexts>
- <doc:TextEmbeddings>
- <doc:ReasoningAndThinking>
- <doc:ChatAndPromptReuse>
- <doc:Troubleshooting>

### Loading and GenerationEvent

- ``ModelBundleLoader``
- ``LanguageModelContainer``
- ``LanguageModelContext``
- ``ModelInput``
- ``TextEmbeddingContainer``
- ``TextEmbeddingContext``
- ``TextEmbeddingInput``
- ``PreparedPrompt``
- ``ExecutablePrompt``
- ``InputMessage``
- ``InputImage``
- ``InputVideo``
- ``GenerationParameters``
- ``GenerationEvent``
- ``CompletionInfo``
- ``PromptSnapshot``

### Supporting Types

- ``ModelConfiguration``
- ``ModelBundleLoaderError``
- ``LanguageModelContextError``
