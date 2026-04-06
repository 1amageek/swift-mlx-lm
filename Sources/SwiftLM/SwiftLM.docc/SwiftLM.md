# ``SwiftLM``

High-performance local language model inference on Apple Silicon using direct Metal compute.

## Overview

Use `SwiftLM` when you want to load a HuggingFace model bundle, compile it for Metal, and stream generated text from Swift.

The public entry points are:

- ``ModelBundleLoader`` to load and compile a model
- ``ModelContainer`` to prepare prompts and stream generations
- ``ModelInput`` and ``InputMessage`` for text, chat, and Qwen-style multimodal prompt preparation
- ``PreparedInput`` and ``ExecutablePrompt`` for the explicit prepare/execute boundary
- ``GenerateParameters`` to control decoding behavior
- ``PromptState`` to reuse prompt prefixes efficiently

## Topics

### Essentials

- <doc:GettingStarted>
- <doc:ChatAndPromptReuse>
- <doc:Troubleshooting>

### Loading and Generation

- ``ModelBundleLoader``
- ``ModelContainer``
- ``ModelInput``
- ``PreparedInput``
- ``ExecutablePrompt``
- ``InputMessage``
- ``InputImage``
- ``InputVideo``
- ``GenerateParameters``
- ``Generation``
- ``CompletionInfo``
- ``PromptState``

### Supporting Types

- ``ModelConfiguration``
- ``ModelBundleLoaderError``
- ``ModelContainerError``
