# ``SwiftLM``

High-performance local language model inference on Apple Silicon using direct Metal compute.

## Overview

Use `SwiftLM` when you want to load a HuggingFace model bundle, compile it for Metal, and stream generated text from Swift.

The public entry points are:

- ``ModelBundleLoader`` to load and compile a model
- ``InferenceSession`` to prepare prompts and stream generations
- ``ModelInput`` and ``InputMessage`` for text, chat, and Qwen-style multimodal prompt preparation
- ``PreparedPrompt`` and ``ExecutablePrompt`` for the explicit prepare/execute boundary
- ``GenerationParameters`` to control decoding behavior
- ``PromptSnapshot`` to reuse prompt prefixes efficiently

## Topics

### Essentials

- <doc:GettingStarted>
- <doc:ChatAndPromptReuse>
- <doc:Troubleshooting>

### Loading and GenerationEvent

- ``ModelBundleLoader``
- ``InferenceSession``
- ``ModelInput``
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
- ``InferenceSessionError``
