# Reasoning and Thinking

## Overview

`SwiftLM` separates two concerns that are often mixed together:

- prompt-time thinking control
- output-time reasoning visibility

These are configured in different places on purpose.

## Prompt-Time Thinking Control

Use ``PromptPreparationOptions`` to control template-time thinking behavior:

```swift
let input = ModelInput(
    chat: [
        .user("Solve this step by step.")
    ],
    promptOptions: PromptPreparationOptions(
        isThinkingEnabled: true
    )
)
```

`isThinkingEnabled` affects prompt rendering for bundles whose chat template exposes an `enable_thinking` variable.

`templateVariables` is also prompt-time configuration. Use it only for template rendering inputs, not for generation policy.

## Output-Time Reasoning Visibility

Use ``GenerationParameters`` and ``ReasoningOptions`` to control how reasoning content is surfaced:

```swift
let parameters = GenerationParameters(
    maxTokens: 128,
    reasoning: .separate
)
```

Supported visibility modes:

- ``ReasoningOptions/hidden`` hides reasoning from user-visible output
- ``ReasoningOptions/inline`` leaves reasoning inline with text output
- ``ReasoningOptions/separate`` emits reasoning as separate ``GenerationEvent/reasoning(_:)`` events

## Streaming Behavior

When reasoning visibility is `.separate`, the stream can contain both visible text and reasoning chunks:

```swift
for await event in try await container.generate(input, parameters: parameters) {
    if let reasoning = event.reasoning {
        print("[reasoning]", reasoning)
    }
    if let text = event.text {
        print(text, terminator: "")
    }
}
```

Use `event.completion` to observe the final ``CompletionInfo`` regardless of reasoning visibility.

## Design Rule

Keep these layers separate:

- ``PromptPreparationOptions`` controls how the prompt is rendered
- ``GenerationParameters`` controls how generated output is surfaced

Do not treat template variables as a substitute for generation policy, and do not use generation policy to smuggle prompt-template configuration.
