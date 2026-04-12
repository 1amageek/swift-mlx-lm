# Documentation

This directory keeps non-API documentation that does not need to live at the repository root.

## Maintained Documents

- `../README.md` — project overview, architecture, and public usage
- `using-swift-lm.md` — application developer guide for generation, embeddings, and public API usage
- `releases/0.1.0.md` — release notes and public support boundary for `0.1.0`
- `releases/0.3.0.md` — release notes for the renamed public generation surface
- `releases/0.4.0.md` — release notes for embedding request values and public API guide alignment
- `../Sources/SwiftLM/SwiftLM.docc/` — DocC catalog for API and guide documentation
- `../AGENTS.md` — repository guidance for coding agents
- `../DESIGN-Metal4.md` — forward-looking Metal 4 design notes
- `../Sources/MetalCompiler/STAF/README.md` — STAF format notes
- `../Sources/MetalCompiler/STAF/KVCacheSpec.md` — KV cache layout specification

The maintained user-facing documentation should agree on these public API rules:

- `Container / Context / Input` is the primary public API shape
- prompt-time thinking control and output-time reasoning visibility are documented separately
- `TextEmbeddingInput` is the preferred embedding request value
- staged generation types such as `PreparedPrompt`, `ExecutablePrompt`, and `PromptSnapshot` are documented as advanced APIs

## Archive

- `archive/reports/` — one-off benchmark notes, migration memos, and experiment write-ups
- `archive/articles/` — article drafts and long-form narrative documents

Archived documents are kept for reference, but they are not part of the actively maintained project surface.
