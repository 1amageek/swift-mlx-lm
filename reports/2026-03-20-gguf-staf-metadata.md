# GGUF To STAF Metadata Investigation

Date: 2026-03-20  
Repository: `swift-lm`

## Goal

Identify which GGUF concepts should influence the next STAF metadata schema, without turning STAF into a GGUF clone and without leaking executor-specific quirks into higher layers.

## Implemented Minimum

The repository now has the first minimal STAF metadata implementation:

- header format versioning
- metadata table count and offset in the 64-byte header
- typed file-level metadata values
- loader roundtrip into `STAFWeightStore.metadata`
- loader-populated architectural metadata for newly converted caches

Specifically:

- legacy STAF remains loadable
- newly written STAF uses format version `1`
- metadata currently supports:
  - `bool`
  - `u32`
  - `u64`
  - `f32`
  - `f64`
  - `string`
- current default metadata includes:
  - source format
  - converter version
  - source shard count
- `ModelBundleLoader` additionally records:
  - architecture family
  - hidden size
  - layer count
  - intermediate size
  - vocab size
  - attention head counts
  - head dimension
  - tied embeddings
  - rope dimension and theta

This minimum contract has now been extended with provenance-aware cache validation:

- config hash
- tokenizer hash
- tokenizer_config hash
- special_tokens_map hash
- resolved chat-template hash and source
- safetensors manifest hash
- compatibility-aware `isValid(...)` using typed metadata, not mtime alone

`ModelBundleLoader` now builds this metadata through a dedicated STAF metadata builder, so bundle provenance stays in the cache/compiler layer instead of being hand-assembled in the public loader.

## Current STAF State

Current STAF already has a solid tensor payload contract:

- file header:
  - magic
  - section count
  - section table offset
  - string table offset
  - string table size
- per-tensor section entry:
  - tensor name
  - quantization scheme identifier
  - semantic role
  - original dtype
  - rank and shape
  - payload offset and size
  - alignment
  - block size
  - group size
  - checksum
  - safetensors shard index

This is enough for zero-copy loading and current kernel selection.

What it does not yet encode well is file-level semantic metadata.

## What GGUF Adds

GGUF is not just a tensor container. It combines:

- typed file-level key/value metadata
- tensor info records
- tensor payloads

The important structural difference versus tensor-only formats is that GGUF stores both:

- tensor storage information
- standardized model metadata

This is the part STAF should learn from.

Relevant GGUF structure:

- header:
  - magic
  - version
  - tensor count
  - metadata key/value count
- metadata table:
  - typed key
  - typed value
- tensor info:
  - tensor name
  - rank
  - shape
  - ggml type
  - offset into data region

## Design Reading

The main lesson is not "use GGUF".

The lesson is:

- execution caches should be self-describing
- metadata should be typed
- model-level semantics should not be reconstructed from tensor names alone
- layout and quantization should be represented as formal contracts

For `swift-lm`, this implies:

- `safetensors` remains canonical source of truth
- STAF remains the regenerable execution cache
- STAF should gain a typed metadata layer similar in spirit to GGUF's key/value table
- importer logic should map from HF or future GGUF into STAF metadata, rather than making STAF mirror external formats directly

## Metadata Categories STAF Should Add

### 1. File-Level Model Identity

Add typed metadata for:

- architecture family
- model name / identifier
- source format
- converter version
- source revision or fingerprint

Why:

- GGUF's `general.*` metadata makes files self-describing
- current STAF files are executable but weakly identifiable
- cache validation and debugging become easier when the file explains what it is

### 2. Architectural Dimensions

Add typed metadata for normalized model config fields, for example:

- context length
- embedding length
- block count
- feed-forward length
- attention head count
- KV head count
- rope settings
- MoE counts where applicable
- hybrid schedule metadata where applicable

Why:

- GGUF commonly stores architecture-defining values as metadata
- STAF currently relies on adjacent HF files and runtime config reconstruction
- a self-describing cache should state the architectural contract it was built from

Constraint:

- use normalized, family-level keys
- do not store product-specific ad hoc names when a family-level field exists

### 3. Tokenizer Contract

Add typed metadata for tokenizer compatibility, at minimum:

- tokenizer family/type
- vocab size
- BOS/EOS/PAD token ids
- unknown token id if present
- special-token contract hash
- chat template hash or presence marker

Why:

- GGUF usage shows tokenizer metadata matters for execution correctness, not just packaging
- current STAF does not describe tokenizer compatibility at all
- `swift-lm` should not have to guess whether a cache was built against a compatible tokenizer setup

Important:

- STAF does not need to embed the full tokenizer payload
- it should record enough metadata to validate compatibility with external tokenizer files

### 4. Typed Metadata Table

Add a generic typed metadata table to STAF instead of expanding only fixed header fields.

Recommended properties:

- key namespace
- scalar types: bool, u32, u64, f32, f64, string
- small arrays for shape-like metadata
- forward-compatible skipping of unknown keys

Why:

- this is the single most useful GGUF idea
- it avoids hardcoding every future model field into the 64-byte file header
- it keeps STAF generic across model families

Recommendation:

- keep the current fixed header minimal
- add `metadataTableOffset` and `metadataEntryCount`
- store richer semantics in the typed metadata section

### 5. Tensor-Level Layout Contract

Current STAF tensor entries already describe quantization and shape well, but they should eventually formalize:

- canonical tensor layout
- available specialized layouts
- execution-phase applicability
- source tensor mapping

Why:

- this is where current specialized decode layout experiments are still too implicit
- GGUF tensor info is simple, but STAF needs one more layer because it is an execution cache, not just a portable model file

Recommendation:

- keep row-major tensor entry semantics as canonical
- represent specialized decode-only layouts as additional access variants, not as silent replacement of canonical payload meaning

### 6. Provenance And Validation

Add metadata for cache validity:

- safetensors source fingerprint
- config fingerprint
- tokenizer fingerprint
- converter build version
- kernel/layout capability version

Why:

- current STAF invalidation is too dependent on out-of-band assumptions
- GGUF's self-description helps users inspect what they have
- STAF needs this for correctness and debuggability more than for portability

## What STAF Should Not Copy From GGUF

### 1. GGML-Specific Type IDs

Do not adopt GGUF/ggml type identifiers directly as STAF's internal execution contract.

Reason:

- STAF's job is to describe the runtime payload used by `swift-lm`
- the internal quantization/layout contract should remain owned by `swift-lm`
- external formats should map into STAF, not define STAF

### 2. Product-Specific Metadata Keys

Do not make STAF depend on product labels when family-level keys exist.

Reason:

- compiler and cache logic must stay generic across model families
- product-specific keys create exactly the kind of special-case logic the current design is trying to avoid

### 3. Full Tokenizer Payload Embedding

Do not turn STAF into a monolithic model bundle that copies all HF tokenizer files.

Reason:

- STAF is an execution cache
- tokenizer artifacts already live next to the model bundle
- compatibility metadata is enough

## Recommended STAF Schema Direction

### Header

Keep a compact fixed header, but extend it with:

- format version
- metadata table offset
- metadata entry count
- string table offset
- section table offset
- payload offset

### Metadata Table

Add a typed table with namespaced keys such as:

- `general.architecture_family`
- `general.model_name`
- `general.source_format`
- `general.converter_version`
- `model.context_length`
- `model.embedding_length`
- `model.block_count`
- `model.feed_forward_length`
- `model.attention.head_count`
- `model.attention.kv_head_count`
- `model.rope.base`
- `model.rope.scale`
- `tokenizer.family`
- `tokenizer.vocab_size`
- `tokenizer.bos_token_id`
- `tokenizer.eos_token_id`
- `tokenizer.pad_token_id`
- `tokenizer.chat_template_hash`
- `cache.source.safetensors_fingerprint`
- `cache.source.config_fingerprint`
- `cache.source.tokenizer_fingerprint`

### Tensor Section Entries

Keep existing fields, but plan for optional extension fields or metadata references for:

- canonical layout id
- specialized layout availability
- specialization phase mask
- source tensor identity

## How This Fits The Current Architecture

This direction preserves current design principles:

- `safetensors` stays canonical
- STAF stays regenerable
- compiler remains generic
- specialized layouts remain backend-layer concerns
- model declarations stay family-level and backend-independent

Most importantly:

- HF input bundle
- future GGUF importer
- future other formats

can all normalize into the same STAF metadata schema.

That keeps importer complexity at the boundary and keeps execution generic.

## Immediate Next Step

The best next implementation step is not GGUF import.

It is:

1. add STAF format versioning
2. add a typed metadata table
3. populate a minimal file-level metadata set from current HF inputs
4. validate tokenizer/config/cache compatibility from STAF metadata during load

After that, GGUF import becomes a mapping problem instead of a format-design problem.

## Sources

- [Hugging Face Hub: GGUF](https://huggingface.co/docs/hub/gguf)
- [Hugging Face Hub: GGUF usage with llama.cpp](https://huggingface.co/docs/hub/gguf-llamacpp)
- [llama.cpp Wiki: dev notes](https://github.com/ggml-org/llama.cpp/wiki/dev-notes)
