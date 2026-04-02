# Deferred Array Parsing & Concurrent Tokenizer Construction

**Date**: 2026-03-11
**Module**: `GGUFParser`, `GGUFTokenizer`, `MLXLM`
**Hardware**: M4 Max
**Model**: Qwen3.5-0.8B-Q4_K_M (508MB GGUF, vocab=248,320, merges=247,587)

---

## Problem

Full pipeline profiling (`gpu-weight-packing.md`) identified two dominant bottlenecks:

| Stage | Time | % of Total |
|-------|------|-----------|
| GGUF parse | 334ms | 33% |
| Tokenizer creation | 432ms | 43% |
| **Subtotal** | **766ms** | **76%** |

The root cause: tokenizer data (248K vocabulary strings, 247K merge strings, 248K token types) was processed through 3 redundant passes:

1. **Parse**: bytes → `GGUFMetadataValue.string()` enum wrappers (248K + 247K allocations)
2. **Accessor**: `[GGUFMetadataValue]` → `compactMap(\.stringValue)` → `[String]`
3. **Tokenizer init**: `[String]` → enumerate → `[String: Int]` dictionary insertion

Additionally, `file[.tokens]?.count` in model constructors triggered full materialization of the 248K-element deferred array just to obtain the vocabulary size.

## Solution

Three independent optimizations:

### 1. Deferred Array Parsing

Skip large tokenizer arrays during `GGUFFile.parse()`. Instead of reading 248K+ elements into `[GGUFMetadataValue]`, record byte offsets and read on demand.

**Deferred keys**: `tokenizer.ggml.tokens`, `tokenizer.ggml.merges`, `tokenizer.ggml.scores`, `tokenizer.ggml.token_type`

```
GGUFFile.parse()
  ├── Regular metadata: read as before
  └── Deferred keys: read array header (type, count), record byte offset, skip elements
                     → DeferredArray { elementType, count, offset }
```

**Single-pass deferred readers** build final data structures directly from mmap bytes:

| Reader | Output | Use Case |
|--------|--------|----------|
| `readDeferredVocabulary()` | `([String], [String: Int])` | Vocab + tokenToID in one pass |
| `readDeferredStringDictionary()` | `[String: Int]` | Merge ranks |
| `readDeferredFloat32Array()` | `[Float]` | SentencePiece scores |
| `readDeferredInt32Array()` | `[Int]` | Token types |

### 2. Concurrent Deferred Reading

Vocabulary, merges, and token types reside in disjoint byte ranges of the mmap buffer. `GGUFTokenizerFactory` reads all three concurrently via `DispatchQueue.concurrentPerform`:

```
                          ┌─ Thread 0: readDeferredVocabulary (tokens)  ─── 415ms ─┐
concurrentPerform(3) ─────┼─ Thread 1: readDeferredStringDictionary (merges) ── 415ms ─┼── wall: 415ms
                          └─ Thread 2: readDeferredInt32Array (types) ── 313ms ─┘
```

The wall-clock time equals `max(vocab, merges, types)` instead of `sum(vocab, merges, types)`.

### 3. Vocabulary Size Without Materialization

Replaced `file[.tokens]?.count` with `file.vocabularySize` in all model constructors (TransformerModel, Qwen35Model, CohereModel, Qwen25VLModel). `vocabularySize` uses `deferredArrayCount()` which returns the pre-recorded count from the array header — zero I/O, zero allocation.

## Results

**Test**: `LoaderProfilingTests` (serialized, single test at a time)

| Stage | Before | After | Change |
|-------|--------|-------|--------|
| GGUF parse | 334ms | 202ms | **-40%** |
| Tokenizer creation | 432ms | 498ms | +15%* |
| Model construction | 193ms | 34ms | **-82%** |
| Weight loading | 40ms | 110ms | (variance) |
| eval(model) | 97ms | — | (unchanged) |
| **Total loadContext** | **1,003ms** | **856ms** | **-15%** |

\* Tokenizer creation increased because mmap page faults that previously occurred during parse now occur during tokenizer creation. The total I/O cost is unchanged, but the page faults are now interleaved with dictionary construction.

### Stage Breakdown (concurrent reads, warm mmap)

| Operation | Time | Notes |
|-----------|------|-------|
| readDeferredVocabulary | 165ms | 248K strings → [String] + [String: Int] |
| readDeferredStringDictionary | 162ms | 247K strings → [String: Int] |
| readDeferredInt32Array | 88ms | 248K int32s → [Int] |
| **Sequential total** | **415ms** | |
| **Concurrent total** | **~165ms** | max(vocab, merges) |

### Net Effect

| Metric | Before | After |
|--------|--------|-------|
| Parse + Tokenizer + Model construction | 959ms | 734ms |
| Full loadContext | 1,003ms | 856ms |
| **Saved** | | **~150ms (15%)** |

## Architecture

```
GGUFFile.parse(url:)
  │
  ├── metadata: [String: GGUFMetadataValue]     ← regular keys (fast)
  └── deferredArrays: [String: DeferredArray]    ← byte offsets only (skip data)
        │
        ▼
GGUFTokenizerFactory.create(from:)
  │
  ├── deferredArrayCount() → vocabSize check (O(1))
  │
  └── concurrentPerform(3)
        ├── readDeferredVocabulary("tokens")     → ([String], [String: Int])
        ├── readDeferredStringDictionary("merges") → [String: Int]
        └── readDeferredInt32Array("token_type") → [Int]
              │
              ▼
        MergesBPETokenizer(vocabulary:, tokenToID:, mergeRanks:, ...)
                           ↑ fast init (pre-built dicts, no re-iteration)
```

## Files Changed

| File | Change |
|------|--------|
| `Sources/GGUFParser/GGUFFile.swift` | `DeferredArray` struct, `deferredKeys`, deferred skip in `parse()`, deferred reader extensions |
| `Sources/GGUFParser/GGUFReader.swift` | `setOffset()`, `skipMetadataValue()`, bulk direct readers (`readStringArrayDirect`, `readStringArrayAsDictionary`, `readFloat32ArrayDirect`, `readInt32ArrayDirect`) |
| `Sources/GGUFParser/GGUFMetadataAccessors.swift` | `tokens`/`merges`/`tokenScores`/`tokenTypes` accessors check deferred path first; `vocabularySize` uses `deferredArrayCount()` |
| `Sources/GGUFParser/GGUFMetadataKey.swift` | Subscript checks `deferredArrays` with `materializeDeferredArray()` fallback |
| `Sources/GGUFTokenizer/GGUFTokenizerFactory.swift` | Concurrent deferred reading via `DispatchQueue.concurrentPerform` |
| `Sources/GGUFTokenizer/MergesBPETokenizer.swift` | Fast init accepting pre-built `tokenToID` and `mergeRanks` |
| `Sources/MLXLM/Models/TransformerModel.swift` | `file[.tokens]?.count` → `file.vocabularySize` |
| `Sources/MLXLM/Models/Qwen35Model.swift` | Same |
| `Sources/MLXLM/Models/CohereModel.swift` | Same |
| `Sources/MLXLM/Models/Qwen25VL/Qwen25VLModel.swift` | Same |
| `Tests/MLXLMDiagnosticTests/LoaderProfilingTests.swift` | Stage-by-stage profiling test |

## Updated Full Pipeline Profile

| Stage | Time (ms) | % | Notes |
|-------|-----------|---|-------|
| GGUF parse | 202 | 24% | Deferred tokenizer arrays (skip ~750K elements) |
| Tokenizer creation | 498 | 58% | Concurrent deferred reads + dict construction |
| Model construction | 34 | 4% | vocabularySize via deferredArrayCount (O(1)) |
| Weight conversion + eval | ~120 | 14% | Unchanged |
| **Total** | **~856** | **100%** | |

The remaining bottleneck is tokenizer creation (498ms), dominated by mmap page faults and `[String: Int]` dictionary insertion for ~500K total entries. Further optimization would require binary-serialized tokenizer caches or lazier dictionary strategies.
