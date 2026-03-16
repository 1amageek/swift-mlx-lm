# KV Cache Runtime Specification

## Overview

KV cache is transient runtime state that lives only during inference. It is NOT persisted to disk and is NOT part of the STAF file format.

This document defines the memory layout and quantization rules for KV cache on Apple Silicon Metal.

## KVCacheSpecification

```swift
struct KVCacheSpecification {
    let keyQuantizationScheme: QuantizationSchemeIdentifier
    let valueQuantizationScheme: QuantizationSchemeIdentifier
    let layoutMode: KVCacheLayoutMode
    let layerCount: Int
    let kvHeadCount: Int
    let headDimension: Int
    let maximumSequenceLength: Int
    let tokenSlotAlignment: Int  // 256 bytes
}
```

## Why K and V Are Quantized Separately

K and V have different numerical properties in attention:

```
K: used for dot product (q · K^T → scores)
   Distribution is relatively uniform.
   Tolerant of quantization.
   → Aggressive quantization reduces DRAM reads.

V: used for weighted sum (softmax(scores) · V → output)
   Outliers directly impact output quality.
   → Conservative quantization preserves accuracy.
```

## Layout Modes

### Sequence-Major (default for decode)

```
[layer][head][seq][dim]

K_cache[layer][head]:
  [tok_0: dim values][tok_1: dim values]...[tok_T: dim values]

Decode writes new token to end — contiguous append.
Optimal for autoregressive generation.
```

### Head-Major (for GQA/MQA parallel processing)

```
[layer][seq][head][dim]

K_cache[layer][tok]:
  [head_0: dim values][head_1: dim values]...[head_H: dim values]

All heads for one token are contiguous — efficient for grouped-query attention.
```

## Recommended Quantization Combinations

| Use Case | K Scheme | V Scheme |
|----------|----------|----------|
| Quality-first | Q8_G64_SF16 | FP16 |
| Balanced | Q8_G32_SF16 | Q8_G64_SF16 |
| Memory-saving (long context) | Q6_G16_SF16 | Q8_G32_SF16 |
| Maximum speed | FP16 | FP16 |

## Runtime Quantization

During decode, projection GEMVs produce FP16 K and V values. These must be quantized before storing in cache when the cache uses a quantized scheme.

The runtime quantization kernel:
1. Reads FP16 values from GEMV output
2. Computes per-group scale (and zero for asymmetric)
3. Quantizes to target bit width
4. Writes interleaved block to cache at the current position

## Alignment

```
Each token slot:     256B aligned
K buffer start:      4KB aligned (storageModePrivate)
V buffer start:      4KB aligned (storageModePrivate)
```

K and V are separate MTLBuffers. Never mixed in a single buffer.
