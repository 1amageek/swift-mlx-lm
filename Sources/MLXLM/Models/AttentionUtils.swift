import MLX
import MLXFast

/// Performs scaled dot-product attention with automatic cache management.
///
/// Routes to quantized or standard attention based on cache type.
/// Models call this single function regardless of cache configuration.
func attentionWithCacheUpdate(
    queries: MLXArray,
    keys: MLXArray,
    values: MLXArray,
    cache: KVCache?,
    scale: Float,
    mask: MLXFast.ScaledDotProductAttentionMaskMode = .none
) -> MLXArray {
    guard let cache else {
        return MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: keys,
            values: values,
            scale: scale,
            mask: mask
        )
    }

    if let quantizedCache = cache as? QuantizedKVCache {
        let (quantizedKeys, quantizedValues) = quantizedCache.updateQuantized(
            keys: keys, values: values)
        // Dequantize and use standard attention
        let dqKeys = MLX.dequantized(
            quantizedKeys.0, scales: quantizedKeys.1, biases: quantizedKeys.2,
            groupSize: quantizedCache.groupSize, bits: quantizedCache.bits
        )
        let dqValues = MLX.dequantized(
            quantizedValues.0, scales: quantizedValues.1, biases: quantizedValues.2,
            groupSize: quantizedCache.groupSize, bits: quantizedCache.bits
        )
        return MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: dqKeys,
            values: dqValues,
            scale: scale,
            mask: mask
        )
    } else {
        let (cachedKeys, cachedValues) = cache.update(keys: keys, values: values)
        return MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: cachedKeys,
            values: cachedValues,
            scale: scale,
            mask: mask
        )
    }
}
