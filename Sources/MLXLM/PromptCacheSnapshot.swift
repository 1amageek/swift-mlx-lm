import MLX

/// Snapshot of KV cache state for prefix reuse across generations.
public struct PromptCacheSnapshot {

    /// Number of prefix tokens represented by this snapshot.
    public let prefixTokenCount: Int

    /// Class names of each cache layer (for reconstruction).
    public let cacheClasses: [String]

    /// Meta-state strings per cache layer.
    public let cacheMetaState: [[String]]

    /// MLXArray state per cache layer.
    public let cacheState: [[MLXArray]]

    public init(
        prefixTokenCount: Int,
        cacheClasses: [String],
        cacheMetaState: [[String]],
        cacheState: [[MLXArray]]
    ) {
        self.prefixTokenCount = prefixTokenCount
        self.cacheClasses = cacheClasses
        self.cacheMetaState = cacheMetaState
        self.cacheState = cacheState
    }
}

// MARK: - Capture / Materialize

/// Capture the current KV cache state into a snapshot for later reuse.
///
/// - Parameters:
///   - cache: Active KV caches (one per layer).
///   - prefixTokenCount: Number of prompt tokens represented by this cache state.
/// - Returns: A snapshot that can be restored via `materializePromptCache`.
public func capturePromptCache(
    cache: [KVCache],
    prefixTokenCount: Int
) -> PromptCacheSnapshot {
    var classes: [String] = []
    var metaStates: [[String]] = []
    var states: [[MLXArray]] = []

    for layer in cache {
        classes.append(cacheClassName(layer))
        metaStates.append(layer.metaState)
        states.append(layer.state)
    }

    return PromptCacheSnapshot(
        prefixTokenCount: prefixTokenCount,
        cacheClasses: classes,
        cacheMetaState: metaStates,
        cacheState: states
    )
}

/// Restore KV caches from a previously captured snapshot.
///
/// - Parameter snapshot: A snapshot created by `capturePromptCache`.
/// - Returns: KV caches initialized with the snapshot's state.
public func materializePromptCache(
    from snapshot: PromptCacheSnapshot
) -> [KVCache] {
    var caches: [KVCache] = []

    for i in 0..<snapshot.cacheClasses.count {
        let cache = instantiateCache(className: snapshot.cacheClasses[i], metaState: snapshot.cacheMetaState[i])
        cache.state = snapshot.cacheState[i]
        cache.metaState = snapshot.cacheMetaState[i]
        caches.append(cache)
    }

    return caches
}

// MARK: - Private Helpers

private func cacheClassName(_ cache: KVCache) -> String {
    switch cache {
    case is KVCacheSimple:
        return "KVCacheSimple"
    case is RotatingKVCache:
        return "RotatingKVCache"
    case is QuantizedKVCache:
        return "QuantizedKVCache"
    default:
        return "KVCacheSimple"
    }
}

private func instantiateCache(className: String, metaState: [String]) -> KVCache {
    switch className {
    case "RotatingKVCache":
        let maxSize = metaState.count >= 2 ? (Int(metaState[1]) ?? 4096) : 4096
        let keep = metaState.count >= 1 ? (Int(metaState[0]) ?? 0) : 0
        let step = metaState.count >= 3 ? (Int(metaState[2]) ?? 256) : 256
        return RotatingKVCache(maxSize: maxSize, keep: keep, step: step)
    case "QuantizedKVCache":
        let groupSize = metaState.count >= 3 ? (Int(metaState[2]) ?? 64) : 64
        let bits = metaState.count >= 4 ? (Int(metaState[3]) ?? 8) : 8
        return QuantizedKVCache(groupSize: groupSize, bits: bits)
    default:
        return KVCacheSimple()
    }
}
