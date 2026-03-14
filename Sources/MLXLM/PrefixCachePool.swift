import MLX

/// Cache layout for a generation request.
///
/// Only slots with the same layout can be reused safely.
enum CacheLayout: Equatable, Sendable {
    case simple
    case rotating(maxSize: Int)
    case quantized(bits: Int, groupSize: Int, startLayer: Int)

    init(parameters: GenerateParameters) {
        if let maxSize = parameters.maxKVSize {
            self = .rotating(maxSize: maxSize)
            return
        }

        if let bits = parameters.kvBits {
            self = .quantized(
                bits: bits,
                groupSize: parameters.kvGroupSize,
                startLayer: parameters.quantizedKVStart
            )
            return
        }

        self = .simple
    }
}

/// A cache slot holding live KV caches and the token sequence they represent.
///
/// Unlike `PromptCacheSnapshot` which captures/restores cache state (expensive copy),
/// cache slots keep KV caches alive in memory. Reuse is achieved by trimming the
/// divergent tail — no serialization cycle needed.
///
/// Analogous to Ollama's `InputCacheSlot` in `runner/llamarunner/cache.go`.
struct CacheSlot {

    /// Live KV caches (one per model layer).
    var cache: [KVCache]

    /// Cache layout for this slot.
    var layout: CacheLayout

    /// Full token sequence processed through this cache (prompt + generated).
    ///
    /// Used for prefix matching: the next request's tokens are compared against
    /// this sequence to find the longest common prefix.
    var tokens: [Int]

    /// When this slot was last used (for LRU eviction).
    var lastUsed: ContinuousClock.Instant
}

/// Manages multiple KV cache slots with token-level prefix matching and LRU eviction.
///
/// On each generation request, finds the slot whose token history shares the longest
/// common prefix with the new request's tokens. The divergent tail is trimmed from the
/// KV cache, and only the remaining tokens need to be prefilled.
///
/// This is the core optimization for multi-turn chat: the system prompt + prior
/// conversation is already in the cache, so only the new user message (and any
/// assistant response from the previous turn) needs to be processed.
///
/// Analogous to Ollama's `InputCache` in `runner/llamarunner/cache.go`.
/// Not thread-safe — designed to be used within `ModelContainer` actor.
public final class PrefixCachePool {

    private var slots: [CacheSlot] = []
    private let maxSlots: Int
    private let clock = ContinuousClock()

    /// Create a prefix cache pool.
    ///
    /// - Parameter maxSlots: Maximum number of cache slots to keep alive.
    ///   Default is 4 (matches typical multi-conversation scenarios).
    public init(maxSlots: Int = 4) {
        precondition(maxSlots > 0, "maxSlots must be positive")
        self.maxSlots = maxSlots
    }

    /// Acquire the best cache for the given token sequence.
    ///
    /// Finds the slot with the longest common prefix, trims divergent tokens from the
    /// KV cache, and returns it ready for generation. The slot is removed from the pool
    /// and must be returned via `release(cache:tokens:)` after generation completes.
    ///
    /// If no matching slot is found, creates a fresh cache via the factory.
    ///
    /// - Parameters:
    ///   - tokens: The full token sequence of the new request.
    ///   - layout: Cache layout for the new request.
    ///   - newCacheFactory: Factory to create fresh caches when no prefix match exists.
    /// - Returns: A tuple of (cache, reusedPrefixLength). The cache is trimmed to the
    ///   common prefix; `reusedPrefixLength` indicates how many tokens are already cached.
    func acquire(
        for tokens: [Int],
        layout: CacheLayout,
        newCacheFactory: () -> [KVCache]
    ) -> (cache: [KVCache], reusedPrefixLength: Int) {
        guard !tokens.isEmpty else {
            return (newCacheFactory(), 0)
        }

        // Build candidates sorted by prefix length, longest first.
        // Unlike the previous single-best approach, this tries multiple slots
        // so that a non-trimmable slot (e.g. DeltaNet) that requires trimming
        // doesn't prevent reuse of a shorter exact-prefix match.
        var candidates: [(index: Int, prefixLen: Int)] = []
        for (i, slot) in slots.enumerated() {
            guard slot.layout == layout else { continue }
            let common = countCommonPrefix(slot.tokens, tokens)
            if common > 0 {
                candidates.append((index: i, prefixLen: common))
            }
        }
        candidates.sort { $0.prefixLen > $1.prefixLen }

        for candidate in candidates {
            let slot = slots[candidate.index]
            let trimCount = slot.tokens.count - candidate.prefixLen

            if trimCount == 0 {
                // Exact prefix match — safe for all cache types including
                // non-trimmable (DeltaNet recurrent state, MLXInferenceKVCache).
                // This is the common multi-turn chat pattern.
                slots.remove(at: candidate.index)
                return (slot.cache, candidate.prefixLen)
            }

            // Trimming required — only possible if all caches support it
            if slot.cache.allSatisfy({ $0.isTrimmable }) {
                slots.remove(at: candidate.index)
                for cache in slot.cache { cache.trim(trimCount) }
                return (slot.cache, candidate.prefixLen)
            }

            // This slot can't be trimmed — keep it in the pool and try next.
            // Previously the slot was removed and lost here (slot leak).
        }

        // No reusable slot found — create fresh cache
        return (newCacheFactory(), 0)
    }

    /// Return a cache to the pool after generation completes.
    ///
    /// The cache and its full token sequence (input + generated) are stored for
    /// potential reuse by future requests. If the pool is at capacity, the
    /// least recently used slot is evicted.
    ///
    /// - Parameters:
    ///   - cache: The KV caches used for generation.
    ///   - layout: Cache layout for the generation request.
    ///   - tokens: The full token sequence now represented by this cache.
    func release(cache: [KVCache], layout: CacheLayout, tokens: [Int]) {
        let slot = CacheSlot(
            cache: cache,
            layout: layout,
            tokens: tokens,
            lastUsed: clock.now
        )
        slots.append(slot)

        // Evict LRU if over capacity
        while slots.count > maxSlots {
            if let minIdx = slots.enumerated()
                .min(by: { $0.element.lastUsed < $1.element.lastUsed })?
                .offset
            {
                slots.remove(at: minIdx)
            }
        }
    }

    /// Clear all cached state.
    ///
    /// Call when switching models or when memory pressure requires freeing GPU resources.
    public func clear() {
        slots.removeAll()
    }

    /// Number of currently cached slots.
    public var count: Int { slots.count }

    // MARK: - Private

    /// Count matching tokens from the beginning of two sequences.
    ///
    /// Analogous to Ollama's `countCommonPrefix` in `cache.go`.
    private func countCommonPrefix(_ a: [Int], _ b: [Int]) -> Int {
        let len = min(a.count, b.count)
        for i in 0..<len {
            if a[i] != b[i] { return i }
        }
        return len
    }
}
