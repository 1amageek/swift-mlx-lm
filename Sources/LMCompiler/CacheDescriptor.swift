import SwiftLM

/// The kind of cache required by a layer.
public enum CacheKind: Sendable, Equatable {

    /// Standard key-value cache for attention layers (unbounded growth).
    case kv

    /// Fixed-size rotating key-value cache with middle token eviction.
    case rotating(maxSize: Int)

    /// Quantized key-value cache for memory reduction.
    case quantized(bits: Int, groupSize: Int)

    /// Recurrent state cache for state-space models (DeltaNet, Mamba).
    case recurrent
}

/// Describes the cache requirements for a single cacheable operation.
///
/// Each descriptor is keyed by a `StructuralPath` that uniquely identifies
/// the operation in the model graph. The executor resolves cache slots by
/// path lookup rather than relying on execution-order counters.
public struct CacheDescriptor: Sendable {

    /// The structural path to the cacheable operation.
    public let path: StructuralPath

    /// The kind of cache required.
    public let kind: CacheKind

    /// The flat index of this cache in the cache array.
    public let slotIndex: Int

    public init(path: StructuralPath, kind: CacheKind, slotIndex: Int) {
        self.path = path
        self.kind = kind
        self.slotIndex = slotIndex
    }
}
