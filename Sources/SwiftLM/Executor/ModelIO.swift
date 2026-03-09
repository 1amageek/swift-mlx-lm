/// Input tensors for model execution.
public struct ModelInputs: Sendable {

    /// Token IDs to process.
    public let tokenIDs: TensorData

    /// Optional position IDs (for models requiring explicit positions).
    public let positionIDs: TensorData?

    /// Optional attention mask.
    public let attentionMask: TensorData?

    /// Optional KV cache from a previous execution step.
    public let cache: KVCacheState?

    public init(
        tokenIDs: TensorData,
        positionIDs: TensorData? = nil,
        attentionMask: TensorData? = nil,
        cache: KVCacheState? = nil
    ) {
        self.tokenIDs = tokenIDs
        self.positionIDs = positionIDs
        self.attentionMask = attentionMask
        self.cache = cache
    }
}

/// Output tensors from model execution.
public struct ModelOutputs: Sendable {

    /// Logits over the vocabulary for each position.
    public let logits: TensorData

    /// Updated KV cache state after execution.
    public let cache: KVCacheState?

    /// Optional hidden states from intermediate layers.
    public let hiddenStates: TensorData?

    public init(
        logits: TensorData,
        cache: KVCacheState? = nil,
        hiddenStates: TensorData? = nil
    ) {
        self.logits = logits
        self.cache = cache
        self.hiddenStates = hiddenStates
    }
}

/// Opaque KV cache state for autoregressive generation.
///
/// The concrete representation depends on the runtime backend.
/// SwiftLM core defines only the contract; backend-specific
/// implementations provide typed access to cache contents.
public struct KVCacheState: Sendable {

    /// Opaque backend-specific cache data.
    public let storage: any Sendable

    /// Number of cached positions (sequence length already processed).
    public let cachedLength: Int

    public init(storage: any Sendable, cachedLength: Int) {
        self.storage = storage
        self.cachedLength = cachedLength
    }
}
