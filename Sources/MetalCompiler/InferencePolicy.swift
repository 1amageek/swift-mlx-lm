/// Deployment-level decisions for model inference.
///
/// Separates deployment intent from model architecture (IR) and
/// implementation details (compiler). Consumers specify WHAT they want;
/// the compiler decides HOW to achieve it.
///
/// ```swift
/// // Default: FP16 KV cache, 4096 max tokens
/// let policy = InferencePolicy.default
///
/// // Custom: Q4 K cache, FP16 V cache, 8192 max tokens
/// let policy = InferencePolicy(
///     maximumSequenceLength: 8192,
///     kvCache: KVCachePolicy(
///         keyScheme: .fixed(.q4Group64ScaleF16),
///         valueScheme: .automatic
///     )
/// )
/// ```
public struct InferencePolicy: Sendable {

    /// Maximum number of tokens in KV cache.
    public var maximumSequenceLength: Int

    /// KV cache quantization and layout policy.
    public var kvCache: KVCachePolicy

    public init(
        maximumSequenceLength: Int = 4096,
        kvCache: KVCachePolicy = .automatic
    ) {
        self.maximumSequenceLength = maximumSequenceLength
        self.kvCache = kvCache
    }

    /// Default policy: automatic scheme selection, 4096 max sequence length.
    public static let `default` = InferencePolicy()
}

/// KV cache quantization and layout configuration.
///
/// K and V caches are configured independently because they have different
/// tolerance for quantization: K is used for dot products (tolerates aggressive
/// quantization), V is used for weighted sums (requires conservative quantization
/// to preserve outlier values).
public struct KVCachePolicy: Sendable {

    /// Quantization scheme for K cache.
    public var keyScheme: SchemeSelection

    /// Quantization scheme for V cache.
    public var valueScheme: SchemeSelection

    /// Memory layout mode.
    public var layoutMode: KVCacheLayoutMode

    public init(
        keyScheme: SchemeSelection = .automatic,
        valueScheme: SchemeSelection = .automatic,
        layoutMode: KVCacheLayoutMode = .sequenceMajor
    ) {
        self.keyScheme = keyScheme
        self.valueScheme = valueScheme
        self.layoutMode = layoutMode
    }

    /// Automatic: compiler selects scheme based on weight format.
    public static let automatic = KVCachePolicy()
}

/// How to select a KV cache quantization scheme.
public enum SchemeSelection: Sendable {
    /// Let the compiler choose based on weight format (current behavior).
    case automatic
    /// Use a specific quantization scheme.
    case fixed(QuantizationSchemeIdentifier)
}
