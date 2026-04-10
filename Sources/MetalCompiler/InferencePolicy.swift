/// Deployment-level decisions for model inference.
///
/// Separates deployment intent from model architecture (IR) and
/// implementation details (compiler). Consumers specify WHAT they want;
/// the compiler decides HOW to achieve it.
///
/// ```swift
/// // Default: model-aware loader default, 4096 max tokens
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

    /// Default policy: conservative automatic KV cache, 4096 max sequence length.
    ///
    /// Loaders may resolve this into a model-aware default after inspecting
    /// the compiled graph. For attention-backed decode paths, `SwiftLM`
    /// upgrades this to RotorQuant Q4 by default.
    public static let `default` = InferencePolicy()
}

/// KV cache quantization and layout configuration.
///
/// K and V caches are configured independently because they have different
/// tolerance for quantization: K is used for dot products (tolerates aggressive
/// quantization), V is used for weighted sums (requires conservative quantization
/// to preserve outlier values).
///
/// For RotorQuant schemes, Clifford algebra Cl(3,0) rotors are applied per group
/// of 3 dimensions before quantization. The rotation smooths outliers, improving
/// quantization quality at the same bit width.
///
/// QJL (Quantized Johnson-Lindenstrauss) correction stores projected residuals
/// alongside the K cache to provide unbiased inner product estimation.
/// Set `qjlDimension > 0` to enable. Typical value: 16.
public struct KVCachePolicy: Sendable {

    /// Quantization scheme for K cache.
    public var keyScheme: SchemeSelection

    /// Quantization scheme for V cache.
    public var valueScheme: SchemeSelection

    /// Memory layout mode.
    public var layoutMode: KVCacheLayoutMode

    /// QJL residual projection dimension for K cache correction.
    ///
    /// 0 = disabled (default). When > 0, stores per-token JL-projected
    /// quantization residuals to correct Q·K inner product estimation.
    /// Only effective when K uses a RotorQuant scheme.
    public var qjlDimension: Int

    public init(
        keyScheme: SchemeSelection = .automatic,
        valueScheme: SchemeSelection = .automatic,
        layoutMode: KVCacheLayoutMode = .sequenceMajor,
        qjlDimension: Int = 0
    ) {
        self.keyScheme = keyScheme
        self.valueScheme = valueScheme
        self.layoutMode = layoutMode
        self.qjlDimension = qjlDimension
    }

    /// Automatic: compiler selects a conservative dense scheme based on weight format.
    public static let automatic = KVCachePolicy()

    /// Whether either K or V requests a RotorQuant scheme explicitly.
    public var usesRotorQuant: Bool {
        switch keyScheme {
        case .fixed(let scheme) where scheme.isRotorScheme:
            return true
        default:
            break
        }
        switch valueScheme {
        case .fixed(let scheme) where scheme.isRotorScheme:
            return true
        default:
            return false
        }
    }
}

/// How to select a KV cache quantization scheme.
public enum SchemeSelection: Sendable {
    /// Let the compiler choose based on weight format (current behavior).
    case automatic
    /// Use a specific quantization scheme.
    case fixed(QuantizationSchemeIdentifier)
}

extension SchemeSelection: Equatable {}
extension KVCachePolicy: Equatable {}
