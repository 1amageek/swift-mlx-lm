import LMArchitecture

/// Gemma 4 attention with value normalization and optional KV sharing.
///
/// Extends standard attention with:
/// - Per-head RMS normalization on V projections with learned scale (GemmaRMSNorm: 1 + weight)
/// - Optional K=V mode: reuses K projection as V source
/// - Optional cross-layer KV cache sharing
struct Gemma4Attention: ModelComponent {

    typealias Attributes = AttentionAttributes

    // MARK: - Projection geometry

    let hiddenSize: Int
    let headCount: Int
    let kvHeadCount: Int
    let headDimension: Int
    let bias: Bool

    // MARK: - Attention computation

    let attentionScale: Float?
    let rope: RoPEAttributes
    let qkNorm: QKNormKind
    let window: AttentionWindow?

    // MARK: - Gemma 4 specific: K/V handling

    /// How the raw V tensor is sourced. `.keyProjection` reuses K as V.
    let valueProjectionSource: AttentionValueProjectionSource

    /// Layer index whose KV cache should be reused instead of computing new K/V.
    let sharedKeyValueSourceLayerIndex: Int?

    var attributes: AttentionAttributes {
        AttentionAttributes(
            hiddenSize: hiddenSize,
            headCount: headCount,
            kvHeadCount: kvHeadCount,
            headDimension: headDimension,
            attentionScale: attentionScale,
            bias: bias,
            rope: rope,
            qkNorm: qkNorm,
            valueNorm: .rmsNormNoScale,
            valueProjectionSource: valueProjectionSource,
            window: window,
            sharedKeyValueSourceLayerIndex: sharedKeyValueSourceLayerIndex
        )
    }
}
