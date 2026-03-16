/// Attributes for a short convolution node.
///
/// Represents double-gated LIV depthwise convolution:
///   in_proj(D -> 3D) -> chunk(B, C, x) -> B*x -> depthwise_conv1d -> C*conv_out -> out_proj
///
/// ## Dimensions
///
/// - `hiddenSize`: input/output dimension D
/// - `kernelSize`: depthwise conv1d kernel width (= conv_L_cache from HF config)
///
/// ShortConv is a distinct family from DeltaNet:
/// - No recurrent state (conv cache only)
/// - Plain element-wise gating (no sigmoid/softplus)
/// - No dtype boundary management (no long-range accumulation)
///
/// Reference: LiquidAI LFM2 (Lfm2ShortConv in HuggingFace transformers)
public struct ShortConvAttributes: OperationAttributes, Codable, Equatable {

    /// Hidden dimension of the short convolution block (input and output size).
    public let hiddenSize: Int

    /// Depthwise conv1d kernel width.
    ///
    /// Corresponds to `conv_L_cache` in HF config.json.
    /// Determines the sliding window size for causal convolution.
    public let kernelSize: Int

    public init(hiddenSize: Int, kernelSize: Int) {
        self.hiddenSize = hiddenSize
        self.kernelSize = kernelSize
    }
}
