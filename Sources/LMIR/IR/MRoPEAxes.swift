/// Multi-axis Rotary Position Embedding (M-RoPE) configuration.
///
/// M-RoPE extends standard 1-axis RoPE by partitioning the rotary
/// dimensions into multiple sections, each driven by a different
/// positional axis (temporal, height, width).
///
/// Used by vision-language models to encode spatial and temporal
/// position information from both text and image modalities.
///
/// ## Variants
///
/// - **Qwen 2.5-VL**: sections=[16,24,24], interleaved=false (contiguous blocks per axis)
/// - **Qwen 3.5**: sections=[11,11,10], interleaved=true (cycling T,H,W,T,H,W,...)
///
/// The sum of `sections` must equal the half-dimension of the rotary
/// embedding (i.e., `RoPEAttributes.dimension / 2`).
public struct MRoPEAxes: Codable, Equatable, Sendable {

    /// Dimension allocation per axis.
    ///
    /// Typically `[temporal, height, width]` with 3 elements.
    /// The sum determines the total rotary half-dimensions used.
    public let sections: [Int]

    /// Whether dimensions cycle across axes (interleaved) or are
    /// allocated in contiguous blocks.
    ///
    /// - `true`: dims cycle as [T,H,W,T,H,W,...] (Qwen 3.5 pattern)
    /// - `false`: dims are contiguous [T...T, H...H, W...W] (Qwen 2.5-VL pattern)
    public let interleaved: Bool

    /// Total number of half-dimensions used by M-RoPE.
    public var totalDimensions: Int { sections.reduce(0, +) }

    public init(sections: [Int], interleaved: Bool = false) {
        precondition(!sections.isEmpty, "sections must not be empty")
        precondition(sections.allSatisfy { $0 > 0 }, "all sections must be positive")
        self.sections = sections
        self.interleaved = interleaved
    }
}
