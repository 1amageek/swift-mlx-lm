/// Attributes for a learned per-layer scalar multiply.
///
/// This semantic node scales the hidden state by a learned scalar after the
/// layer body has completed. Gemma 4 uses this at the end of each text layer.
public struct LayerScaleAttributes: OperationAttributes, Codable, Equatable {

    /// Hidden dimension carried through unchanged.
    public let dimension: Int

    public init(dimension: Int) {
        self.dimension = dimension
    }
}
