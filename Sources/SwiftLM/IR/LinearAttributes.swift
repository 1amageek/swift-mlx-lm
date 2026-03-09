/// Attributes for a linear projection node.
///
/// Available as a semantic primitive for cases where higher-level nodes
/// (attention, MLP) are not applicable. Canonicalization may collapse
/// linear operations into higher-level semantic nodes when legal.
public struct LinearAttributes: Codable, Equatable, Sendable {

    /// Input dimension.
    public let inputSize: Int

    /// Output dimension.
    public let outputSize: Int

    /// Whether a bias term is included.
    public let bias: Bool

    public init(inputSize: Int, outputSize: Int, bias: Bool = false) {
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.bias = bias
    }
}
