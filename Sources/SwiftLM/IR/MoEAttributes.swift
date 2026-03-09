/// Attributes for a Mixture-of-Experts node.
///
/// Routes tokens to a subset of expert MLPs via a gating mechanism.
public struct MoEAttributes: Codable, Equatable, Sendable {

    /// Total number of experts.
    public let expertCount: Int

    /// Number of experts activated per token.
    public let expertsPerToken: Int

    /// Gating mechanism for expert selection.
    public let gateKind: MoEGateKind

    /// MLP attributes shared by all experts.
    public let expertMLP: MLPAttributes

    public init(
        expertCount: Int,
        expertsPerToken: Int,
        gateKind: MoEGateKind = .topK,
        expertMLP: MLPAttributes
    ) {
        self.expertCount = expertCount
        self.expertsPerToken = expertsPerToken
        self.gateKind = gateKind
        self.expertMLP = expertMLP
    }
}

/// Gating mechanism for expert selection.
public enum MoEGateKind: Codable, Equatable, Sendable {
    case topK
    case sigmoidTopK
    case custom(String)
}
