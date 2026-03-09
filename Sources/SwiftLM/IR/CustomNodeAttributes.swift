import JSONSchema

/// Attributes for a custom (escape-hatch) node.
///
/// Allows representation of unsupported or experimental operations
/// in the ModelGraph. Compiler guarantees, canonical equivalence,
/// and optimization support are intentionally limited for custom nodes.
public struct CustomNodeAttributes: Codable, Equatable, Sendable {

    /// Domain namespace for the custom operation (e.g., "research.lab").
    public let domain: String

    /// Operation name within the domain.
    public let name: String

    /// Arbitrary attributes stored as typed JSON values.
    public let attributes: JSONValue

    public init(domain: String, name: String, attributes: JSONValue = .null) {
        self.domain = domain
        self.name = name
        self.attributes = attributes
    }
}
