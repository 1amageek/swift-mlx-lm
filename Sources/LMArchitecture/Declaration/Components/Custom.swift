import JSONSchema

/// Custom operation component (escape hatch).
///
/// Allows representation of unsupported or experimental operations
/// in the DSL. Compiler guarantees, canonical equivalence, and
/// optimization support are intentionally limited for custom nodes.
///
/// ```swift
/// Custom(domain: "research", name: "sparse_attn")
/// ```
public struct Custom: ModelComponent {

    public typealias Body = Never

    public let domain: String
    public let name: String
    public let attributes: JSONValue

    public init(
        domain: String,
        name: String,
        attributes: JSONValue = .null
    ) {
        self.domain = domain
        self.name = name
        self.attributes = attributes
    }
}

extension Custom: PrimitiveComponent {

    package var operationKind: OperationKind {
        .primitive(CustomNodeAttributes(
            domain: domain,
            name: name,
            attributes: attributes
        ))
    }

    package var operationSignature: OperationSignature {
        OperationSignature(operandArity: .variadic, resultArity: .variadic)
    }
}
