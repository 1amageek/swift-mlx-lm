import GGUFParser

public struct GGUFValidationIssue: Sendable, Equatable {
    public enum Severity: String, Sendable, Equatable {
        case error
        case warning
    }

    public enum Kind: String, Sendable, Equatable {
        case missingMetadata
        case invalidMetadata
        case unsupportedFamily
    }

    public let severity: Severity
    public let kind: Kind
    public let metadataKey: String
    public let message: String
    public let evidence: [String]
    public let suggestedValue: GGUFMetadataValue?

    public init(
        severity: Severity,
        kind: Kind,
        metadataKey: String,
        message: String,
        evidence: [String] = [],
        suggestedValue: GGUFMetadataValue? = nil
    ) {
        self.severity = severity
        self.kind = kind
        self.metadataKey = metadataKey
        self.message = message
        self.evidence = evidence
        self.suggestedValue = suggestedValue
    }
}
