public struct GGUFValidationReport: Sendable, Equatable {
    public let architecture: String?
    public let family: String?
    public let issues: [GGUFValidationIssue]
    public let warnings: [String]

    public init(
        architecture: String?,
        family: String?,
        issues: [GGUFValidationIssue],
        warnings: [String] = []
    ) {
        self.architecture = architecture
        self.family = family
        self.issues = issues
        self.warnings = warnings
    }

    public var hasErrors: Bool {
        issues.contains { $0.severity == .error }
    }
}
