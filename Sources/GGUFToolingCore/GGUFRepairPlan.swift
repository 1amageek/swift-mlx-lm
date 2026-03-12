import Foundation

public struct GGUFRepairPlan: Sendable, Equatable {
    public let sourceURL: URL?
    public let architecture: String?
    public let actions: [GGUFRepairAction]
    public let warnings: [String]

    public init(
        sourceURL: URL?,
        architecture: String?,
        actions: [GGUFRepairAction],
        warnings: [String] = []
    ) {
        self.sourceURL = sourceURL
        self.architecture = architecture
        self.actions = actions
        self.warnings = warnings
    }
}
