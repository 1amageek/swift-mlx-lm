import GGUFParser

public struct GGUFMetadataPatch: Sendable, Equatable {
    public let values: [String: GGUFMetadataValue]

    public init(values: [String: GGUFMetadataValue]) {
        self.values = values
    }

    public init(actions: [GGUFRepairAction]) {
        var values: [String: GGUFMetadataValue] = [:]
        values.reserveCapacity(actions.count)
        for action in actions {
            switch action {
            case .addMetadata(let key, let value, _):
                values[key] = value
            case .replaceMetadata(let key, _, let newValue, _):
                values[key] = newValue
            }
        }
        self.values = values
    }

    public func applying(to metadata: [String: GGUFMetadataValue]) -> [String: GGUFMetadataValue] {
        metadata.merging(values) { _, newValue in newValue }
    }
}
