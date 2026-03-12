import GGUFParser

public enum GGUFRepairAction: Sendable, Equatable {
    case addMetadata(key: String, value: GGUFMetadataValue, rationale: String)
    case replaceMetadata(
        key: String,
        oldValue: GGUFMetadataValue,
        newValue: GGUFMetadataValue,
        rationale: String
    )

    public var key: String {
        switch self {
        case .addMetadata(let key, _, _):
            return key
        case .replaceMetadata(let key, _, _, _):
            return key
        }
    }
}
