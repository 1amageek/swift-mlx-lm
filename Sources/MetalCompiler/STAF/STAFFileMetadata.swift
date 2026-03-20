import Foundation

public struct STAFFileMetadata: Sendable, Equatable {
    public static let empty = STAFFileMetadata(values: [:])

    public let values: [String: STAFMetadataValue]

    public init(values: [String: STAFMetadataValue]) {
        self.values = values
    }

    public subscript(key: String) -> STAFMetadataValue? {
        values[key]
    }

    public var isEmpty: Bool {
        values.isEmpty
    }

    public var count: Int {
        values.count
    }

    func containsAllValues(of expected: STAFFileMetadata) -> Bool {
        for (key, expectedValue) in expected.values {
            guard values[key] == expectedValue else {
                return false
            }
        }
        return true
    }

    func merged(with overriding: STAFFileMetadata) -> STAFFileMetadata {
        var mergedValues = values
        for (key, value) in overriding.values {
            mergedValues[key] = value
        }
        return STAFFileMetadata(values: mergedValues)
    }
}

extension STAFFileMetadata {
    static func defaultCacheMetadata(sourceShardCount: Int) -> STAFFileMetadata {
        STAFFileMetadata(values: [
            STAFMetadataKey.sourceFormat: .string("safetensors"),
            STAFMetadataKey.converterVersion: .uint32(STAF.currentConverterVersion),
            STAFMetadataKey.sourceShardCount: .uint64(UInt64(sourceShardCount)),
            STAFMetadataKey.metadataSchemaVersion: .uint32(STAF.currentMetadataSchemaVersion)
        ])
    }
}
