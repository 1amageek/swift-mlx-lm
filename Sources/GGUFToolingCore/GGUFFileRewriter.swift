import Foundation
import GGUFParser

public struct GGUFFileRewriter: Sendable {
    public init() {}

    public func applying(_ patch: GGUFMetadataPatch, to fileURL: URL, outputURL: URL) throws {
        guard fileURL.standardizedFileURL != outputURL.standardizedFileURL else {
            throw GGUFToolingError.inPlaceRewriteNotAllowed
        }

        let file = try GGUFFile.parse(url: fileURL)
        guard file.version == 2 || file.version == 3 else {
            throw GGUFToolingError.unsupportedVersion(file.version)
        }

        let updatedMetadata = patch.applying(to: try file.materializedMetadata())
        let encoded = try encodeFile(
            version: file.version,
            metadata: updatedMetadata,
            tensors: file.tensors,
            tensorPayload: try tensorPayload(in: file)
        )
        try encoded.write(to: outputURL, options: .atomic)
    }

    private func tensorPayload(in file: GGUFFile) throws -> Data {
        let requiredBytes = file.tensors.reduce(0) { current, tensor in
            let end = Int(tensor.offset) + tensor.dataSize
            return max(current, end)
        }
        let start = file.tensorDataOffset
        let end = start + requiredBytes
        guard start >= 0, end >= start, end <= file.data.count else {
            throw GGUFToolingError.outputWouldTruncateTensorData
        }
        return file.data.subdata(in: start..<end)
    }

    private func encodeFile(
        version: UInt32,
        metadata: [String: GGUFMetadataValue],
        tensors: [GGUFTensorInfo],
        tensorPayload: Data
    ) throws -> Data {
        let alignment = metadata["general.alignment"]?.intValue ?? 32
        guard alignment > 0, alignment & (alignment - 1) == 0 else {
            throw GGUFToolingError.invalidAlignment(alignment)
        }

        let orderedMetadata = metadata.keys.sorted().map { ($0, metadata[$0]!) }
        var data = Data()
        append(UInt32(GGUFFile.magic), to: &data)
        append(version, to: &data)

        if version >= 3 {
            append(UInt64(tensors.count), to: &data)
            append(UInt64(orderedMetadata.count), to: &data)
        } else {
            append(UInt32(tensors.count), to: &data)
            append(UInt32(orderedMetadata.count), to: &data)
        }

        for (key, value) in orderedMetadata {
            try appendString(key, version: version, to: &data)
            try appendMetadataValue(value, key: key, version: version, to: &data)
        }

        for tensor in tensors {
            try appendTensorInfo(tensor, version: version, to: &data)
        }

        let paddingCount = alignUp(data.count, to: alignment) - data.count
        if paddingCount > 0 {
            data.append(Data(repeating: 0, count: paddingCount))
        }
        data.append(tensorPayload)
        return data
    }

    private func appendTensorInfo(_ tensor: GGUFTensorInfo, version: UInt32, to data: inout Data) throws {
        try appendString(tensor.name, version: version, to: &data)
        append(UInt32(tensor.dimensions.count), to: &data)
        for dimension in tensor.dimensions {
            append(UInt64(dimension), to: &data)
        }
        append(tensor.quantizationType.rawValue, to: &data)
        append(tensor.offset, to: &data)
    }

    private func appendMetadataValue(
        _ value: GGUFMetadataValue,
        key: String,
        version: UInt32,
        to data: inout Data
    ) throws {
        append(value.ggufType.rawValue, to: &data)
        switch value {
        case .uint8(let rawValue):
            append(rawValue, to: &data)
        case .int8(let rawValue):
            append(UInt8(bitPattern: rawValue), to: &data)
        case .uint16(let rawValue):
            append(rawValue, to: &data)
        case .int16(let rawValue):
            append(UInt16(bitPattern: rawValue), to: &data)
        case .uint32(let rawValue):
            append(rawValue, to: &data)
        case .int32(let rawValue):
            append(UInt32(bitPattern: rawValue), to: &data)
        case .float32(let rawValue):
            append(rawValue.bitPattern, to: &data)
        case .bool(let rawValue):
            append(rawValue ? UInt8(1) : UInt8(0), to: &data)
        case .string(let rawValue):
            try appendString(rawValue, version: version, to: &data)
        case .uint64(let rawValue):
            append(rawValue, to: &data)
        case .int64(let rawValue):
            append(UInt64(bitPattern: rawValue), to: &data)
        case .float64(let rawValue):
            append(rawValue.bitPattern, to: &data)
        case .array(let values):
            let elementType = try arrayElementType(for: values, key: key)
            append(elementType.rawValue, to: &data)
            if version >= 3 {
                append(UInt64(values.count), to: &data)
            } else {
                append(UInt32(values.count), to: &data)
            }
            for element in values {
                try appendArrayElement(element, expectedType: elementType, key: key, version: version, to: &data)
            }
        }
    }

    private func appendArrayElement(
        _ value: GGUFMetadataValue,
        expectedType: GGUFMetadataValueType,
        key: String,
        version: UInt32,
        to data: inout Data
    ) throws {
        guard value.ggufType == expectedType else {
            throw GGUFToolingError.mixedArrayTypes(key: key)
        }
        switch value {
        case .array:
            throw GGUFToolingError.mixedArrayTypes(key: key)
        default:
            break
        }

        switch value {
        case .uint8(let rawValue):
            append(rawValue, to: &data)
        case .int8(let rawValue):
            append(UInt8(bitPattern: rawValue), to: &data)
        case .uint16(let rawValue):
            append(rawValue, to: &data)
        case .int16(let rawValue):
            append(UInt16(bitPattern: rawValue), to: &data)
        case .uint32(let rawValue):
            append(rawValue, to: &data)
        case .int32(let rawValue):
            append(UInt32(bitPattern: rawValue), to: &data)
        case .float32(let rawValue):
            append(rawValue.bitPattern, to: &data)
        case .bool(let rawValue):
            append(rawValue ? UInt8(1) : UInt8(0), to: &data)
        case .string(let rawValue):
            try appendString(rawValue, version: version, to: &data)
        case .uint64(let rawValue):
            append(rawValue, to: &data)
        case .int64(let rawValue):
            append(UInt64(bitPattern: rawValue), to: &data)
        case .float64(let rawValue):
            append(rawValue.bitPattern, to: &data)
        case .array:
            throw GGUFToolingError.mixedArrayTypes(key: key)
        }
    }

    private func arrayElementType(for values: [GGUFMetadataValue], key: String) throws -> GGUFMetadataValueType {
        guard let first = values.first else {
            throw GGUFToolingError.unsupportedEmptyArray(key: key)
        }
        let type = first.ggufType
        guard values.allSatisfy({ $0.ggufType == type && $0.ggufType != .array }) else {
            throw GGUFToolingError.mixedArrayTypes(key: key)
        }
        return type
    }

    private func appendString(_ value: String, version: UInt32, to data: inout Data) throws {
        guard let bytes = value.data(using: .utf8) else {
            throw CocoaError(.fileWriteInapplicableStringEncoding)
        }
        if version >= 3 {
            append(UInt64(bytes.count), to: &data)
        } else {
            append(UInt32(bytes.count), to: &data)
        }
        data.append(bytes)
    }

    private func append<T: FixedWidthInteger>(_ value: T, to data: inout Data) {
        var littleEndian = value.littleEndian
        withUnsafeBytes(of: &littleEndian) { bytes in
            data.append(contentsOf: bytes)
        }
    }

    private func alignUp(_ value: Int, to alignment: Int) -> Int {
        let mask = alignment - 1
        return (value + mask) & ~mask
    }
}
