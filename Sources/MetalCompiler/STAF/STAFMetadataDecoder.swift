import Foundation

enum STAFMetadataDecodeError: Error, CustomStringConvertible {
    case invalidFile(String)

    var description: String {
        switch self {
        case .invalidFile(let message):
            return "STAFMetadataDecodeError.invalidFile: \(message)"
        }
    }
}

struct STAFMetadataDecoder {
    func decode(
        from fileData: Data,
        header: STAFHeader
    ) throws -> STAFFileMetadata {
        try fileData.withUnsafeBytes { rawBuffer in
            guard let baseAddress = rawBuffer.baseAddress else {
                throw STAFMetadataDecodeError.invalidFile("STAF data buffer is empty")
            }
            return try decode(
                at: baseAddress,
                fileSize: fileData.count,
                header: header
            )
        }
    }

    func decode(
        at filePointer: UnsafeRawPointer,
        fileSize: Int,
        header: STAFHeader
    ) throws -> STAFFileMetadata {
        guard header.supportsMetadataTable else {
            return .empty
        }

        let stringTableOffset = Int(header.stringTableOffset)
        let stringTableSize = Int(header.stringTableSize)
        let metadataTableOffset = Int(header.metadataTableOffset)
        let metadataEntryCount = Int(header.metadataEntryCount)
        let metadataTableSize = metadataEntryCount * STAF.metadataEntrySize

        guard metadataTableOffset >= STAF.headerSize,
              metadataTableOffset + metadataTableSize <= fileSize else {
            throw STAFMetadataDecodeError.invalidFile("Metadata table is out of bounds")
        }

        guard stringTableOffset >= 0,
              stringTableSize >= 0,
              stringTableOffset + stringTableSize <= fileSize else {
            throw STAFMetadataDecodeError.invalidFile("String table is out of bounds")
        }

        var values: [String: STAFMetadataValue] = [:]
        for index in 0..<metadataEntryCount {
            let base = filePointer + metadataTableOffset + index * STAF.metadataEntrySize
            let keyOffset = Int(base.loadUnaligned(as: UInt32.self))
            let keyLength = Int((base + 4).loadUnaligned(as: UInt32.self))
            let valueTypeRaw = (base + 8).load(as: UInt8.self)
            let payload0 = (base + 12).loadUnaligned(as: UInt64.self)
            let payload1 = (base + 20).loadUnaligned(as: UInt64.self)

            guard let valueType = STAFMetadataValueType(rawValue: valueTypeRaw) else {
                throw STAFMetadataDecodeError.invalidFile("Unknown metadata value type: \(valueTypeRaw)")
            }

            let key = try decodeString(
                from: filePointer,
                offset: keyOffset,
                length: keyLength,
                stringTableOffset: stringTableOffset,
                stringTableSize: stringTableSize
            )
            let value = try decodeValue(
                type: valueType,
                payload0: payload0,
                payload1: payload1,
                filePointer: filePointer,
                stringTableOffset: stringTableOffset,
                stringTableSize: stringTableSize
            )
            values[key] = value
        }

        return STAFFileMetadata(values: values)
    }

    private func decodeString(
        from filePointer: UnsafeRawPointer,
        offset: Int,
        length: Int,
        stringTableOffset: Int,
        stringTableSize: Int
    ) throws -> String {
        guard offset >= 0, length >= 0, offset + length <= stringTableSize else {
            throw STAFMetadataDecodeError.invalidFile("Metadata string is out of bounds")
        }
        let stringPointer = filePointer + stringTableOffset + offset
        let bytes = UnsafeBufferPointer(
            start: stringPointer.assumingMemoryBound(to: UInt8.self),
            count: length
        )
        guard let string = String(bytes: bytes, encoding: .utf8) else {
            throw STAFMetadataDecodeError.invalidFile("Metadata string is not valid UTF-8")
        }
        return string
    }

    private func decodeValue(
        type: STAFMetadataValueType,
        payload0: UInt64,
        payload1: UInt64,
        filePointer: UnsafeRawPointer,
        stringTableOffset: Int,
        stringTableSize: Int
    ) throws -> STAFMetadataValue {
        switch type {
        case .bool:
            return .bool(payload0 != 0)
        case .uint32:
            return .uint32(UInt32(truncatingIfNeeded: payload0))
        case .uint64:
            return .uint64(payload0)
        case .float32:
            return .float32(Float(bitPattern: UInt32(truncatingIfNeeded: payload0)))
        case .float64:
            return .float64(Double(bitPattern: payload0))
        case .string:
            guard payload0 <= UInt64(Int.max),
                  payload1 <= UInt64(Int.max) else {
                throw STAFMetadataDecodeError.invalidFile("String metadata payload overflow")
            }
            let valueOffset = Int(payload0)
            let valueLength = Int(payload1)
            let string = try decodeString(
                from: filePointer,
                offset: valueOffset,
                length: valueLength,
                stringTableOffset: stringTableOffset,
                stringTableSize: stringTableSize
            )
            return .string(string)
        }
    }
}
