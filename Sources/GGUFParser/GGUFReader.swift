import Foundation

/// Low-level binary reader for GGUF files.
///
/// Reads primitive types and GGUF structures from a `Data` buffer.
/// Maintains a cursor position that advances as values are read.
struct GGUFReader {
    private let data: Data
    private(set) var offset: Int
    /// GGUF version (2 or 3). Affects string length and array count encoding.
    var version: UInt32 = 3

    init(data: Data) {
        self.data = data
        self.offset = 0
    }

    var remainingBytes: Int { data.count - offset }

    // MARK: - Primitives

    mutating func readUInt8() throws -> UInt8 {
        try ensureAvailable(1, context: "UInt8")
        let value = data[data.startIndex + offset]
        offset += 1
        return value
    }

    mutating func readInt8() throws -> Int8 {
        Int8(bitPattern: try readUInt8())
    }

    mutating func readUInt16() throws -> UInt16 {
        try readLittleEndian(UInt16.self)
    }

    mutating func readInt16() throws -> Int16 {
        Int16(bitPattern: try readUInt16())
    }

    mutating func readUInt32() throws -> UInt32 {
        try readLittleEndian(UInt32.self)
    }

    mutating func readInt32() throws -> Int32 {
        Int32(bitPattern: try readUInt32())
    }

    mutating func readUInt64() throws -> UInt64 {
        try readLittleEndian(UInt64.self)
    }

    mutating func readInt64() throws -> Int64 {
        Int64(bitPattern: try readUInt64())
    }

    mutating func readFloat32() throws -> Float {
        Float(bitPattern: try readUInt32())
    }

    mutating func readFloat64() throws -> Double {
        Double(bitPattern: try readUInt64())
    }

    mutating func readBool() throws -> Bool {
        try readUInt8() != 0
    }

    mutating func readString() throws -> String {
        let length: Int
        if version >= 3 {
            length = Int(try readUInt64())
        } else {
            length = Int(try readUInt32())
        }
        try ensureAvailable(length, context: "string of length \(length)")
        let start = data.startIndex + offset
        let end = start + length
        offset += length
        guard let string = String(data: data[start..<end], encoding: .utf8) else {
            throw GGUFError.unexpectedEndOfData(context: "invalid UTF-8 string")
        }
        return string
    }

    // MARK: - GGUF Structures

    mutating func readMetadataValue() throws -> GGUFMetadataValue {
        let rawType = try readUInt32()
        guard let valueType = GGUFMetadataValueType(rawValue: rawType) else {
            throw GGUFError.invalidMetadataValueType(rawType)
        }
        return try readMetadataValue(type: valueType)
    }

    mutating func readMetadataValue(type: GGUFMetadataValueType) throws -> GGUFMetadataValue {
        switch type {
        case .uint8: return .uint8(try readUInt8())
        case .int8: return .int8(try readInt8())
        case .uint16: return .uint16(try readUInt16())
        case .int16: return .int16(try readInt16())
        case .uint32: return .uint32(try readUInt32())
        case .int32: return .int32(try readInt32())
        case .float32: return .float32(try readFloat32())
        case .bool: return .bool(try readBool())
        case .string: return .string(try readString())
        case .uint64: return .uint64(try readUInt64())
        case .int64: return .int64(try readInt64())
        case .float64: return .float64(try readFloat64())
        case .array:
            let rawElementType = try readUInt32()
            guard let elementType = GGUFMetadataValueType(rawValue: rawElementType) else {
                throw GGUFError.invalidMetadataValueType(rawElementType)
            }
            let count: Int
            if version >= 3 {
                count = Int(try readUInt64())
            } else {
                count = Int(try readUInt32())
            }
            var elements: [GGUFMetadataValue] = []
            elements.reserveCapacity(count)
            for _ in 0..<count {
                elements.append(try readMetadataValue(type: elementType))
            }
            return .array(elements)
        }
    }

    mutating func readTensorInfo() throws -> GGUFTensorInfo {
        let name = try readString()
        let nDimensions = Int(try readUInt32())
        var dimensions: [Int] = []
        dimensions.reserveCapacity(nDimensions)
        for _ in 0..<nDimensions {
            dimensions.append(Int(try readUInt64()))
        }
        let rawType = try readUInt32()
        guard let quantizationType = GGUFQuantizationType(rawValue: rawType) else {
            throw GGUFError.invalidQuantizationType(rawType)
        }
        let dataOffset = try readUInt64()
        return GGUFTensorInfo(
            name: name,
            dimensions: dimensions,
            quantizationType: quantizationType,
            offset: dataOffset
        )
    }

    // MARK: - Helpers

    private mutating func readLittleEndian<T: FixedWidthInteger>(_ type: T.Type) throws -> T {
        let size = MemoryLayout<T>.size
        try ensureAvailable(size, context: "\(T.self)")
        let start = data.startIndex + offset
        var value: T = 0
        _ = withUnsafeMutableBytes(of: &value) { dest in
            data.copyBytes(to: dest, from: start..<(start + size))
        }
        offset += size
        return T(littleEndian: value)
    }

    private func ensureAvailable(_ count: Int, context: String) throws {
        guard remainingBytes >= count else {
            throw GGUFError.unexpectedEndOfData(context: context)
        }
    }
}
