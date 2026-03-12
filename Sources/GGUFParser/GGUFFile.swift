import Foundation

/// Parsed representation of a GGUF file.
///
/// Contains the file header, all metadata key-value pairs, and the tensor
/// directory. The actual tensor data is not loaded into memory — tensor
/// offsets point into the memory-mapped file data.
public struct GGUFFile: Sendable {

    /// GGUF magic number: "GGUF" in ASCII (little-endian) = 0x46554747.
    public static let magic: UInt32 = 0x4655_4747

    /// GGUF format version.
    public let version: UInt32

    /// All metadata key-value pairs.
    /// Large tokenizer arrays are deferred (not in this dictionary).
    public let metadata: [String: GGUFMetadataValue]

    /// Tensor descriptors (name, shape, type, offset).
    public let tensors: [GGUFTensorInfo]

    /// Byte offset where tensor data begins in the file.
    public let tensorDataOffset: Int

    /// Raw file data (memory-mapped).
    public let data: Data

    /// Byte offsets for large arrays that were skipped during parsing.
    /// Key is the metadata key, value describes how to read the array.
    public struct DeferredArray: Sendable {
        public let elementType: GGUFMetadataValueType
        public let count: Int
        /// Byte offset in `data` where the first element starts.
        public let offset: Int
    }

    /// Deferred arrays (large tokenizer data skipped during parse).
    public let deferredArrays: [String: DeferredArray]

    // MARK: - Parsing

    /// Parse a GGUF file from a URL.
    ///
    /// The file is memory-mapped for efficient access. Only the header,
    /// metadata, and tensor directory are parsed; tensor data remains on disk
    /// until explicitly accessed.
    public static func parse(url: URL) throws -> GGUFFile {
        let data = try Data(contentsOf: url, options: .mappedIfSafe)
        return try parse(data: data)
    }

    /// Keys whose arrays are large enough to justify deferred parsing.
    /// These are skipped during parse() and read on demand.
    private static let deferredKeys: Set<String> = [
        "tokenizer.ggml.tokens",
        "tokenizer.ggml.scores",
        "tokenizer.ggml.token_type",
        "tokenizer.ggml.merges",
    ]

    /// Parse a GGUF file from raw data.
    public static func parse(data: Data) throws -> GGUFFile {
        var reader = GGUFReader(data: data)

        // Header
        let fileMagic = try reader.readUInt32()
        guard fileMagic == GGUFFile.magic else {
            throw GGUFError.invalidMagic(fileMagic)
        }

        let version = try reader.readUInt32()
        guard version == 2 || version == 3 else {
            throw GGUFError.unsupportedVersion(version)
        }
        reader.version = version

        let tensorCount: Int
        let metadataKVCount: Int
        if version >= 3 {
            tensorCount = Int(try reader.readUInt64())
            metadataKVCount = Int(try reader.readUInt64())
        } else {
            tensorCount = Int(try reader.readUInt32())
            metadataKVCount = Int(try reader.readUInt32())
        }

        // Metadata — defer large tokenizer arrays
        var metadata: [String: GGUFMetadataValue] = [:]
        metadata.reserveCapacity(metadataKVCount)
        var deferred: [String: DeferredArray] = [:]

        for _ in 0..<metadataKVCount {
            let key = try reader.readString()

            if deferredKeys.contains(key) {
                // Read array header but skip element data
                let rawType = try reader.readUInt32()
                guard rawType == GGUFMetadataValueType.array.rawValue else {
                    throw GGUFError.invalidMetadataValueType(rawType)
                }
                let rawElementType = try reader.readUInt32()
                guard let elementType = GGUFMetadataValueType(rawValue: rawElementType) else {
                    throw GGUFError.invalidMetadataValueType(rawElementType)
                }
                let count: Int
                if version >= 3 {
                    count = Int(try reader.readUInt64())
                } else {
                    count = Int(try reader.readUInt32())
                }
                let elementOffset = reader.offset
                // Skip all elements without allocating
                for _ in 0..<count {
                    try reader.skipMetadataValue(type: elementType)
                }
                deferred[key] = DeferredArray(
                    elementType: elementType, count: count, offset: elementOffset)
            } else {
                let value = try reader.readMetadataValue()
                metadata[key] = value
            }
        }

        // Tensor directory
        var tensors: [GGUFTensorInfo] = []
        tensors.reserveCapacity(tensorCount)
        for _ in 0..<tensorCount {
            tensors.append(try reader.readTensorInfo())
        }

        // Alignment
        let alignment: Int
        if let alignValue = metadata["general.alignment"]?.intValue {
            alignment = alignValue
        } else {
            alignment = 32
        }

        // Tensor data starts after header+metadata+tensor_info, aligned
        let headerEnd = reader.offset
        let tensorDataOffset = alignUp(headerEnd, to: alignment)

        return GGUFFile(
            version: version,
            metadata: metadata,
            tensors: tensors,
            tensorDataOffset: tensorDataOffset,
            data: data,
            deferredArrays: deferred
        )
    }

    // MARK: - Tensor Data Access

    /// Get the raw bytes for a tensor.
    ///
    /// Returns a `Data` slice pointing into the memory-mapped file.
    public func tensorData(for tensor: GGUFTensorInfo) throws -> Data {
        guard tensor.offset <= UInt64(Int.max) else {
            throw GGUFError.unexpectedEndOfData(context: "tensor offset overflow for \(tensor.name)")
        }
        let start = tensorDataOffset + Int(tensor.offset)
        let end = start + tensor.dataSize
        guard start >= tensorDataOffset, end >= start, data.startIndex + end <= data.endIndex else {
            throw GGUFError.unexpectedEndOfData(context: "tensor data out of bounds for \(tensor.name)")
        }
        return data[data.startIndex + start ..< data.startIndex + end]
    }
}

// MARK: - Deferred Array Readers

extension GGUFFile {
    /// Materialize all metadata, including deferred tokenizer arrays.
    public func materializedMetadata() throws -> [String: GGUFMetadataValue] {
        var result = metadata
        result.reserveCapacity(metadata.count + deferredArrays.count)
        for key in deferredArrays.keys {
            result[key] = try readDeferredArrayValue(key)
        }
        return result
    }

    /// Read a deferred string array directly into [String].
    /// Single pass, no enum wrapping.
    public func readDeferredStringArray(_ key: String) -> [String]? {
        guard let da = deferredArrays[key], da.elementType == .string else { return nil }
        do {
            var reader = GGUFReader(data: data)
            reader.version = version
            reader.setOffset(da.offset)
            return try reader.readStringArrayDirect(count: da.count)
        } catch {
            return nil
        }
    }

    /// Read a deferred string array and build [String: Int] dictionary in one pass.
    /// Index = position in array.
    public func readDeferredStringDictionary(_ key: String) -> [String: Int]? {
        guard let da = deferredArrays[key], da.elementType == .string else { return nil }
        do {
            var reader = GGUFReader(data: data)
            reader.version = version
            reader.setOffset(da.offset)
            return try reader.readStringArrayAsDictionary(count: da.count)
        } catch {
            return nil
        }
    }

    /// Read a deferred float32 array directly into [Float].
    public func readDeferredFloat32Array(_ key: String) -> [Float]? {
        guard let da = deferredArrays[key], da.elementType == .float32 else { return nil }
        do {
            var reader = GGUFReader(data: data)
            reader.version = version
            reader.setOffset(da.offset)
            return try reader.readFloat32ArrayDirect(count: da.count)
        } catch {
            return nil
        }
    }

    /// Read a deferred int32 array directly into [Int].
    public func readDeferredInt32Array(_ key: String) -> [Int]? {
        guard let da = deferredArrays[key],
              da.elementType == .int32 || da.elementType == .uint32 else { return nil }
        do {
            var reader = GGUFReader(data: data)
            reader.version = version
            reader.setOffset(da.offset)
            return try reader.readInt32ArrayDirect(count: da.count)
        } catch {
            return nil
        }
    }

    /// Read a deferred string array and produce BOTH [String] and [String: Int]
    /// in a single pass (avoids reading the bytes twice).
    public func readDeferredVocabulary(_ key: String) -> (vocabulary: [String], tokenToID: [String: Int])? {
        guard let da = deferredArrays[key], da.elementType == .string else { return nil }
        var reader = GGUFReader(data: data)
        reader.version = version
        reader.setOffset(da.offset)
        var vocab: [String] = []
        var dict: [String: Int] = [:]
        vocab.reserveCapacity(da.count)
        dict.reserveCapacity(da.count)
        for i in 0..<da.count {
            guard let s = try? reader.readString() else { break }
            vocab.append(s)
            dict[s] = i
        }
        return (vocab, dict)
    }

    /// Count of elements in a deferred array, or nil if not deferred.
    public func deferredArrayCount(_ key: String) -> Int? {
        deferredArrays[key]?.count
    }

    private func readDeferredArrayValue(_ key: String) throws -> GGUFMetadataValue {
        guard let deferred = deferredArrays[key] else {
            throw GGUFError.unexpectedEndOfData(context: "deferred array metadata not found for \(key)")
        }
        var reader = GGUFReader(data: data)
        reader.version = version
        reader.setOffset(deferred.offset)
        var elements: [GGUFMetadataValue] = []
        elements.reserveCapacity(deferred.count)
        for _ in 0..<deferred.count {
            elements.append(try reader.readMetadataValue(type: deferred.elementType))
        }
        return .array(elements)
    }
}

private func alignUp(_ value: Int, to alignment: Int) -> Int {
    precondition(alignment > 0 && (alignment & (alignment - 1)) == 0, "alignment must be a power of 2")
    let mask = alignment - 1
    return (value + mask) & ~mask
}
