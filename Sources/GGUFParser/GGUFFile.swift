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
    public let metadata: [String: GGUFMetadataValue]

    /// Tensor descriptors (name, shape, type, offset).
    public let tensors: [GGUFTensorInfo]

    /// Byte offset where tensor data begins in the file.
    public let tensorDataOffset: Int

    /// Raw file data (memory-mapped).
    public let data: Data

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

        // Metadata
        var metadata: [String: GGUFMetadataValue] = [:]
        metadata.reserveCapacity(metadataKVCount)
        for _ in 0..<metadataKVCount {
            let key = try reader.readString()
            let value = try reader.readMetadataValue()
            metadata[key] = value
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
            data: data
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

private func alignUp(_ value: Int, to alignment: Int) -> Int {
    precondition(alignment > 0 && (alignment & (alignment - 1)) == 0, "alignment must be a power of 2")
    let mask = alignment - 1
    return (value + mask) & ~mask
}
