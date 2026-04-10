import Foundation
import Metal

/// Loads a STAF file via zero-copy mmap for direct GPU access.
///
/// The entire payload section is mapped as a single MTLBuffer using
/// `makeBuffer(bytesNoCopy:)`. Individual tensors are accessed via
/// offset within the buffer. No memcpy occurs at load time.
public struct STAFLoader: Sendable {

    public init() {}

    /// Load a STAF file into a `STAFWeightStore`.
    ///
    /// - Parameters:
    ///   - url: Path to the .staf file.
    ///   - device: Metal device for buffer creation.
    /// - Returns: A weight store with zero-copy GPU access to all tensors.
    public func load(at url: URL, device: MTLDevice) throws -> STAFWeightStore {
        // mmap the entire file
        let fileAttributes = try FileManager.default.attributesOfItem(atPath: url.path)
        guard let fileSize = fileAttributes[.size] as? Int, fileSize > STAF.headerSize else {
            throw STAFLoadError.invalidFile("File too small or missing")
        }

        let fileDescriptor = open(url.path, O_RDONLY)
        guard fileDescriptor >= 0 else {
            throw STAFLoadError.invalidFile("Cannot open: \(url.path)")
        }

        let mmapPointer = mmap(nil, fileSize, PROT_READ, MAP_PRIVATE, fileDescriptor, 0)
        close(fileDescriptor)

        guard let validPointer = mmapPointer, validPointer != MAP_FAILED else {
            throw STAFLoadError.mmapFailed("mmap failed for: \(url.path)")
        }

        let header = STAF.parseHeader(at: validPointer)
        guard header.magic == STAF.magic else {
            munmap(validPointer, fileSize)
            throw STAFLoadError.invalidFile("Bad magic: expected STAF")
        }
        guard header.formatVersion == STAF.legacyFormatVersion ||
                header.formatVersion == STAF.currentFormatVersion else {
            munmap(validPointer, fileSize)
            throw STAFLoadError.invalidFile("Unsupported STAF format version: \(header.formatVersion)")
        }

        let sectionCount = Int(header.sectionCount)
        let sectionTableOffset = Int(header.sectionTableOffset)
        let stringTableOffset = Int(header.stringTableOffset)
        let metadata: STAFFileMetadata
        do {
            metadata = try STAFMetadataDecoder().decode(
                at: validPointer,
                fileSize: fileSize,
                header: header
            )
        } catch let error as STAFMetadataDecodeError {
            munmap(validPointer, fileSize)
            throw STAFLoadError.invalidFile(error.description)
        }

        // Parse section table
        var tensorEntries: [String: STAFTensorEntry] = [:]
        var allEntries: [STAFTensorEntry] = []

        // Parse section entries manually (byte-offset based) to avoid
        // Swift struct alignment issues. The on-disk format is packed:
        //
        //   0: nameOffset (uint32)
        //   4: nameLength (uint32)
        //   8: quantSchemeId (uint8)
        //   9: semanticRole (uint8)
        //  10: origDtype (uint8)
        //  11: ndim (uint8)
        //  12: shape (uint32 × 8 = 32 bytes)
        //  44: payloadOffset (uint64)
        //  52: payloadSize (uint64)
        //  60: alignment (uint32)
        //  64: blockSize (uint32)
        //  68: groupSize (uint32)
        //  72: checksum (uint32)
        //  76: shardIndex (uint32)
        //  80: reserved (48 bytes)
        // total: 128 bytes per entry (packed, no alignment padding)
        let packedEntrySize = 128

        for i in 0..<sectionCount {
            let base = validPointer + sectionTableOffset + i * packedEntrySize

            let nameOffset = Int(base.loadUnaligned(as: UInt32.self))
            let nameLength = Int((base + 4).loadUnaligned(as: UInt32.self))
            let quantSchemeRaw = (base + 8).load(as: UInt8.self)
            let semanticRoleRaw = (base + 9).load(as: UInt8.self)
            let ndim = min(Int((base + 11).load(as: UInt8.self)), STAF.maximumDimensions)

            // Shape: 8 × uint32 at offset 12
            var shape: [Int] = []
            for d in 0..<ndim {
                let dimValue = (base + 12 + d * 4).loadUnaligned(as: UInt32.self)
                shape.append(Int(dimValue))
            }

            let rawPayloadOffset = (base + 44).loadUnaligned(as: UInt64.self)
            let rawPayloadSize = (base + 52).loadUnaligned(as: UInt64.self)
            guard rawPayloadOffset <= UInt64(Int.max), rawPayloadSize <= UInt64(Int.max),
                  rawPayloadOffset <= UInt64(fileSize), rawPayloadSize <= UInt64(fileSize) else {
                // Dump diagnostic info before throwing
                let entryBase = sectionTableOffset + i * 128
                let byteDump = (0..<128).map { String(format: "%02x", (validPointer + entryBase + $0).load(as: UInt8.self)) }
                    .chunked(into: 16).map { $0.joined(separator: " ") }.joined(separator: "\n  ")
                munmap(validPointer, fileSize)
                throw STAFLoadError.invalidFile(
                    """
                    Payload overflow in section \(i)/\(sectionCount) of \(url.path)
                    fileSize=\(fileSize), payloadOffset=\(rawPayloadOffset), payloadSize=\(rawPayloadSize)
                    nameOffset=\(nameOffset), nameLen=\(nameLength), ndim=\(ndim), scheme=\(quantSchemeRaw)
                    Entry bytes (hex):
                      \(byteDump)
                    """)
            }
            let payloadOffset = Int(rawPayloadOffset)
            let payloadSize = Int(rawPayloadSize)
            let blockSize = Int((base + 64).loadUnaligned(as: UInt32.self))
            let groupSize = Int((base + 68).loadUnaligned(as: UInt32.self))

            // Tensor name from string table
            let nameStart = stringTableOffset + nameOffset
            let namePointer = validPointer + nameStart
            let name = String(
                bytes: UnsafeBufferPointer(
                    start: namePointer.assumingMemoryBound(to: UInt8.self),
                    count: nameLength),
                encoding: .utf8) ?? ""

            let schemeIdentifier = QuantizationSchemeIdentifier(rawValue: quantSchemeRaw)
                ?? .passthrough

            let tensorEntry = STAFTensorEntry(
                name: name,
                payloadOffset: payloadOffset,
                payloadSize: payloadSize,
                schemeIdentifier: schemeIdentifier,
                semanticRole: SemanticRole(rawValue: semanticRoleRaw) ?? .unknown,
                shape: shape,
                blockSize: blockSize,
                groupSize: groupSize
            )

            tensorEntries[name] = tensorEntry
            allEntries.append(tensorEntry)
        }

        // Find payload region bounds (first tensor offset to end of file)
        guard let firstPayloadOffset = allEntries.map(\.payloadOffset).min() else {
            munmap(validPointer, fileSize)
            throw STAFLoadError.invalidFile("No tensors in STAF")
        }

        let payloadPointer = validPointer + firstPayloadOffset
        let payloadSize = fileSize - firstPayloadOffset

        // Verify 4KB alignment for bytesNoCopy
        let pageSize = Int(sysconf(_SC_PAGESIZE))
        let payloadAlignment = Int(bitPattern: payloadPointer) % pageSize
        let alignedPointer = payloadPointer - payloadAlignment
        let alignedSize = alignUp(payloadSize + payloadAlignment, to: pageSize)

        // Create zero-copy MTLBuffer from mmap'd payload
        let mmapSizeCapture = fileSize
        guard let metalBuffer = device.makeBuffer(
            bytesNoCopy: alignedPointer,
            length: alignedSize,
            options: .storageModeShared,
            deallocator: { pointer, _ in
                munmap(pointer, mmapSizeCapture)
            }
        ) else {
            munmap(validPointer, fileSize)
            throw STAFLoadError.bufferCreationFailed("MTLBuffer bytesNoCopy failed")
        }

        // Adjust tensor offsets by payload alignment
        var adjustedEntries: [String: STAFTensorEntry] = [:]
        for (name, entry) in tensorEntries {
            var adjusted = entry
            adjusted.bufferOffset = entry.payloadOffset - firstPayloadOffset + payloadAlignment
            adjustedEntries[name] = adjusted
        }

        return STAFWeightStore(
            buffer: metalBuffer,
            entries: adjustedEntries,
            metadata: metadata,
            specializedBufferAccesses: [:]
        )
    }

    private func alignUp(_ value: Int, to alignment: Int) -> Int {
        let remainder = value % alignment
        return remainder == 0 ? value : value + (alignment - remainder)
    }

}

// MARK: - STAF Weight Store

/// GPU-ready weight store backed by a single zero-copy MTLBuffer.
///
/// All tensors are accessible via offset within the buffer.
/// The runtime uses `schemeIdentifier` to select the correct GEMV kernel.
public struct STAFWeightStore: @unchecked Sendable {
    /// Single MTLBuffer containing all tensor payloads (zero-copy mmap).
    public let buffer: MTLBuffer
    /// Tensor metadata indexed by name.
    public let entries: [String: STAFTensorEntry]
    /// File-level typed metadata describing the execution cache.
    public let metadata: STAFFileMetadata
    /// Optional specialized GPU-ready buffer accesses keyed by tensor name and layout.
    let specializedBufferAccesses: [STAFSpecializedWeightKey: STAFWeightBufferAccess]

    var residencyCandidateBuffers: [MTLBuffer] {
        var seen = Set<ObjectIdentifier>()
        var buffers: [MTLBuffer] = []
        for buffer in [self.buffer] + specializedBufferAccesses.values.map(\.buffer) {
            let identifier = ObjectIdentifier(buffer as AnyObject)
            guard seen.insert(identifier).inserted else { continue }
            buffers.append(buffer)
        }
        return buffers
    }

    /// Get the buffer offset and quantization format for a named tensor.
    public func tensor(for name: String) -> (offset: Int, format: any QuantizationFormat)? {
        guard let entry = entries[name] else { return nil }
        guard let format = QuantizationFormatRegistry.format(for: entry.schemeIdentifier) else {
            return nil
        }
        return (offset: entry.bufferOffset, format: format)
    }

    /// Get raw buffer access for a named tensor and exact layout.
    public func bufferAccess(
        for name: String,
        layout: STAFWeightLayout = .rowMajor
    ) -> STAFWeightBufferAccess? {
        guard let entry = entries[name] else { return nil }
        switch layout {
        case .rowMajor:
            return STAFWeightBufferAccess(
                buffer: buffer,
                offset: entry.bufferOffset,
                size: entry.payloadSize,
                layout: .rowMajor
            )
        case .blockedRows4Tiles128:
            return specializedBufferAccesses[STAFSpecializedWeightKey(
                tensorName: name,
                layout: .blockedRows4Tiles128
            )]
        case .blockedRows8Tiles128:
            return specializedBufferAccesses[STAFSpecializedWeightKey(
                tensorName: name,
                layout: .blockedRows8Tiles128
            )]
        }
    }

    /// Resolve the best available access for a tensor, falling back to row-major storage.
    public func resolvedBufferAccess(
        for request: STAFWeightAccessRequest
    ) -> STAFWeightBufferAccess? {
        if let preferred = bufferAccess(
            for: request.tensorName,
            layout: request.preferredLayout
        ) {
            return preferred
        }
        return bufferAccess(for: request.tensorName, layout: .rowMajor)
    }

    public func registeringSpecializedBufferAccess(
        _ access: STAFWeightBufferAccess,
        for request: STAFWeightAccessRequest
    ) -> STAFWeightStore {
        var updated = specializedBufferAccesses
        updated[STAFSpecializedWeightKey(
            tensorName: request.tensorName,
            layout: request.preferredLayout
        )] = access
        return STAFWeightStore(
            buffer: buffer,
            entries: entries,
            metadata: metadata,
            specializedBufferAccesses: updated
        )
    }
}

// MARK: - Tensor Entry

/// Metadata for a single tensor in the loaded STAF weight store.
public struct STAFTensorEntry: Sendable {
    public let name: String
    public let payloadOffset: Int
    public let payloadSize: Int
    public let schemeIdentifier: QuantizationSchemeIdentifier
    public let semanticRole: SemanticRole
    public let shape: [Int]
    public let blockSize: Int
    public let groupSize: Int
    /// Adjusted offset within the MTLBuffer (set during loading).
    public var bufferOffset: Int = 0
}

// MARK: - Errors

public enum STAFLoadError: Error, CustomStringConvertible {
    case invalidFile(String)
    case mmapFailed(String)
    case bufferCreationFailed(String)

    public var description: String {
        switch self {
        case .invalidFile(let message): return "STAFLoadError.invalidFile: \(message)"
        case .mmapFailed(let message): return "STAFLoadError.mmapFailed: \(message)"
        case .bufferCreationFailed(let message): return "STAFLoadError.bufferCreationFailed: \(message)"
        }
    }
}

// MARK: - Array Chunking Helper

private extension Array {
    func chunked(into size: Int) -> [[Element]] {
        stride(from: 0, to: count, by: size).map {
            Array(self[$0..<Swift.min($0 + size, count)])
        }
    }
}
