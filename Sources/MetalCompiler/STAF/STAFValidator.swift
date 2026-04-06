import Foundation

struct STAFValidator: Sendable {

    func isValid(
        stafURL: URL,
        safetensorsURLs: [URL],
        expectedMetadata: STAFFileMetadata?
    ) throws -> Bool {
        let fileData = try Data(contentsOf: stafURL, options: [.mappedIfSafe])
        guard fileData.count >= STAF.headerSize else {
            return false
        }

        guard let header = STAF.parseHeader(from: fileData),
              header.magic == STAF.magic,
              header.sectionCount > 0 else {
            return false
        }
        guard header.formatVersion == STAF.currentFormatVersion else {
            return false
        }
        guard header.sectionTableOffset == UInt32(STAF.headerSize) else {
            return false
        }

        let stafFileSize = UInt64(fileData.count)
        if header.supportsMetadataTable {
            let metadataTableOffset = Int(header.metadataTableOffset)
            let metadataEntryCount = Int(header.metadataEntryCount)
            let minimumMetadataOffset = STAF.headerSize + Int(header.sectionCount) * STAF.sectionEntrySize
            if metadataTableOffset < minimumMetadataOffset ||
                metadataTableOffset + metadataEntryCount * STAF.metadataEntrySize > Int(stafFileSize) {
                return false
            }
        }

        if stafFileSize > UInt64(STAF.headerSize + STAF.sectionEntrySize) {
            let payloadSize = fileData.withUnsafeBytes { rawBuffer -> UInt64 in
                guard let baseAddress = rawBuffer.baseAddress else {
                    return UInt64.max
                }
                return (baseAddress + STAF.headerSize + 52).loadUnaligned(as: UInt64.self)
            }
            if payloadSize > stafFileSize {
                return false
            }
        }

        if let expectedMetadata {
            let actualMetadata: STAFFileMetadata
            do {
                actualMetadata = try STAFMetadataDecoder().decode(from: fileData, header: header)
            } catch {
                return false
            }
            guard actualMetadata.containsAllValues(of: expectedMetadata) else {
                return false
            }
        }

        let stafAttributes = try FileManager.default.attributesOfItem(atPath: stafURL.path)
        guard let stafModificationDate = stafAttributes[.modificationDate] as? Date else {
            return false
        }

        for url in safetensorsURLs {
            let resolvedURL = url.resolvingSymlinksInPath()
            let attributes = try FileManager.default.attributesOfItem(atPath: resolvedURL.path)
            if let sourceModificationDate = attributes[.modificationDate] as? Date,
               sourceModificationDate > stafModificationDate {
                return false
            }
        }

        return true
    }
}
