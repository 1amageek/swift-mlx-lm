import Foundation

/// Converts safetensors files to STAF (SafeTensor Accelerated Format).
///
/// The converter performs a one-time offline transformation:
/// 1. Parse safetensors headers to discover tensors
/// 2. Repack quantized weights into interleaved block format
/// 3. Write STAF with 4KB-aligned payload for zero-copy GPU access
///
/// safetensors remain source of truth. STAF is a regenerable cache.
public struct STAFConverter: Sendable {

    public init() {}

    /// Convert one or more safetensors files into a single STAF file.
    ///
    /// - Parameters:
    ///   - safetensorsURLs: Paths to safetensors shards (sorted by filename).
    ///   - outputURL: Destination path for the .staf file.
    /// - Throws: If parsing fails or the output file cannot be written.
    public func convert(
        safetensorsURLs: [URL],
        outputURL: URL
    ) throws {
        let sortedURLs = safetensorsURLs.sorted { $0.lastPathComponent < $1.lastPathComponent }

        // Phase 1: Parse all safetensors headers and collect tensor info
        let loader = SafetensorsLoader()
        var allTensors: [(name: String, info: SafetensorsTensorInfo, shardIndex: Int, shardURL: URL)] = []

        for (shardIndex, url) in sortedURLs.enumerated() {
            let tensors = try loader.parseHeader(at: url)
            for tensor in tensors {
                allTensors.append((name: tensor.name, info: tensor, shardIndex: shardIndex, shardURL: url))
            }
        }

        // Build set of consumed companion tensors (scales/biases that are
        // embedded in their parent weight's interleaved block).
        var consumedCompanions = Set<String>()
        for (name, _, _, _) in allTensors where name.hasSuffix(".weight") {
            let modulePath = String(name.dropLast(".weight".count))
            let scalesName = modulePath + ".scales"
            let biasesName = modulePath + ".biases"
            if allTensors.contains(where: { $0.name == scalesName }),
               allTensors.contains(where: { $0.name == biasesName }) {
                consumedCompanions.insert(scalesName)
                consumedCompanions.insert(biasesName)
            }
        }

        // Phase 2: Determine quantization scheme for each tensor.
        // Skip .scales/.biases that are consumed by quantized weights.
        var tensorEntries: [TensorConversionEntry] = []
        for (name, info, shardIndex, shardURL) in allTensors {
            if consumedCompanions.contains(name) { continue }

            let scheme = determineScheme(name: name, info: info, allTensors: allTensors)
            let role = inferSemanticRole(name: name)
            let originalDType = mapOriginalDType(info.dtype)

            tensorEntries.append(TensorConversionEntry(
                name: name,
                info: info,
                shardIndex: shardIndex,
                shardURL: shardURL,
                schemeIdentifier: scheme,
                semanticRole: role,
                originalDType: originalDType
            ))
        }

        // Phase 3: Build STAF file
        try writeSTAF(
            entries: tensorEntries,
            sortedURLs: sortedURLs,
            outputURL: outputURL
        )
    }

    /// Check if an existing STAF cache is still valid for the given safetensors.
    ///
    /// Uses file metadata (mtime) instead of content hashing.
    /// STAF is a regenerable cache — no cryptographic integrity needed.
    public func isValid(stafURL: URL, safetensorsURLs: [URL]) throws -> Bool {
        // Check 1: STAF file must exist and have valid magic
        let fileHandle = try FileHandle(forReadingFrom: stafURL)
        defer { fileHandle.closeFile() }

        guard let headerData = try fileHandle.read(upToCount: STAF.headerSize),
              headerData.count == STAF.headerSize else {
            return false
        }

        let header = headerData.withUnsafeBytes { $0.load(as: STAFHeader.self) }
        guard header.magic == STAF.magic, header.sectionCount > 0 else {
            return false
        }

        // Check: section table offset must be exactly after the header.
        guard header.sectionTableOffset == UInt32(STAF.headerSize) else {
            return false
        }

        // Check: first section entry must have sane payload values.
        // Detects old-format STAF with different struct alignment.
        let stafFileAttributes = try FileManager.default.attributesOfItem(atPath: stafURL.path)
        let stafFileSize = stafFileAttributes[.size] as? UInt64 ?? 0
        if stafFileSize > UInt64(STAF.headerSize + STAF.sectionEntrySize) {
            let firstEntryData = try fileHandle.read(upToCount: STAF.sectionEntrySize)
            if let entryBytes = firstEntryData, entryBytes.count == STAF.sectionEntrySize {
                let payloadSize = entryBytes.withUnsafeBytes {
                    ($0.baseAddress! + 52).loadUnaligned(as: UInt64.self)
                }
                if payloadSize > stafFileSize {
                    return false  // payload size exceeds file size → corrupted
                }
            }
        }

        // Check 2: STAF must be newer than all safetensors files
        let stafAttributes = try FileManager.default.attributesOfItem(atPath: stafURL.path)
        guard let stafModificationDate = stafAttributes[.modificationDate] as? Date else {
            return false
        }

        for url in safetensorsURLs {
            let attributes = try FileManager.default.attributesOfItem(atPath: url.path)
            if let sourceModificationDate = attributes[.modificationDate] as? Date,
               sourceModificationDate > stafModificationDate {
                return false
            }
        }

        return true
    }

    // MARK: - Internal Types

    private struct TensorConversionEntry {
        let name: String
        let info: SafetensorsTensorInfo
        let shardIndex: Int
        let shardURL: URL
        let schemeIdentifier: QuantizationSchemeIdentifier
        let semanticRole: SemanticRole
        let originalDType: OriginalDType
    }


    // MARK: - Scheme Detection

    private func determineScheme(
        name: String,
        info: SafetensorsTensorInfo,
        allTensors: [(name: String, info: SafetensorsTensorInfo, shardIndex: Int, shardURL: URL)]
    ) -> QuantizationSchemeIdentifier {
        // Check for MLX quantized weight: companion .scales and .biases tensors
        if name.hasSuffix(".weight") {
            let modulePath = String(name.dropLast(".weight".count))
            let hasScales = allTensors.contains { $0.name == modulePath + ".scales" }
            let hasBiases = allTensors.contains { $0.name == modulePath + ".biases" }

            if hasScales && hasBiases {
                // MLX quantized — determine group size from scales shape
                if let scalesInfo = allTensors.first(where: { $0.name == modulePath + ".scales" })?.info {
                    let groupSize = estimateGroupSize(
                        weightShape: info.shape,
                        scalesShape: scalesInfo.shape,
                        bits: 4  // MLX default
                    )
                    switch groupSize {
                    case 64: return .q4Group64ScaleF16
                    case 128: return .q4Group128ScaleF16
                    default: return .q4Group64ScaleF16
                    }
                }
                return .q4Group64ScaleF16
            }
        }

        // Skip scales and biases tensors (consumed by their weight's block)
        if name.hasSuffix(".scales") || name.hasSuffix(".biases") {
            return .passthrough
        }

        // Dense tensor — preserve original dtype
        switch info.dtype {
        case .float16: return .fp16RowMajor
        case .bfloat16: return .bf16RowMajor
        case .float32: return .fp16RowMajor
        default: return .passthrough
        }
    }

    private func estimateGroupSize(
        weightShape: [Int], scalesShape: [Int], bits: Int
    ) -> Int {
        guard weightShape.count >= 2, scalesShape.count >= 2 else { return 64 }
        let packedDimension = weightShape[weightShape.count - 1]
        let numberOfGroups = scalesShape[scalesShape.count - 1]
        let elementsPerUInt32 = 32 / bits
        let inputDimension = packedDimension * elementsPerUInt32
        return numberOfGroups > 0 ? inputDimension / numberOfGroups : 64
    }

    // MARK: - Semantic Role

    private func inferSemanticRole(name: String) -> SemanticRole {
        if name.contains("embed_tokens") || name.contains("token_embd") {
            return .tokenEmbedding
        }
        if name.contains("q_proj") { return .attentionQuery }
        if name.contains("k_proj") { return .attentionKey }
        if name.contains("v_proj") { return .attentionValue }
        if name.contains("o_proj") || name.contains("out_proj") { return .attentionOutput }
        if name.contains("gate_proj") || name.contains(".w1.") { return .mlpGate }
        if name.contains("up_proj") || name.contains(".w3.") { return .mlpUp }
        if name.contains("down_proj") || name.contains(".w2.") { return .mlpDown }
        if name.contains("layernorm") || name.contains("norm") && name.hasSuffix(".weight") {
            return .normWeight
        }
        if name.contains("lm_head") { return .languageModelHead }
        if name.contains("experts") && name.contains("gate") { return .moeExpertGate }
        if name.contains("experts") && name.contains("up") { return .moeExpertUp }
        if name.contains("experts") && name.contains("down") { return .moeExpertDown }
        if name.contains("router") || name.contains("gate.weight") && name.contains("moe") {
            return .moeRouter
        }
        return .unknown
    }

    private func mapOriginalDType(_ dtype: SafetensorsDType) -> OriginalDType {
        switch dtype {
        case .float32: return .float32
        case .float16: return .float16
        case .bfloat16: return .bfloat16
        case .int32: return .int32
        case .int16: return .int16
        case .int8: return .int8
        default: return .unknown
        }
    }

    // MARK: - STAF File Writing

    private func writeSTAF(
        entries: [TensorConversionEntry],
        sortedURLs: [URL],
        outputURL: URL
    ) throws {
        // Calculate layout
        let sectionCount = entries.count
        let sectionTableOffset = STAF.headerSize
        let sectionTableSize = sectionCount * STAF.sectionEntrySize

        // Build string table
        var stringTableData = Data()
        var nameOffsets: [Int] = []
        for entry in entries {
            nameOffsets.append(stringTableData.count)
            stringTableData.append(contentsOf: entry.name.utf8)
            stringTableData.append(0)  // null terminator
        }

        let stringTableOffset = sectionTableOffset + sectionTableSize
        let metadataEnd = stringTableOffset + stringTableData.count

        // Payload starts at next 4KB boundary
        let payloadStart = alignUp(metadataEnd, to: STAF.payloadAlignment)

        // Calculate payload offsets for each tensor
        var payloadOffsets: [UInt64] = []
        var payloadSizes: [UInt64] = []
        var currentOffset = payloadStart

        for entry in entries {
            let tensorSize = computePayloadSize(entry: entry)
            let alignedOffset = alignUp(currentOffset, to: STAF.tensorAlignment)
            payloadOffsets.append(UInt64(alignedOffset))
            payloadSizes.append(UInt64(tensorSize))
            currentOffset = alignedOffset + tensorSize
        }

        // Build the file
        var fileData = Data()

        // Header — packed 64 bytes, manual byte layout.
        let packedHeaderSize = 64
        var headerData = Data(count: packedHeaderSize)
        headerData.withUnsafeMutableBytes { buf in
            let base = buf.baseAddress!
            base.storeBytes(of: STAF.magic, toByteOffset: 0, as: UInt32.self)
            // 4..39: reserved (already zero)
            base.storeBytes(of: UInt32(sectionCount), toByteOffset: 40, as: UInt32.self)
            base.storeBytes(of: UInt32(sectionTableOffset), toByteOffset: 44, as: UInt32.self)
            base.storeBytes(of: UInt32(stringTableOffset), toByteOffset: 48, as: UInt32.self)
            base.storeBytes(of: UInt32(stringTableData.count), toByteOffset: 52, as: UInt32.self)
            // 56..63: reserved (already zero)
        }
        fileData.append(headerData)

        // Section table — write packed 128-byte entries (no Swift alignment padding).
        // Manual byte layout to match the on-disk spec exactly.
        let packedEntrySize = 128
        for (index, entry) in entries.enumerated() {
            let format = QuantizationFormatRegistry.format(for: entry.schemeIdentifier)
            let shapeArray = entry.info.shape + Array(repeating: 0, count: STAF.maximumDimensions - min(entry.info.shape.count, STAF.maximumDimensions))

            var entryData = Data(count: packedEntrySize)
            entryData.withUnsafeMutableBytes { buf in
                let base = buf.baseAddress!
                base.storeBytes(of: UInt32(nameOffsets[index]), toByteOffset: 0, as: UInt32.self)
                base.storeBytes(of: UInt32(entry.name.utf8.count), toByteOffset: 4, as: UInt32.self)
                base.storeBytes(of: entry.schemeIdentifier.rawValue, toByteOffset: 8, as: UInt8.self)
                base.storeBytes(of: entry.semanticRole.rawValue, toByteOffset: 9, as: UInt8.self)
                base.storeBytes(of: entry.originalDType.rawValue, toByteOffset: 10, as: UInt8.self)
                base.storeBytes(of: UInt8(min(entry.info.shape.count, 8)), toByteOffset: 11, as: UInt8.self)
                for d in 0..<8 {
                    base.storeBytes(of: UInt32(shapeArray[d]), toByteOffset: 12 + d * 4, as: UInt32.self)
                }
                base.storeBytes(of: payloadOffsets[index], toByteOffset: 44, as: UInt64.self)
                base.storeBytes(of: payloadSizes[index], toByteOffset: 52, as: UInt64.self)
                base.storeBytes(of: UInt32(STAF.tensorAlignment), toByteOffset: 60, as: UInt32.self)
                base.storeBytes(of: UInt32(format?.weightsPerBlock ?? 1), toByteOffset: 64, as: UInt32.self)
                base.storeBytes(of: UInt32(format?.groupSize ?? 1), toByteOffset: 68, as: UInt32.self)
                base.storeBytes(of: UInt32(0), toByteOffset: 72, as: UInt32.self) // checksum
                base.storeBytes(of: UInt32(entry.shardIndex), toByteOffset: 76, as: UInt32.self)
                // 80..127: reserved (already zero)
            }
            fileData.append(entryData)
        }

        // String table
        fileData.append(stringTableData)

        // Padding to 4KB boundary
        let paddingNeeded = payloadStart - fileData.count
        if paddingNeeded > 0 {
            fileData.append(Data(count: paddingNeeded))
        }

        // Payload: convert and write each tensor
        for (index, entry) in entries.enumerated() {
            // Pad to tensor alignment
            let targetOffset = Int(payloadOffsets[index])
            if fileData.count < targetOffset {
                fileData.append(Data(count: targetOffset - fileData.count))
            }

            let payloadData = try convertTensorPayload(entry: entry, sortedURLs: sortedURLs)
            fileData.append(payloadData)
        }

        try fileData.write(to: outputURL)
    }

    private func computePayloadSize(entry: TensorConversionEntry) -> Int {
        switch entry.schemeIdentifier {
        case .fp16RowMajor, .passthrough:
            let elementCount = entry.info.shape.reduce(1, *)
            return elementCount * 2
        case .bf16RowMajor:
            let elementCount = entry.info.shape.reduce(1, *)
            return elementCount * 2
        case .fp32RowMajor:
            let elementCount = entry.info.shape.reduce(1, *)
            return elementCount * 4
        default:
            // Quantized: safetensors shape is the PACKED shape (uint32).
            // Must expand to logical weight count before computing block count.
            guard let format = QuantizationFormatRegistry.format(for: entry.schemeIdentifier) else {
                return entry.info.shape.reduce(1, *) * 2
            }
            let outputDimension = entry.info.shape[0]
            let packedDimension = entry.info.shape.count >= 2 ? entry.info.shape[1] : 1
            let elementsPerUInt32 = 32 / format.bits
            let inputDimension = packedDimension * elementsPerUInt32
            let blocksPerRow = inputDimension / format.groupSize
            let totalBlocks = outputDimension * blocksPerRow
            return totalBlocks * format.bytesPerBlock
        }
    }

    private func convertTensorPayload(
        entry: TensorConversionEntry,
        sortedURLs: [URL]
    ) throws -> Data {
        // Read raw tensor data from safetensors shard
        let shardURL = entry.shardURL
        let fileHandle = try FileHandle(forReadingFrom: shardURL)
        defer { fileHandle.closeFile() }

        // Parse header to get data section offset
        guard let sizeData = try fileHandle.read(upToCount: 8), sizeData.count == 8 else {
            throw STAFConversionError.readFailed(entry.name)
        }
        let headerSize = sizeData.withUnsafeBytes { Int(UInt64(littleEndian: $0.load(as: UInt64.self))) }
        let dataSectionOffset = 8 + headerSize

        // Seek to tensor data
        try fileHandle.seek(toOffset: UInt64(dataSectionOffset + entry.info.dataOffset))
        guard let tensorData = try fileHandle.read(upToCount: entry.info.byteCount),
              tensorData.count == entry.info.byteCount else {
            throw STAFConversionError.readFailed(entry.name)
        }

        switch entry.schemeIdentifier {
        case .fp16RowMajor, .passthrough:
            if entry.info.dtype == .float32 {
                return convertFloat32ToFloat16(tensorData)
            }
            return tensorData

        case .q4Group64ScaleF16, .q4Group128ScaleF16:
            return try repackMLXQuantized(
                entry: entry, weightData: tensorData,
                sortedURLs: sortedURLs)

        default:
            return tensorData
        }
    }

    // MARK: - Repacking

    /// Repack MLX quantized format (3 separate tensors) into interleaved blocks.
    private func repackMLXQuantized(
        entry: TensorConversionEntry,
        weightData: Data,
        sortedURLs: [URL]
    ) throws -> Data {
        let modulePath = String(entry.name.dropLast(".weight".count))
        let scalesName = modulePath + ".scales"
        let biasesName = modulePath + ".biases"

        // Load scales and biases from safetensors
        let scalesData = try loadTensorFromSafetensors(name: scalesName, shardURL: entry.shardURL)
        let biasesData = try loadTensorFromSafetensors(name: biasesName, shardURL: entry.shardURL)

        guard let format = QuantizationFormatRegistry.format(for: entry.schemeIdentifier) else {
            throw STAFConversionError.unsupportedFormat(entry.schemeIdentifier.rawValue)
        }

        let outputDimension = entry.info.shape[0]
        let packedDimension = entry.info.shape.count >= 2 ? entry.info.shape[1] : 1
        let elementsPerUInt32 = 32 / format.bits
        let inputDimension = packedDimension * elementsPerUInt32
        let blocksPerRow = inputDimension / format.groupSize

        let totalBlocks = outputDimension * blocksPerRow
        var output = Data(count: totalBlocks * format.bytesPerBlock)

        weightData.withUnsafeBytes { weightBuffer in
            scalesData.withUnsafeBytes { scalesBuffer in
                biasesData.withUnsafeBytes { biasesBuffer in
                    output.withUnsafeMutableBytes { outputBuffer in
                        let weights = weightBuffer.bindMemory(to: UInt32.self)
                        let scales = scalesBuffer.bindMemory(to: Float16.self)
                        let biases = biasesBuffer.bindMemory(to: Float16.self)

                        for row in 0..<outputDimension {
                            for block in 0..<blocksPerRow {
                                let blockOffset = (row * blocksPerRow + block) * format.bytesPerBlock
                                let destination = outputBuffer.baseAddress! + blockOffset

                                // Write scale (2 bytes)
                                var scale = scales[row * blocksPerRow + block]
                                memcpy(destination, &scale, 2)

                                // Write zero (2 bytes)
                                var zero = biases[row * blocksPerRow + block]
                                memcpy(destination + 2, &zero, 2)

                                // Write packed quants
                                // MLX packs 8 × 4-bit values per uint32
                                // STAF packs 2 × 4-bit values per uint8 (low nibble first)
                                let qDestination = destination + 4
                                let weightsPerGroup = format.groupSize
                                let uint32PerGroup = weightsPerGroup / elementsPerUInt32

                                for u in 0..<uint32PerGroup {
                                    let packedIndex = row * packedDimension + block * uint32PerGroup + u
                                    let packed = weights[packedIndex]

                                    // Unpack uint32 → 8 nibbles → repack as uint8 pairs
                                    for nibblePair in 0..<4 {
                                        let lowNibble = UInt8((packed >> (nibblePair * 8)) & 0xF)
                                        let highNibble = UInt8((packed >> (nibblePair * 8 + 4)) & 0xF)
                                        let byte = lowNibble | (highNibble << 4)
                                        qDestination.storeBytes(
                                            of: byte,
                                            toByteOffset: u * 4 + nibblePair,
                                            as: UInt8.self)
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        return output
    }

    private func loadTensorFromSafetensors(name: String, shardURL: URL) throws -> Data {
        let loader = SafetensorsLoader()
        let tensors = try loader.parseHeader(at: shardURL)

        guard let tensor = tensors.first(where: { $0.name == name }) else {
            throw STAFConversionError.tensorNotFound(name)
        }

        let fileHandle = try FileHandle(forReadingFrom: shardURL)
        defer { fileHandle.closeFile() }

        guard let sizeData = try fileHandle.read(upToCount: 8), sizeData.count == 8 else {
            throw STAFConversionError.readFailed(name)
        }
        let headerSize = sizeData.withUnsafeBytes { Int(UInt64(littleEndian: $0.load(as: UInt64.self))) }
        let dataSectionOffset = 8 + headerSize

        try fileHandle.seek(toOffset: UInt64(dataSectionOffset + tensor.dataOffset))
        guard let data = try fileHandle.read(upToCount: tensor.byteCount),
              data.count == tensor.byteCount else {
            throw STAFConversionError.readFailed(name)
        }

        return data
    }

    private func convertFloat32ToFloat16(_ data: Data) -> Data {
        let count = data.count / 4
        var output = Data(count: count * 2)

        data.withUnsafeBytes { source in
            output.withUnsafeMutableBytes { destination in
                let floats = source.bindMemory(to: Float.self)
                let halfs = destination.bindMemory(to: Float16.self)
                for i in 0..<count {
                    halfs[i] = Float16(floats[i])
                }
            }
        }

        return output
    }

    private func convertBFloat16ToFloat16(_ data: Data) -> Data {
        let count = data.count / 2
        var output = Data(count: count * 2)

        data.withUnsafeBytes { source in
            output.withUnsafeMutableBytes { destination in
                let bf16 = source.bindMemory(to: UInt16.self)
                let fp16 = destination.bindMemory(to: Float16.self)
                for i in 0..<count {
                    // BF16 → Float32 → Float16
                    let f32Bits = UInt32(bf16[i]) << 16
                    let f32 = Float(bitPattern: f32Bits)
                    fp16[i] = Float16(f32)
                }
            }
        }

        return output
    }

    // MARK: - Helpers

    private func alignUp(_ value: Int, to alignment: Int) -> Int {
        let remainder = value % alignment
        return remainder == 0 ? value : value + (alignment - remainder)
    }
}

// MARK: - Errors

public enum STAFConversionError: Error, CustomStringConvertible {
    case readFailed(String)
    case tensorNotFound(String)
    case unsupportedFormat(UInt8)

    public var description: String {
        switch self {
        case .readFailed(let name): return "STAFConversionError: failed to read tensor '\(name)'"
        case .tensorNotFound(let name): return "STAFConversionError: tensor '\(name)' not found"
        case .unsupportedFormat(let id): return "STAFConversionError: unsupported format 0x\(String(id, radix: 16))"
        }
    }
}
