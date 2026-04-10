import Foundation

struct STAFPayloadConverter: Sendable {

    func convertPayload(for entry: STAFConversionEntry) throws -> Data {
        let tensorData = try loadRawTensorData(entry: entry)

        switch entry.schemeIdentifier {
        case .fp16RowMajor, .passthrough:
            return try convertDensePayload(entry: entry, tensorData: tensorData)
        case .bf16RowMajor:
            return tensorData
        case .fp32RowMajor:
            return tensorData
        case .q4Group64ScaleF16, .q4Group128ScaleF16:
            return try repackMLXQuantized(entry: entry, weightData: tensorData)
        default:
            return tensorData
        }
    }

    private func convertDensePayload(entry: STAFConversionEntry, tensorData: Data) throws -> Data {
        if entry.info.dtype != .float32 {
            return tensorData
        }

        let count = tensorData.count / MemoryLayout<Float>.size
        var output = Data(count: count * MemoryLayout<Float16>.size)
        tensorData.withUnsafeBytes { source in
            output.withUnsafeMutableBytes { destination in
                let floats = source.bindMemory(to: Float.self)
                let halfs = destination.bindMemory(to: Float16.self)
                for index in 0..<count {
                    halfs[index] = Float16(floats[index])
                }
            }
        }
        return output
    }

    private func loadRawTensorData(entry: STAFConversionEntry) throws -> Data {
        let fileHandle = try FileHandle(forReadingFrom: entry.shardURL)
        defer { fileHandle.closeFile() }

        guard let sizeData = try fileHandle.read(upToCount: 8), sizeData.count == 8 else {
            throw STAFConversionError.readFailed(entry.name)
        }
        let headerSize = sizeData.withUnsafeBytes { Int(UInt64(littleEndian: $0.load(as: UInt64.self))) }
        let dataSectionOffset = 8 + headerSize

        try fileHandle.seek(toOffset: UInt64(dataSectionOffset + entry.info.dataOffset))
        guard let tensorData = try fileHandle.read(upToCount: entry.info.byteCount),
              tensorData.count == entry.info.byteCount else {
            throw STAFConversionError.readFailed(entry.name)
        }
        return tensorData
    }

    private func repackMLXQuantized(entry: STAFConversionEntry, weightData: Data) throws -> Data {
        let modulePath = String(entry.name.dropLast(".weight".count))
        let scalesData = try loadTensorFromSafetensors(name: modulePath + ".scales", shardURL: entry.shardURL)
        let biasesData = try loadTensorFromSafetensors(name: modulePath + ".biases", shardURL: entry.shardURL)

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

                                var scale = scales[row * blocksPerRow + block]
                                memcpy(destination, &scale, 2)

                                var zero = biases[row * blocksPerRow + block]
                                memcpy(destination + 2, &zero, 2)

                                let qDestination = destination + 4
                                let weightsPerGroup = format.groupSize
                                let uint32PerGroup = weightsPerGroup / elementsPerUInt32

                                for packedIndexOffset in 0..<uint32PerGroup {
                                    let packedIndex = row * packedDimension + block * uint32PerGroup + packedIndexOffset
                                    let packed = weights[packedIndex]
                                    for nibblePair in 0..<4 {
                                        let lowNibble = UInt8((packed >> (nibblePair * 8)) & 0xF)
                                        let highNibble = UInt8((packed >> (nibblePair * 8 + 4)) & 0xF)
                                        let byte = lowNibble | (highNibble << 4)
                                        qDestination.storeBytes(
                                            of: byte,
                                            toByteOffset: packedIndexOffset * 4 + nibblePair,
                                            as: UInt8.self
                                        )
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
}
