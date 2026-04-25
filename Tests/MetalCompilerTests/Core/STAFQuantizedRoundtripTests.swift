import Foundation
import Testing
@testable import MetalCompiler

/// End-to-end round-trip tests for MLX → STAF quantized payload conversion.
///
/// Covers formats recently added to `STAFPayloadConverter.convertPayload`
/// dispatch: Q3G16/G32, Q5G32/G64, Q8G128, Q4G128Zero, plus anchor formats
/// Q2G16, Q4G64, Q6G32 for regression coverage.
///
/// For each format the test:
///   1. Packs a known bit-stream of weights with known scale/zero
///      using the MLX layout (uint32-packed LSB-first row).
///   2. Writes a minimal safetensors file containing `.weight`, `.scales`,
///      `.biases` tensors for a single module.
///   3. Invokes `STAFPayloadConverter.convertPayload` with a manually
///      constructed `STAFConversionEntry`.
///   4. Verifies the repacked STAF block layout: scale F16 (2B) + zero F16
///      (2B) + original bit-stream bytes — byte-exact.
@Suite("STAF MLX→STAF Quantized Round-trip", .serialized)
struct STAFQuantizedRoundtripTests {

    // MARK: - Q3

    // Q3G16 has 16 × 3 = 48 bits per block which is not a whole uint32 word,
    // so MLX packs 2 blocks per row (96 bits = 3 uint32 words) as the minimum
    // row-aligned unit. The multi-block driver covers this case.
    @Test("Q3G16 MLX→STAF repack preserves bit-stream (2 blocks for uint32 alignment)")
    func q3Group16() throws {
        try runMultiRowRoundtripTest(
            scheme: .q3Group16ScaleF16,
            bits: 3,
            groupSize: 16,
            numberOfRows: 1,
            blocksPerRow: 2,
            scaleBase: 0.125,
            zeroBase: -0.5
        )
    }

    @Test("Q3G32 MLX→STAF repack preserves bit-stream and scale/zero")
    func q3Group32() throws {
        try runRoundtripTest(
            scheme: .q3Group32ScaleF16,
            bits: 3,
            groupSize: 32,
            weights: (0..<32).map { UInt32($0 % 8) },
            scale: 0.0625,
            zero: 0.0
        )
    }

    @Test("Q3G64 MLX→STAF repack preserves bit-stream and scale/zero")
    func q3Group64() throws {
        try runRoundtripTest(
            scheme: .q3Group64ScaleF16,
            bits: 3,
            groupSize: 64,
            weights: (0..<64).map { UInt32($0 % 8) },
            scale: 0.03125,
            zero: 0.25
        )
    }

    // MARK: - Q5

    @Test("Q5G32 MLX→STAF repack preserves bit-stream and scale/zero")
    func q5Group32() throws {
        try runRoundtripTest(
            scheme: .q5Group32ScaleF16,
            bits: 5,
            groupSize: 32,
            weights: (0..<32).map { UInt32($0 % 32) },
            scale: 0.03125,
            zero: -0.25
        )
    }

    @Test("Q5G64 MLX→STAF repack preserves bit-stream and scale/zero")
    func q5Group64() throws {
        try runRoundtripTest(
            scheme: .q5Group64ScaleF16,
            bits: 5,
            groupSize: 64,
            weights: (0..<64).map { UInt32($0 % 32) },
            scale: 0.015625,
            zero: 0.125
        )
    }

    // MARK: - Q8G128

    @Test("Q8G128 MLX→STAF repack preserves bit-stream and scale/zero")
    func q8Group128() throws {
        try runRoundtripTest(
            scheme: .q8Group128ScaleF16,
            bits: 8,
            groupSize: 128,
            weights: (0..<128).map { UInt32($0 % 256) },
            scale: 0.0078125,
            zero: -16.0
        )
    }

    // MARK: - Q4G128Zero (recently added dispatch case)

    @Test("Q4G128Zero MLX→STAF repack preserves bit-stream and scale/zero")
    func q4Group128Zero() throws {
        try runRoundtripTest(
            scheme: .q4Group128ScaleF16Zero,
            bits: 4,
            groupSize: 128,
            weights: (0..<128).map { UInt32($0 % 16) },
            scale: 0.0625,
            zero: -0.5
        )
    }

    // MARK: - Regression anchors

    @Test("Q2G16 MLX→STAF repack (regression anchor)")
    func q2Group16Baseline() throws {
        try runRoundtripTest(
            scheme: .q2Group16ScaleF16,
            bits: 2,
            groupSize: 16,
            weights: (0..<16).map { UInt32($0 % 4) },
            scale: 0.25,
            zero: -0.5
        )
    }

    @Test("Q4G64 MLX→STAF repack (regression anchor)")
    func q4Group64Baseline() throws {
        try runRoundtripTest(
            scheme: .q4Group64ScaleF16,
            bits: 4,
            groupSize: 64,
            weights: (0..<64).map { UInt32($0 % 16) },
            scale: 0.0625,
            zero: -0.5
        )
    }

    @Test("Q6G32 MLX→STAF repack (regression anchor)")
    func q6Group32Baseline() throws {
        try runRoundtripTest(
            scheme: .q6Group32ScaleF16,
            bits: 6,
            groupSize: 32,
            weights: (0..<32).map { UInt32($0 % 64) },
            scale: 0.015625,
            zero: -0.5
        )
    }

    // MARK: - Multi-row / multi-block coverage

    @Test("Q4G64 MLX→STAF repack with 2 rows × 2 blocks preserves row/block ordering")
    func q4Group64MultiRow() throws {
        try runMultiRowRoundtripTest(
            scheme: .q4Group64ScaleF16,
            bits: 4,
            groupSize: 64,
            numberOfRows: 2,
            blocksPerRow: 2,
            scaleBase: 0.0625,
            zeroBase: -0.5
        )
    }

    @Test("Q5G32 MLX→STAF repack with 2 rows × 2 blocks preserves row/block ordering")
    func q5Group32MultiRow() throws {
        try runMultiRowRoundtripTest(
            scheme: .q5Group32ScaleF16,
            bits: 5,
            groupSize: 32,
            numberOfRows: 2,
            blocksPerRow: 2,
            scaleBase: 0.03125,
            zeroBase: -0.25
        )
    }

    // MARK: - BF16 scale/bias path

    @Test("Q4G64 MLX→STAF with BF16 scales/biases converts to F16 in-block")
    func q4Group64BFloat16Scales() throws {
        try runBFloat16ScaleRoundtripTest(
            scheme: .q4Group64ScaleF16,
            bits: 4,
            groupSize: 64,
            weights: (0..<64).map { UInt32($0 % 16) },
            scale: 0.0625,
            zero: -0.5
        )
    }

    // MARK: - Single-block driver

    private func runRoundtripTest(
        scheme: QuantizationSchemeIdentifier,
        bits: Int,
        groupSize: Int,
        weights: [UInt32],
        scale: Float,
        zero: Float
    ) throws {
        try runMultiBlockRoundtripTest(
            scheme: scheme,
            bits: bits,
            groupSize: groupSize,
            rows: [[RowBlock(weights: weights, scale: scale, zero: zero)]]
        )
    }

    // MARK: - Multi-row driver

    private func runMultiRowRoundtripTest(
        scheme: QuantizationSchemeIdentifier,
        bits: Int,
        groupSize: Int,
        numberOfRows: Int,
        blocksPerRow: Int,
        scaleBase: Float,
        zeroBase: Float
    ) throws {
        let mask = (UInt32(1) << bits) - 1
        var rows: [[RowBlock]] = []
        for row in 0..<numberOfRows {
            var rowBlocks: [RowBlock] = []
            for block in 0..<blocksPerRow {
                let seed = row * 37 + block * 11
                let weights: [UInt32] = (0..<groupSize).map { i in
                    UInt32(seed + i) & mask
                }
                let scale = scaleBase * Float(block + 1)
                let zero = zeroBase + Float(row) * 0.125
                rowBlocks.append(RowBlock(weights: weights, scale: scale, zero: zero))
            }
            rows.append(rowBlocks)
        }
        try runMultiBlockRoundtripTest(
            scheme: scheme,
            bits: bits,
            groupSize: groupSize,
            rows: rows
        )
    }

    // MARK: - BF16 driver

    private func runBFloat16ScaleRoundtripTest(
        scheme: QuantizationSchemeIdentifier,
        bits: Int,
        groupSize: Int,
        weights: [UInt32],
        scale: Float,
        zero: Float
    ) throws {
        let blocksPerRow = 1
        let totalWeightBits = groupSize * bits
        #expect(totalWeightBits % 32 == 0)
        let packedDimension = totalWeightBits / 32

        let rowBitStream = packLSBFirstBitStream(
            weights: weights,
            bits: bits,
            expectedBytes: packedDimension * 4
        )

        let scaleBF16 = Data(floatToBF16Bytes(scale))
        let zeroBF16 = Data(floatToBF16Bytes(zero))

        let weightTensor = TestTensor(
            name: "layers.0.mlp.gate_proj.weight",
            dtype: "U32",
            shape: [1, packedDimension],
            data: rowBitStream
        )
        let scalesTensor = TestTensor(
            name: "layers.0.mlp.gate_proj.scales",
            dtype: "BF16",
            shape: [1, blocksPerRow],
            data: scaleBF16
        )
        let biasesTensor = TestTensor(
            name: "layers.0.mlp.gate_proj.biases",
            dtype: "BF16",
            shape: [1, blocksPerRow],
            data: zeroBF16
        )

        let tempURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("staf-bf16-scale-\(UUID().uuidString).safetensors")
        try writeSafetensors(
            tensors: [weightTensor, scalesTensor, biasesTensor],
            to: tempURL
        )
        defer { try? FileManager.default.removeItem(at: tempURL) }

        let output = try convertWeightTensor(
            at: tempURL,
            tensorName: "layers.0.mlp.gate_proj.weight",
            scheme: scheme
        )

        guard let format = QuantizationFormatRegistry.format(for: scheme) else {
            Issue.record("no format registered for \(scheme)")
            return
        }
        #expect(output.count == format.bytesPerBlock)

        // Scale/zero must be converted from BF16 → F16, not raw-copied.
        let scaleF16Bits = Float16(scale).bitPattern
        let zeroF16Bits = Float16(zero).bitPattern
        let outScale = output.withUnsafeBytes { $0.load(fromByteOffset: 0, as: UInt16.self) }
        let outZero = output.withUnsafeBytes { $0.load(fromByteOffset: 2, as: UInt16.self) }
        #expect(outScale == scaleF16Bits)
        #expect(outZero == zeroF16Bits)

        let qsStart = 4
        let qsEnd = qsStart + format.bytesPerBlock - 4
        let outQs = output.subdata(in: qsStart..<qsEnd)
        let expectedQs = rowBitStream  // single block, whole row
        #expect(outQs == expectedQs)
    }

    // MARK: - Core driver

    private struct RowBlock {
        let weights: [UInt32]
        let scale: Float
        let zero: Float
    }

    private func runMultiBlockRoundtripTest(
        scheme: QuantizationSchemeIdentifier,
        bits: Int,
        groupSize: Int,
        rows: [[RowBlock]]
    ) throws {
        #expect(!rows.isEmpty)
        let numberOfRows = rows.count
        let blocksPerRow = rows[0].count
        #expect(rows.allSatisfy { $0.count == blocksPerRow })
        for row in rows {
            for block in row {
                #expect(block.weights.count == groupSize)
            }
        }

        guard let format = QuantizationFormatRegistry.format(for: scheme) else {
            Issue.record("no format registered for \(scheme)")
            return
        }
        #expect(format.bits == bits)
        #expect(format.groupSize == groupSize)

        let bytesPerGroup = groupSize * bits / 8
        let bytesPerRow = blocksPerRow * bytesPerGroup
        #expect(bytesPerRow * 8 == blocksPerRow * groupSize * bits)
        #expect(bytesPerRow % 4 == 0, "MLX U32 layout requires row to be a whole number of uint32 words")
        let packedDimension = bytesPerRow / 4

        // Assemble MLX weight bytes: [row][block][group bit-stream]
        var weightBytes = Data(count: numberOfRows * bytesPerRow)
        weightBytes.withUnsafeMutableBytes { ptr in
            guard let base = ptr.baseAddress else { return }
            for row in 0..<numberOfRows {
                for block in 0..<blocksPerRow {
                    let blockBytes = packLSBFirstBitStream(
                        weights: rows[row][block].weights,
                        bits: bits,
                        expectedBytes: bytesPerGroup
                    )
                    let destOffset = row * bytesPerRow + block * bytesPerGroup
                    blockBytes.withUnsafeBytes { (srcPtr: UnsafeRawBufferPointer) -> Void in
                        memcpy(base + destOffset, srcPtr.baseAddress!, bytesPerGroup)
                    }
                }
            }
        }

        // Assemble F16 scales/biases: [row × blocksPerRow]
        let totalBlocks = numberOfRows * blocksPerRow
        var scaleBytes = Data(count: totalBlocks * 2)
        var zeroBytes = Data(count: totalBlocks * 2)
        scaleBytes.withUnsafeMutableBytes { sp in
            zeroBytes.withUnsafeMutableBytes { zp in
                let sBase = sp.bindMemory(to: UInt16.self).baseAddress!
                let zBase = zp.bindMemory(to: UInt16.self).baseAddress!
                for row in 0..<numberOfRows {
                    for block in 0..<blocksPerRow {
                        let index = row * blocksPerRow + block
                        sBase[index] = Float16(rows[row][block].scale).bitPattern
                        zBase[index] = Float16(rows[row][block].zero).bitPattern
                    }
                }
            }
        }

        let tensorName = "layers.0.mlp.gate_proj.weight"
        let modulePath = "layers.0.mlp.gate_proj"
        let weightTensor = TestTensor(
            name: tensorName,
            dtype: "U32",
            shape: [numberOfRows, packedDimension],
            data: weightBytes
        )
        let scalesTensor = TestTensor(
            name: modulePath + ".scales",
            dtype: "F16",
            shape: [numberOfRows, blocksPerRow],
            data: scaleBytes
        )
        let biasesTensor = TestTensor(
            name: modulePath + ".biases",
            dtype: "F16",
            shape: [numberOfRows, blocksPerRow],
            data: zeroBytes
        )

        let tempURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("staf-\(scheme.rawValue)-\(UUID().uuidString).safetensors")
        try writeSafetensors(
            tensors: [weightTensor, scalesTensor, biasesTensor],
            to: tempURL
        )
        defer { try? FileManager.default.removeItem(at: tempURL) }

        let output = try convertWeightTensor(
            at: tempURL,
            tensorName: tensorName,
            scheme: scheme
        )

        #expect(output.count == totalBlocks * format.bytesPerBlock)

        // Verify each block byte-exactly.
        for row in 0..<numberOfRows {
            for block in 0..<blocksPerRow {
                let blockOffset = (row * blocksPerRow + block) * format.bytesPerBlock
                let slice = output.subdata(in: blockOffset..<(blockOffset + format.bytesPerBlock))

                let outScale = slice.withUnsafeBytes {
                    $0.load(fromByteOffset: 0, as: UInt16.self)
                }
                let outZero = slice.withUnsafeBytes {
                    $0.load(fromByteOffset: 2, as: UInt16.self)
                }
                let expectedScale = Float16(rows[row][block].scale).bitPattern
                let expectedZero = Float16(rows[row][block].zero).bitPattern
                #expect(
                    outScale == expectedScale,
                    "\(scheme) row=\(row) block=\(block) scale mismatch: got \(outScale), expected \(expectedScale)"
                )
                #expect(
                    outZero == expectedZero,
                    "\(scheme) row=\(row) block=\(block) zero mismatch: got \(outZero), expected \(expectedZero)"
                )

                let outQs = slice.subdata(in: 4..<format.bytesPerBlock)
                let expectedQs = packLSBFirstBitStream(
                    weights: rows[row][block].weights,
                    bits: bits,
                    expectedBytes: bytesPerGroup
                )
                #expect(
                    outQs == expectedQs,
                    "\(scheme) row=\(row) block=\(block) qs mismatch"
                )
            }
        }
    }

    // MARK: - STAF converter invocation

    private func convertWeightTensor(
        at url: URL,
        tensorName: String,
        scheme: QuantizationSchemeIdentifier
    ) throws -> Data {
        let loader = SafetensorsLoader()
        let tensors = try loader.parseHeader(at: url)
        guard let info = tensors.first(where: { $0.name == tensorName }) else {
            Issue.record("tensor \(tensorName) not in header")
            return Data()
        }
        let entry = STAFConversionEntry(
            name: tensorName,
            sourceName: tensorName,
            info: info,
            shardIndex: 0,
            shardURL: url,
            schemeIdentifier: scheme,
            semanticRole: .mlpGate,
            originalDType: .float16
        )
        let converter = STAFPayloadConverter()
        return try converter.convertPayload(for: entry)
    }

    // MARK: - Bit-stream packing

    private func packLSBFirstBitStream(
        weights: [UInt32],
        bits: Int,
        expectedBytes: Int
    ) -> Data {
        let totalBits = weights.count * bits
        let byteCount = (totalBits + 7) / 8
        #expect(byteCount == expectedBytes, "totalBits=\(totalBits) → bytes=\(byteCount) != expected=\(expectedBytes)")
        var result = [UInt8](repeating: 0, count: byteCount)
        let mask = (UInt64(1) << bits) - 1
        for (k, weight) in weights.enumerated() {
            let value = UInt64(weight) & mask
            let bitOffset = k * bits
            let byteIndex = bitOffset / 8
            let bitIndex = bitOffset % 8
            let shifted = value << bitIndex
            let spannedBytes = (bitIndex + bits + 7) / 8
            for offset in 0..<spannedBytes {
                let byte = UInt8((shifted >> (offset * 8)) & 0xFF)
                result[byteIndex + offset] |= byte
            }
        }
        return Data(result)
    }

    private func floatToBF16Bytes(_ value: Float) -> [UInt8] {
        let bits32 = value.bitPattern
        let upper16 = UInt16(bits32 >> 16)
        return [UInt8(upper16 & 0xFF), UInt8((upper16 >> 8) & 0xFF)]
    }
}

// MARK: - Safetensors writer

private struct TestTensor {
    let name: String
    let dtype: String
    let shape: [Int]
    let data: Data
}

private func writeSafetensors(tensors: [TestTensor], to url: URL) throws {
    var dataSection = Data()
    var tensorOffsets: [(name: String, begin: Int, end: Int)] = []
    for tensor in tensors {
        let begin = dataSection.count
        dataSection.append(tensor.data)
        let end = dataSection.count
        tensorOffsets.append((name: tensor.name, begin: begin, end: end))
    }

    var headerObject: [String: Any] = [:]
    for (index, tensor) in tensors.enumerated() {
        let offsets = tensorOffsets[index]
        headerObject[tensor.name] = [
            "dtype": tensor.dtype,
            "shape": tensor.shape,
            "data_offsets": [offsets.begin, offsets.end]
        ] as [String: Any]
    }

    let headerJSON = try JSONSerialization.data(withJSONObject: headerObject, options: .sortedKeys)
    let headerSize = UInt64(headerJSON.count)

    var fileData = Data()
    var headerSizeLE = headerSize.littleEndian
    fileData.append(Data(bytes: &headerSizeLE, count: 8))
    fileData.append(headerJSON)
    fileData.append(dataSection)

    try fileData.write(to: url)
}
