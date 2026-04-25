import Metal
import Foundation
import Testing
@testable import MetalCompiler

/// End-to-end correctness: MLX safetensors → STAFConverter → STAFLoader →
/// unified quantized GEMV kernel dispatch → software reference.
///
/// `UnifiedGEMVMultiRowTests` validates the kernel with weight blocks hand-packed
/// in STAF layout (`[scale|zero|qs]`). It never exercises the MLX→STAF repacking
/// in `STAFPayloadConverter.repackMLXQuantized`. Those two can disagree on bit
/// order, byte alignment, or per-group scale/bias ordering — a bug that only
/// surfaces when weights arrive from a real MLX bundle. These tests close that
/// gap by synthesizing valid MLX safetensors (row-major U32 weight bit-stream
/// with `.scales` / `.biases` companions), running the full converter + loader,
/// and dispatching the kernel against the resulting STAF-backed buffer.
@Suite("STAF End-to-End Unified GEMV", .serialized)
struct STAFEndToEndGEMVTests {

    // MARK: - Q2 (aligned)

    @Test("Q2G16 MLX→STAF→GEMV end-to-end")
    func q2Group16EndToEnd() throws {
        try runEndToEndTest(format: AffineQ2Group16Format())
    }

    @Test("Q2G32 MLX→STAF→GEMV end-to-end")
    func q2Group32EndToEnd() throws {
        try runEndToEndTest(format: AffineQ2Group32Format())
    }

    // MARK: - Q3 (non-aligned)

    @Test("Q3G16 MLX→STAF→GEMV end-to-end")
    func q3Group16EndToEnd() throws {
        try runEndToEndTest(format: AffineQ3Group16Format())
    }

    @Test("Q3G32 MLX→STAF→GEMV end-to-end")
    func q3Group32EndToEnd() throws {
        try runEndToEndTest(format: AffineQ3Group32Format())
    }

    @Test("Q3G64 MLX→STAF→GEMV end-to-end")
    func q3Group64EndToEnd() throws {
        try runEndToEndTest(format: AffineQ3Group64Format())
    }

    // MARK: - Q4 (aligned)

    @Test("Q4G64 MLX→STAF→GEMV end-to-end")
    func q4Group64EndToEnd() throws {
        try runEndToEndTest(format: AffineQ4Group64Format())
    }

    @Test("Q4G128 MLX→STAF→GEMV end-to-end")
    func q4Group128EndToEnd() throws {
        try runEndToEndTest(format: AffineQ4Group128Format())
    }

    // MARK: - Q5 (non-aligned)

    @Test("Q5G32 MLX→STAF→GEMV end-to-end")
    func q5Group32EndToEnd() throws {
        try runEndToEndTest(format: AffineQ5Group32Format())
    }

    @Test("Q5G64 MLX→STAF→GEMV end-to-end")
    func q5Group64EndToEnd() throws {
        try runEndToEndTest(format: AffineQ5Group64Format())
    }

    // MARK: - Q6 (non-aligned)

    @Test("Q6G16 MLX→STAF→GEMV end-to-end")
    func q6Group16EndToEnd() throws {
        try runEndToEndTest(format: AffineQ6Group16Format())
    }

    @Test("Q6G32 MLX→STAF→GEMV end-to-end")
    func q6Group32EndToEnd() throws {
        try runEndToEndTest(format: AffineQ6Group32Format())
    }

    // MARK: - Q8 (aligned)

    @Test("Q8G32 MLX→STAF→GEMV end-to-end")
    func q8Group32EndToEnd() throws {
        try runEndToEndTest(format: AffineQ8Group32Format())
    }

    @Test("Q8G64 MLX→STAF→GEMV end-to-end")
    func q8Group64EndToEnd() throws {
        try runEndToEndTest(format: AffineQ8Group64Format())
    }

    @Test("Q8G128 MLX→STAF→GEMV end-to-end")
    func q8Group128EndToEnd() throws {
        try runEndToEndTest(format: AffineQ8Group128Format())
    }

    // MARK: - Driver

    /// Synthesize an MLX-shaped safetensors file for `outputDimension × numBlocksPerRow`
    /// blocks, convert it through `STAFConverter`, load it, and dispatch the unified
    /// GEMV kernel. Verify every output lane against the software reference.
    private func runEndToEndTest(
        format: any QuantizationFormat,
        numBlocksPerRow: Int = 4,
        outputDimension: Int = 4
    ) throws {
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }
        guard let device = MTLCreateSystemDefaultDevice() else { return }

        let weightsPerBlock = format.weightsPerBlock
        let bitRange = UInt32(1) << format.bits
        let inclusiveMax = bitRange - 1
        let weightsPerRow = weightsPerBlock * numBlocksPerRow

        // MLX .weight stores a row-major U32 bit-stream, one row at a time.
        // `packedWordsPerRow = weightsPerRow × bits / 32` must divide evenly;
        // this is guaranteed for all 13 MLX-reachable formats when
        // numBlocksPerRow ≥ 1 and groupSize × bits is a whole number of bytes.
        let totalBitsPerRow = weightsPerRow * format.bits
        #expect(totalBitsPerRow % 32 == 0,
                "weightsPerRow × bits must be a multiple of 32 for U32 packing")
        let packedWordsPerRow = totalBitsPerRow / 32

        var perRowBlockWeights: [[[UInt32]]] = []
        var perRowBlockScale: [[Float]] = []
        var perRowBlockZero: [[Float]] = []

        var weightWords: [UInt32] = []
        weightWords.reserveCapacity(outputDimension * packedWordsPerRow)

        for row in 0..<outputDimension {
            var rowBlockWeights: [[UInt32]] = []
            var rowScales: [Float] = []
            var rowZeros: [Float] = []
            var rowBitStream: [UInt32] = []
            rowBitStream.reserveCapacity(weightsPerRow)

            for block in 0..<numBlocksPerRow {
                let scale = 0.0625 * Float(block + 1) + 0.015625 * Float(row)
                let zero = -0.25 + 0.125 * Float(block) + 0.03125 * Float(row)
                let weights: [UInt32] = (0..<weightsPerBlock).map { k in
                    UInt32((k &+ block &* 7 &+ row &* 3) % Int(bitRange))
                }
                for (index, value) in weights.enumerated() {
                    #expect(
                        value <= inclusiveMax,
                        "row \(row) block \(block) weight[\(index)]=\(value) exceeds \(format.bits)-bit range"
                    )
                }
                rowBlockWeights.append(weights)
                rowScales.append(scale)
                rowZeros.append(zero)
                rowBitStream.append(contentsOf: weights)
            }

            perRowBlockWeights.append(rowBlockWeights)
            perRowBlockScale.append(rowScales)
            perRowBlockZero.append(rowZeros)

            let rowBytes = packRowBitStream(
                weights: rowBitStream,
                bits: format.bits,
                expectedWords: packedWordsPerRow
            )
            weightWords.append(contentsOf: rowBytes)
        }

        // Build MLX-shaped tensors.
        let tensorName = "test.layers.0.gate_proj"
        var weightData = Data()
        weightData.reserveCapacity(weightWords.count * 4)
        for word in weightWords {
            var w = word.littleEndian
            weightData.append(Data(bytes: &w, count: 4))
        }

        var scalesData = Data()
        var biasesData = Data()
        for row in 0..<outputDimension {
            for block in 0..<numBlocksPerRow {
                var s = Float16(perRowBlockScale[row][block]).bitPattern.littleEndian
                scalesData.append(Data(bytes: &s, count: 2))
                var z = Float16(perRowBlockZero[row][block]).bitPattern.littleEndian
                biasesData.append(Data(bytes: &z, count: 2))
            }
        }

        let tempDirectory = makeTempDirectory()
        defer { cleanup(tempDirectory) }

        let safetensorsURL = tempDirectory.appendingPathComponent("model.safetensors")
        try writeSafetensors(
            tensors: [
                TestTensor(
                    name: tensorName + ".weight",
                    dtype: "U32",
                    shape: [outputDimension, packedWordsPerRow],
                    data: weightData
                ),
                TestTensor(
                    name: tensorName + ".scales",
                    dtype: "F16",
                    shape: [outputDimension, numBlocksPerRow],
                    data: scalesData
                ),
                TestTensor(
                    name: tensorName + ".biases",
                    dtype: "F16",
                    shape: [outputDimension, numBlocksPerRow],
                    data: biasesData
                ),
            ],
            to: safetensorsURL
        )

        let stafURL = tempDirectory.appendingPathComponent("model.staf")
        try STAFConverter().convert(
            safetensorsURLs: [safetensorsURL],
            outputURL: stafURL,
            quantization: MLXQuantizationHint(bits: format.bits, groupSize: format.groupSize)
        )

        let store = try STAFLoader().load(at: stafURL, device: device)
        let access = try #require(
            store.bufferAccess(for: tensorName + ".weight"),
            "STAF weight '\(tensorName).weight' not found after conversion"
        )

        let expectedBlocks = outputDimension * numBlocksPerRow
        let expectedBytes = expectedBlocks * format.bytesPerBlock
        #expect(
            access.size == expectedBytes,
            "STAF payload size \(access.size) != expected \(expectedBytes) for \(format.schemeIdentifier)"
        )

        // Compile and dispatch the unified GEMV kernel.
        let kernelName = "test_staf_e2e_gemv_\(format.schemeIdentifier.rawValue)"
        let source = MetalSourceGenerator.commonHeader + "\n\n"
            + MetalSourceGenerator.generateUnifiedQuantizedGEMV(
                name: kernelName,
                format: format,
                bufferPrecision: .float16
            )
        let pipeline = try makePipeline(device: device, source: source, functionName: kernelName)

        let inputValues: [Float16] = (0..<weightsPerRow).map { k in
            let tenths = Float(k % 13) * 0.046875 - 0.25
            return Float16(tenths)
        }

        let inputBuffer = try #require(device.makeBuffer(
            bytes: inputValues,
            length: inputValues.count * MemoryLayout<Float16>.stride,
            options: .storageModeShared
        ))
        let outputBuffer = try #require(device.makeBuffer(
            length: outputDimension * MemoryLayout<Float16>.stride,
            options: .storageModeShared
        ))

        let queue = try #require(device.makeCommandQueue())
        let commandBuffer = try #require(queue.makeCommandBuffer())
        let encoder = try #require(commandBuffer.makeComputeCommandEncoder())
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(inputBuffer, offset: 0, index: 0)
        encoder.setBuffer(access.buffer, offset: access.offset, index: 1)
        encoder.setBuffer(outputBuffer, offset: 0, index: 2)
        var inputDimension = UInt32(weightsPerRow)
        var outputDim = UInt32(outputDimension)
        encoder.setBytes(&inputDimension, length: MemoryLayout<UInt32>.stride, index: 3)
        encoder.setBytes(&outputDim, length: MemoryLayout<UInt32>.stride, index: 4)
        // 32 threads/TG → rowsPerThreadgroup = 1 → grid.width = outputDimension.
        encoder.dispatchThreadgroups(
            MTLSize(width: outputDimension, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1)
        )
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        if let error = commandBuffer.error {
            Issue.record("GPU error for \(format.schemeIdentifier): \(error)")
            return
        }

        let outputPointer = outputBuffer.contents()
            .bindMemory(to: Float16.self, capacity: outputDimension)

        let absoluteTolerance: Float = 0.03
        let relativeTolerance: Float = 0.03

        for row in 0..<outputDimension {
            var expected: Float = 0
            for block in 0..<numBlocksPerRow {
                let scale = perRowBlockScale[row][block]
                let zero = perRowBlockZero[row][block]
                for k in 0..<weightsPerBlock {
                    let dequantized = scale * Float(perRowBlockWeights[row][block][k]) + zero
                    expected += dequantized * Float(inputValues[block * weightsPerBlock + k])
                }
            }

            let actual = Float(outputPointer[row])
            let tolerance = max(absoluteTolerance, relativeTolerance * abs(expected))

            #expect(
                abs(actual - expected) < tolerance,
                """
                \(format.schemeIdentifier) end-to-end drift (row=\(row))
                actual=\(actual), expected=\(expected)
                diff=\(actual - expected), tolerance=\(tolerance)
                """
            )
        }
    }

    // MARK: - Bit-stream packing

    /// Pack `weights` as an LSB-first bit-stream into `expectedWords` uint32 words.
    /// Matches MLX `quantize()` packing: word[i] holds bits `[i×32, (i+1)×32)`
    /// of the stream, with each weight occupying `bits` contiguous bits.
    private func packRowBitStream(
        weights: [UInt32],
        bits: Int,
        expectedWords: Int
    ) -> [UInt32] {
        let totalBits = weights.count * bits
        let wordCount = (totalBits + 31) / 32
        #expect(
            wordCount == expectedWords,
            "totalBits=\(totalBits) → words=\(wordCount) != expected=\(expectedWords)"
        )
        var result = [UInt32](repeating: 0, count: wordCount)
        let mask = (UInt64(1) << bits) - 1
        for (k, weight) in weights.enumerated() {
            let value = UInt64(weight) & mask
            let bitOffset = k * bits
            let wordIndex = bitOffset / 32
            let bitIndex = bitOffset % 32
            let shifted = value << bitIndex
            let spannedWords = (bitIndex + bits + 31) / 32
            for offset in 0..<spannedWords {
                let word = UInt32((shifted >> (offset * 32)) & 0xFFFF_FFFF)
                result[wordIndex + offset] |= word
            }
        }
        return result
    }

    // MARK: - Metal pipeline

    private func makePipeline(
        device: MTLDevice,
        source: String,
        functionName: String
    ) throws -> MTLComputePipelineState {
        let options = MTLCompileOptions()
        options.languageVersion = .version4_0
        let library = try device.makeLibrary(source: source, options: options)
        let function = try #require(library.makeFunction(name: functionName))
        return try device.makeComputePipelineState(function: function)
    }

    // MARK: - Temp directory

    private func makeTempDirectory() -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("staf_e2e_gemv_\(UUID().uuidString)")
        try! FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    private func cleanup(_ url: URL) {
        try? FileManager.default.removeItem(at: url)
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
