import Metal
import Testing
@testable import MetalCompiler

/// Multi-row GEMV correctness tests for the unified quantized kernel generator.
///
/// The multi-block tests in `UnifiedGEMVMultiBlockTests` exercise the kernel's
/// inner block-loop with a single output row. They cannot catch regressions in
/// the outer row-dispatch path: a stale `rowBase` pointer, a miscomputed
/// `row = gid.x * rowsPerThreadgroup + sgitg`, or a row-stride mismatch in the
/// packed weight buffer only manifests when `outputDimension ≥ 2`.
///
/// These tests pack `outputDimension` rows each with `numBlocksPerRow` blocks
/// and verify every output lane against a software reference. Rows share the
/// input vector but have distinct `(scale, zero)` and weight patterns so a
/// row-to-row cross-contamination bug surfaces as a value mismatch.
@Suite("Unified Quantized GEMV Multi-Row Correctness", .serialized)
struct UnifiedGEMVMultiRowTests {

    // MARK: - Q2 (aligned)

    @Test("Q2G16 multi-row GEMV aggregates per row")
    func q2Group16MultiRow() throws {
        try runMultiRowGEMVTest(format: AffineQ2Group16Format())
    }

    @Test("Q2G32 multi-row GEMV aggregates per row")
    func q2Group32MultiRow() throws {
        try runMultiRowGEMVTest(format: AffineQ2Group32Format())
    }

    // MARK: - Q3 (non-aligned)

    @Test("Q3G16 multi-row GEMV aggregates per row")
    func q3Group16MultiRow() throws {
        try runMultiRowGEMVTest(format: AffineQ3Group16Format())
    }

    @Test("Q3G32 multi-row GEMV aggregates per row")
    func q3Group32MultiRow() throws {
        try runMultiRowGEMVTest(format: AffineQ3Group32Format())
    }

    @Test("Q3G64 multi-row GEMV aggregates per row")
    func q3Group64MultiRow() throws {
        try runMultiRowGEMVTest(format: AffineQ3Group64Format())
    }

    // MARK: - Q4 (aligned)

    @Test("Q4G64 multi-row GEMV aggregates per row")
    func q4Group64MultiRow() throws {
        try runMultiRowGEMVTest(format: AffineQ4Group64Format())
    }

    @Test("Q4G128 multi-row GEMV aggregates per row")
    func q4Group128MultiRow() throws {
        try runMultiRowGEMVTest(format: AffineQ4Group128Format())
    }

    // MARK: - Q5 (non-aligned)

    @Test("Q5G32 multi-row GEMV aggregates per row")
    func q5Group32MultiRow() throws {
        try runMultiRowGEMVTest(format: AffineQ5Group32Format())
    }

    @Test("Q5G64 multi-row GEMV aggregates per row")
    func q5Group64MultiRow() throws {
        try runMultiRowGEMVTest(format: AffineQ5Group64Format())
    }

    // MARK: - Q6 (non-aligned)

    @Test("Q6G16 multi-row GEMV aggregates per row")
    func q6Group16MultiRow() throws {
        try runMultiRowGEMVTest(format: AffineQ6Group16Format())
    }

    @Test("Q6G32 multi-row GEMV aggregates per row")
    func q6Group32MultiRow() throws {
        try runMultiRowGEMVTest(format: AffineQ6Group32Format())
    }

    // MARK: - Q8 (aligned)

    @Test("Q8G32 multi-row GEMV aggregates per row")
    func q8Group32MultiRow() throws {
        try runMultiRowGEMVTest(format: AffineQ8Group32Format())
    }

    @Test("Q8G64 multi-row GEMV aggregates per row")
    func q8Group64MultiRow() throws {
        try runMultiRowGEMVTest(format: AffineQ8Group64Format())
    }

    @Test("Q8G128 multi-row GEMV aggregates per row")
    func q8Group128MultiRow() throws {
        try runMultiRowGEMVTest(format: AffineQ8Group128Format())
    }

    // MARK: - Test driver

    /// Pack `outputDimension` rows of `numBlocksPerRow` blocks each with
    /// distinct per-(row, block) `(scale, zero)` and weight patterns, dispatch
    /// the kernel with `grid.width = outputDimension`, and verify every output
    /// lane against the software reference.
    private func runMultiRowGEMVTest(
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

        // Per-row × per-block data. Row-major layout: row 0 blocks first, then
        // row 1, etc. This matches the kernel's `rowBase = weight + row *
        // blocksPerRow * bytesPerBlock` addressing.
        var perRowBlockWeights: [[[UInt32]]] = []
        var perRowBlockScale: [[Float]] = []
        var perRowBlockZero: [[Float]] = []
        var packedWeightBytes: [UInt8] = []

        for row in 0..<outputDimension {
            var rowWeights: [[UInt32]] = []
            var rowScales: [Float] = []
            var rowZeros: [Float] = []
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
                rowWeights.append(weights)
                rowScales.append(scale)
                rowZeros.append(zero)
                let blockBytes = packSingleBlock(
                    format: format,
                    weights: weights,
                    scale: scale,
                    zero: zero
                )
                #expect(blockBytes.count == format.bytesPerBlock)
                packedWeightBytes.append(contentsOf: blockBytes)
            }
            perRowBlockWeights.append(rowWeights)
            perRowBlockScale.append(rowScales)
            perRowBlockZero.append(rowZeros)
        }

        let weightsPerRow = weightsPerBlock * numBlocksPerRow
        let kernelName = "test_multirow_gemv_\(format.schemeIdentifier.rawValue)_\(outputDimension)x\(numBlocksPerRow)"
        let source = MetalSourceGenerator.commonHeader + "\n\n"
            + MetalSourceGenerator.generateUnifiedQuantizedGEMV(
                name: kernelName,
                format: format,
                bufferPrecision: .float16
            )
        let pipeline = try makePipeline(device: device, source: source, functionName: kernelName)

        // Input is shared across rows.
        let inputValues: [Float16] = (0..<weightsPerRow).map { k in
            let tenths = Float(k % 13) * 0.046875 - 0.25
            return Float16(tenths)
        }

        let inputBuffer = try #require(device.makeBuffer(
            bytes: inputValues,
            length: inputValues.count * MemoryLayout<Float16>.stride,
            options: .storageModeShared
        ))
        let weightBuffer = try #require(device.makeBuffer(
            bytes: packedWeightBytes,
            length: packedWeightBytes.count,
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
        encoder.setBuffer(weightBuffer, offset: 0, index: 1)
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

        let outputPointer = outputBuffer.contents().bindMemory(to: Float16.self, capacity: outputDimension)

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
                \(format.schemeIdentifier) multi-row GEMV drift (row=\(row), rows=\(outputDimension), blocksPerRow=\(numBlocksPerRow))
                actual=\(actual), expected=\(expected)
                diff=\(actual - expected), tolerance=\(tolerance)
                """
            )
        }
    }

    // MARK: - Block packing helpers

    private func packSingleBlock(
        format: any QuantizationFormat,
        weights: [UInt32],
        scale: Float,
        zero: Float
    ) -> [UInt8] {
        var bytes = [UInt8](repeating: 0, count: format.bytesPerBlock)
        let scaleBits = Float16(scale).bitPattern
        let zeroBits = Float16(zero).bitPattern
        bytes[0] = UInt8(scaleBits & 0xFF)
        bytes[1] = UInt8((scaleBits >> 8) & 0xFF)
        bytes[2] = UInt8(zeroBits & 0xFF)
        bytes[3] = UInt8((zeroBits >> 8) & 0xFF)

        let packed = packLSBFirstBitStream(weights: weights, bits: format.bits)
        for (index, byte) in packed.enumerated() {
            bytes[4 + index] = byte
        }
        return bytes
    }

    private func packLSBFirstBitStream(weights: [UInt32], bits: Int) -> [UInt8] {
        let totalBits = weights.count * bits
        let byteCount = (totalBits + 7) / 8
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
        return result
    }

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
}
