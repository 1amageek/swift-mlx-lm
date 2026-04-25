import Metal
import Testing
@testable import MetalCompiler

/// Multi-block GEMV correctness tests for the unified quantized kernel generator.
///
/// A weight row consists of `blocksPerRow` concatenated quantization blocks.
/// Each block carries its own `(scale, zero)` pair and its own packed quants.
/// The GEMV kernel must iterate over every block, read the per-block metadata,
/// dequantize the weights, and accumulate `Σ (scale_b · q_bk + zero_b) · x[k]`
/// across all blocks.
///
/// The single-block driver in `UnifiedGEMVBitLevelTests` cannot catch block-loop
/// regressions: a reset-between-blocks bug or a stale per-block pointer only
/// manifests when `numBlocks ≥ 2`. These tests exercise `numBlocks = 4` with
/// distinct `(scale, zero)` and weight patterns per block.
@Suite("Unified Quantized GEMV Multi-Block Correctness", .serialized)
struct UnifiedGEMVMultiBlockTests {

    // MARK: - Q2 (aligned)

    @Test("Q2G16 multi-block GEMV aggregates across blocks")
    func q2Group16MultiBlock() throws {
        try runMultiBlockGEMVTest(format: AffineQ2Group16Format())
    }

    @Test("Q2G32 multi-block GEMV aggregates across blocks")
    func q2Group32MultiBlock() throws {
        try runMultiBlockGEMVTest(format: AffineQ2Group32Format())
    }

    // MARK: - Q3 (non-aligned)

    @Test("Q3G16 multi-block GEMV aggregates across blocks")
    func q3Group16MultiBlock() throws {
        try runMultiBlockGEMVTest(format: AffineQ3Group16Format())
    }

    @Test("Q3G32 multi-block GEMV aggregates across blocks")
    func q3Group32MultiBlock() throws {
        try runMultiBlockGEMVTest(format: AffineQ3Group32Format())
    }

    @Test("Q3G64 multi-block GEMV aggregates across blocks")
    func q3Group64MultiBlock() throws {
        try runMultiBlockGEMVTest(format: AffineQ3Group64Format())
    }

    // MARK: - Q4 (aligned)

    @Test("Q4G64 multi-block GEMV aggregates across blocks")
    func q4Group64MultiBlock() throws {
        try runMultiBlockGEMVTest(format: AffineQ4Group64Format())
    }

    @Test("Q4G128 multi-block GEMV aggregates across blocks")
    func q4Group128MultiBlock() throws {
        try runMultiBlockGEMVTest(format: AffineQ4Group128Format())
    }

    // MARK: - Q5 (non-aligned)

    @Test("Q5G32 multi-block GEMV aggregates across blocks")
    func q5Group32MultiBlock() throws {
        try runMultiBlockGEMVTest(format: AffineQ5Group32Format())
    }

    @Test("Q5G64 multi-block GEMV aggregates across blocks")
    func q5Group64MultiBlock() throws {
        try runMultiBlockGEMVTest(format: AffineQ5Group64Format())
    }

    // MARK: - Q6 (non-aligned)

    @Test("Q6G16 multi-block GEMV aggregates across blocks")
    func q6Group16MultiBlock() throws {
        try runMultiBlockGEMVTest(format: AffineQ6Group16Format())
    }

    @Test("Q6G32 multi-block GEMV aggregates across blocks")
    func q6Group32MultiBlock() throws {
        try runMultiBlockGEMVTest(format: AffineQ6Group32Format())
    }

    // MARK: - Q8 (aligned)

    @Test("Q8G32 multi-block GEMV aggregates across blocks")
    func q8Group32MultiBlock() throws {
        try runMultiBlockGEMVTest(format: AffineQ8Group32Format())
    }

    @Test("Q8G64 multi-block GEMV aggregates across blocks")
    func q8Group64MultiBlock() throws {
        try runMultiBlockGEMVTest(format: AffineQ8Group64Format())
    }

    @Test("Q8G128 multi-block GEMV aggregates across blocks")
    func q8Group128MultiBlock() throws {
        try runMultiBlockGEMVTest(format: AffineQ8Group128Format())
    }

    // MARK: - Test driver

    /// Pack `numBlocks` concatenated blocks with distinct `(scale, zero)` per
    /// block and verify the kernel's aggregation matches a software reference.
    private func runMultiBlockGEMVTest(
        format: any QuantizationFormat,
        numBlocks: Int = 4
    ) throws {
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }
        guard let device = MTLCreateSystemDefaultDevice() else { return }

        let weightsPerBlock = format.weightsPerBlock
        let bitRange = UInt32(1) << format.bits
        let inclusiveMax = bitRange - 1

        var perBlockWeights: [[UInt32]] = []
        var perBlockScale: [Float] = []
        var perBlockZero: [Float] = []
        var concatenatedBlockBytes: [UInt8] = []

        for block in 0..<numBlocks {
            let scale = 0.0625 * Float(block + 1)
            let zero = -0.25 + 0.125 * Float(block)
            let weights: [UInt32] = (0..<weightsPerBlock).map { k in
                UInt32((k &+ block &* 7) % Int(bitRange))
            }
            for (index, value) in weights.enumerated() {
                #expect(
                    value <= inclusiveMax,
                    "block \(block) weight[\(index)]=\(value) exceeds \(format.bits)-bit range"
                )
            }
            perBlockWeights.append(weights)
            perBlockScale.append(scale)
            perBlockZero.append(zero)
            let blockBytes = packSingleBlock(
                format: format,
                weights: weights,
                scale: scale,
                zero: zero
            )
            #expect(blockBytes.count == format.bytesPerBlock)
            concatenatedBlockBytes.append(contentsOf: blockBytes)
        }

        let totalWeights = weightsPerBlock * numBlocks

        let kernelName = "test_multi_gemv_\(format.schemeIdentifier.rawValue)_\(numBlocks)"
        let source = MetalSourceGenerator.commonHeader + "\n\n"
            + MetalSourceGenerator.generateUnifiedQuantizedGEMV(
                name: kernelName,
                format: format,
                bufferPrecision: .float16
            )
        let pipeline = try makePipeline(device: device, source: source, functionName: kernelName)

        let inputValues: [Float16] = (0..<totalWeights).map { k in
            let tenths = Float(k % 11) * 0.0625 - 0.3
            return Float16(tenths)
        }

        let inputBuffer = try #require(device.makeBuffer(
            bytes: inputValues,
            length: inputValues.count * MemoryLayout<Float16>.stride,
            options: .storageModeShared
        ))
        let weightBuffer = try #require(device.makeBuffer(
            bytes: concatenatedBlockBytes,
            length: concatenatedBlockBytes.count,
            options: .storageModeShared
        ))
        let outputBuffer = try #require(device.makeBuffer(
            length: MemoryLayout<Float16>.stride,
            options: .storageModeShared
        ))

        let queue = try #require(device.makeCommandQueue())
        let commandBuffer = try #require(queue.makeCommandBuffer())
        let encoder = try #require(commandBuffer.makeComputeCommandEncoder())
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(inputBuffer, offset: 0, index: 0)
        encoder.setBuffer(weightBuffer, offset: 0, index: 1)
        encoder.setBuffer(outputBuffer, offset: 0, index: 2)
        var inputDimension = UInt32(totalWeights)
        var outputDimension: UInt32 = 1
        encoder.setBytes(&inputDimension, length: MemoryLayout<UInt32>.stride, index: 3)
        encoder.setBytes(&outputDimension, length: MemoryLayout<UInt32>.stride, index: 4)
        encoder.dispatchThreadgroups(
            MTLSize(width: 1, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1)
        )
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        let outputPointer = outputBuffer.contents().bindMemory(to: Float16.self, capacity: 1)
        let actual = Float(outputPointer[0])

        var expected: Float = 0
        for block in 0..<numBlocks {
            let scale = perBlockScale[block]
            let zero = perBlockZero[block]
            for k in 0..<weightsPerBlock {
                let dequantized = scale * Float(perBlockWeights[block][k]) + zero
                expected += dequantized * Float(inputValues[block * weightsPerBlock + k])
            }
        }

        let absoluteTolerance: Float = 0.03
        let relativeTolerance: Float = 0.03
        let tolerance = max(absoluteTolerance, relativeTolerance * abs(expected))

        #expect(
            abs(actual - expected) < tolerance,
            """
            \(format.schemeIdentifier) multi-block GEMV drift (blocks=\(numBlocks), weights=\(totalWeights))
            actual=\(actual), expected=\(expected)
            diff=\(actual - expected), tolerance=\(tolerance)
            """
        )
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
