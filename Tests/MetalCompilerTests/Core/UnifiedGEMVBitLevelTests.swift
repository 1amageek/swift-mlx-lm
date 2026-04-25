import Metal
import Testing
@testable import MetalCompiler

/// Bit-level correctness tests for the unified quantized GEMV kernel generator.
///
/// Validates that `generateUnifiedQuantizedGEMV` produces a kernel whose
/// output matches a reference dot product of the dequantized weights against
/// an input vector, for every newly added format (Q3G16/G32, Q5G32/G64,
/// Q8G128) plus regression anchors (Q2G16, Q4G64, Q6G32).
@Suite("Unified Quantized GEMV Bit-Level Correctness", .serialized)
struct UnifiedGEMVBitLevelTests {

    // MARK: - Q3 (non-aligned)

    @Test("Q3G16 GEMV matches reference dot product")
    func q3Group16() throws {
        try runSingleRowGEMVTest(
            format: AffineQ3Group16Format(),
            weights: (0..<16).map { UInt32($0 % 8) },
            scale: 0.125,
            zero: -0.5
        )
    }

    @Test("Q3G32 GEMV matches reference dot product")
    func q3Group32() throws {
        try runSingleRowGEMVTest(
            format: AffineQ3Group32Format(),
            weights: (0..<32).map { UInt32($0 % 8) },
            scale: 0.0625,
            zero: 0.0
        )
    }

    @Test("Q3G64 GEMV matches reference dot product")
    func q3Group64() throws {
        try runSingleRowGEMVTest(
            format: AffineQ3Group64Format(),
            weights: (0..<64).map { UInt32($0 % 8) },
            scale: 0.03125,
            zero: 0.25
        )
    }

    // MARK: - Q5 (non-aligned)

    @Test("Q5G32 GEMV matches reference dot product")
    func q5Group32() throws {
        try runSingleRowGEMVTest(
            format: AffineQ5Group32Format(),
            weights: (0..<32).map { UInt32($0 % 32) },
            scale: 0.03125,
            zero: -0.25
        )
    }

    @Test("Q5G64 GEMV matches reference dot product")
    func q5Group64() throws {
        try runSingleRowGEMVTest(
            format: AffineQ5Group64Format(),
            weights: (0..<64).map { UInt32($0 % 32) },
            scale: 0.015625,
            zero: 0.125
        )
    }

    // MARK: - Q8G128 (aligned)

    @Test("Q8G128 GEMV matches reference dot product")
    func q8Group128() throws {
        try runSingleRowGEMVTest(
            format: AffineQ8Group128Format(),
            weights: (0..<128).map { UInt32($0 % 256) },
            scale: 0.0078125,
            zero: -16.0
        )
    }

    // MARK: - Q6G16 (non-aligned)

    @Test("Q6G16 GEMV matches reference dot product")
    func q6Group16() throws {
        try runSingleRowGEMVTest(
            format: AffineQ6Group16Format(),
            weights: (0..<16).map { UInt32($0 % 64) },
            scale: 0.03125,
            zero: -1.0
        )
    }

    // MARK: - Aligned formats (bit-exact variants across group sizes)

    @Test("Q2G16 GEMV baseline (regression anchor)")
    func q2Group16Baseline() throws {
        try runSingleRowGEMVTest(
            format: AffineQ2Group16Format(),
            weights: (0..<16).map { UInt32($0 % 4) },
            scale: 0.25,
            zero: -0.5
        )
    }

    @Test("Q2G32 GEMV matches reference dot product")
    func q2Group32() throws {
        try runSingleRowGEMVTest(
            format: AffineQ2Group32Format(),
            weights: (0..<32).map { UInt32($0 % 4) },
            scale: 0.125,
            zero: 0.25
        )
    }

    @Test("Q4G64 GEMV baseline (regression anchor)")
    func q4Group64Baseline() throws {
        try runSingleRowGEMVTest(
            format: AffineQ4Group64Format(),
            weights: (0..<64).map { UInt32($0 % 16) },
            scale: 0.0625,
            zero: -0.5
        )
    }

    @Test("Q4G128 GEMV matches reference dot product")
    func q4Group128() throws {
        try runSingleRowGEMVTest(
            format: AffineQ4Group128Format(),
            weights: (0..<128).map { UInt32($0 % 16) },
            scale: 0.03125,
            zero: -0.25
        )
    }

    @Test("Q6G32 GEMV baseline (regression anchor)")
    func q6Group32Baseline() throws {
        try runSingleRowGEMVTest(
            format: AffineQ6Group32Format(),
            weights: (0..<32).map { UInt32($0 % 64) },
            scale: 0.015625,
            zero: -0.5
        )
    }

    @Test("Q8G32 GEMV matches reference dot product")
    func q8Group32() throws {
        try runSingleRowGEMVTest(
            format: AffineQ8Group32Format(),
            weights: (0..<32).map { UInt32($0 % 256) },
            scale: 0.00390625,
            zero: -0.5
        )
    }

    @Test("Q8G64 GEMV matches reference dot product")
    func q8Group64() throws {
        try runSingleRowGEMVTest(
            format: AffineQ8Group64Format(),
            weights: (0..<64).map { UInt32($0 % 256) },
            scale: 0.0078125,
            zero: -32.0
        )
    }

    // MARK: - Test driver

    /// Pack a single-row single-block weight, run GEMV with a known input
    /// vector, and verify the output matches `sum(w[k] * x[k])` where
    /// `w[k] = scale * float(q[k]) + zero`.
    private func runSingleRowGEMVTest(
        format: any QuantizationFormat,
        weights: [UInt32],
        scale: Float,
        zero: Float
    ) throws {
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }
        guard let device = MTLCreateSystemDefaultDevice() else { return }

        #expect(weights.count == format.weightsPerBlock)
        let bitMask = (UInt32(1) << format.bits) - 1
        for (index, weight) in weights.enumerated() {
            #expect(
                weight <= bitMask,
                "weight[\(index)]=\(weight) exceeds \(format.bits)-bit range (max \(bitMask))"
            )
        }

        let weightBytes = packSingleBlock(
            format: format,
            weights: weights,
            scale: scale,
            zero: zero
        )
        #expect(weightBytes.count == format.bytesPerBlock)

        let kernelName = "test_unified_gemv_\(format.schemeIdentifier.rawValue)"
        let source = MetalSourceGenerator.commonHeader + "\n\n"
            + MetalSourceGenerator.generateUnifiedQuantizedGEMV(
                name: kernelName,
                format: format,
                bufferPrecision: .float16
            )
        let pipeline = try makePipeline(device: device, source: source, functionName: kernelName)

        let inputValues: [Float16] = (0..<format.weightsPerBlock).map { k in
            let tenths = Float(k % 10) * 0.125 - 0.5
            return Float16(tenths)
        }
        let inputBuffer = try #require(device.makeBuffer(
            bytes: inputValues,
            length: inputValues.count * MemoryLayout<Float16>.stride,
            options: .storageModeShared
        ))
        let weightBuffer = try #require(device.makeBuffer(
            bytes: weightBytes,
            length: weightBytes.count,
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
        var inputDimension = UInt32(format.weightsPerBlock)
        var outputDimension: UInt32 = 1
        encoder.setBytes(&inputDimension, length: MemoryLayout<UInt32>.stride, index: 3)
        encoder.setBytes(&outputDimension, length: MemoryLayout<UInt32>.stride, index: 4)
        // 1 simdgroup = 1 row. 1 threadgroup = 1 row.
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
        for k in 0..<format.weightsPerBlock {
            let w = scale * Float(weights[k]) + zero
            expected += w * Float(inputValues[k])
        }

        let absoluteTolerance: Float = 0.01
        let relativeTolerance: Float = 0.02
        let tolerance = max(absoluteTolerance, relativeTolerance * abs(expected))

        #expect(
            abs(actual - expected) < tolerance,
            """
            \(format.schemeIdentifier) GEMV drift exceeded tolerance
            actual=\(actual), expected=\(expected), tolerance=\(tolerance)
            diff=\(actual - expected)
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
