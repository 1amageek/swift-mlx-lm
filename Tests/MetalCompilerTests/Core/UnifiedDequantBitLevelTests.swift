import Metal
import Testing
@testable import MetalCompiler

/// Bit-level correctness tests for the unified dequant→BF16 kernel generator.
///
/// Validates that the MSL dequant math emitted by `QuantizationFormat`
/// conformances matches the affine formula `w = scale * float(q) + zero` for
/// every weight index in a block, using the MLX LSB-first bit-stream packing
/// layout.
///
/// Covers the newly added non-aligned formats (Q3G16/G32, Q5G32/G64) and the
/// newly added Q8G128 aligned format, plus existing formats as regression
/// anchors.
@Suite("Unified Dequant Bit-Level Correctness", .serialized)
struct UnifiedDequantBitLevelTests {

    // MARK: - Q3 (non-aligned: 8 weights per 3 bytes)

    @Test("Q3G16 dequant matches scale*q+zero for all q ∈ [0,7]")
    func q3Group16() throws {
        try runSingleBlockDequantTest(
            format: AffineQ3Group16Format(),
            weights: (0..<16).map { UInt32($0 % 8) },
            scale: 0.125,
            zero: -0.5
        )
    }

    @Test("Q3G32 dequant matches scale*q+zero for full bit range")
    func q3Group32() throws {
        try runSingleBlockDequantTest(
            format: AffineQ3Group32Format(),
            weights: (0..<32).map { UInt32($0 % 8) },
            scale: 0.0625,
            zero: 0.0
        )
    }

    @Test("Q3G64 dequant matches scale*q+zero across full block")
    func q3Group64() throws {
        try runSingleBlockDequantTest(
            format: AffineQ3Group64Format(),
            weights: (0..<64).map { UInt32($0 % 8) },
            scale: 0.03125,
            zero: -0.125
        )
    }

    @Test("Q3G64 dequant preserves row stride for multiple rows")
    func q3Group64MultiRow() throws {
        try runMultiRowSingleBlockDequantTest(
            format: AffineQ3Group64Format(),
            rows: [
                (weights: (0..<64).map { UInt32($0 % 8) }, scale: 0.03125, zero: -0.125),
                (weights: (0..<64).map { UInt32(7 - ($0 % 8)) }, scale: 0.0625, zero: 0.25),
            ]
        )
    }

    // MARK: - Q5 (non-aligned: 8 weights per 5 bytes)

    @Test("Q5G32 dequant matches scale*q+zero for all q ∈ [0,31]")
    func q5Group32() throws {
        try runSingleBlockDequantTest(
            format: AffineQ5Group32Format(),
            weights: (0..<32).map { UInt32($0 % 32) },
            scale: 0.03125,
            zero: -0.25
        )
    }

    @Test("Q5G64 dequant matches scale*q+zero across boundaries")
    func q5Group64() throws {
        try runSingleBlockDequantTest(
            format: AffineQ5Group64Format(),
            weights: (0..<64).map { UInt32($0 % 32) },
            scale: 0.015625,
            zero: 0.125
        )
    }

    // MARK: - Q8G128 (aligned)

    @Test("Q8G128 dequant matches scale*q+zero for 128-element block")
    func q8Group128() throws {
        try runSingleBlockDequantTest(
            format: AffineQ8Group128Format(),
            weights: (0..<128).map { UInt32($0 % 256) },
            scale: 0.0078125,
            zero: -16.0
        )
    }

    // MARK: - Q6G16 (non-aligned: 4 weights per 3 bytes)

    @Test("Q6G16 dequant matches scale*q+zero for all q ∈ [0,63]")
    func q6Group16() throws {
        try runSingleBlockDequantTest(
            format: AffineQ6Group16Format(),
            weights: (0..<16).map { UInt32($0 % 64) },
            scale: 0.03125,
            zero: -1.0
        )
    }

    // MARK: - Aligned formats (bit-exact variants across group sizes)

    @Test("Q2G16 dequant baseline (regression anchor)")
    func q2Group16Baseline() throws {
        try runSingleBlockDequantTest(
            format: AffineQ2Group16Format(),
            weights: (0..<16).map { UInt32($0 % 4) },
            scale: 0.25,
            zero: -0.5
        )
    }

    @Test("Q2G32 dequant matches scale*q+zero across 2-bit range")
    func q2Group32() throws {
        try runSingleBlockDequantTest(
            format: AffineQ2Group32Format(),
            weights: (0..<32).map { UInt32($0 % 4) },
            scale: 0.125,
            zero: 0.25
        )
    }

    @Test("Q4G64 dequant baseline (regression anchor)")
    func q4Group64Baseline() throws {
        try runSingleBlockDequantTest(
            format: AffineQ4Group64Format(),
            weights: (0..<64).map { UInt32($0 % 16) },
            scale: 0.0625,
            zero: -0.5
        )
    }

    @Test("Q4G128 dequant matches scale*q+zero for 128-element block")
    func q4Group128() throws {
        try runSingleBlockDequantTest(
            format: AffineQ4Group128Format(),
            weights: (0..<128).map { UInt32($0 % 16) },
            scale: 0.03125,
            zero: -0.25
        )
    }

    @Test("Q6G32 dequant baseline (regression anchor)")
    func q6Group32Baseline() throws {
        try runSingleBlockDequantTest(
            format: AffineQ6Group32Format(),
            weights: (0..<32).map { UInt32($0 % 64) },
            scale: 0.015625,
            zero: -0.5
        )
    }

    @Test("Q8G32 dequant matches scale*q+zero for 32-element block")
    func q8Group32() throws {
        try runSingleBlockDequantTest(
            format: AffineQ8Group32Format(),
            weights: (0..<32).map { UInt32($0 % 256) },
            scale: 0.00390625,
            zero: -0.5
        )
    }

    @Test("Q8G64 dequant matches scale*q+zero for 64-element block")
    func q8Group64() throws {
        try runSingleBlockDequantTest(
            format: AffineQ8Group64Format(),
            weights: (0..<64).map { UInt32($0 % 256) },
            scale: 0.0078125,
            zero: -32.0
        )
    }

    // MARK: - Test driver

    /// Pack a single block with the given weights/scale/zero, run the unified
    /// dequant kernel, and verify each dequantized weight matches
    /// `scale * float(q) + zero` within BFloat16 precision.
    private func runSingleBlockDequantTest(
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

        let blockBytes = packSingleBlock(
            format: format,
            weights: weights,
            scale: scale,
            zero: zero
        )
        #expect(blockBytes.count == format.bytesPerBlock)

        let kernelName = MetalSourceGenerator.unifiedDequantKernelName(for: format)
        let source = MetalSourceGenerator.generateUnifiedDequantToBFloat(
            name: kernelName,
            format: format
        )
        let pipeline = try makePipeline(device: device, source: source, functionName: kernelName)

        let inputBuffer = try #require(device.makeBuffer(
            bytes: blockBytes,
            length: blockBytes.count,
            options: .storageModeShared
        ))
        let outputByteCount = format.weightsPerBlock * MemoryLayout<UInt16>.stride
        let outputBuffer = try #require(device.makeBuffer(
            length: outputByteCount,
            options: .storageModeShared
        ))

        let queue = try #require(device.makeCommandQueue())
        let commandBuffer = try #require(queue.makeCommandBuffer())
        let encoder = try #require(commandBuffer.makeComputeCommandEncoder())
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(inputBuffer, offset: 0, index: 0)
        encoder.setBuffer(outputBuffer, offset: 0, index: 1)
        var inputDimension = UInt32(format.weightsPerBlock)
        var outputDimension: UInt32 = 1
        encoder.setBytes(&inputDimension, length: MemoryLayout<UInt32>.stride, index: 2)
        encoder.setBytes(&outputDimension, length: MemoryLayout<UInt32>.stride, index: 3)
        encoder.dispatchThreadgroups(
            MTLSize(width: 1, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)
        )
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        let pointer = outputBuffer.contents().bindMemory(
            to: UInt16.self,
            capacity: format.weightsPerBlock
        )
        let decoded = (0..<format.weightsPerBlock).map { bf16BitsToFloat(pointer[$0]) }
        let expected = weights.map { scale * Float($0) + zero }

        let maxAbsDiff = zip(decoded, expected).map { abs($0 - $1) }.max() ?? 0
        let expectedRange = expected.map { abs($0) }.max() ?? 1
        let relativeTolerance: Float = 0.01
        let absoluteTolerance: Float = 0.005
        let tolerance = max(absoluteTolerance, relativeTolerance * expectedRange)

        #expect(
            maxAbsDiff < tolerance,
            """
            \(format.schemeIdentifier) dequant drift exceeded tolerance
            maxAbsDiff=\(maxAbsDiff), tolerance=\(tolerance)
            head decoded=\(decoded.prefix(8).map { String(format: "%.6f", $0) }.joined(separator: ", "))
            head expected=\(expected.prefix(8).map { String(format: "%.6f", $0) }.joined(separator: ", "))
            tail decoded=\(decoded.suffix(8).map { String(format: "%.6f", $0) }.joined(separator: ", "))
            tail expected=\(expected.suffix(8).map { String(format: "%.6f", $0) }.joined(separator: ", "))
            """
        )
    }

    private func runMultiRowSingleBlockDequantTest(
        format: any QuantizationFormat,
        rows: [(weights: [UInt32], scale: Float, zero: Float)]
    ) throws {
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }
        guard let device = MTLCreateSystemDefaultDevice() else { return }

        var blockBytes: [UInt8] = []
        var expected: [Float] = []
        for row in rows {
            #expect(row.weights.count == format.weightsPerBlock)
            blockBytes.append(contentsOf: packSingleBlock(
                format: format,
                weights: row.weights,
                scale: row.scale,
                zero: row.zero
            ))
            expected.append(contentsOf: row.weights.map { row.scale * Float($0) + row.zero })
        }

        let kernelName = MetalSourceGenerator.unifiedDequantKernelName(for: format)
        let source = MetalSourceGenerator.generateUnifiedDequantToBFloat(
            name: kernelName,
            format: format
        )
        let pipeline = try makePipeline(device: device, source: source, functionName: kernelName)

        let inputBuffer = try #require(device.makeBuffer(
            bytes: blockBytes,
            length: blockBytes.count,
            options: .storageModeShared
        ))
        let outputByteCount = expected.count * MemoryLayout<UInt16>.stride
        let outputBuffer = try #require(device.makeBuffer(
            length: outputByteCount,
            options: .storageModeShared
        ))

        let queue = try #require(device.makeCommandQueue())
        let commandBuffer = try #require(queue.makeCommandBuffer())
        let encoder = try #require(commandBuffer.makeComputeCommandEncoder())
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(inputBuffer, offset: 0, index: 0)
        encoder.setBuffer(outputBuffer, offset: 0, index: 1)
        var inputDimension = UInt32(format.weightsPerBlock)
        var outputDimension = UInt32(rows.count)
        encoder.setBytes(&inputDimension, length: MemoryLayout<UInt32>.stride, index: 2)
        encoder.setBytes(&outputDimension, length: MemoryLayout<UInt32>.stride, index: 3)
        encoder.dispatchThreadgroups(
            MTLSize(width: rows.count, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)
        )
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        let pointer = outputBuffer.contents().bindMemory(to: UInt16.self, capacity: expected.count)
        let decoded = (0..<expected.count).map { bf16BitsToFloat(pointer[$0]) }
        let maxAbsDiff = zip(decoded, expected).map { abs($0 - $1) }.max() ?? 0
        #expect(maxAbsDiff < 0.005)
    }

    // MARK: - Block packing helpers

    /// Pack a single block: 2-byte scale (F16) + 2-byte zero (F16) + LSB-first
    /// bit-stream of weights. Matches MLX's `extract_bits<N>` convention which
    /// is what `QuantizationFormat.perWeightReadExpression` is written to
    /// decode.
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

    /// LSB-first bit-stream packing: weight k occupies bits [k*bits, k*bits+bits).
    /// This is what MLX's `extract_bits<N>` kernels produce and consume.
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

    // MARK: - BFloat16 helpers

    /// Reinterpret a raw 16-bit BFloat16 value as Float32 (upper 16 bits of
    /// IEEE 754 binary32 representation).
    private func bf16BitsToFloat(_ bits: UInt16) -> Float {
        Float(bitPattern: UInt32(bits) << 16)
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
