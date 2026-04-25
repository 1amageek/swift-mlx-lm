import Testing
import Metal
@testable import MetalCompiler

@Suite("Quantized Embedding Kernels")
struct QuantizedEmbeddingKernelTests {
    @Test("Gather fragment resolves q4 prefill embedding kernel")
    func q4PrefillKernel() {
        let fragment = GatherFragment(vocabularySize: 1024, embeddingDimension: 256)
        let context = KernelContext(
            bufferPrecision: .float32,
            weightFormat: WeightFormats.quantized4Bit(groupSize: 64)
        )

        let name = fragment.kernelName(context: context)
        #expect(name == "embedding_lookup_seq_q4_g64_f32")

        let source = fragment.kernelSource(
            name: name,
            bufferPrecision: .float32,
            weightFormat: WeightFormats.quantized4Bit(groupSize: 64)
        )
        #expect(source.contains("const uint GROUP_SIZE = 64;"))
        #expect(source.contains("tokenIDs[seqPos]"))
        #expect(source.contains("packed & 0x0F"))
    }

    @Test("Gather fragment resolves q8 decode embedding kernel")
    func q8DecodeKernel() {
        let fragment = GatherFragment(vocabularySize: 1024, embeddingDimension: 256)
        let context = KernelContext(
            bufferPrecision: .float16,
            weightFormat: WeightFormats.quantized8Bit(groupSize: 32)
        )

        let name = fragment.kernelName(context: context)
        #expect(name == "embedding_lookup_q8_g32")

        let source = fragment.kernelSource(
            name: name,
            bufferPrecision: .float16,
            weightFormat: WeightFormats.quantized8Bit(groupSize: 32)
        )
        #expect(source.contains("const uint GROUP_SIZE = 32;"))
        #expect(source.contains("quantized[indexInGroup]"))
    }

    @Test("Q4 embedding lookup reads quantized rows directly")
    func q4EmbeddingExecution() throws {
        let output = try executeQ4EmbeddingLookup(scale: nil)
        #expect(output.count == 64)
        #expect(output.allSatisfy { abs($0 - 2) < 0.001 })
    }

    @Test("Q4 embedding lookup applies embedding scale")
    func q4ScaledEmbeddingExecution() throws {
        let output = try executeQ4EmbeddingLookup(scale: 3)
        #expect(output.count == 64)
        #expect(output.allSatisfy { abs($0 - 6) < 0.001 })
    }

    @Test("Q3G64 sequence embedding lookup matches affine reference")
    func q3Group64SequenceEmbeddingExecution() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw MetalCompilerError.deviceSetupFailed("No Metal device")
        }

        let format = AffineQ3Group64Format()
        let kernelName = "test_embedding_lookup_seq_q3_g64"
        let source = MetalSourceGenerator.commonHeader + "\n"
            + MetalSourceGenerator.generateUnifiedQuantizedEmbeddingLookup(
                name: kernelName,
                format: format,
                bufferPrecision: .float32,
                isSequence: true,
                embeddingScale: nil
            )
        let options = MTLCompileOptions()
        options.languageVersion = .version4_0
        let library = try device.makeLibrary(source: source, options: options)
        let function = try #require(library.makeFunction(name: kernelName))
        let pipeline = try device.makeComputePipelineState(function: function)

        let embeddingDim = 128
        let tokens = [Int32(2), Int32(0), Int32(1)]
        let rows = makeQ3Rows(
            tokenCount: 3,
            embeddingDim: embeddingDim,
            groupSize: format.groupSize
        )
        let table = rows.flatMap(\.bytes)

        let tokenBuffer = try #require(device.makeBuffer(
            bytes: tokens,
            length: tokens.count * MemoryLayout<Int32>.stride,
            options: .storageModeShared
        ))
        let tableBuffer = try #require(device.makeBuffer(
            bytes: table,
            length: table.count,
            options: .storageModeShared
        ))
        let outputBuffer = try #require(device.makeBuffer(
            length: tokens.count * embeddingDim * MemoryLayout<Float>.stride,
            options: .storageModeShared
        ))
        memset(outputBuffer.contents(), 0, outputBuffer.length)

        var embeddingDimValue = UInt32(embeddingDim)
        var sequenceLength = UInt32(tokens.count)

        let queue = try #require(device.makeCommandQueue())
        let commandBuffer = try #require(queue.makeCommandBuffer())
        let encoder = try #require(commandBuffer.makeComputeCommandEncoder())
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(tokenBuffer, offset: 0, index: 0)
        encoder.setBuffer(tableBuffer, offset: 0, index: 1)
        encoder.setBuffer(outputBuffer, offset: 0, index: 2)
        encoder.setBytes(&embeddingDimValue, length: MemoryLayout<UInt32>.stride, index: 3)
        encoder.setBytes(&sequenceLength, length: MemoryLayout<UInt32>.stride, index: 4)
        encoder.dispatchThreads(
            MTLSize(width: embeddingDim, height: tokens.count, depth: 1),
            threadsPerThreadgroup: MTLSize(
                width: min(embeddingDim, pipeline.maxTotalThreadsPerThreadgroup),
                height: 1,
                depth: 1
            )
        )
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        if let error = commandBuffer.error {
            throw error
        }

        let pointer = outputBuffer.contents()
            .bindMemory(to: Float.self, capacity: tokens.count * embeddingDim)
        for seq in tokens.indices {
            let rowIndex = Int(tokens[seq])
            for dim in 0..<embeddingDim {
                let actual = pointer[seq * embeddingDim + dim]
                let expected = rows[rowIndex].values[dim]
                #expect(
                    abs(actual - expected) < 0.01,
                    "Q3 embedding mismatch seq=\(seq) dim=\(dim): actual=\(actual) expected=\(expected)"
                )
            }
        }
    }

    private func executeQ4EmbeddingLookup(scale: Float?) throws -> [Float] {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw MetalCompilerError.deviceSetupFailed("No Metal device")
        }

        let kernelName = scale == nil ? "test_embedding_lookup_q4" : "test_embedding_lookup_q4_scaled"
        let source = MetalSourceGenerator.commonHeader + "\n" + MetalSourceGenerator.generateQuantizedEmbeddingLookupQ4(
            name: kernelName,
            bufferPrecision: .float32,
            groupSize: 64,
            isSequence: true,
            embeddingScale: scale
        )
        let options = MTLCompileOptions()
        options.languageVersion = .version4_0
        let library = try device.makeLibrary(source: source, options: options)
        let function = try #require(library.makeFunction(name: kernelName))
        let pipeline = try device.makeComputePipelineState(function: function)

        var tokenIDs: [Int32] = [1]
        let tokenBuffer = try #require(
            device.makeBuffer(
                bytes: &tokenIDs,
                length: tokenIDs.count * MemoryLayout<Int32>.stride,
                options: .storageModeShared
            )
        )
        var table = makeQ4Table(tokens: [1, 2], groupSize: 64)
        let tableBuffer = try #require(
            device.makeBuffer(
                bytes: &table,
                length: table.count,
                options: .storageModeShared
            )
        )
        let outputBuffer = try #require(
            device.makeBuffer(
                length: 64 * MemoryLayout<Float>.stride,
                options: .storageModeShared
            )
        )
        memset(outputBuffer.contents(), 0, outputBuffer.length)

        var embeddingDim: UInt32 = 64
        var sequenceLength: UInt32 = 1
        var embeddingScale = scale ?? 1

        let queue = try #require(device.makeCommandQueue())
        let commandBuffer = try #require(queue.makeCommandBuffer())
        let encoder = try #require(commandBuffer.makeComputeCommandEncoder())
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(tokenBuffer, offset: 0, index: 0)
        encoder.setBuffer(tableBuffer, offset: 0, index: 1)
        encoder.setBuffer(outputBuffer, offset: 0, index: 2)
        encoder.setBytes(&embeddingDim, length: MemoryLayout<UInt32>.stride, index: 3)
        encoder.setBytes(&sequenceLength, length: MemoryLayout<UInt32>.stride, index: 4)
        if scale != nil {
            encoder.setBytes(&embeddingScale, length: MemoryLayout<Float>.stride, index: 5)
        }

        let threadCount = min(64, pipeline.maxTotalThreadsPerThreadgroup)
        encoder.dispatchThreadgroups(
            MTLSize(width: 1, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: threadCount, height: 1, depth: 1)
        )
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        if let error = commandBuffer.error {
            throw error
        }

        let pointer = outputBuffer.contents().bindMemory(to: Float.self, capacity: 64)
        return Array(UnsafeBufferPointer(start: pointer, count: 64))
    }

    private func makeQ4Table(tokens: [UInt8], groupSize: Int) -> [UInt8] {
        precondition(groupSize == 64)
        return tokens.flatMap { value in
            let scale = Float16(1).bitPattern
            let zero = Float16(0).bitPattern
            let packed = value | (value << 4)
            var row = [UInt8]()
            row.reserveCapacity(4 + groupSize / 2)
            row.append(UInt8(scale & 0x00FF))
            row.append(UInt8((scale >> 8) & 0x00FF))
            row.append(UInt8(zero & 0x00FF))
            row.append(UInt8((zero >> 8) & 0x00FF))
            row.append(contentsOf: repeatElement(packed, count: groupSize / 2))
            return row
        }
    }

    private func makeQ3Rows(
        tokenCount: Int,
        embeddingDim: Int,
        groupSize: Int
    ) -> [(bytes: [UInt8], values: [Float])] {
        precondition(groupSize == 64)
        precondition(embeddingDim % groupSize == 0)
        return (0..<tokenCount).map { token in
            var rowBytes: [UInt8] = []
            var values: [Float] = []
            for group in 0..<(embeddingDim / groupSize) {
                let scale = Float(0.03125 * Float(group + 1) + 0.0078125 * Float(token))
                let zero = Float(-0.125 + 0.0625 * Float(token) - 0.015625 * Float(group))
                let weights = (0..<groupSize).map { UInt32(($0 + group * 3 + token * 5) % 8) }
                rowBytes.append(contentsOf: makeQuantizedBlock(
                    weights: weights,
                    bits: 3,
                    scale: scale,
                    zero: zero,
                    payloadByteCount: 24
                ))
                values.append(contentsOf: weights.map { scale * Float($0) + zero })
            }
            return (rowBytes, values)
        }
    }

    private func makeQuantizedBlock(
        weights: [UInt32],
        bits: Int,
        scale: Float,
        zero: Float,
        payloadByteCount: Int
    ) -> [UInt8] {
        var bytes = [UInt8]()
        bytes.reserveCapacity(4 + payloadByteCount)
        let scaleBits = Float16(scale).bitPattern
        let zeroBits = Float16(zero).bitPattern
        bytes.append(UInt8(scaleBits & 0x00FF))
        bytes.append(UInt8((scaleBits >> 8) & 0x00FF))
        bytes.append(UInt8(zeroBits & 0x00FF))
        bytes.append(UInt8((zeroBits >> 8) & 0x00FF))
        bytes.append(contentsOf: packLSBFirstBitStream(weights: weights, bits: bits))
        precondition(bytes.count == 4 + payloadByteCount)
        return bytes
    }

    private func packLSBFirstBitStream(weights: [UInt32], bits: Int) -> [UInt8] {
        let totalBits = weights.count * bits
        let byteCount = (totalBits + 7) / 8
        var result = [UInt8](repeating: 0, count: byteCount)
        let mask = (UInt64(1) << bits) - 1
        for (index, weight) in weights.enumerated() {
            let value = UInt64(weight) & mask
            let bitOffset = index * bits
            let byteIndex = bitOffset / 8
            let bitIndex = bitOffset % 8
            let shifted = value << bitIndex
            let spannedBytes = (bitIndex + bits + 7) / 8
            for offset in 0..<spannedBytes {
                result[byteIndex + offset] |= UInt8((shifted >> (offset * 8)) & 0xFF)
            }
        }
        return result
    }
}
