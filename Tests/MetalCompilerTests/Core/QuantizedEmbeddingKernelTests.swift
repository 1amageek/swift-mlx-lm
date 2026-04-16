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
            weightFormat: .quantized4Bit(groupSize: 64)
        )

        let name = fragment.kernelName(context: context)
        #expect(name == "embedding_lookup_seq_q4_g64_f32")

        let source = fragment.kernelSource(
            name: name,
            bufferPrecision: .float32,
            weightFormat: .quantized4Bit(groupSize: 64)
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
            weightFormat: .quantized8Bit(groupSize: 32)
        )

        let name = fragment.kernelName(context: context)
        #expect(name == "embedding_lookup_q8_g32")

        let source = fragment.kernelSource(
            name: name,
            bufferPrecision: .float16,
            weightFormat: .quantized8Bit(groupSize: 32)
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
}
