import Metal
import Testing
@testable import MetalCompiler

@Suite("Quantized Q8 Correctness", .serialized)
struct QuantizedQ8CorrectnessTests {
    @Test("Q8 KV quantize/dequantize preserves unsigned payload above 127")
    func q8KVQuantizeDequantizePreservesUnsignedPayloadAbove127() throws {
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }
        guard let device = MTLCreateSystemDefaultDevice() else { return }

        let source = MetalSourceGenerator.commonHeader + "\n\n" + MetalSourceGenerator.kvQuantizationSource
        let quantize = try makePipeline(device: device, source: source, functionName: "quantize_kv_q8")
        let dequantize = try makePipeline(device: device, source: source, functionName: "dequantize_kv_q8")

        let groupSize = 32
        let bytesPerBlock = 4 + groupSize
        let inputValues: [Float16] = (0..<groupSize).map { index in
            Float16(Float(index) * 8.0)
        }
        let inputBuffer = try #require(device.makeBuffer(
            bytes: inputValues,
            length: inputValues.count * MemoryLayout<Float16>.stride,
            options: .storageModeShared
        ))
        let quantizedBuffer = try #require(device.makeBuffer(
            length: bytesPerBlock,
            options: .storageModeShared
        ))
        let outputBuffer = try #require(device.makeBuffer(
            length: inputValues.count * MemoryLayout<Float16>.stride,
            options: .storageModeShared
        ))

        let queue = try #require(device.makeCommandQueue())
        try run(
            queue: queue,
            pipeline: quantize,
            threadCount: 1
        ) { encoder in
            encoder.setBuffer(inputBuffer, offset: 0, index: 0)
            encoder.setBuffer(quantizedBuffer, offset: 0, index: 1)
            var totalElements = UInt32(groupSize)
            var groupSizeValue = UInt32(groupSize)
            var bytesPerBlockValue = UInt32(bytesPerBlock)
            encoder.setBytes(&totalElements, length: MemoryLayout<UInt32>.stride, index: 2)
            encoder.setBytes(&groupSizeValue, length: MemoryLayout<UInt32>.stride, index: 3)
            encoder.setBytes(&bytesPerBlockValue, length: MemoryLayout<UInt32>.stride, index: 4)
        }
        try run(
            queue: queue,
            pipeline: dequantize,
            threadCount: 1
        ) { encoder in
            encoder.setBuffer(quantizedBuffer, offset: 0, index: 0)
            encoder.setBuffer(outputBuffer, offset: 0, index: 1)
            var totalElements = UInt32(groupSize)
            var groupSizeValue = UInt32(groupSize)
            var bytesPerBlockValue = UInt32(bytesPerBlock)
            encoder.setBytes(&totalElements, length: MemoryLayout<UInt32>.stride, index: 2)
            encoder.setBytes(&groupSizeValue, length: MemoryLayout<UInt32>.stride, index: 3)
            encoder.setBytes(&bytesPerBlockValue, length: MemoryLayout<UInt32>.stride, index: 4)
        }

        let outputPointer = outputBuffer.contents().bindMemory(to: Float16.self, capacity: groupSize)
        let decoded = (0..<groupSize).map { Float(outputPointer[$0]) }
        let expected = inputValues.map(Float.init)
        let maxError = zip(decoded, expected).map { abs($0 - $1) }.max() ?? .zero
        let tail = decoded.suffix(8).map { String(format: "%.1f", $0) }.joined(separator: ", ")

        #expect(
            maxError < 1.5,
            """
            Q8 KV quantize/dequantize corrupted unsigned payloads
            maxError=\(maxError)
            decoded tail=\(tail)
            expected tail=\(expected.suffix(8).map { String(format: "%.1f", $0) }.joined(separator: ", "))
            """
        )
    }

    @Test("flash attention helper reads Q8 payload as unsigned bytes")
    func flashAttentionHelperReadsQ8PayloadAsUnsignedBytes() throws {
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }
        guard let device = MTLCreateSystemDefaultDevice() else { return }

        let source = MetalSourceGenerator.commonHeader + "\n\n"
            + MetalSourceGenerator.flashAttentionHelperSource + "\n\n"
            + """
            kernel void test_read_q8(
                device const uchar* cache [[buffer(0)]],
                device float* output [[buffer(1)]],
                constant uint& headDim [[buffer(2)]],
                constant uint& headSlotBytes [[buffer(3)]],
                constant uint& scheme [[buffer(4)]],
                uint gid [[thread_position_in_grid]]
            ) {
                if (gid >= headDim) return;
                output[gid] = read_kv_element(cache, gid, scheme, headSlotBytes, headDim);
            }
            """
        let pipeline = try makePipeline(device: device, source: source, functionName: "test_read_q8")

        let groupSize = 32
        let bytesPerBlock = 4 + groupSize
        var block = [UInt8](repeating: 0, count: bytesPerBlock)
        let scale = Float16(1.0).bitPattern
        let zero = Float16(0.0).bitPattern
        block[0] = UInt8(scale & 0xFF)
        block[1] = UInt8(scale >> 8)
        block[2] = UInt8(zero & 0xFF)
        block[3] = UInt8(zero >> 8)
        for index in 0..<groupSize {
            block[4 + index] = UInt8(200 + index)
        }

        let cacheBuffer = try #require(device.makeBuffer(
            bytes: block,
            length: block.count,
            options: .storageModeShared
        ))
        let outputBuffer = try #require(device.makeBuffer(
            length: groupSize * MemoryLayout<Float>.stride,
            options: .storageModeShared
        ))

        let queue = try #require(device.makeCommandQueue())
        try run(
            queue: queue,
            pipeline: pipeline,
            threadCount: groupSize
        ) { encoder in
            encoder.setBuffer(cacheBuffer, offset: 0, index: 0)
            encoder.setBuffer(outputBuffer, offset: 0, index: 1)
            var headDim = UInt32(groupSize)
            var headSlotBytes = UInt32(bytesPerBlock)
            var scheme = UInt32(QuantizationSchemeIdentifier.q8Group32ScaleF16.rawValue)
            encoder.setBytes(&headDim, length: MemoryLayout<UInt32>.stride, index: 2)
            encoder.setBytes(&headSlotBytes, length: MemoryLayout<UInt32>.stride, index: 3)
            encoder.setBytes(&scheme, length: MemoryLayout<UInt32>.stride, index: 4)
        }

        let outputPointer = outputBuffer.contents().bindMemory(to: Float.self, capacity: groupSize)
        let decoded = (0..<groupSize).map { outputPointer[$0] }
        let expected = (0..<groupSize).map { Float(200 + $0) }
        let maxError = zip(decoded, expected).map { abs($0 - $1) }.max() ?? .zero

        #expect(
            maxError < 0.01,
            """
            flash attention helper decoded Q8 payload as signed bytes
            maxError=\(maxError)
            decoded head=\(decoded.prefix(8).map { String(format: "%.1f", $0) }.joined(separator: ", "))
            expected head=\(expected.prefix(8).map { String(format: "%.1f", $0) }.joined(separator: ", "))
            """
        )
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

    private func run(
        queue: MTLCommandQueue,
        pipeline: MTLComputePipelineState,
        threadCount: Int,
        encode: (MTLComputeCommandEncoder) throws -> Void
    ) throws {
        let commandBuffer = try #require(queue.makeCommandBuffer())
        let encoder = try #require(commandBuffer.makeComputeCommandEncoder())
        encoder.setComputePipelineState(pipeline)
        try encode(encoder)
        let threads = MTLSize(width: max(threadCount, 1), height: 1, depth: 1)
        encoder.dispatchThreads(
            threads,
            threadsPerThreadgroup: MTLSize(width: min(max(threadCount, 1), pipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1)
        )
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
    }
}
