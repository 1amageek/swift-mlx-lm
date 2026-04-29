import Metal
import Testing
@testable import MetalCompiler

@Suite("Attention Sequence Equivalence", .serialized)
struct AttentionSequenceEquivalenceTests {
    @Test("BF16 KV cache fill and batch attention match repeated decode attention")
    func bf16KVCacheFillAndBatchAttentionMatchRepeatedDecodeAttention() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device")
            return
        }
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }

        let harness = try SequenceKernelEquivalenceHarness(device: device)
        let decodePipeline = try harness.pipeline(named: "flash_attn_decode_f32")
        let fillPipeline = try harness.pipeline(named: "kv_cache_fill_seq_f32")
        let batchPipeline = try harness.pipeline(named: "flash_attn_batch_bf16_f32")

        let headCount = 8
        let kvHeadCount = 2
        let headDimension = 256
        let maximumSequenceLength = 8
        let sequenceLength = 5
        let headSlotBytes = headDimension * MemoryLayout<BFloat16>.stride
        let queryValues = roundedBFloat16Values(
            count: sequenceLength * headCount * headDimension,
            multiplier: 17,
            modulus: 29,
            scale: 0.03125
        )
        let keyValues = roundedBFloat16Values(
            count: sequenceLength * kvHeadCount * headDimension,
            multiplier: 13,
            modulus: 23,
            scale: 0.03125
        )
        let valueValues = roundedBFloat16Values(
            count: sequenceLength * kvHeadCount * headDimension,
            multiplier: 19,
            modulus: 31,
            scale: 0.03125
        )

        let decode = try runDecodeAttentionTrace(
            harness: harness,
            pipeline: decodePipeline,
            queryValues: queryValues,
            keyValues: keyValues,
            valueValues: valueValues,
            headCount: headCount,
            kvHeadCount: kvHeadCount,
            headDimension: headDimension,
            maximumSequenceLength: maximumSequenceLength,
            sequenceLength: sequenceLength,
            headSlotBytes: headSlotBytes
        )
        let sequence = try runSequenceAttentionTrace(
            harness: harness,
            fillPipeline: fillPipeline,
            batchPipeline: batchPipeline,
            queryValues: queryValues,
            keyValues: keyValues,
            valueValues: valueValues,
            headCount: headCount,
            kvHeadCount: kvHeadCount,
            headDimension: headDimension,
            maximumSequenceLength: maximumSequenceLength,
            sequenceLength: sequenceLength,
            headSlotBytes: headSlotBytes
        )

        let roundedDecodeOutput = decode.output.map { Float(BFloat16($0)) }
        let outputMismatch = harness.firstMismatch(
            expected: roundedDecodeOutput,
            actual: sequence.output,
            tolerance: 0.000_001
        )
        #expect(
            outputMismatch == nil,
            "attention output drifted: \(String(describing: outputMismatch)), maxError=\(harness.maxAbsoluteError(expected: roundedDecodeOutput, actual: sequence.output))"
        )
        #expect(
            decode.keyCacheBytes == sequence.keyCacheBytes,
            "key cache drifted between decode and sequence fill"
        )
        #expect(
            decode.valueCacheBytes == sequence.valueCacheBytes,
            "value cache drifted between decode and sequence fill"
        )
    }

    private func roundedBFloat16Values(
        count: Int,
        multiplier: Int,
        modulus: Int,
        scale: Float
    ) -> [Float] {
        (0..<count).map { index in
            Float(BFloat16(Float((index * multiplier) % modulus - modulus / 2) * scale))
        }
    }

    private func runDecodeAttentionTrace(
        harness: SequenceKernelEquivalenceHarness,
        pipeline: MTLComputePipelineState,
        queryValues: [Float],
        keyValues: [Float],
        valueValues: [Float],
        headCount: Int,
        kvHeadCount: Int,
        headDimension: Int,
        maximumSequenceLength: Int,
        sequenceLength: Int,
        headSlotBytes: Int
    ) throws -> (output: [Float], keyCacheBytes: [UInt8], valueCacheBytes: [UInt8]) {
        let cacheByteLength = kvHeadCount * maximumSequenceLength * headSlotBytes
        let keyCache = try harness.makeZeroedSharedBuffer(byteLength: cacheByteLength)
        let valueCache = try harness.makeZeroedSharedBuffer(byteLength: cacheByteLength)
        let positionBuffer = try harness.makeZeroedSharedBuffer(byteLength: MemoryLayout<UInt32>.stride)
        let placeholder = try harness.makeZeroedSharedBuffer(byteLength: 4)
        var trace = [Float](repeating: .zero, count: sequenceLength * headCount * headDimension)
        let threads = min(
            max(headDimension, 32),
            pipeline.maxTotalThreadsPerThreadgroup
        )
        let threadgroup = MTLSize(width: threads, height: 1, depth: 1)
        let grid = MTLSize(width: headCount, height: 1, depth: 1)

        for position in 0..<sequenceLength {
            positionBuffer.contents().storeBytes(of: UInt32(position), as: UInt32.self)
            let queryOffset = position * headCount * headDimension
            let kvOffset = position * kvHeadCount * headDimension
            let queryBuffer = try harness.makeSharedBuffer(
                values: Array(queryValues[queryOffset..<(queryOffset + headCount * headDimension)])
            )
            let keyBuffer = try harness.makeSharedBuffer(
                values: Array(keyValues[kvOffset..<(kvOffset + kvHeadCount * headDimension)])
            )
            let valueBuffer = try harness.makeSharedBuffer(
                values: Array(valueValues[kvOffset..<(kvOffset + kvHeadCount * headDimension)])
            )
            let outputBuffer = try harness.makeZeroedSharedBuffer(
                byteLength: headCount * headDimension * MemoryLayout<Float>.stride
            )

            let (commandBuffer, encoder) = try harness.makeCommandEncoder()
            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(queryBuffer, offset: 0, index: 0)
            encoder.setBuffer(keyBuffer, offset: 0, index: 1)
            encoder.setBuffer(valueBuffer, offset: 0, index: 2)
            encoder.setBuffer(keyCache, offset: 0, index: 3)
            encoder.setBuffer(valueCache, offset: 0, index: 4)
            encoder.setBuffer(outputBuffer, offset: 0, index: 5)
            encoder.setBuffer(positionBuffer, offset: 0, index: 6)
            encoder.setBuffer(placeholder, offset: 0, index: 17)
            encoder.setBuffer(placeholder, offset: 0, index: 18)
            encoder.setBuffer(placeholder, offset: 0, index: 19)
            setDecodeAttentionConstants(
                encoder: encoder,
                headCount: headCount,
                kvHeadCount: kvHeadCount,
                headDimension: headDimension,
                maximumSequenceLength: maximumSequenceLength,
                headSlotBytes: headSlotBytes
            )
            encoder.dispatchThreadgroups(grid, threadsPerThreadgroup: threadgroup)
            encoder.endEncoding()
            try harness.complete(commandBuffer)

            let output = harness.readFloat32(outputBuffer, count: headCount * headDimension)
            trace.replaceSubrange(
                queryOffset..<(queryOffset + headCount * headDimension),
                with: output
            )
        }

        return (
            output: trace,
            keyCacheBytes: readBytes(keyCache, count: cacheByteLength),
            valueCacheBytes: readBytes(valueCache, count: cacheByteLength)
        )
    }

    private func runSequenceAttentionTrace(
        harness: SequenceKernelEquivalenceHarness,
        fillPipeline: MTLComputePipelineState,
        batchPipeline: MTLComputePipelineState,
        queryValues: [Float],
        keyValues: [Float],
        valueValues: [Float],
        headCount: Int,
        kvHeadCount: Int,
        headDimension: Int,
        maximumSequenceLength: Int,
        sequenceLength: Int,
        headSlotBytes: Int
    ) throws -> (output: [Float], keyCacheBytes: [UInt8], valueCacheBytes: [UInt8]) {
        let cacheByteLength = kvHeadCount * maximumSequenceLength * headSlotBytes
        let queryBuffer = try harness.makeSharedBuffer(values: queryValues)
        let keyBuffer = try harness.makeSharedBuffer(values: keyValues)
        let valueBuffer = try harness.makeSharedBuffer(values: valueValues)
        let keyCache = try harness.makeZeroedSharedBuffer(byteLength: cacheByteLength)
        let valueCache = try harness.makeZeroedSharedBuffer(byteLength: cacheByteLength)
        let outputBuffer = try harness.makeZeroedSharedBuffer(
            byteLength: sequenceLength * headCount * headDimension * MemoryLayout<Float>.stride
        )
        let placeholder = try harness.makeZeroedSharedBuffer(byteLength: 4)

        let fillThreads = min(max(headDimension, 32), fillPipeline.maxTotalThreadsPerThreadgroup)
        let (fillCommandBuffer, fillEncoder) = try harness.makeCommandEncoder()
        fillEncoder.setComputePipelineState(fillPipeline)
        fillEncoder.setBuffer(keyBuffer, offset: 0, index: 0)
        fillEncoder.setBuffer(valueBuffer, offset: 0, index: 1)
        fillEncoder.setBuffer(keyCache, offset: 0, index: 2)
        fillEncoder.setBuffer(valueCache, offset: 0, index: 3)
        fillEncoder.setBuffer(placeholder, offset: 0, index: 13)
        fillEncoder.setBuffer(placeholder, offset: 0, index: 14)
        fillEncoder.setBuffer(placeholder, offset: 0, index: 15)
        setKVCacheFillConstants(
            encoder: fillEncoder,
            kvHeadCount: kvHeadCount,
            headDimension: headDimension,
            maximumSequenceLength: maximumSequenceLength,
            sequenceLength: sequenceLength,
            headSlotBytes: headSlotBytes
        )
        fillEncoder.dispatchThreadgroups(
            MTLSize(width: sequenceLength, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: fillThreads, height: 1, depth: 1)
        )
        fillEncoder.endEncoding()
        try harness.complete(fillCommandBuffer)

        let batchThreads = min(max(headDimension, 32), batchPipeline.maxTotalThreadsPerThreadgroup)
        let (batchCommandBuffer, batchEncoder) = try harness.makeCommandEncoder()
        batchEncoder.setComputePipelineState(batchPipeline)
        batchEncoder.setBuffer(queryBuffer, offset: 0, index: 0)
        batchEncoder.setBuffer(keyCache, offset: 0, index: 1)
        batchEncoder.setBuffer(valueCache, offset: 0, index: 2)
        batchEncoder.setBuffer(outputBuffer, offset: 0, index: 3)
        batchEncoder.setBuffer(placeholder, offset: 0, index: 15)
        batchEncoder.setBuffer(placeholder, offset: 0, index: 16)
        batchEncoder.setBuffer(placeholder, offset: 0, index: 17)
        batchEncoder.setBuffer(keyBuffer, offset: 0, index: 23)
        batchEncoder.setBuffer(valueBuffer, offset: 0, index: 24)
        setBatchAttentionConstants(
            encoder: batchEncoder,
            headCount: headCount,
            kvHeadCount: kvHeadCount,
            headDimension: headDimension,
            maximumSequenceLength: maximumSequenceLength,
            sequenceLength: sequenceLength,
            headSlotBytes: headSlotBytes
        )
        batchEncoder.dispatchThreadgroups(
            MTLSize(width: sequenceLength * headCount, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: batchThreads, height: 1, depth: 1)
        )
        batchEncoder.endEncoding()
        try harness.complete(batchCommandBuffer)

        return (
            output: harness.readFloat32(
                outputBuffer,
                count: sequenceLength * headCount * headDimension
            ),
            keyCacheBytes: readBytes(keyCache, count: cacheByteLength),
            valueCacheBytes: readBytes(valueCache, count: cacheByteLength)
        )
    }

    private func setDecodeAttentionConstants(
        encoder: MTLComputeCommandEncoder,
        headCount: Int,
        kvHeadCount: Int,
        headDimension: Int,
        maximumSequenceLength: Int,
        headSlotBytes: Int
    ) {
        var heads = UInt32(headCount)
        var kvHeads = UInt32(kvHeadCount)
        var headDim = UInt32(headDimension)
        var scale = 1.0 / Float(headDimension).squareRoot()
        var layout = UInt32(KVCacheLayoutMode.sequenceMajor.rawValue)
        var maxSeq = UInt32(maximumSequenceLength)
        var scheme = UInt32(QuantizationSchemeIdentifier.bf16RowMajor.rawValue)
        var slotBytes = UInt32(headSlotBytes)
        var zero = UInt32(0)
        var flags = UInt32(0)
        var windowLeft = UInt32.max
        encoder.setBytes(&heads, length: MemoryLayout<UInt32>.stride, index: 7)
        encoder.setBytes(&kvHeads, length: MemoryLayout<UInt32>.stride, index: 8)
        encoder.setBytes(&headDim, length: MemoryLayout<UInt32>.stride, index: 9)
        encoder.setBytes(&scale, length: MemoryLayout<Float>.stride, index: 10)
        encoder.setBytes(&layout, length: MemoryLayout<UInt32>.stride, index: 11)
        encoder.setBytes(&maxSeq, length: MemoryLayout<UInt32>.stride, index: 12)
        encoder.setBytes(&scheme, length: MemoryLayout<UInt32>.stride, index: 13)
        encoder.setBytes(&scheme, length: MemoryLayout<UInt32>.stride, index: 14)
        encoder.setBytes(&slotBytes, length: MemoryLayout<UInt32>.stride, index: 15)
        encoder.setBytes(&slotBytes, length: MemoryLayout<UInt32>.stride, index: 16)
        encoder.setBytes(&zero, length: MemoryLayout<UInt32>.stride, index: 20)
        encoder.setBytes(&zero, length: MemoryLayout<UInt32>.stride, index: 21)
        encoder.setBytes(&flags, length: MemoryLayout<UInt32>.stride, index: 29)
        encoder.setBytes(&windowLeft, length: MemoryLayout<UInt32>.stride, index: 30)
    }

    private func setKVCacheFillConstants(
        encoder: MTLComputeCommandEncoder,
        kvHeadCount: Int,
        headDimension: Int,
        maximumSequenceLength: Int,
        sequenceLength: Int,
        headSlotBytes: Int
    ) {
        var kvHeads = UInt32(kvHeadCount)
        var headDim = UInt32(headDimension)
        var maxSeq = UInt32(maximumSequenceLength)
        var seqLen = UInt32(sequenceLength)
        var layout = UInt32(KVCacheLayoutMode.sequenceMajor.rawValue)
        var scheme = UInt32(QuantizationSchemeIdentifier.bf16RowMajor.rawValue)
        var slotBytes = UInt32(headSlotBytes)
        var zero = UInt32(0)
        encoder.setBytes(&kvHeads, length: MemoryLayout<UInt32>.stride, index: 4)
        encoder.setBytes(&headDim, length: MemoryLayout<UInt32>.stride, index: 5)
        encoder.setBytes(&maxSeq, length: MemoryLayout<UInt32>.stride, index: 6)
        encoder.setBytes(&seqLen, length: MemoryLayout<UInt32>.stride, index: 7)
        encoder.setBytes(&layout, length: MemoryLayout<UInt32>.stride, index: 8)
        encoder.setBytes(&scheme, length: MemoryLayout<UInt32>.stride, index: 9)
        encoder.setBytes(&scheme, length: MemoryLayout<UInt32>.stride, index: 10)
        encoder.setBytes(&slotBytes, length: MemoryLayout<UInt32>.stride, index: 11)
        encoder.setBytes(&slotBytes, length: MemoryLayout<UInt32>.stride, index: 12)
        encoder.setBytes(&zero, length: MemoryLayout<UInt32>.stride, index: 16)
        encoder.setBytes(&zero, length: MemoryLayout<UInt32>.stride, index: 17)
    }

    private func setBatchAttentionConstants(
        encoder: MTLComputeCommandEncoder,
        headCount: Int,
        kvHeadCount: Int,
        headDimension: Int,
        maximumSequenceLength: Int,
        sequenceLength: Int,
        headSlotBytes: Int
    ) {
        var heads = UInt32(headCount)
        var kvHeads = UInt32(kvHeadCount)
        var headDim = UInt32(headDimension)
        var scale = 1.0 / Float(headDimension).squareRoot()
        var layout = UInt32(KVCacheLayoutMode.sequenceMajor.rawValue)
        var maxSeq = UInt32(maximumSequenceLength)
        var seqLen = UInt32(sequenceLength)
        var scheme = UInt32(QuantizationSchemeIdentifier.bf16RowMajor.rawValue)
        var slotBytes = UInt32(headSlotBytes)
        var zero = UInt32(0)
        var causal = UInt32(1)
        var window = UInt32.max
        encoder.setBytes(&heads, length: MemoryLayout<UInt32>.stride, index: 4)
        encoder.setBytes(&kvHeads, length: MemoryLayout<UInt32>.stride, index: 5)
        encoder.setBytes(&headDim, length: MemoryLayout<UInt32>.stride, index: 6)
        encoder.setBytes(&scale, length: MemoryLayout<Float>.stride, index: 7)
        encoder.setBytes(&layout, length: MemoryLayout<UInt32>.stride, index: 8)
        encoder.setBytes(&maxSeq, length: MemoryLayout<UInt32>.stride, index: 9)
        encoder.setBytes(&seqLen, length: MemoryLayout<UInt32>.stride, index: 10)
        encoder.setBytes(&scheme, length: MemoryLayout<UInt32>.stride, index: 11)
        encoder.setBytes(&scheme, length: MemoryLayout<UInt32>.stride, index: 12)
        encoder.setBytes(&slotBytes, length: MemoryLayout<UInt32>.stride, index: 13)
        encoder.setBytes(&slotBytes, length: MemoryLayout<UInt32>.stride, index: 14)
        encoder.setBytes(&zero, length: MemoryLayout<UInt32>.stride, index: 18)
        encoder.setBytes(&zero, length: MemoryLayout<UInt32>.stride, index: 19)
        encoder.setBytes(&causal, length: MemoryLayout<UInt32>.stride, index: 20)
        encoder.setBytes(&window, length: MemoryLayout<UInt32>.stride, index: 21)
        encoder.setBytes(&window, length: MemoryLayout<UInt32>.stride, index: 22)
    }

    private func readBytes(_ buffer: MTLBuffer, count: Int) -> [UInt8] {
        let pointer = buffer.contents().bindMemory(to: UInt8.self, capacity: count)
        return Array(UnsafeBufferPointer(start: pointer, count: count))
    }
}
