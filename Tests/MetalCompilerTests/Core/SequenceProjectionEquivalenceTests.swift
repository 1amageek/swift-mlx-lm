import Metal
import Testing
@testable import MetalCompiler

@Suite("Sequence Projection Equivalence", .serialized)
struct SequenceProjectionEquivalenceTests {
    @Test("BF16 batched sequence GEMV matches decode GEMV with padded scratch slots")
    func bf16BatchedSequenceGEMVMatchesDecodeGEMVWithPaddedScratchSlots() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device")
            return
        }
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }

        let inputDimension = 1024
        let slotDimension = 6144
        let sequenceLength = 5
        let outputDimensions = [6144, 2048, 16, 16]
        let decodeKernelName = "batched_gemv4_bf16_padded_decode_equivalence"
        let sequenceKernelName = "batched_gemv4_bf16_padded_sequence_equivalence"
        let source = [
            MetalSourceGenerator.commonHeader,
            MetalSourceGenerator.generateBatchedGEMV4(
                name: decodeKernelName,
                bufferPrecision: BufferPrecision.bfloat16,
                weightFormat: WeightFormats.bfloat16
            ),
            MetalSourceGenerator.generateBatchedSequenceGEMV(
                name: sequenceKernelName,
                count: outputDimensions.count,
                bufferPrecision: BufferPrecision.float32,
                weightFormat: WeightFormats.bfloat16
            ),
        ].joined(separator: "\n")
        let harness = try SequenceKernelEquivalenceHarness(device: device, source: source)
        let decodePipeline = try harness.pipeline(named: decodeKernelName)
        let sequencePipeline = try harness.pipeline(named: sequenceKernelName)

        let packedInput = (0..<(sequenceLength * inputDimension)).map { index in
            Float(BFloat16(Float((index * 37) % 29 - 14) * 0.03125))
        }
        let paddedInput = paddedRows(
            packedInput,
            rowCount: sequenceLength,
            logicalWidth: inputDimension,
            rowStride: slotDimension
        )
        let weights = outputDimensions.enumerated().map { projection, outputDimension in
            (0..<(outputDimension * inputDimension)).map { index in
                BFloat16(Float((index * (projection + 3) + projection * 11) % 31 - 15) * 0.015625)
            }
        }

        let expected = try runDecodeProjectionTraceInScratch(
            harness: harness,
            pipeline: decodePipeline,
            inputValues: paddedInput,
            weights: weights,
            inputDimension: inputDimension,
            inputRowStride: slotDimension,
            slotDimension: slotDimension,
            sequenceLength: sequenceLength,
            outputDimensions: outputDimensions
        )
        let actual = try runSequenceProjectionTraceInScratch(
            harness: harness,
            pipeline: sequencePipeline,
            inputValues: paddedInput,
            weights: weights,
            inputDimension: inputDimension,
            inputRowStride: slotDimension,
            outputRowStride: slotDimension,
            slotDimension: slotDimension,
            sequenceLength: sequenceLength,
            outputDimensions: outputDimensions
        )

        for projection in outputDimensions.indices {
            let mismatch = harness.firstMismatch(
                expected: expected[projection],
                actual: actual[projection],
                tolerance: 0.000_001
            )
            #expect(
                mismatch == nil,
                "padded projection \(projection) drifted: \(String(describing: mismatch)), maxError=\(harness.maxAbsoluteError(expected: expected[projection], actual: actual[projection]))"
            )
        }
    }

    @Test("BF16 batched sequence GEMV matches repeated decode GEMV")
    func bf16BatchedSequenceGEMVMatchesRepeatedDecodeGEMV() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device")
            return
        }
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }

        let inputDimension = 64
        let sequenceLength = 5
        let outputDimensions = [17, 9, 13, 7]
        let decodeKernelName = "batched_gemv4_bf16_decode_equivalence"
        let sequenceKernelName = "batched_gemv4_bf16_sequence_equivalence"
        let source = [
            MetalSourceGenerator.commonHeader,
            MetalSourceGenerator.generateBatchedGEMV4(
                name: decodeKernelName,
                bufferPrecision: BufferPrecision.bfloat16,
                weightFormat: WeightFormats.bfloat16
            ),
            MetalSourceGenerator.generateBatchedSequenceGEMV(
                name: sequenceKernelName,
                count: outputDimensions.count,
                bufferPrecision: BufferPrecision.float32,
                weightFormat: WeightFormats.bfloat16
            ),
        ].joined(separator: "\n")
        let harness = try SequenceKernelEquivalenceHarness(device: device, source: source)
        let decodePipeline = try harness.pipeline(named: decodeKernelName)
        let sequencePipeline = try harness.pipeline(named: sequenceKernelName)

        let inputValues = (0..<(sequenceLength * inputDimension)).map { index in
            Float(BFloat16(Float((index * 37) % 29 - 14) * 0.03125))
        }
        let weights = outputDimensions.enumerated().map { projection, outputDimension in
            (0..<(outputDimension * inputDimension)).map { index in
                BFloat16(Float((index * (projection + 3) + projection * 11) % 31 - 15) * 0.015625)
            }
        }

        let expected = try runDecodeProjectionTrace(
            harness: harness,
            pipeline: decodePipeline,
            inputValues: inputValues,
            weights: weights,
            inputDimension: inputDimension,
            sequenceLength: sequenceLength,
            outputDimensions: outputDimensions
        )
        let actual = try runSequenceProjectionTrace(
            harness: harness,
            pipeline: sequencePipeline,
            inputValues: inputValues,
            weights: weights,
            inputDimension: inputDimension,
            sequenceLength: sequenceLength,
            outputDimensions: outputDimensions
        )

        for projection in outputDimensions.indices {
            let mismatch = harness.firstMismatch(
                expected: expected[projection],
                actual: actual[projection],
                tolerance: 0.000_001
            )
            #expect(
                mismatch == nil,
                "projection \(projection) drifted: \(String(describing: mismatch)), maxError=\(harness.maxAbsoluteError(expected: expected[projection], actual: actual[projection]))"
            )
        }
    }

    private func runDecodeProjectionTraceInScratch(
        harness: SequenceKernelEquivalenceHarness,
        pipeline: MTLComputePipelineState,
        inputValues: [Float],
        weights: [[BFloat16]],
        inputDimension: Int,
        inputRowStride: Int,
        slotDimension: Int,
        sequenceLength: Int,
        outputDimensions: [Int]
    ) throws -> [[Float]] {
        let weightBuffers = try weights.map { try harness.makeSharedBuffer(values: $0) }
        var traces = outputDimensions.map { [Float](repeating: .zero, count: sequenceLength * $0) }
        let simdWidth = 32
        let threads = min(simdWidth * 2, pipeline.maxTotalThreadsPerThreadgroup)
        let rowsPerThreadgroup = max(1, threads / simdWidth)
        let totalRows = outputDimensions.reduce(0, +)
        let grid = MTLSize(
            width: (totalRows + rowsPerThreadgroup - 1) / rowsPerThreadgroup,
            height: 1,
            depth: 1
        )
        let threadgroup = MTLSize(width: threads, height: 1, depth: 1)

        for position in 0..<sequenceLength {
            let inputStart = position * inputRowStride
            let tokenInput = inputValues[inputStart..<(inputStart + inputDimension)].map { BFloat16($0) }
            let scratch = try harness.makeZeroedSharedBuffer(
                byteLength: 5 * slotDimension * MemoryLayout<BFloat16>.stride
            )
            scratch.contents()
                .bindMemory(to: BFloat16.self, capacity: 5 * slotDimension)
                .update(from: Array(tokenInput), count: inputDimension)

            let (commandBuffer, encoder) = try harness.makeCommandEncoder()
            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(scratch, offset: 0, index: 0)
            for index in 0..<4 {
                encoder.setBuffer(weightBuffers[index], offset: 0, index: 1 + index)
                encoder.setBuffer(
                    scratch,
                    offset: (index + 1) * slotDimension * MemoryLayout<BFloat16>.stride,
                    index: 5 + index
                )
            }
            var inputDim = UInt32(inputDimension)
            var outputDim0 = UInt32(outputDimensions[0])
            var outputDim1 = UInt32(outputDimensions[1])
            var outputDim2 = UInt32(outputDimensions[2])
            var outputDim3 = UInt32(outputDimensions[3])
            encoder.setBytes(&inputDim, length: MemoryLayout<UInt32>.stride, index: 9)
            encoder.setBytes(&outputDim0, length: MemoryLayout<UInt32>.stride, index: 10)
            encoder.setBytes(&outputDim1, length: MemoryLayout<UInt32>.stride, index: 11)
            encoder.setBytes(&outputDim2, length: MemoryLayout<UInt32>.stride, index: 12)
            encoder.setBytes(&outputDim3, length: MemoryLayout<UInt32>.stride, index: 13)
            encoder.dispatchThreadgroups(grid, threadsPerThreadgroup: threadgroup)
            encoder.endEncoding()
            try harness.complete(commandBuffer)

            let scratchValues = harness.readBFloat16AsFloat(
                scratch,
                count: 5 * slotDimension
            )
            for projection in outputDimensions.indices {
                let sourceOffset = (projection + 1) * slotDimension
                let destinationOffset = position * outputDimensions[projection]
                traces[projection].replaceSubrange(
                    destinationOffset..<(destinationOffset + outputDimensions[projection]),
                    with: scratchValues[sourceOffset..<(sourceOffset + outputDimensions[projection])]
                )
            }
        }
        return traces
    }

    private func runSequenceProjectionTraceInScratch(
        harness: SequenceKernelEquivalenceHarness,
        pipeline: MTLComputePipelineState,
        inputValues: [Float],
        weights: [[BFloat16]],
        inputDimension: Int,
        inputRowStride: Int,
        outputRowStride: Int,
        slotDimension: Int,
        sequenceLength: Int,
        outputDimensions: [Int]
    ) throws -> [[Float]] {
        let weightBuffers = try weights.map { try harness.makeSharedBuffer(values: $0) }
        let scratch = try harness.makeZeroedSharedBuffer(
            byteLength: 5 * sequenceLength * slotDimension * MemoryLayout<Float>.stride
        )
        scratch.contents()
            .bindMemory(to: Float.self, capacity: 5 * sequenceLength * slotDimension)
            .update(from: inputValues, count: inputValues.count)

        let simdWidth = 32
        let threads = min(simdWidth * 2, pipeline.maxTotalThreadsPerThreadgroup)
        let rowsPerThreadgroup = max(1, threads / simdWidth)
        let totalRows = outputDimensions.reduce(0, +)
        let grid = MTLSize(
            width: (totalRows + rowsPerThreadgroup - 1) / rowsPerThreadgroup,
            height: sequenceLength,
            depth: 1
        )
        let threadgroup = MTLSize(width: threads, height: 1, depth: 1)

        let (commandBuffer, encoder) = try harness.makeCommandEncoder()
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(scratch, offset: 0, index: 0)
        for index in 0..<4 {
            encoder.setBuffer(weightBuffers[index], offset: 0, index: 1 + index)
            encoder.setBuffer(
                scratch,
                offset: (index + 1) * sequenceLength * slotDimension * MemoryLayout<Float>.stride,
                index: 5 + index
            )
        }
        var inputDim = UInt32(inputDimension)
        var outputDim0 = UInt32(outputDimensions[0])
        var outputDim1 = UInt32(outputDimensions[1])
        var outputDim2 = UInt32(outputDimensions[2])
        var outputDim3 = UInt32(outputDimensions[3])
        var seqLen = UInt32(sequenceLength)
        var inputStride = UInt32(inputRowStride)
        var outputStride = UInt32(outputRowStride)
        encoder.setBytes(&inputDim, length: MemoryLayout<UInt32>.stride, index: 9)
        encoder.setBytes(&outputDim0, length: MemoryLayout<UInt32>.stride, index: 10)
        encoder.setBytes(&outputDim1, length: MemoryLayout<UInt32>.stride, index: 11)
        encoder.setBytes(&outputDim2, length: MemoryLayout<UInt32>.stride, index: 12)
        encoder.setBytes(&outputDim3, length: MemoryLayout<UInt32>.stride, index: 13)
        encoder.setBytes(&seqLen, length: MemoryLayout<UInt32>.stride, index: 14)
        encoder.setBytes(&inputStride, length: MemoryLayout<UInt32>.stride, index: 15)
        encoder.setBytes(&outputStride, length: MemoryLayout<UInt32>.stride, index: 16)
        encoder.dispatchThreadgroups(grid, threadsPerThreadgroup: threadgroup)
        encoder.endEncoding()
        try harness.complete(commandBuffer)

        let scratchValues = harness.readFloat32(
            scratch,
            count: 5 * sequenceLength * slotDimension
        )
        return outputDimensions.indices.map { projection in
            var packed: [Float] = []
            packed.reserveCapacity(sequenceLength * outputDimensions[projection])
            let slotOffset = (projection + 1) * sequenceLength * slotDimension
            for position in 0..<sequenceLength {
                let sourceOffset = slotOffset + position * outputRowStride
                packed.append(contentsOf: scratchValues[sourceOffset..<(sourceOffset + outputDimensions[projection])])
            }
            return packed
        }
    }

    private func runDecodeProjectionTrace(
        harness: SequenceKernelEquivalenceHarness,
        pipeline: MTLComputePipelineState,
        inputValues: [Float],
        weights: [[BFloat16]],
        inputDimension: Int,
        sequenceLength: Int,
        outputDimensions: [Int]
    ) throws -> [[Float]] {
        let weightBuffers = try weights.map { try harness.makeSharedBuffer(values: $0) }
        var traces = outputDimensions.map { [Float](repeating: .zero, count: sequenceLength * $0) }
        let simdWidth = 32
        let threads = min(simdWidth * 2, pipeline.maxTotalThreadsPerThreadgroup)
        let rowsPerThreadgroup = max(1, threads / simdWidth)
        let totalRows = outputDimensions.reduce(0, +)
        let grid = MTLSize(
            width: (totalRows + rowsPerThreadgroup - 1) / rowsPerThreadgroup,
            height: 1,
            depth: 1
        )
        let threadgroup = MTLSize(width: threads, height: 1, depth: 1)

        for position in 0..<sequenceLength {
            let inputStart = position * inputDimension
            let tokenInput = inputValues[inputStart..<(inputStart + inputDimension)].map { BFloat16($0) }
            let inputBuffer = try harness.makeSharedBuffer(values: Array(tokenInput))
            let outputBuffers = try outputDimensions.map {
                try harness.makeZeroedSharedBuffer(byteLength: $0 * MemoryLayout<BFloat16>.stride)
            }

            let (commandBuffer, encoder) = try harness.makeCommandEncoder()
            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(inputBuffer, offset: 0, index: 0)
            for index in 0..<4 {
                encoder.setBuffer(weightBuffers[index], offset: 0, index: 1 + index)
                encoder.setBuffer(outputBuffers[index], offset: 0, index: 5 + index)
            }
            var inputDim = UInt32(inputDimension)
            var outputDim0 = UInt32(outputDimensions[0])
            var outputDim1 = UInt32(outputDimensions[1])
            var outputDim2 = UInt32(outputDimensions[2])
            var outputDim3 = UInt32(outputDimensions[3])
            encoder.setBytes(&inputDim, length: MemoryLayout<UInt32>.stride, index: 9)
            encoder.setBytes(&outputDim0, length: MemoryLayout<UInt32>.stride, index: 10)
            encoder.setBytes(&outputDim1, length: MemoryLayout<UInt32>.stride, index: 11)
            encoder.setBytes(&outputDim2, length: MemoryLayout<UInt32>.stride, index: 12)
            encoder.setBytes(&outputDim3, length: MemoryLayout<UInt32>.stride, index: 13)
            encoder.dispatchThreadgroups(grid, threadsPerThreadgroup: threadgroup)
            encoder.endEncoding()
            try harness.complete(commandBuffer)

            for projection in outputDimensions.indices {
                let values = harness.readBFloat16AsFloat(
                    outputBuffers[projection],
                    count: outputDimensions[projection]
                )
                let offset = position * outputDimensions[projection]
                traces[projection].replaceSubrange(offset..<(offset + outputDimensions[projection]), with: values)
            }
        }
        return traces
    }

    private func runSequenceProjectionTrace(
        harness: SequenceKernelEquivalenceHarness,
        pipeline: MTLComputePipelineState,
        inputValues: [Float],
        weights: [[BFloat16]],
        inputDimension: Int,
        sequenceLength: Int,
        outputDimensions: [Int]
    ) throws -> [[Float]] {
        let inputBuffer = try harness.makeSharedBuffer(values: inputValues)
        let weightBuffers = try weights.map { try harness.makeSharedBuffer(values: $0) }
        let outputRowStride = outputDimensions.max() ?? 0
        let outputBuffers = try outputDimensions.map { _ in
            try harness.makeZeroedSharedBuffer(byteLength: sequenceLength * outputRowStride * MemoryLayout<Float>.stride)
        }
        let simdWidth = 32
        let threads = min(simdWidth * 2, pipeline.maxTotalThreadsPerThreadgroup)
        let rowsPerThreadgroup = max(1, threads / simdWidth)
        let totalRows = outputDimensions.reduce(0, +)
        let grid = MTLSize(
            width: (totalRows + rowsPerThreadgroup - 1) / rowsPerThreadgroup,
            height: sequenceLength,
            depth: 1
        )
        let threadgroup = MTLSize(width: threads, height: 1, depth: 1)

        let (commandBuffer, encoder) = try harness.makeCommandEncoder()
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(inputBuffer, offset: 0, index: 0)
        for index in 0..<4 {
            encoder.setBuffer(weightBuffers[index], offset: 0, index: 1 + index)
            encoder.setBuffer(outputBuffers[index], offset: 0, index: 5 + index)
        }
        var inputDim = UInt32(inputDimension)
        var outputDim0 = UInt32(outputDimensions[0])
        var outputDim1 = UInt32(outputDimensions[1])
        var outputDim2 = UInt32(outputDimensions[2])
        var outputDim3 = UInt32(outputDimensions[3])
        var seqLen = UInt32(sequenceLength)
        var inputRowStride = UInt32(inputDimension)
        var outputStride = UInt32(outputRowStride)
        encoder.setBytes(&inputDim, length: MemoryLayout<UInt32>.stride, index: 9)
        encoder.setBytes(&outputDim0, length: MemoryLayout<UInt32>.stride, index: 10)
        encoder.setBytes(&outputDim1, length: MemoryLayout<UInt32>.stride, index: 11)
        encoder.setBytes(&outputDim2, length: MemoryLayout<UInt32>.stride, index: 12)
        encoder.setBytes(&outputDim3, length: MemoryLayout<UInt32>.stride, index: 13)
        encoder.setBytes(&seqLen, length: MemoryLayout<UInt32>.stride, index: 14)
        encoder.setBytes(&inputRowStride, length: MemoryLayout<UInt32>.stride, index: 15)
        encoder.setBytes(&outputStride, length: MemoryLayout<UInt32>.stride, index: 16)
        encoder.dispatchThreadgroups(grid, threadsPerThreadgroup: threadgroup)
        encoder.endEncoding()
        try harness.complete(commandBuffer)

        return outputDimensions.indices.map { projection in
            let padded = harness.readFloat32(
                outputBuffers[projection],
                count: sequenceLength * outputRowStride
            )
            var packed: [Float] = []
            packed.reserveCapacity(sequenceLength * outputDimensions[projection])
            for position in 0..<sequenceLength {
                let start = position * outputRowStride
                packed.append(contentsOf: padded[start..<(start + outputDimensions[projection])])
            }
            return packed
        }
    }

    private func paddedRows(
        _ values: [Float],
        rowCount: Int,
        logicalWidth: Int,
        rowStride: Int
    ) -> [Float] {
        var padded = [Float](repeating: .zero, count: rowCount * rowStride)
        for row in 0..<rowCount {
            let sourceStart = row * logicalWidth
            let destinationStart = row * rowStride
            padded.replaceSubrange(
                destinationStart..<(destinationStart + logicalWidth),
                with: values[sourceStart..<(sourceStart + logicalWidth)]
            )
        }
        return padded
    }
}
