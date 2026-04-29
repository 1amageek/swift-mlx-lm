import Metal
import Testing
@testable import MetalCompiler

@Suite("Conv1d Sequence Equivalence", .serialized)
struct Conv1dSequenceEquivalenceTests {
    @Test("BF16 conv sequence kernels match repeated decode state updates")
    func bf16ConvSequenceKernelsMatchRepeatedDecodeStateUpdates() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device")
            return
        }
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }

        let harness = try SequenceKernelEquivalenceHarness(device: device)
        let decodePipeline = try harness.pipeline(named: "conv_state_update_bf16")
        let sequencePipeline = try harness.pipeline(named: "conv1d_causal_seq_f32")
        let extractPipeline = try harness.pipeline(named: "extract_conv_state_f32")

        let convDimension = 8
        let inputProjectionDimension = convDimension * 3
        let kernelSize = 4
        let sequenceLength = 6
        let inputValues = (0..<(sequenceLength * inputProjectionDimension)).map { index in
            Float(Float16(Float((index * 19) % 23 - 11) * 0.0625))
        }
        let weights = (0..<(convDimension * kernelSize)).map { index in
            BFloat16(Float((index * 7) % 17 - 8) * 0.03125)
        }

        let decode = try runDecodeConvTrace(
            harness: harness,
            pipeline: decodePipeline,
            inputValues: inputValues,
            weights: weights,
            convDimension: convDimension,
            inputProjectionDimension: inputProjectionDimension,
            kernelSize: kernelSize,
            sequenceLength: sequenceLength
        )
        let sequence = try runSequenceConvTrace(
            harness: harness,
            sequencePipeline: sequencePipeline,
            extractPipeline: extractPipeline,
            inputValues: inputValues,
            weights: weights,
            convDimension: convDimension,
            inputProjectionDimension: inputProjectionDimension,
            kernelSize: kernelSize,
            sequenceLength: sequenceLength
        )

        let roundedSequenceOutput = sequence.output.map { Float(Float16($0)) }
        let mismatch = harness.firstMismatch(
            expected: decode.output,
            actual: roundedSequenceOutput,
            tolerance: 0.000_001
        )
        #expect(
            mismatch == nil,
            "conv output drifted: \(String(describing: mismatch)), maxError=\(harness.maxAbsoluteError(expected: decode.output, actual: roundedSequenceOutput))"
        )
        #expect(
            decode.convStateBits == sequence.convStateBits,
            "conv state drifted: decode=\(decode.convStateBits), sequence=\(sequence.convStateBits)"
        )
    }

    private func runDecodeConvTrace(
        harness: SequenceKernelEquivalenceHarness,
        pipeline: MTLComputePipelineState,
        inputValues: [Float],
        weights: [BFloat16],
        convDimension: Int,
        inputProjectionDimension: Int,
        kernelSize: Int,
        sequenceLength: Int
    ) throws -> (output: [Float], convStateBits: [UInt16]) {
        let convState = try harness.makeZeroedSharedBuffer(
            byteLength: convDimension * kernelSize * MemoryLayout<BFloat16>.stride
        )
        let weightBuffer = try harness.makeSharedBuffer(values: weights)
        var trace = [Float](repeating: .zero, count: sequenceLength * convDimension)
        let threads = min(max(pipeline.threadExecutionWidth, 32), pipeline.maxTotalThreadsPerThreadgroup)
        let grid = MTLSize(
            width: (convDimension + threads - 1) / threads,
            height: 1,
            depth: 1
        )
        let threadgroup = MTLSize(width: threads, height: 1, depth: 1)

        for position in 0..<sequenceLength {
            let inputStart = position * inputProjectionDimension
            let tokenInput = inputValues[inputStart..<(inputStart + inputProjectionDimension)].map {
                Float16($0)
            }
            let inputBuffer = try harness.makeSharedBuffer(values: Array(tokenInput))
            let outputBuffer = try harness.makeZeroedSharedBuffer(
                byteLength: convDimension * MemoryLayout<Float16>.stride
            )
            let (commandBuffer, encoder) = try harness.makeCommandEncoder()
            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(convState, offset: 0, index: 0)
            encoder.setBuffer(inputBuffer, offset: 0, index: 1)
            encoder.setBuffer(weightBuffer, offset: 0, index: 2)
            encoder.setBuffer(outputBuffer, offset: 0, index: 3)
            var dimension = UInt32(convDimension)
            var kernel = UInt32(kernelSize)
            encoder.setBytes(&dimension, length: MemoryLayout<UInt32>.stride, index: 4)
            encoder.setBytes(&kernel, length: MemoryLayout<UInt32>.stride, index: 5)
            encoder.dispatchThreadgroups(grid, threadsPerThreadgroup: threadgroup)
            encoder.endEncoding()
            try harness.complete(commandBuffer)

            let output = harness.readFloat16AsFloat(outputBuffer, count: convDimension)
            let offset = position * convDimension
            trace.replaceSubrange(offset..<(offset + convDimension), with: output)
        }
        return (
            output: trace,
            convStateBits: harness.readBFloat16Bits(
                convState,
                count: convDimension * kernelSize
            )
        )
    }

    private func runSequenceConvTrace(
        harness: SequenceKernelEquivalenceHarness,
        sequencePipeline: MTLComputePipelineState,
        extractPipeline: MTLComputePipelineState,
        inputValues: [Float],
        weights: [BFloat16],
        convDimension: Int,
        inputProjectionDimension: Int,
        kernelSize: Int,
        sequenceLength: Int
    ) throws -> (output: [Float], convStateBits: [UInt16]) {
        let inputBuffer = try harness.makeSharedBuffer(values: inputValues)
        let weightBuffer = try harness.makeSharedBuffer(values: weights)
        let outputBuffer = try harness.makeZeroedSharedBuffer(
            byteLength: sequenceLength * convDimension * MemoryLayout<Float>.stride
        )
        let convState = try harness.makeZeroedSharedBuffer(
            byteLength: convDimension * kernelSize * MemoryLayout<BFloat16>.stride
        )
        var convDim = UInt32(convDimension)
        var inputProjDim = UInt32(inputProjectionDimension)
        var kernel = UInt32(kernelSize)
        var seqLen = UInt32(sequenceLength)

        let sequenceThreads = MTLSize(width: 8, height: 1, depth: 1)
        let sequenceGrid = MTLSize(
            width: (convDimension + sequenceThreads.width - 1) / sequenceThreads.width,
            height: sequenceLength,
            depth: 1
        )
        let (sequenceCommandBuffer, sequenceEncoder) = try harness.makeCommandEncoder()
        sequenceEncoder.setComputePipelineState(sequencePipeline)
        sequenceEncoder.setBuffer(inputBuffer, offset: 0, index: 0)
        sequenceEncoder.setBuffer(weightBuffer, offset: 0, index: 1)
        sequenceEncoder.setBuffer(outputBuffer, offset: 0, index: 2)
        sequenceEncoder.setBytes(&convDim, length: MemoryLayout<UInt32>.stride, index: 3)
        sequenceEncoder.setBytes(&inputProjDim, length: MemoryLayout<UInt32>.stride, index: 4)
        sequenceEncoder.setBytes(&kernel, length: MemoryLayout<UInt32>.stride, index: 5)
        sequenceEncoder.setBytes(&seqLen, length: MemoryLayout<UInt32>.stride, index: 6)
        sequenceEncoder.dispatchThreadgroups(sequenceGrid, threadsPerThreadgroup: sequenceThreads)
        sequenceEncoder.endEncoding()
        try harness.complete(sequenceCommandBuffer)

        let extractThreads = MTLSize(width: 8, height: 1, depth: 1)
        let extractGrid = MTLSize(
            width: (convDimension + extractThreads.width - 1) / extractThreads.width,
            height: kernelSize,
            depth: 1
        )
        let (extractCommandBuffer, extractEncoder) = try harness.makeCommandEncoder()
        extractEncoder.setComputePipelineState(extractPipeline)
        extractEncoder.setBuffer(inputBuffer, offset: 0, index: 0)
        extractEncoder.setBuffer(convState, offset: 0, index: 1)
        extractEncoder.setBytes(&convDim, length: MemoryLayout<UInt32>.stride, index: 2)
        extractEncoder.setBytes(&inputProjDim, length: MemoryLayout<UInt32>.stride, index: 3)
        extractEncoder.setBytes(&kernel, length: MemoryLayout<UInt32>.stride, index: 4)
        extractEncoder.setBytes(&seqLen, length: MemoryLayout<UInt32>.stride, index: 5)
        extractEncoder.dispatchThreadgroups(extractGrid, threadsPerThreadgroup: extractThreads)
        extractEncoder.endEncoding()
        try harness.complete(extractCommandBuffer)

        return (
            output: harness.readFloat32(
                outputBuffer,
                count: sequenceLength * convDimension
            ),
            convStateBits: harness.readBFloat16Bits(
                convState,
                count: convDimension * kernelSize
            )
        )
    }
}
