import Metal
import Testing
@testable import MetalCompiler

@Suite("SSM Recurrence Sequence Equivalence", .serialized)
struct SSMRecurrenceSequenceEquivalenceTests {
    @Test("BF16 SSM sequence recurrence matches repeated decode recurrence")
    func bf16SSMSequenceRecurrenceMatchesRepeatedDecodeRecurrence() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device")
            return
        }
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }

        let headCount = 16
        let groupCount = 16
        let keyDimension = 128
        let valueDimension = 128
        let convKernelSize = 4
        let sequenceLength = 19
        let keyGroupDimension = groupCount * keyDimension
        let convDimension = 2 * keyGroupDimension + headCount * valueDimension
        let outputDimension = headCount * valueDimension
        let decodeKernelName = "ssm_recurrence_bf16_decode_equivalence"
        let sequenceKernelName = "ssm_recurrence_bf16_sequence_equivalence"
        let source = [
            MetalSourceGenerator.commonHeader,
            MetalSourceGenerator.generateSSMWeightIndependentHelpers(),
            MetalSourceGenerator.generateSSMConvSiluHelper(weightFormat: .bfloat16),
            MetalSourceGenerator.generateSSMRecurrence(
                name: decodeKernelName,
                bufferPrecision: .bfloat16,
                weightFormat: .bfloat16,
                convDimension: convDimension,
                maxThreadgroupSize: SSMRecurrenceFragment.maxThreadgroupSize,
                headCount: headCount,
                groupCount: groupCount,
                keyHeadDimension: keyDimension,
                valueHeadDimension: valueDimension
            ),
            MetalSourceGenerator.generateSSMRecurrenceSequence(
                name: sequenceKernelName,
                bufferPrecision: .float32,
                weightFormat: .bfloat16,
                convDimension: convDimension,
                maxThreadgroupSize: SSMRecurrenceFragment.maxThreadgroupSize,
                headCount: headCount,
                groupCount: groupCount,
                keyHeadDimension: keyDimension,
                valueHeadDimension: valueDimension
            ),
        ].joined(separator: "\n")
        let harness = try SequenceKernelEquivalenceHarness(device: device, source: source)
        let decodePipeline = try harness.pipeline(named: decodeKernelName)
        let sequencePipeline = try harness.pipeline(named: sequenceKernelName)

        let projectedQKV = roundedBFloat16Values(
            count: sequenceLength * convDimension,
            multiplier: 13,
            modulus: 23,
            scale: 0.125
        )
        let projectedZ = roundedBFloat16Values(
            count: sequenceLength * outputDimension,
            multiplier: 17,
            modulus: 19,
            scale: 0.125
        )
        let projectedBeta = roundedBFloat16Values(
            count: sequenceLength * headCount,
            multiplier: 7,
            modulus: 11,
            scale: 0.125
        )
        let projectedAlpha = roundedBFloat16Values(
            count: sequenceLength * headCount,
            multiplier: 5,
            modulus: 13,
            scale: 0.125
        )
        let convWeight = (0..<(convDimension * convKernelSize)).map { index in
            BFloat16(Float((index * 11) % 17 - 8) * 0.03125)
        }
        let normWeight = (0..<valueDimension).map { index in
            0.75 + Float(index) * 0.0625
        }
        let dtBias = (0..<headCount).map { index in
            BFloat16(Float(index - 1) * 0.03125)
        }
        let aLog = (0..<headCount).map { index in
            Float(index) * 0.0625 - 0.125
        }

        let decode = try runDecodeSSMTrace(
            harness: harness,
            pipeline: decodePipeline,
            projectedQKV: projectedQKV,
            projectedZ: projectedZ,
            projectedBeta: projectedBeta,
            projectedAlpha: projectedAlpha,
            convWeight: convWeight,
            normWeight: normWeight,
            dtBias: dtBias,
            aLog: aLog,
            headCount: headCount,
            groupCount: groupCount,
            keyDimension: keyDimension,
            valueDimension: valueDimension,
            convKernelSize: convKernelSize,
            sequenceLength: sequenceLength,
            convDimension: convDimension,
            outputDimension: outputDimension
        )
        let sequence = try runSequenceSSMTrace(
            harness: harness,
            pipeline: sequencePipeline,
            projectedQKV: projectedQKV,
            projectedZ: projectedZ,
            projectedBeta: projectedBeta,
            projectedAlpha: projectedAlpha,
            convWeight: convWeight,
            normWeight: normWeight,
            dtBias: dtBias,
            aLog: aLog,
            headCount: headCount,
            groupCount: groupCount,
            keyDimension: keyDimension,
            valueDimension: valueDimension,
            convKernelSize: convKernelSize,
            sequenceLength: sequenceLength,
            convDimension: convDimension,
            outputDimension: outputDimension
        )

        let outputMismatch = harness.firstMismatch(
            expected: decode.output,
            actual: sequence.output,
            tolerance: 0.000_01
        )
        #expect(
            outputMismatch == nil,
            "SSM output drifted: \(String(describing: outputMismatch)), maxError=\(harness.maxAbsoluteError(expected: decode.output, actual: sequence.output))"
        )
        let recurrentMismatch = harness.firstMismatch(
            expected: decode.recurrentState,
            actual: sequence.recurrentState,
            tolerance: 0.000_01
        )
        #expect(
            recurrentMismatch == nil,
            "SSM recurrent state drifted: \(String(describing: recurrentMismatch)), maxError=\(harness.maxAbsoluteError(expected: decode.recurrentState, actual: sequence.recurrentState))"
        )
        #expect(
            decode.convStateBits == sequence.convStateBits,
            "SSM conv state drifted: decode=\(decode.convStateBits), sequence=\(sequence.convStateBits)"
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

    private func runDecodeSSMTrace(
        harness: SequenceKernelEquivalenceHarness,
        pipeline: MTLComputePipelineState,
        projectedQKV: [Float],
        projectedZ: [Float],
        projectedBeta: [Float],
        projectedAlpha: [Float],
        convWeight: [BFloat16],
        normWeight: [Float],
        dtBias: [BFloat16],
        aLog: [Float],
        headCount: Int,
        groupCount: Int,
        keyDimension: Int,
        valueDimension: Int,
        convKernelSize: Int,
        sequenceLength: Int,
        convDimension: Int,
        outputDimension: Int
    ) throws -> (output: [Float], recurrentState: [Float], convStateBits: [UInt16]) {
        let convWeightBuffer = try harness.makeSharedBuffer(values: convWeight)
        let normWeightBuffer = try harness.makeSharedBuffer(values: normWeight)
        let dtBiasBuffer = try harness.makeSharedBuffer(values: dtBias)
        let aLogBuffer = try harness.makeSharedBuffer(values: aLog)
        let recurrentState = try harness.makeZeroedSharedBuffer(
            byteLength: headCount * keyDimension * valueDimension * MemoryLayout<Float>.stride
        )
        let convState = try harness.makeZeroedSharedBuffer(
            byteLength: convKernelSize * convDimension * MemoryLayout<BFloat16>.stride
        )
        var trace = [Float](repeating: .zero, count: sequenceLength * outputDimension)
        let threads = ssmThreadCount(
            pipeline: pipeline,
            headCount: headCount,
            groupCount: groupCount,
            keyDimension: keyDimension,
            valueDimension: valueDimension
        )
        let grid = MTLSize(width: max(groupCount, 1), height: 1, depth: 1)
        let threadgroup = MTLSize(width: threads, height: 1, depth: 1)

        for position in 0..<sequenceLength {
            let qkvOffset = position * convDimension
            let zOffset = position * outputDimension
            let headOffset = position * headCount
            let qkvInput = projectedQKV[qkvOffset..<(qkvOffset + convDimension)].map { BFloat16($0) }
            let zInput = projectedZ[zOffset..<(zOffset + outputDimension)].map { BFloat16($0) }
            let betaInput = projectedBeta[headOffset..<(headOffset + headCount)].map { BFloat16($0) }
            let alphaInput = projectedAlpha[headOffset..<(headOffset + headCount)].map { BFloat16($0) }
            let qkvBuffer = try harness.makeSharedBuffer(values: Array(qkvInput))
            let zBuffer = try harness.makeSharedBuffer(values: Array(zInput))
            let betaBuffer = try harness.makeSharedBuffer(values: Array(betaInput))
            let alphaBuffer = try harness.makeSharedBuffer(values: Array(alphaInput))
            let outputBuffer = try harness.makeZeroedSharedBuffer(
                byteLength: outputDimension * MemoryLayout<BFloat16>.stride
            )

            let (commandBuffer, encoder) = try harness.makeCommandEncoder()
            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(qkvBuffer, offset: 0, index: 0)
            encoder.setBuffer(zBuffer, offset: 0, index: 1)
            encoder.setBuffer(betaBuffer, offset: 0, index: 2)
            encoder.setBuffer(alphaBuffer, offset: 0, index: 3)
            encoder.setBuffer(convWeightBuffer, offset: 0, index: 4)
            encoder.setBuffer(normWeightBuffer, offset: 0, index: 5)
            encoder.setBuffer(dtBiasBuffer, offset: 0, index: 6)
            encoder.setBuffer(aLogBuffer, offset: 0, index: 7)
            encoder.setBuffer(recurrentState, offset: 0, index: 8)
            encoder.setBuffer(convState, offset: 0, index: 9)
            encoder.setBuffer(outputBuffer, offset: 0, index: 10)
            setSSMConstants(
                encoder: encoder,
                headCount: headCount,
                groupCount: groupCount,
                keyDimension: keyDimension,
                valueDimension: valueDimension,
                convKernelSize: convKernelSize
            )
            encoder.dispatchThreadgroups(grid, threadsPerThreadgroup: threadgroup)
            encoder.endEncoding()
            try harness.complete(commandBuffer)

            let output = harness.readBFloat16AsFloat(outputBuffer, count: outputDimension)
            trace.replaceSubrange(zOffset..<(zOffset + outputDimension), with: output)
        }

        return (
            output: trace,
            recurrentState: harness.readFloat32(
                recurrentState,
                count: headCount * keyDimension * valueDimension
            ),
            convStateBits: harness.readBFloat16Bits(
                convState,
                count: convKernelSize * convDimension
            )
        )
    }

    private func runSequenceSSMTrace(
        harness: SequenceKernelEquivalenceHarness,
        pipeline: MTLComputePipelineState,
        projectedQKV: [Float],
        projectedZ: [Float],
        projectedBeta: [Float],
        projectedAlpha: [Float],
        convWeight: [BFloat16],
        normWeight: [Float],
        dtBias: [BFloat16],
        aLog: [Float],
        headCount: Int,
        groupCount: Int,
        keyDimension: Int,
        valueDimension: Int,
        convKernelSize: Int,
        sequenceLength: Int,
        convDimension: Int,
        outputDimension: Int
    ) throws -> (output: [Float], recurrentState: [Float], convStateBits: [UInt16]) {
        let activationRowStride = max(convDimension, outputDimension, headCount)
        let qkvBuffer = try harness.makeSharedBuffer(values: paddedRows(
            projectedQKV,
            rowCount: sequenceLength,
            logicalWidth: convDimension,
            rowStride: activationRowStride
        ))
        let zBuffer = try harness.makeSharedBuffer(values: paddedRows(
            projectedZ,
            rowCount: sequenceLength,
            logicalWidth: outputDimension,
            rowStride: activationRowStride
        ))
        let betaBuffer = try harness.makeSharedBuffer(values: paddedRows(
            projectedBeta,
            rowCount: sequenceLength,
            logicalWidth: headCount,
            rowStride: activationRowStride
        ))
        let alphaBuffer = try harness.makeSharedBuffer(values: paddedRows(
            projectedAlpha,
            rowCount: sequenceLength,
            logicalWidth: headCount,
            rowStride: activationRowStride
        ))
        let convWeightBuffer = try harness.makeSharedBuffer(values: convWeight)
        let normWeightBuffer = try harness.makeSharedBuffer(values: normWeight)
        let dtBiasBuffer = try harness.makeSharedBuffer(values: dtBias)
        let aLogBuffer = try harness.makeSharedBuffer(values: aLog)
        let recurrentState = try harness.makeZeroedSharedBuffer(
            byteLength: headCount * keyDimension * valueDimension * MemoryLayout<Float>.stride
        )
        let convState = try harness.makeZeroedSharedBuffer(
            byteLength: convKernelSize * convDimension * MemoryLayout<BFloat16>.stride
        )
        let outputBuffer = try harness.makeZeroedSharedBuffer(
            byteLength: sequenceLength * activationRowStride * MemoryLayout<Float>.stride
        )
        let threads = ssmThreadCount(
            pipeline: pipeline,
            headCount: headCount,
            groupCount: groupCount,
            keyDimension: keyDimension,
            valueDimension: valueDimension
        )
        let grid = MTLSize(width: max(groupCount, 1), height: 1, depth: 1)
        let threadgroup = MTLSize(width: threads, height: 1, depth: 1)

        let (commandBuffer, encoder) = try harness.makeCommandEncoder()
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(qkvBuffer, offset: 0, index: 0)
        encoder.setBuffer(zBuffer, offset: 0, index: 1)
        encoder.setBuffer(betaBuffer, offset: 0, index: 2)
        encoder.setBuffer(alphaBuffer, offset: 0, index: 3)
        encoder.setBuffer(convWeightBuffer, offset: 0, index: 4)
        encoder.setBuffer(normWeightBuffer, offset: 0, index: 5)
        encoder.setBuffer(dtBiasBuffer, offset: 0, index: 6)
        encoder.setBuffer(aLogBuffer, offset: 0, index: 7)
        encoder.setBuffer(recurrentState, offset: 0, index: 8)
        encoder.setBuffer(convState, offset: 0, index: 9)
        encoder.setBuffer(outputBuffer, offset: 0, index: 10)
        setSSMConstants(
            encoder: encoder,
            headCount: headCount,
            groupCount: groupCount,
            keyDimension: keyDimension,
            valueDimension: valueDimension,
            convKernelSize: convKernelSize
        )
        var seqLen = UInt32(sequenceLength)
        var rowStride = UInt32(activationRowStride)
        encoder.setBytes(&seqLen, length: MemoryLayout<UInt32>.stride, index: 16)
        encoder.setBytes(&rowStride, length: MemoryLayout<UInt32>.stride, index: 17)
        encoder.dispatchThreadgroups(grid, threadsPerThreadgroup: threadgroup)
        encoder.endEncoding()
        try harness.complete(commandBuffer)

        let paddedOutput = harness.readFloat32(
            outputBuffer,
            count: sequenceLength * activationRowStride
        )
        var output: [Float] = []
        output.reserveCapacity(sequenceLength * outputDimension)
        for position in 0..<sequenceLength {
            let start = position * activationRowStride
            output.append(contentsOf: paddedOutput[start..<(start + outputDimension)])
        }

        return (
            output: output,
            recurrentState: harness.readFloat32(
                recurrentState,
                count: headCount * keyDimension * valueDimension
            ),
            convStateBits: harness.readBFloat16Bits(
                convState,
                count: convKernelSize * convDimension
            )
        )
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

    private func ssmThreadCount(
        pipeline: MTLComputePipelineState,
        headCount: Int,
        groupCount: Int,
        keyDimension: Int,
        valueDimension: Int
    ) -> Int {
        let safeGroupCount = max(groupCount, 1)
        let headsPerGroup = max(1, headCount / safeGroupCount)
        let localDimension = 2 * keyDimension + headsPerGroup * valueDimension
        let phase2Threads = headsPerGroup * min(valueDimension, 256)
        let desiredThreads = max(localDimension, phase2Threads)
        return min(
            min(SSMRecurrenceFragment.maxThreadgroupSize, desiredThreads),
            pipeline.maxTotalThreadsPerThreadgroup
        )
    }

    private func setSSMConstants(
        encoder: MTLComputeCommandEncoder,
        headCount: Int,
        groupCount: Int,
        keyDimension: Int,
        valueDimension: Int,
        convKernelSize: Int
    ) {
        var heads = UInt32(headCount)
        var groups = UInt32(groupCount)
        var keyDim = UInt32(keyDimension)
        var valueDim = UInt32(valueDimension)
        var kernel = UInt32(convKernelSize)
        encoder.setBytes(&heads, length: MemoryLayout<UInt32>.stride, index: 11)
        encoder.setBytes(&groups, length: MemoryLayout<UInt32>.stride, index: 12)
        encoder.setBytes(&keyDim, length: MemoryLayout<UInt32>.stride, index: 13)
        encoder.setBytes(&valueDim, length: MemoryLayout<UInt32>.stride, index: 14)
        encoder.setBytes(&kernel, length: MemoryLayout<UInt32>.stride, index: 15)
    }
}
