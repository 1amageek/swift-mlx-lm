import Testing
import Metal
@testable import MetalCompiler

@Suite("Naive GEMM Argument Table", .serialized)
struct NaiveGEMMArgumentTableTests {
    @Test("Naive BF16 GEMM matches CPU reference with classic bindings")
    func naiveBF16GEMMMatchesCPUReferenceWithClassicBindings() throws {
        try runScenario(
            inputDimension: 7,
            outputDimension: 5,
            sequenceLength: 4,
            mode: .classic,
            tolerance: 0.01
        )
    }

    @Test("Naive BF16 GEMM matches CPU reference with Metal 4 argument table")
    func naiveBF16GEMMMatchesCPUReferenceWithArgumentTable() throws {
        try runScenario(
            inputDimension: 7,
            outputDimension: 5,
            sequenceLength: 4,
            mode: .argumentTable,
            tolerance: 0.01
        )
    }

    @Test("Naive BF16 GEMM with Metal 4 argument table writes correctly to a non-zero output offset")
    func naiveBF16GEMMWritesCorrectlyToNonZeroOutputOffsetWithArgumentTable() throws {
        try runScenario(
            inputDimension: 7,
            outputDimension: 5,
            sequenceLength: 4,
            mode: .argumentTable,
            outputOffset: 2_097_152,
            tolerance: 0.01
        )
    }

    @Test("Naive BF16 GEMM with Metal 4 argument table writes correctly to a private output buffer")
    func naiveBF16GEMMWritesCorrectlyToPrivateOutputBufferWithArgumentTable() throws {
        try runScenario(
            inputDimension: 7,
            outputDimension: 5,
            sequenceLength: 4,
            mode: .argumentTablePrivateOutput,
            tolerance: 0.01
        )
    }

    @Test("Naive BF16 GEMM with Metal 4 argument table writes correctly to a private output buffer at a non-zero offset")
    func naiveBF16GEMMWritesCorrectlyToPrivateOutputBufferAtNonZeroOffsetWithArgumentTable() throws {
        try runScenario(
            inputDimension: 7,
            outputDimension: 5,
            sequenceLength: 4,
            mode: .argumentTablePrivateOutput,
            outputOffset: 2_097_152,
            tolerance: 0.01
        )
    }

    @Test("Representative prefill BF16 GEMM matches CPU reference with runtime sequence-length buffer rebinding")
    func representativePrefillBF16GEMMMatchesCPUReferenceWithRuntimeSequenceLengthBufferRebinding() throws {
        try runScenario(
            inputDimension: 2_048,
            outputDimension: 3_072,
            sequenceLength: 5,
            mode: .argumentTableRuntimeSequenceLengthBuffer,
            outputOffset: 2_097_152,
            maximumSequenceLength: 64,
            tolerance: 0.05
        )
    }

    @Test("Representative prefill BF16 GEMM matches CPU reference with private aliased IO buffer")
    func representativePrefillBF16GEMMMatchesCPUReferenceWithPrivateAliasedIOBuffer() throws {
        try runScenario(
            inputDimension: 2_048,
            outputDimension: 3_072,
            sequenceLength: 5,
            mode: .argumentTablePrivateAliasedIOBuffer,
            outputOffset: 2_097_152,
            maximumSequenceLength: 64,
            tolerance: 0.05
        )
    }

    @Test("Representative prefill RMSNorm to BF16 GEMM chain matches CPU reference with Metal 4 argument table")
    func representativePrefillRMSNormToBF16GEMMChainMatchesCPUReferenceWithMetal4ArgumentTable() throws {
        try runRMSNormToGEMMChainScenario(
            inputDimension: 2_048,
            slotDimension: 8_192,
            outputDimension: 6_144,
            sequenceLength: 5,
            maximumSequenceLength: 64,
            outputOffset: 2_097_152,
            epsilon: 1e-6,
            tolerance: 0.05
        )
    }

    @Test("Representative prefill BF16 GEMM matches CPU reference with Metal 4 argument table")
    func representativePrefillBF16GEMMMatchesCPUReferenceWithArgumentTable() throws {
        try runScenario(
            inputDimension: 2_048,
            outputDimension: 3_072,
            sequenceLength: 2,
            mode: .argumentTable,
            tolerance: 0.05
        )
    }

    private enum ExecutionMode {
        case classic
        case argumentTable
        case argumentTablePrivateOutput
        case argumentTableRuntimeSequenceLengthBuffer
        case argumentTablePrivateAliasedIOBuffer
    }

    private func runScenario(
        inputDimension: Int,
        outputDimension: Int,
        sequenceLength: Int,
        mode: ExecutionMode,
        outputOffset: Int = 0,
        maximumSequenceLength: Int? = nil,
        tolerance: Float
    ) throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device")
            return
        }

        let input = makeInput(inputDimension: inputDimension, sequenceLength: sequenceLength)
        let weight = makeWeight(inputDimension: inputDimension, outputDimension: outputDimension)
        let expected = cpuReference(
            input: input,
            weight: weight,
            inputDimension: inputDimension,
            outputDimension: outputDimension,
            sequenceLength: sequenceLength
        )

        let inputBuffer = try #require(device.makeBuffer(
            bytes: input,
            length: input.count * MemoryLayout<Float>.stride,
            options: .storageModeShared
        ))
        let weightBuffer = try #require(device.makeBuffer(
            bytes: weight,
            length: weight.count * MemoryLayout<BFloat16>.stride,
            options: .storageModeShared
        ))
        let outputLength = outputDimension * sequenceLength
        let outputBuffer = try #require(device.makeBuffer(
            length: outputOffset + outputLength * MemoryLayout<Float>.stride,
            options: .storageModeShared
        ))
        outputBuffer.contents().initializeMemory(
            as: UInt8.self,
            repeating: 0,
            count: outputOffset + outputLength * MemoryLayout<Float>.stride
        )

        let pipeline = try makePipeline(device: device)
        let actual: [Float]
        switch mode {
        case .classic:
            actual = try executeClassic(
                device: device,
                pipeline: pipeline,
                inputBuffer: inputBuffer,
                weightBuffer: weightBuffer,
                outputBuffer: outputBuffer,
                outputOffset: outputOffset,
                inputDimension: inputDimension,
                outputDimension: outputDimension,
                sequenceLength: sequenceLength
            )
        case .argumentTable:
            actual = try executeArgumentTable(
                device: device,
                pipeline: pipeline,
                inputBuffer: inputBuffer,
                weightBuffer: weightBuffer,
                outputBuffer: outputBuffer,
                outputOffset: outputOffset,
                inputDimension: inputDimension,
                outputDimension: outputDimension,
                sequenceLength: sequenceLength
            )
        case .argumentTablePrivateOutput:
            actual = try executeArgumentTablePrivateOutput(
                device: device,
                pipeline: pipeline,
                inputBuffer: inputBuffer,
                weightBuffer: weightBuffer,
                outputOffset: outputOffset,
                inputDimension: inputDimension,
                outputDimension: outputDimension,
                sequenceLength: sequenceLength
            )
        case .argumentTableRuntimeSequenceLengthBuffer:
            actual = try executeArgumentTableRuntimeSequenceLengthBuffer(
                device: device,
                pipeline: pipeline,
                inputBuffer: inputBuffer,
                weightBuffer: weightBuffer,
                outputOffset: outputOffset,
                inputDimension: inputDimension,
                outputDimension: outputDimension,
                sequenceLength: sequenceLength,
                maximumSequenceLength: maximumSequenceLength ?? sequenceLength
            )
        case .argumentTablePrivateAliasedIOBuffer:
            actual = try executeArgumentTablePrivateAliasedIOBuffer(
                device: device,
                pipeline: pipeline,
                inputBuffer: inputBuffer,
                weightBuffer: weightBuffer,
                outputOffset: outputOffset,
                inputDimension: inputDimension,
                outputDimension: outputDimension,
                sequenceLength: sequenceLength,
                maximumSequenceLength: maximumSequenceLength ?? sequenceLength
            )
        }

        let maxError = zip(actual, expected).reduce(Float.zero) { partial, pair in
            max(partial, abs(pair.0 - pair.1))
        }
        #expect(
            maxError < tolerance,
            """
            naive BF16 GEMM drifted
            mode=\(mode)
            input=\(inputDimension) output=\(outputDimension) seq=\(sequenceLength)
            maxError=\(maxError)
            actualPrefix=\(actual.prefix(8).map { String(format: "%.4f", $0) }.joined(separator: ", "))
            expectedPrefix=\(expected.prefix(8).map { String(format: "%.4f", $0) }.joined(separator: ", "))
            """
        )
    }

    private func makePipeline(device: MTLDevice) throws -> MTLComputePipelineState {
        let source = MetalSourceGenerator.commonHeader + "\n\n"
            + MetalSourceGenerator.generateGEMM(
                name: "test_naive_gemm_bf16_f32s",
                bufferPrecision: .float32,
                weightFormat: .bfloat16
            )
        let options = MTLCompileOptions()
        options.languageVersion = .version4_0
        let library = try device.makeLibrary(source: source, options: options)
        let function = try #require(library.makeFunction(name: "test_naive_gemm_bf16_f32s"))
        return try device.makeComputePipelineState(function: function)
    }

    private func makeRMSNormPipeline(device: MTLDevice) throws -> MTLComputePipelineState {
        let source = MetalSourceGenerator.commonHeader + "\n\n"
            + MetalSourceGenerator.generateReduction(
                name: "test_rms_norm_seq_bf16_f32_inplace",
                dimension: 0,
                epsilon: 0,
                bufferPrecision: .float32,
                weightFormat: .bfloat16
            )
        let options = MTLCompileOptions()
        options.languageVersion = .version4_0
        let library = try device.makeLibrary(source: source, options: options)
        let function = try #require(library.makeFunction(name: "test_rms_norm_seq_bf16_f32_inplace"))
        return try device.makeComputePipelineState(function: function)
    }

    private func executeClassic(
        device: MTLDevice,
        pipeline: MTLComputePipelineState,
        inputBuffer: MTLBuffer,
        weightBuffer: MTLBuffer,
        outputBuffer: MTLBuffer,
        outputOffset: Int,
        inputDimension: Int,
        outputDimension: Int,
        sequenceLength: Int
    ) throws -> [Float] {
        let queue = try #require(device.makeCommandQueue())
        let commandBuffer = try #require(queue.makeCommandBuffer())
        let encoder = try #require(commandBuffer.makeComputeCommandEncoder())
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(inputBuffer, offset: 0, index: 0)
        encoder.setBuffer(weightBuffer, offset: 0, index: 1)
        encoder.setBuffer(outputBuffer, offset: outputOffset, index: 2)
        var inDim = UInt32(inputDimension)
        var outDim = UInt32(outputDimension)
        var seqLen = UInt32(sequenceLength)
        encoder.setBytes(&inDim, length: MemoryLayout<UInt32>.stride, index: 3)
        encoder.setBytes(&outDim, length: MemoryLayout<UInt32>.stride, index: 4)
        encoder.setBytes(&seqLen, length: MemoryLayout<UInt32>.stride, index: 5)
        let geometry = naiveGeometry(pipeline: pipeline, outputDimension: outputDimension, sequenceLength: sequenceLength)
        encoder.dispatchThreadgroups(
            geometry.gridSize,
            threadsPerThreadgroup: geometry.threadgroupSize
        )
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        return readOutput(outputBuffer, offset: outputOffset, count: outputDimension * sequenceLength)
    }

    private func executeArgumentTable(
        device: MTLDevice,
        pipeline: MTLComputePipelineState,
        inputBuffer: MTLBuffer,
        weightBuffer: MTLBuffer,
        outputBuffer: MTLBuffer,
        outputOffset: Int,
        inputDimension: Int,
        outputDimension: Int,
        sequenceLength: Int
    ) throws -> [Float] {
        let allocator = MetalConstantBindingAllocator(device: device)
        let bindingTable = try allocator.makeBindingTable(
            bufferBindings: [
                (index: 0, buffer: inputBuffer, offset: 0),
                (index: 1, buffer: weightBuffer, offset: 0),
                (index: 2, buffer: outputBuffer, offset: outputOffset),
            ],
            bytesBindings: [
                (index: 3, value: uint32Bytes(UInt32(inputDimension))),
                (index: 4, value: uint32Bytes(UInt32(outputDimension))),
                (index: 5, value: uint32Bytes(UInt32(sequenceLength))),
            ],
            argumentPolicy: .argumentTable
        )
        let residency = try MetalResidencyLease.required(
            device: device,
            label: "swift-lm.test.naive-gemm",
            buffers: [inputBuffer, weightBuffer, outputBuffer] + bindingTable.ownedResidencyBuffers
        )

        var submission = try MetalSubmissionContext(device: device)
        let geometry = naiveGeometry(pipeline: pipeline, outputDimension: outputDimension, sequenceLength: sequenceLength)
        try submission.withCompute(ephemeralResidency: residency) { encoder, argumentTable in
            bindingTable.bind(to: argumentTable)
            encoder.setArgumentTable(argumentTable)
            encoder.setComputePipelineState(pipeline)
            encoder.dispatchThreadgroups(
                threadgroupsPerGrid: geometry.gridSize,
                threadsPerThreadgroup: geometry.threadgroupSize
            )
        }
        return readOutput(outputBuffer, offset: outputOffset, count: outputDimension * sequenceLength)
    }

    private func executeArgumentTablePrivateOutput(
        device: MTLDevice,
        pipeline: MTLComputePipelineState,
        inputBuffer: MTLBuffer,
        weightBuffer: MTLBuffer,
        outputOffset: Int,
        inputDimension: Int,
        outputDimension: Int,
        sequenceLength: Int
    ) throws -> [Float] {
        let outputLength = outputDimension * sequenceLength * MemoryLayout<Float>.stride
        let outputBuffer = try #require(device.makeBuffer(
            length: outputOffset + outputLength,
            options: .storageModePrivate
        ))
        let stagingBuffer = try #require(device.makeBuffer(
            length: outputLength,
            options: .storageModeShared
        ))
        stagingBuffer.contents().initializeMemory(
            as: UInt8.self,
            repeating: 0,
            count: outputLength
        )

        let allocator = MetalConstantBindingAllocator(device: device)
        let bindingTable = try allocator.makeBindingTable(
            bufferBindings: [
                (index: 0, buffer: inputBuffer, offset: 0),
                (index: 1, buffer: weightBuffer, offset: 0),
                (index: 2, buffer: outputBuffer, offset: outputOffset),
            ],
            bytesBindings: [
                (index: 3, value: uint32Bytes(UInt32(inputDimension))),
                (index: 4, value: uint32Bytes(UInt32(outputDimension))),
                (index: 5, value: uint32Bytes(UInt32(sequenceLength))),
            ],
            argumentPolicy: .argumentTable
        )
        let residency = try MetalResidencyLease.required(
            device: device,
            label: "swift-lm.test.naive-gemm-private-output",
            buffers: [inputBuffer, weightBuffer, outputBuffer, stagingBuffer] + bindingTable.ownedResidencyBuffers
        )

        var submission = try MetalSubmissionContext(device: device)
        let geometry = naiveGeometry(pipeline: pipeline, outputDimension: outputDimension, sequenceLength: sequenceLength)
        try submission.withCompute(ephemeralResidency: residency) { encoder, argumentTable in
            bindingTable.bind(to: argumentTable)
            encoder.setArgumentTable(argumentTable)
            encoder.setComputePipelineState(pipeline)
            encoder.dispatchThreadgroups(
                threadgroupsPerGrid: geometry.gridSize,
                threadsPerThreadgroup: geometry.threadgroupSize
            )
            encoder.barrier(
                afterEncoderStages: .dispatch,
                beforeEncoderStages: .blit,
                visibilityOptions: []
            )
            encoder.copy(
                sourceBuffer: outputBuffer,
                sourceOffset: outputOffset,
                destinationBuffer: stagingBuffer,
                destinationOffset: 0,
                size: outputLength
            )
        }
        return readOutput(stagingBuffer, offset: 0, count: outputDimension * sequenceLength)
    }

    private func executeArgumentTableRuntimeSequenceLengthBuffer(
        device: MTLDevice,
        pipeline: MTLComputePipelineState,
        inputBuffer: MTLBuffer,
        weightBuffer: MTLBuffer,
        outputOffset: Int,
        inputDimension: Int,
        outputDimension: Int,
        sequenceLength: Int,
        maximumSequenceLength: Int
    ) throws -> [Float] {
        let outputLength = outputDimension * maximumSequenceLength * MemoryLayout<Float>.stride
        let outputBuffer = try #require(device.makeBuffer(
            length: outputOffset + outputLength,
            options: .storageModePrivate
        ))
        let stagingBuffer = try #require(device.makeBuffer(
            length: outputDimension * sequenceLength * MemoryLayout<Float>.stride,
            options: .storageModeShared
        ))
        let runtimeConstantBuffer = try #require(device.makeBuffer(
            length: MemoryLayout<UInt32>.stride,
            options: .storageModeShared
        ))
        runtimeConstantBuffer.contents().storeBytes(of: UInt32(sequenceLength), as: UInt32.self)
        stagingBuffer.contents().initializeMemory(
            as: UInt8.self,
            repeating: 0,
            count: outputDimension * sequenceLength * MemoryLayout<Float>.stride
        )

        let allocator = MetalConstantBindingAllocator(device: device)
        let bindingTable = try allocator.makeBindingTable(
            bufferBindings: [
                (index: 0, buffer: inputBuffer, offset: 0),
                (index: 1, buffer: weightBuffer, offset: 0),
                (index: 2, buffer: outputBuffer, offset: outputOffset),
            ],
            bytesBindings: [
                (index: 3, value: uint32Bytes(UInt32(inputDimension))),
                (index: 4, value: uint32Bytes(UInt32(outputDimension))),
                (index: 5, value: uint32Bytes(UInt32(maximumSequenceLength))),
            ],
            argumentPolicy: .argumentTable
        )
        let residency = try MetalResidencyLease.required(
            device: device,
            label: "swift-lm.test.naive-gemm-runtime-seq",
            buffers: [inputBuffer, weightBuffer, outputBuffer, stagingBuffer, runtimeConstantBuffer]
                + bindingTable.ownedResidencyBuffers
        )

        var submission = try MetalSubmissionContext(device: device)
        let geometry = naiveGeometry(
            pipeline: pipeline,
            outputDimension: outputDimension,
            sequenceLength: sequenceLength
        )
        try submission.withCompute(ephemeralResidency: residency) { encoder, argumentTable in
            bindingTable.bind(to: argumentTable)
            argumentTable.setAddress(runtimeConstantBuffer.gpuAddress, index: 5)
            encoder.setArgumentTable(argumentTable)
            encoder.setComputePipelineState(pipeline)
            encoder.dispatchThreadgroups(
                threadgroupsPerGrid: geometry.gridSize,
                threadsPerThreadgroup: geometry.threadgroupSize
            )
            encoder.barrier(
                afterEncoderStages: .dispatch,
                beforeEncoderStages: .blit,
                visibilityOptions: []
            )
            encoder.copy(
                sourceBuffer: outputBuffer,
                sourceOffset: outputOffset,
                destinationBuffer: stagingBuffer,
                destinationOffset: 0,
                size: outputDimension * sequenceLength * MemoryLayout<Float>.stride
            )
        }
        return readOutput(stagingBuffer, offset: 0, count: outputDimension * sequenceLength)
    }

    private func executeArgumentTablePrivateAliasedIOBuffer(
        device: MTLDevice,
        pipeline: MTLComputePipelineState,
        inputBuffer: MTLBuffer,
        weightBuffer: MTLBuffer,
        outputOffset: Int,
        inputDimension: Int,
        outputDimension: Int,
        sequenceLength: Int,
        maximumSequenceLength: Int
    ) throws -> [Float] {
        let inputLength = inputDimension * sequenceLength * MemoryLayout<Float>.stride
        let outputLength = outputDimension * maximumSequenceLength * MemoryLayout<Float>.stride
        let ioBufferLength = max(inputLength, outputOffset + outputLength)
        let ioBuffer = try #require(device.makeBuffer(
            length: ioBufferLength,
            options: .storageModePrivate
        ))
        let stagingBuffer = try #require(device.makeBuffer(
            length: outputDimension * sequenceLength * MemoryLayout<Float>.stride,
            options: .storageModeShared
        ))
        let runtimeConstantBuffer = try #require(device.makeBuffer(
            length: MemoryLayout<UInt32>.stride,
            options: .storageModeShared
        ))
        runtimeConstantBuffer.contents().storeBytes(of: UInt32(sequenceLength), as: UInt32.self)
        stagingBuffer.contents().initializeMemory(
            as: UInt8.self,
            repeating: 0,
            count: outputDimension * sequenceLength * MemoryLayout<Float>.stride
        )

        let allocator = MetalConstantBindingAllocator(device: device)
        let bindingTable = try allocator.makeBindingTable(
            bufferBindings: [
                (index: 0, buffer: ioBuffer, offset: 0),
                (index: 1, buffer: weightBuffer, offset: 0),
                (index: 2, buffer: ioBuffer, offset: outputOffset),
            ],
            bytesBindings: [
                (index: 3, value: uint32Bytes(UInt32(inputDimension))),
                (index: 4, value: uint32Bytes(UInt32(outputDimension))),
                (index: 5, value: uint32Bytes(UInt32(maximumSequenceLength))),
            ],
            argumentPolicy: .argumentTable
        )
        let residency = try MetalResidencyLease.required(
            device: device,
            label: "swift-lm.test.naive-gemm-private-aliased-io",
            buffers: [inputBuffer, weightBuffer, ioBuffer, stagingBuffer, runtimeConstantBuffer]
                + bindingTable.ownedResidencyBuffers
        )

        var submission = try MetalSubmissionContext(device: device)
        let geometry = naiveGeometry(
            pipeline: pipeline,
            outputDimension: outputDimension,
            sequenceLength: sequenceLength
        )
        try submission.withCompute(ephemeralResidency: residency) { encoder, argumentTable in
            encoder.copy(
                sourceBuffer: inputBuffer,
                sourceOffset: 0,
                destinationBuffer: ioBuffer,
                destinationOffset: 0,
                size: inputLength
            )
            encoder.barrier(
                afterEncoderStages: .blit,
                beforeEncoderStages: .dispatch,
                visibilityOptions: []
            )

            bindingTable.bind(to: argumentTable)
            argumentTable.setAddress(runtimeConstantBuffer.gpuAddress, index: 5)
            encoder.setArgumentTable(argumentTable)
            encoder.setComputePipelineState(pipeline)
            encoder.dispatchThreadgroups(
                threadgroupsPerGrid: geometry.gridSize,
                threadsPerThreadgroup: geometry.threadgroupSize
            )
            encoder.barrier(
                afterEncoderStages: .dispatch,
                beforeEncoderStages: .blit,
                visibilityOptions: []
            )
            encoder.copy(
                sourceBuffer: ioBuffer,
                sourceOffset: outputOffset,
                destinationBuffer: stagingBuffer,
                destinationOffset: 0,
                size: outputDimension * sequenceLength * MemoryLayout<Float>.stride
            )
        }
        return readOutput(stagingBuffer, offset: 0, count: outputDimension * sequenceLength)
    }

    private func runRMSNormToGEMMChainScenario(
        inputDimension: Int,
        slotDimension: Int,
        outputDimension: Int,
        sequenceLength: Int,
        maximumSequenceLength: Int,
        outputOffset: Int,
        epsilon: Float,
        tolerance: Float
    ) throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device")
            return
        }

        let input = makeInput(inputDimension: inputDimension, sequenceLength: sequenceLength)
        let normWeight = makeNormWeight(dimension: inputDimension)
        let gemmWeight = makeWeight(inputDimension: inputDimension, outputDimension: outputDimension)
        let normalized = cpuRMSNormReference(
            input: input,
            weight: normWeight,
            dimension: inputDimension,
            sequenceLength: sequenceLength,
            epsilon: epsilon
        )
        let expected = cpuReference(
            input: normalized,
            weight: gemmWeight,
            inputDimension: inputDimension,
            outputDimension: outputDimension,
            sequenceLength: sequenceLength
        )

        let hiddenBuffer = try #require(device.makeBuffer(
            bytes: input,
            length: input.count * MemoryLayout<Float>.stride,
            options: .storageModeShared
        ))
        let normWeightBuffer = try #require(device.makeBuffer(
            bytes: normWeight,
            length: normWeight.count * MemoryLayout<BFloat16>.stride,
            options: .storageModeShared
        ))
        let gemmWeightBuffer = try #require(device.makeBuffer(
            bytes: gemmWeight,
            length: gemmWeight.count * MemoryLayout<BFloat16>.stride,
            options: .storageModeShared
        ))
        let scratchLength = maximumSequenceLength * slotDimension * 2 * MemoryLayout<Float>.stride
        let scratchBuffer = try #require(device.makeBuffer(
            length: scratchLength,
            options: .storageModePrivate
        ))
        let stagingBuffer = try #require(device.makeBuffer(
            length: outputDimension * sequenceLength * MemoryLayout<Float>.stride,
            options: .storageModeShared
        ))
        let runtimeConstantBuffer = try #require(device.makeBuffer(
            length: MemoryLayout<UInt32>.stride,
            options: .storageModeShared
        ))
        runtimeConstantBuffer.contents().storeBytes(of: UInt32(sequenceLength), as: UInt32.self)
        stagingBuffer.contents().initializeMemory(
            as: UInt8.self,
            repeating: 0,
            count: outputDimension * sequenceLength * MemoryLayout<Float>.stride
        )

        let gemmPipeline = try makePipeline(device: device)
        let normPipeline = try makeRMSNormPipeline(device: device)
        let allocator = MetalConstantBindingAllocator(device: device)
        let normBindings = try allocator.makeBindingTable(
            bufferBindings: [
                (index: 0, buffer: hiddenBuffer, offset: 0),
                (index: 1, buffer: normWeightBuffer, offset: 0),
                (index: 2, buffer: scratchBuffer, offset: 0),
            ],
            bytesBindings: [
                (index: 3, value: uint32Bytes(UInt32(inputDimension))),
                (index: 4, value: floatBytes(epsilon)),
                (index: 5, value: floatBytes(0)),
                (index: 6, value: uint32Bytes(UInt32(maximumSequenceLength))),
            ],
            argumentPolicy: .argumentTable
        )
        let gemmBindings = try allocator.makeBindingTable(
            bufferBindings: [
                (index: 0, buffer: scratchBuffer, offset: 0),
                (index: 1, buffer: gemmWeightBuffer, offset: 0),
                (index: 2, buffer: scratchBuffer, offset: outputOffset),
            ],
            bytesBindings: [
                (index: 3, value: uint32Bytes(UInt32(inputDimension))),
                (index: 4, value: uint32Bytes(UInt32(outputDimension))),
                (index: 5, value: uint32Bytes(UInt32(maximumSequenceLength))),
            ],
            argumentPolicy: .argumentTable
        )
        let residency = try MetalResidencyLease.required(
            device: device,
            label: "swift-lm.test.rmsnorm-gemm-chain",
            buffers: [
                hiddenBuffer,
                normWeightBuffer,
                gemmWeightBuffer,
                scratchBuffer,
                stagingBuffer,
                runtimeConstantBuffer,
            ] + normBindings.ownedResidencyBuffers + gemmBindings.ownedResidencyBuffers
        )

        var submission = try MetalSubmissionContext(device: device)
        let normThreadgroupSize = MTLSize(
            width: min(1024, max(normPipeline.threadExecutionWidth, 1) * 32),
            height: 1,
            depth: 1
        )
        let normGridSize = MTLSize(width: maximumSequenceLength, height: 1, depth: 1)
        let gemmGeometry = naiveGeometry(
            pipeline: gemmPipeline,
            outputDimension: outputDimension,
            sequenceLength: sequenceLength
        )

        let normDescriptor = MetalDispatchDescriptor(
            pipeline: normPipeline,
            gridSize: normGridSize,
            threadgroupSize: normThreadgroupSize,
            threadgroupMemoryLength: 0,
            barrierPolicy: .bufferBarrier
        )
        let gemmDescriptor = MetalDispatchDescriptor(
            pipeline: gemmPipeline,
            gridSize: gemmGeometry.gridSize,
            threadgroupSize: gemmGeometry.threadgroupSize,
            threadgroupMemoryLength: 0,
            barrierPolicy: .bufferBarrier
        )

        try submission.withCompute(ephemeralResidency: residency) { encoder, argumentTable in
            normBindings.bind(to: argumentTable)
            argumentTable.setAddress(runtimeConstantBuffer.gpuAddress, index: 6)
            normDescriptor.encode(on: encoder, argumentTable: argumentTable)

            gemmBindings.bind(to: argumentTable)
            argumentTable.setAddress(runtimeConstantBuffer.gpuAddress, index: 5)
            gemmDescriptor.encode(on: encoder, argumentTable: argumentTable)
            encoder.barrier(
                afterEncoderStages: .dispatch,
                beforeEncoderStages: .blit,
                visibilityOptions: []
            )
            encoder.copy(
                sourceBuffer: scratchBuffer,
                sourceOffset: outputOffset,
                destinationBuffer: stagingBuffer,
                destinationOffset: 0,
                size: outputDimension * sequenceLength * MemoryLayout<Float>.stride
            )
        }

        let actual = readOutput(stagingBuffer, offset: 0, count: outputDimension * sequenceLength)
        let maxError = zip(actual, expected).reduce(Float.zero) { partial, pair in
            max(partial, abs(pair.0 - pair.1))
        }
        #expect(
            maxError < tolerance,
            """
            representative RMSNorm->GEMM chain drifted
            maxError=\(maxError)
            actualPrefix=\(actual.prefix(8).map { String(format: "%.4f", $0) }.joined(separator: ", "))
            expectedPrefix=\(expected.prefix(8).map { String(format: "%.4f", $0) }.joined(separator: ", "))
            """
        )
    }

    private func naiveGeometry(
        pipeline: MTLComputePipelineState,
        outputDimension: Int,
        sequenceLength: Int
    ) -> (gridSize: MTLSize, threadgroupSize: MTLSize) {
        let rowsPerThreadgroup = 2
        let simdWidth = max(pipeline.threadExecutionWidth, 1)
        let threads = min(
            simdWidth * rowsPerThreadgroup,
            pipeline.maxTotalThreadsPerThreadgroup
        )
        return (
            gridSize: MTLSize(
                width: (outputDimension + rowsPerThreadgroup - 1) / rowsPerThreadgroup,
                height: sequenceLength,
                depth: 1
            ),
            threadgroupSize: MTLSize(width: threads, height: 1, depth: 1)
        )
    }

    private func readOutput(_ buffer: MTLBuffer, offset: Int, count: Int) -> [Float] {
        let pointer = buffer.contents().advanced(by: offset).bindMemory(to: Float.self, capacity: count)
        return (0..<count).map { pointer[$0] }
    }

    private func cpuReference(
        input: [Float],
        weight: [BFloat16],
        inputDimension: Int,
        outputDimension: Int,
        sequenceLength: Int
    ) -> [Float] {
        var expected = [Float](repeating: .zero, count: outputDimension * sequenceLength)
        for seq in 0..<sequenceLength {
            for row in 0..<outputDimension {
                var sum: Float = 0
                for column in 0..<inputDimension {
                    sum += input[seq * inputDimension + column] * Float(weight[row * inputDimension + column])
                }
                expected[seq * outputDimension + row] = sum
            }
        }
        return expected
    }

    private func makeInput(inputDimension: Int, sequenceLength: Int) -> [Float] {
        (0..<(inputDimension * sequenceLength)).map { index in
            Float(((index * 5) % 19) - 9) * 0.125
        }
    }

    private func makeWeight(inputDimension: Int, outputDimension: Int) -> [BFloat16] {
        (0..<(inputDimension * outputDimension)).map { index in
            BFloat16(Float(((index * 7) % 23) - 11) * 0.0625)
        }
    }

    private func makeNormWeight(dimension: Int) -> [BFloat16] {
        (0..<dimension).map { index in
            BFloat16(1.0 + Float((index % 17) - 8) * 0.015625)
        }
    }

    private func cpuRMSNormReference(
        input: [Float],
        weight: [BFloat16],
        dimension: Int,
        sequenceLength: Int,
        epsilon: Float
    ) -> [Float] {
        var output = [Float](repeating: .zero, count: input.count)
        for seq in 0..<sequenceLength {
            let rowBase = seq * dimension
            var sumSquares: Float = 0
            for index in 0..<dimension {
                let value = input[rowBase + index]
                sumSquares += value * value
            }
            let scale = 1 / sqrt(sumSquares / Float(dimension) + epsilon)
            for index in 0..<dimension {
                output[rowBase + index] = input[rowBase + index] * scale * Float(weight[index])
            }
        }
        return output
    }

    private func uint32Bytes(_ value: UInt32) -> [UInt8] {
        withUnsafeBytes(of: value.littleEndian) { rawBuffer in
            Array(rawBuffer)
        }
    }

    private func floatBytes(_ value: Float) -> [UInt8] {
        withUnsafeBytes(of: value) { rawBuffer in
            Array(rawBuffer)
        }
    }
}
