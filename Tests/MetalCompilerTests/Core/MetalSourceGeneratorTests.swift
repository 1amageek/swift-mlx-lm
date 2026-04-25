import Testing
import Metal
@testable import MetalCompiler

/// Verify MetalSourceGenerator produces valid, compilable MSL
/// that computes the same results as the hardcoded kernels.
@Suite("Metal Source Generator", .serialized)
struct MetalSourceGeneratorTests {

    @Test("Generated RMSNorm compiles for all precision × weight format combinations")
    func rmsNormCompiles() throws {
        guard let device = MTLCreateSystemDefaultDevice() else { return }

        let precisions: [(MetalSourceGenerator.BufferPrecision, String)] = [
            (.float16, "f16"),
            (.float32, "f32"),
        ]
        let weightFormats: [(MetalSourceGenerator.WeightFormat, String)] = [
            (.float16, "fp16"),
            (.bfloat16, "bf16"),
        ]

        for (precision, precisionLabel) in precisions {
            for (weightFormat, weightLabel) in weightFormats {
                let name = "rms_norm_\(precisionLabel)_\(weightLabel)"
                let source = MetalSourceGenerator.commonHeader + "\n\n"
                    + MetalSourceGenerator.generateReduction(
                        name: name, dimension: 2048, epsilon: 1e-5,
                        bufferPrecision: precision, weightFormat: weightFormat)

                let options = MTLCompileOptions()
                options.languageVersion = .version4_0
                let library = try device.makeLibrary(source: source, options: options)
                let function = library.makeFunction(name: name)
                #expect(function != nil, "Failed to compile \(name)")
            }
        }
    }

    @Test("Generated SwiGLU compiles for both precisions")
    func swigluCompiles() throws {
        guard let device = MTLCreateSystemDefaultDevice() else { return }

        for precision in [MetalSourceGenerator.BufferPrecision.float16, .float32] {
            let name = "swiglu_\(precision)"
            let source = MetalSourceGenerator.commonHeader + "\n\n"
                + MetalSourceGenerator.generateSwiGLU(name: name, bufferPrecision: precision)

            let options = MTLCompileOptions()
            options.languageVersion = .version4_0
            let library = try device.makeLibrary(source: source, options: options)
            #expect(library.makeFunction(name: name) != nil, "Failed to compile \(name)")
        }
    }

    @Test("Generated GEMM compiles for all weight formats")
    func gemmCompiles() throws {
        guard let device = MTLCreateSystemDefaultDevice() else { return }

        let formats: [(MetalSourceGenerator.WeightFormat, String)] = [
            (.float16, "fp16"), (.bfloat16, "bf16")
        ]

        for (format, label) in formats {
            for precision in [MetalSourceGenerator.BufferPrecision.float16, .float32] {
                let name = "gemm_\(label)_\(precision)"
                let source = MetalSourceGenerator.commonHeader + "\n\n"
                    + MetalSourceGenerator.generateGEMM(
                        name: name, bufferPrecision: precision, weightFormat: format)

                let options = MTLCompileOptions()
                options.languageVersion = .version4_0
                let library = try device.makeLibrary(source: source, options: options)
                #expect(library.makeFunction(name: name) != nil, "Failed to compile \(name)")
            }
        }
    }

    @Test("MPP GEMM matches CPU reference for BF16 prefill projection")
    func mppGEMMMatchesCPUReference() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device")
            return
        }
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }

        let inputDimension = 7
        let outputDimension = 5
        let sequenceLength = 4

        var input: [Float] = (0..<(inputDimension * sequenceLength)).map {
            Float(($0 % 11) - 5) * 0.25
        }
        var weight: [BFloat16] = (0..<(outputDimension * inputDimension)).map {
            BFloat16(Float((($0 * 3) % 13) - 6) * 0.125)
        }
        var output = [Float](repeating: .zero, count: outputDimension * sequenceLength)

        let inputBuffer = try #require(device.makeBuffer(
            bytes: &input,
            length: input.count * MemoryLayout<Float>.size,
            options: .storageModeShared
        ))
        let weightBuffer = try #require(device.makeBuffer(
            bytes: &weight,
            length: weight.count * MemoryLayout<BFloat16>.size,
            options: .storageModeShared
        ))
        let outputBuffer = try #require(device.makeBuffer(
            bytes: &output,
            length: output.count * MemoryLayout<Float>.size,
            options: .storageModeShared
        ))

        let source = MetalSourceGenerator.commonHeader + "\n\n"
            + MetalSourceGenerator.generateMPPGEMM(
                name: "test_mpp_gemm_bf16_f32s",
                bufferPrecision: .float32,
                weightFormat: .bfloat16
            )
        let options = MTLCompileOptions()
        options.languageVersion = .version4_0
        let library = try device.makeLibrary(source: source, options: options)
        let pipeline = try device.makeComputePipelineState(
            function: try #require(library.makeFunction(name: "test_mpp_gemm_bf16_f32s"))
        )

        let queue = try #require(device.makeCommandQueue())
        let commandBuffer = try #require(queue.makeCommandBuffer())
        let encoder = try #require(commandBuffer.makeComputeCommandEncoder())

        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(inputBuffer, offset: 0, index: 0)
        encoder.setBuffer(weightBuffer, offset: 0, index: 1)
        encoder.setBuffer(outputBuffer, offset: 0, index: 2)
        var inDim = UInt32(inputDimension)
        var outDim = UInt32(outputDimension)
        var seqLen = UInt32(sequenceLength)
        var rowStride = UInt32(inputDimension)
        encoder.setBytes(&inDim, length: MemoryLayout<UInt32>.size, index: 3)
        encoder.setBytes(&outDim, length: MemoryLayout<UInt32>.size, index: 4)
        encoder.setBytes(&seqLen, length: MemoryLayout<UInt32>.size, index: 5)
        encoder.setBytes(&rowStride, length: MemoryLayout<UInt32>.size, index: 6)
        encoder.dispatchThreadgroups(
            MTLSize(
                width: (outputDimension + 31) / 32,
                height: (sequenceLength + 63) / 64,
                depth: 1
            ),
            threadsPerThreadgroup: MTLSize(
                width: min(pipeline.threadExecutionWidth * 4, pipeline.maxTotalThreadsPerThreadgroup),
                height: 1,
                depth: 1
            )
        )
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        let result = outputBuffer.contents().bindMemory(
            to: Float.self,
            capacity: output.count
        )

        var expected = [Float](repeating: .zero, count: output.count)
        for seq in 0..<sequenceLength {
            for row in 0..<outputDimension {
                var sum: Float = 0
                for column in 0..<inputDimension {
                    sum += input[seq * inputDimension + column] * Float(weight[row * inputDimension + column])
                }
                expected[seq * outputDimension + row] = sum
            }
        }

        let actual = (0..<output.count).map { result[$0] }
        let maxError = zip(actual, expected).reduce(Float.zero) { partial, pair in
            max(partial, abs(pair.0 - pair.1))
        }
        #expect(maxError < 0.01, "MPP GEMM drifted: maxError=\(maxError)")
    }

    @Test("Q3G64 dequant then MPP GEMM matches CPU reference")
    func q3Group64DequantThenMPPGEMMMatchesCPUReference() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device")
            return
        }
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }

        let format = AffineQ3Group64Format()
        let inputDimension = 128
        let outputDimension = 5
        let sequenceLength = 4
        let blocksPerRow = inputDimension / format.groupSize

        var packedWeights: [UInt8] = []
        var dequantizedWeights: [Float] = []
        for row in 0..<outputDimension {
            for block in 0..<blocksPerRow {
                let scale = 0.03125 * Float(block + 1) + 0.0078125 * Float(row)
                let zero = -0.125 + 0.0625 * Float(row) - 0.015625 * Float(block)
                let weights = (0..<format.groupSize).map {
                    UInt32(($0 + block * 3 + row * 5) % 8)
                }
                packedWeights.append(contentsOf: makeQuantizedBlock(
                    weights: weights,
                    bits: format.bits,
                    scale: scale,
                    zero: zero,
                    payloadByteCount: format.bytesPerBlock - 4
                ))
                dequantizedWeights.append(contentsOf: weights.map { scale * Float($0) + zero })
            }
        }

        var input: [Float] = (0..<(inputDimension * sequenceLength)).map {
            Float(($0 % 17) - 8) * 0.0625
        }
        var output = [Float](repeating: .zero, count: outputDimension * sequenceLength)

        let packedBuffer = try #require(device.makeBuffer(
            bytes: packedWeights,
            length: packedWeights.count,
            options: .storageModeShared
        ))
        let scratchBuffer = try #require(device.makeBuffer(
            length: outputDimension * inputDimension * MemoryLayout<UInt16>.stride,
            options: .storageModePrivate
        ))
        let inputBuffer = try #require(device.makeBuffer(
            bytes: &input,
            length: input.count * MemoryLayout<Float>.stride,
            options: .storageModeShared
        ))
        let outputBuffer = try #require(device.makeBuffer(
            bytes: &output,
            length: output.count * MemoryLayout<Float>.stride,
            options: .storageModeShared
        ))

        let dequantName = "test_dequant_q3_g64_bf16"
        let gemmName = "test_mpp_q3_g64_bf16_f32s"
        let source = MetalSourceGenerator.commonHeader + "\n\n"
            + MetalSourceGenerator.generateUnifiedDequantToBFloat(
                name: dequantName,
                format: format
            ) + "\n\n"
            + MetalSourceGenerator.generateMPPGEMM(
                name: gemmName,
                bufferPrecision: .float32,
                weightFormat: .bfloat16
            )
        let options = MTLCompileOptions()
        options.languageVersion = .version4_0
        let library = try device.makeLibrary(source: source, options: options)
        let dequantPipeline = try device.makeComputePipelineState(
            function: try #require(library.makeFunction(name: dequantName))
        )
        let gemmPipeline = try device.makeComputePipelineState(
            function: try #require(library.makeFunction(name: gemmName))
        )

        let queue = try #require(device.makeCommandQueue())
        let commandBuffer = try #require(queue.makeCommandBuffer())
        let encoder = try #require(commandBuffer.makeComputeCommandEncoder())

        var inDim = UInt32(inputDimension)
        var outDim = UInt32(outputDimension)
        var seqLen = UInt32(sequenceLength)
        var rowStride = UInt32(inputDimension)

        encoder.setComputePipelineState(dequantPipeline)
        encoder.setBuffer(packedBuffer, offset: 0, index: 0)
        encoder.setBuffer(scratchBuffer, offset: 0, index: 1)
        encoder.setBytes(&inDim, length: MemoryLayout<UInt32>.stride, index: 2)
        encoder.setBytes(&outDim, length: MemoryLayout<UInt32>.stride, index: 3)
        encoder.dispatchThreadgroups(
            MTLSize(width: outputDimension, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)
        )
        encoder.memoryBarrier(resources: [scratchBuffer])

        encoder.setComputePipelineState(gemmPipeline)
        encoder.setBuffer(inputBuffer, offset: 0, index: 0)
        encoder.setBuffer(scratchBuffer, offset: 0, index: 1)
        encoder.setBuffer(outputBuffer, offset: 0, index: 2)
        encoder.setBytes(&inDim, length: MemoryLayout<UInt32>.stride, index: 3)
        encoder.setBytes(&outDim, length: MemoryLayout<UInt32>.stride, index: 4)
        encoder.setBytes(&seqLen, length: MemoryLayout<UInt32>.stride, index: 5)
        encoder.setBytes(&rowStride, length: MemoryLayout<UInt32>.stride, index: 6)
        encoder.dispatchThreadgroups(
            MTLSize(
                width: (outputDimension + 31) / 32,
                height: (sequenceLength + 63) / 64,
                depth: 1
            ),
            threadsPerThreadgroup: MTLSize(
                width: min(gemmPipeline.threadExecutionWidth * 4, gemmPipeline.maxTotalThreadsPerThreadgroup),
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

        let actualPointer = outputBuffer.contents().bindMemory(to: Float.self, capacity: output.count)
        let actual = (0..<output.count).map { actualPointer[$0] }
        var expected = [Float](repeating: .zero, count: output.count)
        for seq in 0..<sequenceLength {
            for row in 0..<outputDimension {
                var sum: Float = 0
                for column in 0..<inputDimension {
                    sum += input[seq * inputDimension + column]
                        * dequantizedWeights[row * inputDimension + column]
                }
                expected[seq * outputDimension + row] = sum
            }
        }

        let maxError = zip(actual, expected).reduce(Float.zero) { partial, pair in
            max(partial, abs(pair.0 - pair.1))
        }
        #expect(maxError < 0.02, "Q3G64 dequant to MPP GEMM drifted: maxError=\(maxError)")
    }

    @Test("MPP GEMM matches CPU reference for FP16 prefill projection")
    func mppFP16GEMMMatchesCPUReference() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device")
            return
        }
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }

        let inputDimension = 7
        let outputDimension = 5
        let sequenceLength = 4

        var input: [Float] = (0..<(inputDimension * sequenceLength)).map {
            Float(($0 % 11) - 5) * 0.25
        }
        var weight: [Float16] = (0..<(outputDimension * inputDimension)).map {
            Float16(Float((($0 * 3) % 13) - 6) * 0.125)
        }
        var output = [Float](repeating: .zero, count: outputDimension * sequenceLength)

        let inputBuffer = try #require(device.makeBuffer(
            bytes: &input,
            length: input.count * MemoryLayout<Float>.size,
            options: .storageModeShared
        ))
        let weightBuffer = try #require(device.makeBuffer(
            bytes: &weight,
            length: weight.count * MemoryLayout<Float16>.size,
            options: .storageModeShared
        ))
        let outputBuffer = try #require(device.makeBuffer(
            bytes: &output,
            length: output.count * MemoryLayout<Float>.size,
            options: .storageModeShared
        ))

        let source = MetalSourceGenerator.commonHeader + "\n\n"
            + MetalSourceGenerator.generateMPPGEMM(
                name: "test_mpp_gemm_f32s",
                bufferPrecision: .float32,
                weightFormat: .float16
            )
        let options = MTLCompileOptions()
        options.languageVersion = .version4_0
        let library = try device.makeLibrary(source: source, options: options)
        let pipeline = try device.makeComputePipelineState(
            function: try #require(library.makeFunction(name: "test_mpp_gemm_f32s"))
        )

        let queue = try #require(device.makeCommandQueue())
        let commandBuffer = try #require(queue.makeCommandBuffer())
        let encoder = try #require(commandBuffer.makeComputeCommandEncoder())

        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(inputBuffer, offset: 0, index: 0)
        encoder.setBuffer(weightBuffer, offset: 0, index: 1)
        encoder.setBuffer(outputBuffer, offset: 0, index: 2)
        var inDim = UInt32(inputDimension)
        var outDim = UInt32(outputDimension)
        var seqLen = UInt32(sequenceLength)
        var rowStride = UInt32(inputDimension)
        encoder.setBytes(&inDim, length: MemoryLayout<UInt32>.size, index: 3)
        encoder.setBytes(&outDim, length: MemoryLayout<UInt32>.size, index: 4)
        encoder.setBytes(&seqLen, length: MemoryLayout<UInt32>.size, index: 5)
        encoder.setBytes(&rowStride, length: MemoryLayout<UInt32>.size, index: 6)
        encoder.dispatchThreadgroups(
            MTLSize(
                width: (outputDimension + 31) / 32,
                height: (sequenceLength + 63) / 64,
                depth: 1
            ),
            threadsPerThreadgroup: MTLSize(
                width: min(pipeline.threadExecutionWidth * 4, pipeline.maxTotalThreadsPerThreadgroup),
                height: 1,
                depth: 1
            )
        )
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        let result = outputBuffer.contents().bindMemory(
            to: Float.self,
            capacity: output.count
        )

        var expected = [Float](repeating: .zero, count: output.count)
        for seq in 0..<sequenceLength {
            for row in 0..<outputDimension {
                var sum: Float = 0
                for column in 0..<inputDimension {
                    sum += input[seq * inputDimension + column] * Float(weight[row * inputDimension + column])
                }
                expected[seq * outputDimension + row] = sum
            }
        }

        let actual = (0..<output.count).map { result[$0] }
        let maxError = zip(actual, expected).reduce(Float.zero) { partial, pair in
            max(partial, abs(pair.0 - pair.1))
        }
        #expect(maxError < 0.01, "MPP FP16 GEMM drifted: maxError=\(maxError)")
    }

    @Test("Quantized Q4 GEMM matches CPU reference with padded scratch input stride")
    func quantizedQ4GEMMMatchesCPUReferenceWithPaddedScratchInputStride() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device")
            return
        }
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }

        let inputDimension = 64
        let outputDimension = 4
        let sequenceLength = 4
        let inputRowStride = 256

        var input = [Float](repeating: 9_999, count: sequenceLength * inputRowStride)
        for seq in 0..<sequenceLength {
            for column in 0..<inputDimension {
                input[seq * inputRowStride + column] = Float(((seq + 3) * ((column % 7) - 3))) * 0.125
            }
        }

        var weightBytes: [UInt8] = []
        weightBytes.reserveCapacity(outputDimension * 36)
        func appendBytes<T>(_ value: T) {
            withUnsafeBytes(of: value) { weightBytes.append(contentsOf: $0) }
        }
        for row in 0..<outputDimension {
            let scale = Float16(0.25)
            let zero = Float16(0)
            appendBytes(scale)
            appendBytes(zero)
            let nibble = UInt8(row + 1)
            let packed = nibble | (nibble << 4)
            weightBytes.append(contentsOf: repeatElement(packed, count: inputDimension / 2))
        }

        let inputBuffer = try #require(device.makeBuffer(
            bytes: input,
            length: input.count * MemoryLayout<Float>.size,
            options: .storageModeShared
        ))
        let weightBuffer = try #require(device.makeBuffer(
            bytes: weightBytes,
            length: weightBytes.count,
            options: .storageModeShared
        ))
        let outputBuffer = try #require(device.makeBuffer(
            length: outputDimension * sequenceLength * MemoryLayout<Float>.size,
            options: .storageModeShared
        ))

        let source = MetalSourceGenerator.commonHeader + "\n\n"
            + MetalSourceGenerator.generateQuantizedGEMM_Q4(
                name: "test_quantized_gemm_q4_g64_f32s",
                bufferPrecision: .float32,
                groupSize: 64
            )
        let options = MTLCompileOptions()
        options.languageVersion = .version4_0
        let library = try device.makeLibrary(source: source, options: options)
        let pipeline = try device.makeComputePipelineState(
            function: try #require(library.makeFunction(name: "test_quantized_gemm_q4_g64_f32s"))
        )

        let queue = try #require(device.makeCommandQueue())
        let commandBuffer = try #require(queue.makeCommandBuffer())
        let encoder = try #require(commandBuffer.makeComputeCommandEncoder())
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(inputBuffer, offset: 0, index: 0)
        encoder.setBuffer(weightBuffer, offset: 0, index: 1)
        encoder.setBuffer(outputBuffer, offset: 0, index: 2)
        var inDim = UInt32(inputDimension)
        var outDim = UInt32(outputDimension)
        var seqLen = UInt32(sequenceLength)
        var rowStride = UInt32(inputRowStride)
        encoder.setBytes(&inDim, length: MemoryLayout<UInt32>.size, index: 3)
        encoder.setBytes(&outDim, length: MemoryLayout<UInt32>.size, index: 4)
        encoder.setBytes(&seqLen, length: MemoryLayout<UInt32>.size, index: 5)
        encoder.setBytes(&rowStride, length: MemoryLayout<UInt32>.size, index: 6)
        let simdWidth = pipeline.threadExecutionWidth
        let threads = min(2 * simdWidth, pipeline.maxTotalThreadsPerThreadgroup)
        encoder.dispatchThreadgroups(
            MTLSize(width: (outputDimension + 1) / 2, height: sequenceLength, depth: 1),
            threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1)
        )
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        let actualPointer = outputBuffer.contents().bindMemory(
            to: Float.self,
            capacity: outputDimension * sequenceLength
        )
        let actual = (0..<(outputDimension * sequenceLength)).map { actualPointer[$0] }

        var expected = [Float](repeating: .zero, count: outputDimension * sequenceLength)
        for seq in 0..<sequenceLength {
            let inputSum = (0..<inputDimension).reduce(Float.zero) { partial, column in
                partial + input[seq * inputRowStride + column]
            }
            for row in 0..<outputDimension {
                expected[seq * outputDimension + row] = inputSum * (0.25 * Float(row + 1))
            }
        }

        let maxError = zip(actual, expected).reduce(Float.zero) { partial, pair in
            max(partial, abs(pair.0 - pair.1))
        }
        #expect(
            maxError < 0.001,
            """
            Quantized Q4 GEMM drifted with padded scratch stride
            maxError=\(maxError)
            actualPrefix=\(actual.prefix(8).map { String(format: "%.4f", $0) }.joined(separator: ", "))
            expectedPrefix=\(expected.prefix(8).map { String(format: "%.4f", $0) }.joined(separator: ", "))
            """
        )
    }

    @Test("Generated structural kernels compile")
    func structuralCompiles() throws {
        guard let device = MTLCreateSystemDefaultDevice() else { return }

        for precision in [MetalSourceGenerator.BufferPrecision.float16, .float32] {
            var allSource = MetalSourceGenerator.commonHeader + "\n\n"
            allSource += MetalSourceGenerator.generateCopy(
                name: "copy_\(precision)", bufferPrecision: precision) + "\n\n"
            allSource += MetalSourceGenerator.generateResidualAdd(
                name: "add_\(precision)", bufferPrecision: precision) + "\n\n"
            allSource += MetalSourceGenerator.generateArgmax(
                name: "argmax_\(precision)", bufferPrecision: precision) + "\n\n"
            allSource += MetalSourceGenerator.generateEmbeddingLookup(
                name: "emb_\(precision)", bufferPrecision: precision, weightFormat: .bfloat16)

            let options = MTLCompileOptions()
            options.languageVersion = .version4_0
            let library = try device.makeLibrary(source: allSource, options: options)
            #expect(library.makeFunction(name: "copy_\(precision)") != nil)
            #expect(library.makeFunction(name: "add_\(precision)") != nil)
            #expect(library.makeFunction(name: "argmax_\(precision)") != nil)
            #expect(library.makeFunction(name: "emb_\(precision)") != nil)
        }
    }

    @Test("All precision × weight format combinations produce unique kernels")
    func noDuplicateVariants() {
        // Same computation, different precision/format → different source
        let a = MetalSourceGenerator.generateReduction(
            name: "norm", dimension: 2048, epsilon: 1e-5,
            bufferPrecision: .float16, weightFormat: .float16)
        let b = MetalSourceGenerator.generateReduction(
            name: "norm", dimension: 2048, epsilon: 1e-5,
            bufferPrecision: .float32, weightFormat: .bfloat16)
        #expect(a != b, "Different precision/format should produce different source")

        // Verify BF16 source uses bf16_to_float
        #expect(b.contains("bf16_to_float"))
        #expect(!a.contains("bf16_to_float"))
    }

    @Test("SSM recurrence resolves distinct BF16 kernel variants")
    func ssmRecurrenceKernelNamesIncludeWeightFormat() {
        let fragment = SSMRecurrenceFragment(
            headCount: 8,
            groupCount: 8,
            keyHeadDimension: 64,
            valueHeadDimension: 64,
            convKernelSize: 4
        )

        #expect(fragment.kernelName(context: KernelContext(bufferPrecision: .float16, weightFormat: .float16)) == "ssm_recurrence")
        #expect(fragment.kernelName(context: KernelContext(bufferPrecision: .float16, weightFormat: .bfloat16)) == "ssm_recurrence_bf16")
        #expect(fragment.kernelName(context: KernelContext(bufferPrecision: .float32, weightFormat: .float16)) == "ssm_recurrence_f32")
        #expect(fragment.kernelName(context: KernelContext(bufferPrecision: .float32, weightFormat: .bfloat16)) == "ssm_recurrence_bf16_f32")
        #expect(SSMRecurrenceFragment.sequenceKernelName(bufferPrecision: .float32, weightFormat: .bfloat16) == "ssm_recurrence_seq_bf16_f32")
    }

    @Test("SSM recurrence reads Qwen norm weight as float unit offset")
    func ssmRecurrenceUsesFloatUnitOffsetNormWeight() {
        let decodeSource = MetalSourceGenerator.generateSSMRecurrence(
            name: "ssm_recurrence_bf16",
            bufferPrecision: .float16,
            weightFormat: .bfloat16,
            convDimension: 4096,
            maxThreadgroupSize: 1024,
            headCount: 16,
            groupCount: 16,
            keyHeadDimension: 128,
            valueHeadDimension: 128
        )
        let sequenceSource = MetalSourceGenerator.generateSSMRecurrenceSequence(
            name: "ssm_recurrence_seq_bf16_f32",
            bufferPrecision: .float32,
            weightFormat: .bfloat16,
            convDimension: 4096,
            maxThreadgroupSize: 1024,
            headCount: 16,
            groupCount: 16,
            keyHeadDimension: 128,
            valueHeadDimension: 128
        )

        for source in [decodeSource, sequenceSource] {
            #expect(source.contains("device const float* normWeight [[buffer(5)]]"))
            #expect(source.contains("1.0f + normWeight[d]"))
            #expect(!source.contains("bf16_to_float(normWeight[d])"))
        }
    }

    @Test("Batched QK norm decode applies unit-offset weight bias")
    func batchedQKNormDecodeAppliesWeightBias() {
        let source = MetalSourceGenerator.generateBatchedPerHead2(
            name: "batched_qk_rms_norm_bf16_2",
            bufferPrecision: .float16,
            weightFormat: .bfloat16
        )

        #expect(source.contains("constant float& weightBias     [[buffer(8)]]"))
        #expect(source.contains("float affine = bf16_to_float(weight[i]) + weightBias;"))
        #expect(source.contains("scale * affine"))
    }

    @Test("Unified quantized GEMV emits MLX-compatible Q6 bit extraction")
    func unifiedQuantizedGEMVQ6EmitsMLXBitPattern() throws {
        let formats: [any QuantizationFormat] = [
            AffineQ6Group16Format(),
            AffineQ6Group32Format(),
        ]
        for format in formats {
            let source = MetalSourceGenerator.generateUnifiedQuantizedGEMV(
                name: "test_\(format.gemvKernelName)",
                format: format,
                bufferPrecision: .float16
            )
            // MLX extract_bits<6>: 4 weights span 3 bytes. Each weight slot is
            // selected by a ternary chain keyed on `k & 3`.
            #expect(source.contains("qs[(((k)) >> 2) * 3 + 0] & 0x3f"),
                "Missing w[0] extraction for \(format.schemeIdentifier)")
            #expect(source.contains("((qs[(((k)) >> 2) * 3 + 0] >> 6) & 0x03) | ((qs[(((k)) >> 2) * 3 + 1] & 0x0f) << 2)"),
                "Missing w[1] extraction for \(format.schemeIdentifier)")
            #expect(source.contains("((qs[(((k)) >> 2) * 3 + 1] >> 4) & 0x0f) | ((qs[(((k)) >> 2) * 3 + 2] & 0x03) << 4)"),
                "Missing w[2] extraction for \(format.schemeIdentifier)")
            #expect(source.contains("qs[(((k)) >> 2) * 3 + 2] >> 2"),
                "Missing w[3] extraction for \(format.schemeIdentifier)")
        }
    }

    @Test("Unified quantized GEMV emits MLX-compatible Q2 bit extraction")
    func unifiedQuantizedGEMVQ2EmitsMLXBitPattern() throws {
        let formats: [any QuantizationFormat] = [
            AffineQ2Group16Format(),
            AffineQ2Group32Format(),
        ]
        for format in formats {
            let source = MetalSourceGenerator.generateUnifiedQuantizedGEMV(
                name: "test_\(format.gemvKernelName)",
                format: format,
                bufferPrecision: .float16
            )
            // Aligned Q2: 4 weights per byte. perWeightExpression inlined in per-k loop.
            // `qs[(k) >> 2]` selects the byte; `((k) & 3) * 2` shifts to low bits; `& 0x3` masks.
            #expect(source.contains("qs[(k) >> 2]"),
                "Missing Q2 byte index for \(format.schemeIdentifier)")
            #expect(source.contains("((k) & 3) * 2"),
                "Missing Q2 sub-byte shift for \(format.schemeIdentifier)")
            #expect(source.contains("& 0x3)"),
                "Missing Q2 2-bit mask for \(format.schemeIdentifier)")
        }
    }

    @Test("Unified quantized GEMV emits MLX-compatible Q4 bit extraction")
    func unifiedQuantizedGEMVQ4EmitsMLXBitPattern() throws {
        let formats: [any QuantizationFormat] = [
            AffineQ4Group64Format(),
            AffineQ4Group128Format(),
        ]
        for format in formats {
            let source = MetalSourceGenerator.generateUnifiedQuantizedGEMV(
                name: "test_\(format.gemvKernelName)",
                format: format,
                bufferPrecision: .float16
            )
            // Aligned Q4: 2 weights per byte.
            // `qs[(k) >> 1]` selects the byte; `((k) & 1) * 4` shifts to low nibble; `& 0xF` masks.
            #expect(source.contains("qs[(k) >> 1]"),
                "Missing Q4 byte index for \(format.schemeIdentifier)")
            #expect(source.contains("((k) & 1) * 4"),
                "Missing Q4 nibble shift for \(format.schemeIdentifier)")
            #expect(source.contains("& 0xF)"),
                "Missing Q4 4-bit mask for \(format.schemeIdentifier)")
        }
    }

    @Test("Unified quantized GEMV emits MLX-compatible Q8 byte read")
    func unifiedQuantizedGEMVQ8EmitsMLXBitPattern() throws {
        let formats: [any QuantizationFormat] = [
            AffineQ8Group32Format(),
            AffineQ8Group64Format(),
            AffineQ8Group128Format(),
        ]
        for format in formats {
            let source = MetalSourceGenerator.generateUnifiedQuantizedGEMV(
                name: "test_\(format.gemvKernelName)",
                format: format,
                bufferPrecision: .float16
            )
            // Aligned Q8: 1 weight per byte, direct read.
            #expect(source.contains("float(qs[k])"),
                "Missing Q8 direct byte read for \(format.schemeIdentifier)")
        }
    }

    @Test("Unified quantized GEMV emits MLX-compatible Q3 bit extraction")
    func unifiedQuantizedGEMVQ3EmitsMLXBitPattern() throws {
        let formats: [any QuantizationFormat] = [
            AffineQ3Group16Format(),
            AffineQ3Group32Format(),
            AffineQ3Group64Format(),
        ]
        for format in formats {
            let source = MetalSourceGenerator.generateUnifiedQuantizedGEMV(
                name: "test_\(format.gemvKernelName)",
                format: format,
                bufferPrecision: .float16
            )
            // MLX extract_bits<3>: 8 weights span 3 bytes.
            #expect(source.contains("qs[(((k)) >> 3) * 3 + 0] & 0x07"),
                "Missing Q3 w[0] extraction for \(format.schemeIdentifier)")
            #expect(source.contains("(qs[(((k)) >> 3) * 3 + 0] >> 3) & 0x07"),
                "Missing Q3 w[1] extraction for \(format.schemeIdentifier)")
            #expect(source.contains("((qs[(((k)) >> 3) * 3 + 0] >> 6) & 0x03) | ((qs[(((k)) >> 3) * 3 + 1] & 0x01) << 2)"),
                "Missing Q3 w[2] extraction for \(format.schemeIdentifier)")
            #expect(source.contains("(qs[(((k)) >> 3) * 3 + 1] >> 1) & 0x07"),
                "Missing Q3 w[3] extraction for \(format.schemeIdentifier)")
            #expect(source.contains("(qs[(((k)) >> 3) * 3 + 1] >> 4) & 0x07"),
                "Missing Q3 w[4] extraction for \(format.schemeIdentifier)")
            #expect(source.contains("((qs[(((k)) >> 3) * 3 + 1] >> 7) & 0x01) | ((qs[(((k)) >> 3) * 3 + 2] & 0x03) << 1)"),
                "Missing Q3 w[5] extraction for \(format.schemeIdentifier)")
            #expect(source.contains("(qs[(((k)) >> 3) * 3 + 2] >> 2) & 0x07"),
                "Missing Q3 w[6] extraction for \(format.schemeIdentifier)")
            #expect(source.contains("(qs[(((k)) >> 3) * 3 + 2] >> 5) & 0x07"),
                "Missing Q3 w[7] extraction for \(format.schemeIdentifier)")
        }
    }

    @Test("Unified quantized GEMV emits MLX-compatible Q5 bit extraction")
    func unifiedQuantizedGEMVQ5EmitsMLXBitPattern() throws {
        let formats: [any QuantizationFormat] = [
            AffineQ5Group32Format(),
            AffineQ5Group64Format(),
        ]
        for format in formats {
            let source = MetalSourceGenerator.generateUnifiedQuantizedGEMV(
                name: "test_\(format.gemvKernelName)",
                format: format,
                bufferPrecision: .float16
            )
            // MLX extract_bits<5>: 8 weights span 5 bytes.
            #expect(source.contains("qs[(((k)) >> 3) * 5 + 0] & 0x1f"),
                "Missing Q5 w[0] extraction for \(format.schemeIdentifier)")
            #expect(source.contains("((qs[(((k)) >> 3) * 5 + 0] >> 5) & 0x07) | ((qs[(((k)) >> 3) * 5 + 1] & 0x03) << 3)"),
                "Missing Q5 w[1] extraction for \(format.schemeIdentifier)")
            #expect(source.contains("(qs[(((k)) >> 3) * 5 + 1] >> 2) & 0x1f"),
                "Missing Q5 w[2] extraction for \(format.schemeIdentifier)")
            #expect(source.contains("((qs[(((k)) >> 3) * 5 + 1] >> 7) & 0x01) | ((qs[(((k)) >> 3) * 5 + 2] & 0x0f) << 1)"),
                "Missing Q5 w[3] extraction for \(format.schemeIdentifier)")
            #expect(source.contains("((qs[(((k)) >> 3) * 5 + 2] >> 4) & 0x0f) | ((qs[(((k)) >> 3) * 5 + 3] & 0x01) << 4)"),
                "Missing Q5 w[4] extraction for \(format.schemeIdentifier)")
            #expect(source.contains("(qs[(((k)) >> 3) * 5 + 3] >> 1) & 0x1f"),
                "Missing Q5 w[5] extraction for \(format.schemeIdentifier)")
            #expect(source.contains("((qs[(((k)) >> 3) * 5 + 3] >> 6) & 0x03) | ((qs[(((k)) >> 3) * 5 + 4] & 0x07) << 2)"),
                "Missing Q5 w[6] extraction for \(format.schemeIdentifier)")
            #expect(source.contains("(qs[(((k)) >> 3) * 5 + 4] >> 3) & 0x1f"),
                "Missing Q5 w[7] extraction for \(format.schemeIdentifier)")
        }
    }

    @Test("Unified quantized GEMV compiles for Q6 group16 and group32")
    func unifiedQuantizedGEMVQ6Compiles() throws {
        guard let device = MTLCreateSystemDefaultDevice() else { return }
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }

        let formats: [any QuantizationFormat] = [
            AffineQ6Group16Format(),
            AffineQ6Group32Format(),
        ]
        for format in formats {
            for precision in [MetalSourceGenerator.BufferPrecision.float16, .float32] {
                let name = "test_q6_g\(format.groupSize)_\(precision)"
                let source = MetalSourceGenerator.commonHeader + "\n\n"
                    + MetalSourceGenerator.generateUnifiedQuantizedGEMV(
                        name: name,
                        format: format,
                        bufferPrecision: precision
                    )
                let options = MTLCompileOptions()
                options.languageVersion = .version4_0
                let library = try device.makeLibrary(source: source, options: options)
                #expect(library.makeFunction(name: name) != nil,
                    "Failed to compile \(name)")
            }
        }
    }

    @Test("Q6 group dequant matches CPU reference for all packed values")
    func q6GroupDequantMatchesReference() throws {
        guard let device = MTLCreateSystemDefaultDevice() else { return }

        let formats: [any QuantizationFormat] = [
            AffineQ6Group16Format(),
            AffineQ6Group32Format(),
        ]

        for format in formats {
            // Sweep quant values spanning the full 6-bit range [0, 63] by
            // using a scale=1.0, zero=0.0 block whose packed weights cycle
            // through 0..63 so every 4-weight group exercises one bit offset.
            let wpb = format.weightsPerBlock
            let scale: Float16 = 1.0
            let zero: Float16 = 0.0

            // Build expected weight values [0, 1, 2, ..., wpb-1] — all
            // distinct values < 64 so none overflow 6 bits.
            let expectedWeights: [UInt8] = (0..<wpb).map { UInt8($0) }
            precondition(expectedWeights.allSatisfy { $0 < 64 })

            // Pack into Q6 layout: 4 weights share 3 bytes per MLX spec.
            let packedBytes = (wpb / 4) * 3
            var blockBytes = Data(count: 4 + packedBytes)
            blockBytes.withUnsafeMutableBytes { raw in
                var scaleCopy = scale
                memcpy(raw.baseAddress!, &scaleCopy, 2)
                var zeroCopy = zero
                memcpy(raw.baseAddress! + 2, &zeroCopy, 2)
                let qs = raw.baseAddress!.advanced(by: 4)
                    .assumingMemoryBound(to: UInt8.self)
                for g in 0..<(wpb / 4) {
                    let w0 = expectedWeights[g * 4 + 0]
                    let w1 = expectedWeights[g * 4 + 1]
                    let w2 = expectedWeights[g * 4 + 2]
                    let w3 = expectedWeights[g * 4 + 3]
                    qs[g * 3 + 0] = (w0 & 0x3f) | ((w1 & 0x03) << 6)
                    qs[g * 3 + 1] = ((w1 >> 2) & 0x0f) | ((w2 & 0x0f) << 4)
                    qs[g * 3 + 2] = ((w2 >> 4) & 0x03) | ((w3 & 0x3f) << 2)
                }
            }

            // Compile a dequant-only kernel that wraps perWeightReadExpression.
            let name = "dequant_q6_g\(format.groupSize)_test"
            let readExpression = format.perWeightReadExpression(
                blocksVar: "qs",
                weightIndexVar: "k"
            )!
            let source = """
            #include <metal_stdlib>
            using namespace metal;

            kernel void \(name)(
                device const uchar* block [[buffer(0)]],
                device float* output      [[buffer(1)]]
            ) {
                float scale = float(*(device const half*)(block));
                float zero  = float(*(device const half*)(block + 2));
                device const uchar* qs = block + 4;
                for (uint k = 0; k < \(wpb); k++) output[k] = \(readExpression);
            }
            """
            let options = MTLCompileOptions()
            options.languageVersion = .version4_0
            let library = try device.makeLibrary(source: source, options: options)
            let pipeline = try device.makeComputePipelineState(
                function: try #require(library.makeFunction(name: name))
            )

            let blockBuffer = try #require(device.makeBuffer(
                bytes: [UInt8](blockBytes),
                length: blockBytes.count,
                options: .storageModeShared
            ))
            let outputBuffer = try #require(device.makeBuffer(
                length: wpb * MemoryLayout<Float>.size,
                options: .storageModeShared
            ))

            let queue = try #require(device.makeCommandQueue())
            let commandBuffer = try #require(queue.makeCommandBuffer())
            let encoder = try #require(commandBuffer.makeComputeCommandEncoder())
            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(blockBuffer, offset: 0, index: 0)
            encoder.setBuffer(outputBuffer, offset: 0, index: 1)
            encoder.dispatchThreads(
                MTLSize(width: 1, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1)
            )
            encoder.endEncoding()
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()

            let result = outputBuffer.contents().bindMemory(
                to: Float.self, capacity: wpb
            )
            for k in 0..<wpb {
                let expected = Float(expectedWeights[k])
                let actual = result[k]
                #expect(
                    abs(actual - expected) < 1e-5,
                    "\(format.schemeIdentifier) dequant mismatch at k=\(k): expected=\(expected) actual=\(actual)"
                )
            }
        }
    }

    @Test("Generated complete library includes BF16 SSM kernels")
    func completeLibraryIncludesBF16SSMVariants() throws {
        guard let device = MTLCreateSystemDefaultDevice() else { return }

        let source = MetalSourceGenerator.generateCompleteLibrary(weightFormat: .bfloat16)
        let options = MTLCompileOptions()
        options.languageVersion = .version4_0
        let library = try device.makeLibrary(source: source, options: options)

        for name in [
            "ssm_recurrence_bf16",
            "ssm_recurrence_bf16_f32",
            "ssm_recurrence_seq_bf16",
            "ssm_recurrence_seq_bf16_f32",
        ] {
            #expect(library.makeFunction(name: name) != nil, "Missing: \(name)")
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
        #expect(bytes.count == 4 + payloadByteCount)
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
