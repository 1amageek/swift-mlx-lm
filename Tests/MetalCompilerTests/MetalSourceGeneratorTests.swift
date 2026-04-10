import Testing
import Metal
@testable import MetalCompiler

/// Verify MetalSourceGenerator produces valid, compilable MSL
/// that computes the same results as the hardcoded kernels.
@Suite("Metal Source Generator")
struct MetalSourceGeneratorTests {

    @Test("Generated RMSNorm compiles for all precision × weight format combinations")
    func rmsNormCompiles() throws {
        guard let device = MTLCreateSystemDefaultDevice() else { return }

        let precisions: [MetalSourceGenerator.BufferPrecision] = [.float16, .float32]
        let weightFormats: [MetalSourceGenerator.WeightFormat] = [.float16, .bfloat16]

        for precision in precisions {
            for weightFormat in weightFormats {
                let name = "rms_norm_\(precision)_\(weightFormat)"
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

    @Test("Generated fused SwiGLU projection compiles for both weight formats")
    func fusedSwiGLUProjectionCompiles() throws {
        guard let device = MTLCreateSystemDefaultDevice() else { return }

        let formats: [(MetalSourceGenerator.WeightFormat, String)] = [
            (.float16, "fp16"), (.bfloat16, "bf16")
        ]

        for (format, label) in formats {
            let name = "fused_swiglu_projection_\(label)"
            let source = MetalSourceGenerator.commonHeader + "\n\n"
                + MetalSourceGenerator.generateFusedSwiGLUProjection(
                    name: name,
                    bufferPrecision: .float16,
                    weightFormat: format)

            let options = MTLCompileOptions()
            options.languageVersion = .version4_0
            let library = try device.makeLibrary(source: source, options: options)
            #expect(library.makeFunction(name: name) != nil, "Failed to compile \(name)")
        }
    }

    @Test("Generated specialized fused SwiGLU argument-table kernel keeps runtime outputDimension")
    func specializedFusedSwiGLUArgumentTableCompilesWithRuntimeOutputDimension() throws {
        guard let device = MTLCreateSystemDefaultDevice() else { return }

        let name = "fused_swiglu_projection_2048_argbuf_test"
        let source = MetalSourceGenerator.commonHeader + "\n\n"
            + MetalSourceGenerator.generateInput2048FusedSwiGLUProjectionArgumentTableVariant(
                name: name,
                argumentBufferIndex: 30,
                bufferPrecision: .float16,
                weightFormat: .float16,
                stagesInputAsFloat: false,
                fixedRowsPerThreadgroup: 8,
                fixedSimdgroups: 8,
                unrollFactor: 8
            )

        #expect(source.contains("constant uint& outputDimension"))
        #expect(!source.contains("if (row >= 8192u) return;"))

        let options = MTLCompileOptions()
        options.languageVersion = .version4_0
        let library = try device.makeLibrary(source: source, options: options)
        #expect(library.makeFunction(name: name) != nil, "Failed to compile \(name)")
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
        encoder.setBytes(&inDim, length: MemoryLayout<UInt32>.size, index: 3)
        encoder.setBytes(&outDim, length: MemoryLayout<UInt32>.size, index: 4)
        encoder.setBytes(&seqLen, length: MemoryLayout<UInt32>.size, index: 5)
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
}
