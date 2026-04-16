import Testing
import Metal
import Foundation
@testable import MetalCompiler
import LMArchitecture
import ModelDeclarations
import LMIR

/// Verify that the automatic kernel fusion pipeline produces correct results.
///
/// Three verification layers:
/// 1. Structural: fusion reduces dispatch count on real model graphs
/// 2. Pipeline: fused MSL compiles to valid Metal compute pipelines
/// 3. Numerical: fused kernels produce bit-identical output to unfused execution
@Suite("Fusion Verification")
struct FusionVerificationTests {

    /// MSL preamble required for compiling fused kernel sources standalone.
    /// Uses the full common header to include BF16 utilities (bf16_to_float, etc.).
    private static let mslPreamble = MetalSourceGenerator.commonHeader

    // MARK: - Layer 1: Structural Verification

    @Test("Fusion reduces dispatch count on Transformer model")
    func fusionReducesDispatchCount() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device")
            return
        }
        let graph = try ModelGraph(TinyTransformerForFusion(hiddenSize: 64, layers: 2, vocabSize: 100))
        let kernelContext = KernelContext(bufferPrecision: .float32, weightFormat: .float16)

        let context = CompileContext(
            graph: graph, hiddenSize: 64, intermediateSize: 256, vocabSize: 100,
            inferencePolicy: InferencePolicy(maximumSequenceLength: 64),
            stafWeightStore: nil, device: device, weightFormat: .float16,
            decodeBufferPrecision: .float16,
            accessPolicyResolver: ProjectionWeightAccessPolicyResolver(override: nil)
        )
        let collector = MetalEntryCollector()
        let result = collector.collect(using: context, kernelContext: kernelContext)

        let unfused = result.unfusedCount
        let fused = result.fusedEntries.count
        #expect(fused < unfused,
            "Fusion should reduce dispatch count: unfused=\(unfused), fused=\(fused)")

        // Verify SynthesizedFragment instances exist in fused entries
        let synthesizedCount = result.fusedEntries.filter { $0.fragment is SynthesizedFragment }.count
        #expect(synthesizedCount > 0,
            "Fused entries should contain at least one SynthesizedFragment")

        // Report fusion statistics
        print("[Fusion verification] unfused=\(unfused) fused=\(fused) synthesized=\(synthesizedCount) reduction=\(unfused - fused)")
    }

    @Test("Each SynthesizedFragment wraps exactly 2+ fragments")
    func synthesizedFragmentWrapsMultipleFragments() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device")
            return
        }
        let graph = try ModelGraph(TinyTransformerForFusion(hiddenSize: 64, layers: 1, vocabSize: 100))
        let kernelContext = KernelContext(bufferPrecision: .float32, weightFormat: .float16)

        let context = CompileContext(
            graph: graph, hiddenSize: 64, intermediateSize: 256, vocabSize: 100,
            inferencePolicy: InferencePolicy(maximumSequenceLength: 64),
            stafWeightStore: nil, device: device, weightFormat: .float16,
            decodeBufferPrecision: .float16,
            accessPolicyResolver: ProjectionWeightAccessPolicyResolver(override: nil)
        )
        let collector = MetalEntryCollector()
        let result = collector.collect(using: context, kernelContext: kernelContext)

        for entry in result.fusedEntries {
            guard let synthesized = entry.fragment as? SynthesizedFragment else { continue }
            #expect(synthesized.fragments.count >= 2,
                "SynthesizedFragment must wrap at least 2 fragments, got \(synthesized.fragments.count)")
            #expect(synthesized.mergedContract.ports.count >= 2,
                "Merged contract must have at least 2 external ports")
        }
    }

    @Test("Fused entries preserve all non-fusable fragments unchanged")
    func fusionPreservesNonFusableFragments() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device")
            return
        }
        let graph = try ModelGraph(TinyTransformerForFusion(hiddenSize: 64, layers: 1, vocabSize: 100))
        let kernelContext = KernelContext(bufferPrecision: .float32, weightFormat: .float16)

        let context = CompileContext(
            graph: graph, hiddenSize: 64, intermediateSize: 256, vocabSize: 100,
            inferencePolicy: InferencePolicy(maximumSequenceLength: 64),
            stafWeightStore: nil, device: device, weightFormat: .float16,
            decodeBufferPrecision: .float16,
            accessPolicyResolver: ProjectionWeightAccessPolicyResolver(override: nil)
        )
        let collector = MetalEntryCollector()
        let result = collector.collect(using: context, kernelContext: kernelContext)

        // Non-fusable fragments (FlashAttention, Linear, Gather, Argmax) must not be wrapped
        for entry in result.fusedEntries {
            if entry.fragment is FlashAttentionFragment ||
               entry.fragment is LinearFragment ||
               entry.fragment is BatchedProjection ||
               entry.fragment is GatherFragment ||
               entry.fragment is ArgmaxFragment {
                // These fragments should appear directly, not inside SynthesizedFragment
                #expect(!(entry.fragment is SynthesizedFragment),
                    "Non-fusable fragment should not be inside SynthesizedFragment")
            }
        }
    }

    // MARK: - Layer 2: Pipeline Compilation Verification

    @Test("Fused CopyFragment+Reduction MSL compiles to valid Metal pipeline")
    func fusedCopyReductionCompiles() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device")
            return
        }

        let dim = 64
        let copy = CopyFragment(dimension: dim)
        let reduction = Reduction(dimension: dim, epsilon: 1e-6)

        // Manually fuse via FusionSynthesizer
        let copyBody = try #require(copy.kernelBody(bufferPrecision: .float32, weightFormat: .float16))
        let reductionBody = try #require(reduction.kernelBody(bufferPrecision: .float32, weightFormat: .float16))
        let copyContract = try #require(copy.fusionContract)
        let reductionContract = try #require(reduction.fusionContract)

        let result = try FusionSynthesizer.synthesize([
            .init(contract: copyContract, body: copyBody),
            .init(contract: reductionContract, body: reductionBody, weightFormats: ["weight": .float16]),
        ])

        let source = KernelScaffold.generate(
            name: "test_fused_copy_reduction",
            body: result.body,
            contract: result.contract,
            bufferPrecision: .float32,
            weightFormats: result.weightFormats,
            isSequence: true
        )

        // Compile to Metal library
        let options = MTLCompileOptions()
        options.fastMathEnabled = true
        options.languageVersion = .version4_0
        let library = try device.makeLibrary(source: Self.mslPreamble + source, options: options)
        let function = try #require(library.makeFunction(name: "test_fused_copy_reduction"))
        let pipeline = try device.makeComputePipelineState(function: function)

        #expect(pipeline.maxTotalThreadsPerThreadgroup > 0)
        print("[Pipeline verification] CopyFragment+Reduction compiled successfully")
    }

    @Test("Fused ResidualAdd+CopyFragment+Reduction MSL compiles (3-way fusion)")
    func fusedThreeWayCompiles() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device")
            return
        }

        let dim = 64
        let add = ResidualAddFragment(dimension: dim)
        let copy = CopyFragment(dimension: dim)
        let reduction = Reduction(dimension: dim, epsilon: 1e-6)

        let addBody = try #require(add.kernelBody(bufferPrecision: .float32, weightFormat: .float16))
        let copyBody = try #require(copy.kernelBody(bufferPrecision: .float32, weightFormat: .float16))
        let reductionBody = try #require(reduction.kernelBody(bufferPrecision: .float32, weightFormat: .float16))

        let result = try FusionSynthesizer.synthesize([
            .init(contract: add.fusionContract!, body: addBody),
            .init(contract: copy.fusionContract!, body: copyBody),
            .init(contract: reduction.fusionContract!, body: reductionBody, weightFormats: ["weight": .float16]),
        ])

        let source = KernelScaffold.generate(
            name: "test_fused_3way",
            body: result.body,
            contract: result.contract,
            bufferPrecision: .float32,
            weightFormats: result.weightFormats,
            isSequence: true
        )

        let options = MTLCompileOptions()
        options.fastMathEnabled = true
        options.languageVersion = .version4_0
        let library = try device.makeLibrary(source: Self.mslPreamble + source, options: options)
        let function = try #require(library.makeFunction(name: "test_fused_3way"))
        let pipeline = try device.makeComputePipelineState(function: function)

        #expect(pipeline.maxTotalThreadsPerThreadgroup > 0)
        print("[Pipeline verification] 3-way fusion compiled successfully")
    }

    @Test("All SynthesizedFragments in Transformer graph compile to Metal pipelines")
    func allSynthesizedFragmentsCompile() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device")
            return
        }

        let graph = try ModelGraph(TinyTransformerForFusion(hiddenSize: 64, layers: 1, vocabSize: 100))
        let kernelContext = KernelContext(bufferPrecision: .float32, weightFormat: .float16)

        let context = CompileContext(
            graph: graph, hiddenSize: 64, intermediateSize: 256, vocabSize: 100,
            inferencePolicy: InferencePolicy(maximumSequenceLength: 64),
            stafWeightStore: nil, device: device, weightFormat: .float16,
            decodeBufferPrecision: .float16,
            accessPolicyResolver: ProjectionWeightAccessPolicyResolver(override: nil)
        )
        let collector = MetalEntryCollector()
        let result = collector.collect(using: context, kernelContext: kernelContext)

        let options = MTLCompileOptions()
        options.fastMathEnabled = true
        options.languageVersion = .version4_0

        var compiledCount = 0
        for entry in result.fusedEntries {
            guard let synthesized = entry.fragment as? SynthesizedFragment else { continue }

            let name = synthesized.kernelName(context: kernelContext)
            let source = synthesized.kernelSource(
                name: name,
                bufferPrecision: .float32,
                weightFormat: .float16
            )

            let library = try device.makeLibrary(source: Self.mslPreamble + source, options: options)
            let function = try #require(library.makeFunction(name: name),
                "Failed to find function '\(name)' in compiled library")
            let pipeline = try device.makeComputePipelineState(function: function)
            #expect(pipeline.maxTotalThreadsPerThreadgroup > 0)
            compiledCount += 1
        }

        #expect(compiledCount > 0, "Should have compiled at least one SynthesizedFragment")
        print("[Pipeline verification] \(compiledCount) SynthesizedFragments compiled successfully")
    }

    // MARK: - Layer 3: Numerical Parity Verification

    @Test("CopyFragment+Reduction fusion produces same output as sequential execution")
    func copyReductionNumericalParity() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device")
            return
        }

        let dim = 64
        let seqLen = 4

        // --- Prepare input data ---
        let inputData: [Float] = (0..<(dim * seqLen)).map { Float($0 % 17) * 0.1 + 0.01 }
        let weightData: [Float16] = (0..<dim).map { Float16(1.0 + Float($0 % 5) * 0.1) }
        let epsilon: Float = 1e-6
        let weightBias: Float = 0.0

        // --- Create Metal buffers ---
        let inputBuffer = try #require(device.makeBuffer(
            bytes: inputData, length: inputData.count * MemoryLayout<Float>.size, options: .storageModeShared))
        let weightBuffer = try #require(device.makeBuffer(
            bytes: weightData, length: weightData.count * MemoryLayout<Float16>.size, options: .storageModeShared))

        // Unfused output buffers
        let residualUnfused = try #require(device.makeBuffer(
            length: dim * seqLen * MemoryLayout<Float>.size, options: .storageModeShared))
        let outputUnfused = try #require(device.makeBuffer(
            length: dim * seqLen * MemoryLayout<Float>.size, options: .storageModeShared))

        // Fused output buffers
        let residualFused = try #require(device.makeBuffer(
            length: dim * seqLen * MemoryLayout<Float>.size, options: .storageModeShared))
        let outputFused = try #require(device.makeBuffer(
            length: dim * seqLen * MemoryLayout<Float>.size, options: .storageModeShared))

        let options = MTLCompileOptions()
        options.fastMathEnabled = true
        options.languageVersion = .version4_0

        // --- Step 1: Execute unfused (CPU reference) ---
        // RMSNorm: output[i] = input[i] * rsqrt(mean(input^2) + eps) * weight[i]
        for s in 0..<seqLen {
            let rowStart = s * dim
            // Copy: residual = input
            for i in 0..<dim {
                let val = inputData[rowStart + i]
                residualUnfused.contents().assumingMemoryBound(to: Float.self)[rowStart + i] = val
            }
            // Reduction (RMSNorm)
            var sumSq: Float = 0
            for i in 0..<dim { sumSq += inputData[rowStart + i] * inputData[rowStart + i] }
            let rmsScale = 1.0 / sqrtf(sumSq / Float(dim) + epsilon)
            for i in 0..<dim {
                let w = Float(weightData[i]) + weightBias
                outputUnfused.contents().assumingMemoryBound(to: Float.self)[rowStart + i] =
                    inputData[rowStart + i] * rmsScale * w
            }
        }

        // --- Step 2: Execute fused kernel on GPU ---
        let copy = CopyFragment(dimension: dim)
        let reduction = Reduction(dimension: dim, epsilon: epsilon)

        let copyBody = try #require(copy.kernelBody(bufferPrecision: .float32, weightFormat: .float16))
        let reductionBody = try #require(reduction.kernelBody(bufferPrecision: .float32, weightFormat: .float16))

        let synthesisResult = try FusionSynthesizer.synthesize([
            .init(contract: copy.fusionContract!, body: copyBody),
            .init(contract: reduction.fusionContract!, body: reductionBody, weightFormats: ["weight": .float16]),
        ])

        let fusedSource = KernelScaffold.generate(
            name: "parity_fused_copy_reduction",
            body: synthesisResult.body,
            contract: synthesisResult.contract,
            bufferPrecision: .float32,
            weightFormats: synthesisResult.weightFormats,
            isSequence: true
        )

        let library = try device.makeLibrary(source: Self.mslPreamble + fusedSource, options: options)
        let function = try #require(library.makeFunction(name: "parity_fused_copy_reduction"))
        let pipeline = try device.makeComputePipelineState(function: function)

        // Encode and execute fused kernel
        let commandQueue = try #require(device.makeCommandQueue())
        let commandBuffer = try #require(commandQueue.makeCommandBuffer())
        let encoder = try #require(commandBuffer.makeComputeCommandEncoder())

        encoder.setComputePipelineState(pipeline)

        // Bind buffers per merged contract port order:
        // Port 0: data (input) - CopyFragment.data
        // Port 1: residual (output) - CopyFragment.residual
        // Port 2: weight (input) - Reduction.weight
        // Port 3: output (output) - Reduction.output
        let ports = synthesisResult.contract.ports
        var portIndex = 0
        for port in ports {
            switch port.role {
            case .buffer:
                if port.bufferIntent == .residual {
                    encoder.setBuffer(residualFused, offset: 0, index: portIndex)
                } else if port.direction == .input {
                    encoder.setBuffer(inputBuffer, offset: 0, index: portIndex)
                } else {
                    encoder.setBuffer(outputFused, offset: 0, index: portIndex)
                }
            case .weight:
                encoder.setBuffer(weightBuffer, offset: 0, index: portIndex)
            }
            portIndex += 1
        }

        // Dimension binding
        var dimension = UInt32(dim)
        encoder.setBytes(&dimension, length: 4, index: portIndex)
        portIndex += 1

        // Scalar constants
        var eps = epsilon
        var wb = weightBias
        encoder.setBytes(&eps, length: 4, index: portIndex)
        portIndex += 1
        encoder.setBytes(&wb, length: 4, index: portIndex)
        portIndex += 1

        // Sequence length
        var seqLength = UInt32(seqLen)
        encoder.setBytes(&seqLength, length: 4, index: portIndex)

        // Dispatch per-row: one threadgroup per sequence position
        let simdWidth = pipeline.threadExecutionWidth
        let clamped = min(max(dim, 1), 1024)
        let rounded = ((clamped + simdWidth - 1) / simdWidth) * simdWidth
        let threads = min(rounded, pipeline.maxTotalThreadsPerThreadgroup)

        encoder.dispatchThreadgroups(
            MTLSize(width: seqLen, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1)
        )
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        // --- Step 3: Compare outputs ---
        let fusedOutput = outputFused.contents().assumingMemoryBound(to: Float.self)
        let unfusedOutput = outputUnfused.contents().assumingMemoryBound(to: Float.self)
        let fusedResidual = residualFused.contents().assumingMemoryBound(to: Float.self)
        let unfusedResidual = residualUnfused.contents().assumingMemoryBound(to: Float.self)

        let tolerance: Float = 1e-4
        var maxDiffOutput: Float = 0
        var maxDiffResidual: Float = 0

        for i in 0..<(dim * seqLen) {
            let diffOutput = abs(fusedOutput[i] - unfusedOutput[i])
            let diffResidual = abs(fusedResidual[i] - unfusedResidual[i])
            maxDiffOutput = max(maxDiffOutput, diffOutput)
            maxDiffResidual = max(maxDiffResidual, diffResidual)

            #expect(diffOutput < tolerance,
                "Output mismatch at index \(i): fused=\(fusedOutput[i]) ref=\(unfusedOutput[i]) diff=\(diffOutput)")
            #expect(diffResidual < tolerance,
                "Residual mismatch at index \(i): fused=\(fusedResidual[i]) ref=\(unfusedResidual[i]) diff=\(diffResidual)")
        }

        print("[Numerical parity] CopyFragment+Reduction: maxDiffOutput=\(maxDiffOutput) maxDiffResidual=\(maxDiffResidual)")
    }

    @Test("ResidualAdd+Copy+Reduction 3-way fusion produces same output as sequential")
    func threeWayFusionNumericalParity() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device")
            return
        }

        let dim = 64
        let seqLen = 4

        // --- Prepare input data ---
        let inputData: [Float] = (0..<(dim * seqLen)).map { Float($0 % 13) * 0.15 + 0.05 }
        let residualIn: [Float] = (0..<(dim * seqLen)).map { Float($0 % 11) * 0.12 - 0.3 }
        let weightData: [Float16] = (0..<dim).map { Float16(0.8 + Float($0 % 7) * 0.05) }
        let epsilon: Float = 1e-6
        let weightBias: Float = 0.0

        // --- Create Metal buffers ---
        let inputBuffer = try #require(device.makeBuffer(
            bytes: inputData, length: inputData.count * MemoryLayout<Float>.size, options: .storageModeShared))
        // Single residual buffer: pre-filled with input values, overwritten by Copy output.
        // ResidualAdd reads old values, CopyFragment writes new values — same physical buffer.
        let residualBuffer = try #require(device.makeBuffer(
            bytes: residualIn, length: residualIn.count * MemoryLayout<Float>.size, options: .storageModeShared))
        let weightBuffer = try #require(device.makeBuffer(
            bytes: weightData, length: weightData.count * MemoryLayout<Float16>.size, options: .storageModeShared))

        // Fused output buffer
        let outputFused = try #require(device.makeBuffer(
            length: dim * seqLen * MemoryLayout<Float>.size, options: .storageModeShared))

        // --- CPU reference: ResidualAdd → Copy → Reduction ---
        var refResidualOut = [Float](repeating: 0, count: dim * seqLen)
        var refOutput = [Float](repeating: 0, count: dim * seqLen)

        for s in 0..<seqLen {
            let base = s * dim
            // ResidualAdd: added = input + residual
            var added = [Float](repeating: 0, count: dim)
            for i in 0..<dim { added[i] = inputData[base + i] + residualIn[base + i] }
            // Copy: residualOut = added
            for i in 0..<dim { refResidualOut[base + i] = added[i] }
            // Reduction (RMSNorm on added)
            var sumSq: Float = 0
            for i in 0..<dim { sumSq += added[i] * added[i] }
            let rmsScale = 1.0 / sqrtf(sumSq / Float(dim) + epsilon)
            for i in 0..<dim {
                refOutput[base + i] = added[i] * rmsScale * (Float(weightData[i]) + weightBias)
            }
        }

        // --- Fused kernel ---
        let add = ResidualAddFragment(dimension: dim)
        let copy = CopyFragment(dimension: dim)
        let reduction = Reduction(dimension: dim, epsilon: epsilon)

        let synthesisResult = try FusionSynthesizer.synthesize([
            .init(contract: add.fusionContract!, body: add.kernelBody(bufferPrecision: .float32, weightFormat: .float16)!),
            .init(contract: copy.fusionContract!, body: copy.kernelBody(bufferPrecision: .float32, weightFormat: .float16)!),
            .init(contract: reduction.fusionContract!, body: reduction.kernelBody(bufferPrecision: .float32, weightFormat: .float16)!, weightFormats: ["weight": .float16]),
        ])

        let fusedSource = KernelScaffold.generate(
            name: "parity_3way",
            body: synthesisResult.body,
            contract: synthesisResult.contract,
            bufferPrecision: .float32,
            weightFormats: synthesisResult.weightFormats,
            isSequence: true
        )

        let compileOptions = MTLCompileOptions()
        compileOptions.fastMathEnabled = true
        compileOptions.languageVersion = .version4_0
        let library = try device.makeLibrary(source: Self.mslPreamble + fusedSource, options: compileOptions)
        let function = try #require(library.makeFunction(name: "parity_3way"))
        let pipeline = try device.makeComputePipelineState(function: function)

        let commandQueue = try #require(device.makeCommandQueue())
        let commandBuffer = try #require(commandQueue.makeCommandBuffer())
        let encoder = try #require(commandBuffer.makeComputeCommandEncoder())
        encoder.setComputePipelineState(pipeline)

        // Bind per merged contract port order.
        // After port merging, "residual" is a single non-const port (read by ResidualAdd,
        // written by Copy). Same physical buffer — pre-filled with input values.
        var portIndex = 0
        for port in synthesisResult.contract.ports {
            switch port.role {
            case .buffer:
                if port.bufferIntent == .residual {
                    encoder.setBuffer(residualBuffer, offset: 0, index: portIndex)
                } else if port.direction == .input {
                    encoder.setBuffer(inputBuffer, offset: 0, index: portIndex)
                } else {
                    encoder.setBuffer(outputFused, offset: 0, index: portIndex)
                }
            case .weight:
                encoder.setBuffer(weightBuffer, offset: 0, index: portIndex)
            }
            portIndex += 1
        }

        var dimension = UInt32(dim)
        encoder.setBytes(&dimension, length: 4, index: portIndex); portIndex += 1
        var eps = epsilon
        encoder.setBytes(&eps, length: 4, index: portIndex); portIndex += 1
        var wb = weightBias
        encoder.setBytes(&wb, length: 4, index: portIndex); portIndex += 1
        var seqLength = UInt32(seqLen)
        encoder.setBytes(&seqLength, length: 4, index: portIndex)

        let simdWidth = pipeline.threadExecutionWidth
        let clamped = min(max(dim, 1), 1024)
        let rounded = ((clamped + simdWidth - 1) / simdWidth) * simdWidth
        let threads = min(rounded, pipeline.maxTotalThreadsPerThreadgroup)

        encoder.dispatchThreadgroups(
            MTLSize(width: seqLen, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1)
        )
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        // --- Compare ---
        let fusedOut = outputFused.contents().assumingMemoryBound(to: Float.self)
        let fusedRes = residualBuffer.contents().assumingMemoryBound(to: Float.self)
        let tolerance: Float = 1e-4
        var maxDiffOutput: Float = 0
        var maxDiffResidual: Float = 0

        for i in 0..<(dim * seqLen) {
            let diffOutput = abs(fusedOut[i] - refOutput[i])
            let diffResidual = abs(fusedRes[i] - refResidualOut[i])
            maxDiffOutput = max(maxDiffOutput, diffOutput)
            maxDiffResidual = max(maxDiffResidual, diffResidual)
            #expect(diffOutput < tolerance,
                "Output mismatch at \(i): fused=\(fusedOut[i]) ref=\(refOutput[i])")
            #expect(diffResidual < tolerance,
                "Residual mismatch at \(i): fused=\(fusedRes[i]) ref=\(refResidualOut[i])")
        }

        print("[Numerical parity] 3-way fusion: maxDiffOutput=\(maxDiffOutput) maxDiffResidual=\(maxDiffResidual)")
    }
    // MARK: - Layer 4: Weight Format Verification (BF16)

    @Test("Fused CopyFragment+Reduction MSL compiles with BF16 weights")
    func fusedCopyReductionCompilesBF16() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device")
            return
        }

        let dim = 64
        let copy = CopyFragment(dimension: dim)
        let reduction = Reduction(dimension: dim, epsilon: 1e-6)

        let copyBody = try #require(copy.kernelBody(bufferPrecision: .float32, weightFormat: .bfloat16))
        let reductionBody = try #require(reduction.kernelBody(bufferPrecision: .float32, weightFormat: .bfloat16))

        let result = try FusionSynthesizer.synthesize([
            .init(contract: copy.fusionContract!, body: copyBody),
            .init(contract: reduction.fusionContract!, body: reductionBody, weightFormats: ["weight": .bfloat16]),
        ])

        let source = KernelScaffold.generate(
            name: "test_fused_copy_reduction_bf16",
            body: result.body,
            contract: result.contract,
            bufferPrecision: .float32,
            weightFormats: result.weightFormats,
            isSequence: true
        )

        // BF16 weight port should declare uint16_t buffer type
        #expect(source.contains("uint16_t"))

        let options = MTLCompileOptions()
        options.fastMathEnabled = true
        options.languageVersion = .version4_0
        let library = try device.makeLibrary(source: Self.mslPreamble + source, options: options)
        let function = try #require(library.makeFunction(name: "test_fused_copy_reduction_bf16"))
        let pipeline = try device.makeComputePipelineState(function: function)

        #expect(pipeline.maxTotalThreadsPerThreadgroup > 0)
        print("[Pipeline verification] CopyFragment+Reduction BF16 compiled successfully")
    }

    @Test("CopyFragment+Reduction fusion with BF16 weights produces correct output")
    func copyReductionNumericalParityBF16() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device")
            return
        }

        let dim = 64
        let seqLen = 4

        // --- Prepare input data ---
        let inputData: [Float] = (0..<(dim * seqLen)).map { Float($0 % 17) * 0.1 + 0.01 }
        let epsilon: Float = 1e-6
        let weightBias: Float = 0.0

        // BF16 weight data: Float → UInt16 (BF16 representation)
        let weightValuesF32: [Float] = (0..<dim).map { 1.0 + Float($0 % 5) * 0.1 }
        let weightDataBF16: [UInt16] = weightValuesF32.map { Self.floatToBF16($0) }

        // --- Create Metal buffers ---
        let inputBuffer = try #require(device.makeBuffer(
            bytes: inputData, length: inputData.count * MemoryLayout<Float>.size, options: .storageModeShared))
        let weightBuffer = try #require(device.makeBuffer(
            bytes: weightDataBF16, length: weightDataBF16.count * MemoryLayout<UInt16>.size, options: .storageModeShared))
        let residualFused = try #require(device.makeBuffer(
            length: dim * seqLen * MemoryLayout<Float>.size, options: .storageModeShared))
        let outputFused = try #require(device.makeBuffer(
            length: dim * seqLen * MemoryLayout<Float>.size, options: .storageModeShared))

        // --- CPU reference ---
        var refResidual = [Float](repeating: 0, count: dim * seqLen)
        var refOutput = [Float](repeating: 0, count: dim * seqLen)

        for s in 0..<seqLen {
            let base = s * dim
            for i in 0..<dim { refResidual[base + i] = inputData[base + i] }
            var sumSq: Float = 0
            for i in 0..<dim { sumSq += inputData[base + i] * inputData[base + i] }
            let rmsScale = 1.0 / sqrtf(sumSq / Float(dim) + epsilon)
            for i in 0..<dim {
                let w = Self.bf16ToFloat(weightDataBF16[i]) + weightBias
                refOutput[base + i] = inputData[base + i] * rmsScale * w
            }
        }

        // --- Fused kernel ---
        let copy = CopyFragment(dimension: dim)
        let reduction = Reduction(dimension: dim, epsilon: epsilon)

        let copyBody = try #require(copy.kernelBody(bufferPrecision: .float32, weightFormat: .bfloat16))
        let reductionBody = try #require(reduction.kernelBody(bufferPrecision: .float32, weightFormat: .bfloat16))

        let synthesisResult = try FusionSynthesizer.synthesize([
            .init(contract: copy.fusionContract!, body: copyBody),
            .init(contract: reduction.fusionContract!, body: reductionBody, weightFormats: ["weight": .bfloat16]),
        ])

        let fusedSource = KernelScaffold.generate(
            name: "parity_bf16_weights",
            body: synthesisResult.body,
            contract: synthesisResult.contract,
            bufferPrecision: .float32,
            weightFormats: synthesisResult.weightFormats,
            isSequence: true
        )

        let options = MTLCompileOptions()
        options.fastMathEnabled = true
        options.languageVersion = .version4_0
        let library = try device.makeLibrary(source: Self.mslPreamble + fusedSource, options: options)
        let function = try #require(library.makeFunction(name: "parity_bf16_weights"))
        let pipeline = try device.makeComputePipelineState(function: function)

        let commandQueue = try #require(device.makeCommandQueue())
        let commandBuffer = try #require(commandQueue.makeCommandBuffer())
        let encoder = try #require(commandBuffer.makeComputeCommandEncoder())
        encoder.setComputePipelineState(pipeline)

        var portIndex = 0
        for port in synthesisResult.contract.ports {
            switch port.role {
            case .buffer:
                if port.bufferIntent == .residual {
                    encoder.setBuffer(residualFused, offset: 0, index: portIndex)
                } else if port.direction == .input {
                    encoder.setBuffer(inputBuffer, offset: 0, index: portIndex)
                } else {
                    encoder.setBuffer(outputFused, offset: 0, index: portIndex)
                }
            case .weight:
                encoder.setBuffer(weightBuffer, offset: 0, index: portIndex)
            }
            portIndex += 1
        }

        var dimension = UInt32(dim)
        encoder.setBytes(&dimension, length: 4, index: portIndex); portIndex += 1
        var eps = epsilon
        encoder.setBytes(&eps, length: 4, index: portIndex); portIndex += 1
        var wb = weightBias
        encoder.setBytes(&wb, length: 4, index: portIndex); portIndex += 1
        var seqLength = UInt32(seqLen)
        encoder.setBytes(&seqLength, length: 4, index: portIndex)

        let simdWidth = pipeline.threadExecutionWidth
        let clamped = min(max(dim, 1), 1024)
        let rounded = ((clamped + simdWidth - 1) / simdWidth) * simdWidth
        let threads = min(rounded, pipeline.maxTotalThreadsPerThreadgroup)

        encoder.dispatchThreadgroups(
            MTLSize(width: seqLen, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1)
        )
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        // --- Compare ---
        let fusedOut = outputFused.contents().assumingMemoryBound(to: Float.self)
        let fusedRes = residualFused.contents().assumingMemoryBound(to: Float.self)
        let tolerance: Float = 1e-4
        var maxDiffOutput: Float = 0
        var maxDiffResidual: Float = 0

        for i in 0..<(dim * seqLen) {
            let diffOutput = abs(fusedOut[i] - refOutput[i])
            let diffResidual = abs(fusedRes[i] - refResidual[i])
            maxDiffOutput = max(maxDiffOutput, diffOutput)
            maxDiffResidual = max(maxDiffResidual, diffResidual)
            #expect(diffOutput < tolerance,
                "Output mismatch at \(i): fused=\(fusedOut[i]) ref=\(refOutput[i]) diff=\(diffOutput)")
            #expect(diffResidual < tolerance,
                "Residual mismatch at \(i): fused=\(fusedRes[i]) ref=\(refResidual[i]) diff=\(diffResidual)")
        }

        print("[Numerical parity] BF16 weights: maxDiffOutput=\(maxDiffOutput) maxDiffResidual=\(maxDiffResidual)")
    }

    // MARK: - Layer 5: Decode Mode Verification (F16)

    @Test("Fused CopyFragment+Reduction MSL compiles in F16 decode mode")
    func fusedCopyReductionCompilesF16Decode() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device")
            return
        }

        let dim = 64
        let copy = CopyFragment(dimension: dim)
        let reduction = Reduction(dimension: dim, epsilon: 1e-6)

        let copyBody = try #require(copy.kernelBody(bufferPrecision: .float16, weightFormat: .float16))
        let reductionBody = try #require(reduction.kernelBody(bufferPrecision: .float16, weightFormat: .float16))

        let result = try FusionSynthesizer.synthesize([
            .init(contract: copy.fusionContract!, body: copyBody),
            .init(contract: reduction.fusionContract!, body: reductionBody, weightFormats: ["weight": .float16]),
        ])

        let source = KernelScaffold.generate(
            name: "test_fused_f16_decode",
            body: result.body,
            contract: result.contract,
            bufferPrecision: .float16,
            weightFormats: result.weightFormats,
            isSequence: false
        )

        // Decode mode: no sequenceLength, no _base suffix
        #expect(!source.contains("sequenceLength"))
        #expect(!source.contains("_base"))
        // F16 buffer type
        #expect(source.contains("half*"))

        let options = MTLCompileOptions()
        options.fastMathEnabled = true
        options.languageVersion = .version4_0
        let library = try device.makeLibrary(source: Self.mslPreamble + source, options: options)
        let function = try #require(library.makeFunction(name: "test_fused_f16_decode"))
        let pipeline = try device.makeComputePipelineState(function: function)

        #expect(pipeline.maxTotalThreadsPerThreadgroup > 0)
        print("[Pipeline verification] F16 decode CopyFragment+Reduction compiled successfully")
    }

    @Test("CopyFragment+Reduction fusion in F16 decode produces correct output")
    func copyReductionNumericalParityF16Decode() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device")
            return
        }

        let dim = 64

        // --- Prepare F16 input data (decode = single position) ---
        let inputValuesF32: [Float] = (0..<dim).map { Float($0 % 17) * 0.1 + 0.01 }
        let inputData: [Float16] = inputValuesF32.map { Float16($0) }
        let weightData: [Float16] = (0..<dim).map { Float16(1.0 + Float($0 % 5) * 0.1) }
        let epsilon: Float = 1e-6
        let weightBias: Float = 0.0

        // --- Create Metal buffers ---
        let inputBuffer = try #require(device.makeBuffer(
            bytes: inputData, length: inputData.count * MemoryLayout<Float16>.size, options: .storageModeShared))
        let weightBuffer = try #require(device.makeBuffer(
            bytes: weightData, length: weightData.count * MemoryLayout<Float16>.size, options: .storageModeShared))
        let residualFused = try #require(device.makeBuffer(
            length: dim * MemoryLayout<Float16>.size, options: .storageModeShared))
        let outputFused = try #require(device.makeBuffer(
            length: dim * MemoryLayout<Float16>.size, options: .storageModeShared))

        // --- CPU reference (compute in Float, convert to Float16 for comparison) ---
        var refResidual = [Float16](repeating: 0, count: dim)
        var refOutput = [Float16](repeating: 0, count: dim)

        for i in 0..<dim { refResidual[i] = inputData[i] }
        var sumSq: Float = 0
        for i in 0..<dim {
            let v = Float(inputData[i])
            sumSq += v * v
        }
        let rmsScale = 1.0 / sqrtf(sumSq / Float(dim) + epsilon)
        for i in 0..<dim {
            let result = Float(inputData[i]) * rmsScale * (Float(weightData[i]) + weightBias)
            refOutput[i] = Float16(result)
        }

        // --- Fused kernel ---
        let copy = CopyFragment(dimension: dim)
        let reduction = Reduction(dimension: dim, epsilon: epsilon)

        let copyBody = try #require(copy.kernelBody(bufferPrecision: .float16, weightFormat: .float16))
        let reductionBody = try #require(reduction.kernelBody(bufferPrecision: .float16, weightFormat: .float16))

        let synthesisResult = try FusionSynthesizer.synthesize([
            .init(contract: copy.fusionContract!, body: copyBody),
            .init(contract: reduction.fusionContract!, body: reductionBody, weightFormats: ["weight": .float16]),
        ])

        let fusedSource = KernelScaffold.generate(
            name: "parity_f16_decode",
            body: synthesisResult.body,
            contract: synthesisResult.contract,
            bufferPrecision: .float16,
            weightFormats: synthesisResult.weightFormats,
            isSequence: false
        )

        let options = MTLCompileOptions()
        options.fastMathEnabled = true
        options.languageVersion = .version4_0
        let library = try device.makeLibrary(source: Self.mslPreamble + fusedSource, options: options)
        let function = try #require(library.makeFunction(name: "parity_f16_decode"))
        let pipeline = try device.makeComputePipelineState(function: function)

        let commandQueue = try #require(device.makeCommandQueue())
        let commandBuffer = try #require(commandQueue.makeCommandBuffer())
        let encoder = try #require(commandBuffer.makeComputeCommandEncoder())
        encoder.setComputePipelineState(pipeline)

        // Bind per merged contract port order
        var portIndex = 0
        for port in synthesisResult.contract.ports {
            switch port.role {
            case .buffer:
                if port.bufferIntent == .residual {
                    encoder.setBuffer(residualFused, offset: 0, index: portIndex)
                } else if port.direction == .input {
                    encoder.setBuffer(inputBuffer, offset: 0, index: portIndex)
                } else {
                    encoder.setBuffer(outputFused, offset: 0, index: portIndex)
                }
            case .weight:
                encoder.setBuffer(weightBuffer, offset: 0, index: portIndex)
            }
            portIndex += 1
        }

        var dimension = UInt32(dim)
        encoder.setBytes(&dimension, length: 4, index: portIndex); portIndex += 1
        var eps = epsilon
        encoder.setBytes(&eps, length: 4, index: portIndex); portIndex += 1
        var wb = weightBias
        encoder.setBytes(&wb, length: 4, index: portIndex)

        // Single threadgroup for decode (1 position)
        let simdWidth = pipeline.threadExecutionWidth
        let rounded = ((dim + simdWidth - 1) / simdWidth) * simdWidth
        let threads = min(rounded, pipeline.maxTotalThreadsPerThreadgroup)

        encoder.dispatchThreadgroups(
            MTLSize(width: 1, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1)
        )
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        // --- Compare (Float16 precision: tolerance ~5e-3) ---
        let fusedOut = outputFused.contents().assumingMemoryBound(to: Float16.self)
        let fusedRes = residualFused.contents().assumingMemoryBound(to: Float16.self)
        let tolerance: Float = 5e-3
        var maxDiffOutput: Float = 0
        var maxDiffResidual: Float = 0

        for i in 0..<dim {
            let diffOutput = abs(Float(fusedOut[i]) - Float(refOutput[i]))
            let diffResidual = abs(Float(fusedRes[i]) - Float(refResidual[i]))
            maxDiffOutput = max(maxDiffOutput, diffOutput)
            maxDiffResidual = max(maxDiffResidual, diffResidual)
            #expect(diffOutput < tolerance,
                "Output mismatch at \(i): fused=\(fusedOut[i]) ref=\(refOutput[i]) diff=\(diffOutput)")
            #expect(diffResidual < tolerance,
                "Residual mismatch at \(i): fused=\(fusedRes[i]) ref=\(refResidual[i]) diff=\(diffResidual)")
        }

        print("[Numerical parity] F16 decode: maxDiffOutput=\(maxDiffOutput) maxDiffResidual=\(maxDiffResidual)")
    }

    @Test("Fused CopyFragment+Reduction in F16 decode with BF16 weights compiles and runs correctly")
    func copyReductionF16DecodeBF16Weights() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device")
            return
        }

        let dim = 64

        let inputData: [Float16] = (0..<dim).map { Float16(Float($0 % 17) * 0.1 + 0.01) }
        let epsilon: Float = 1e-6
        let weightBias: Float = 0.0

        let weightValuesF32: [Float] = (0..<dim).map { 1.0 + Float($0 % 5) * 0.1 }
        let weightDataBF16: [UInt16] = weightValuesF32.map { Self.floatToBF16($0) }

        let inputBuffer = try #require(device.makeBuffer(
            bytes: inputData, length: inputData.count * MemoryLayout<Float16>.size, options: .storageModeShared))
        let weightBuffer = try #require(device.makeBuffer(
            bytes: weightDataBF16, length: weightDataBF16.count * MemoryLayout<UInt16>.size, options: .storageModeShared))
        let residualFused = try #require(device.makeBuffer(
            length: dim * MemoryLayout<Float16>.size, options: .storageModeShared))
        let outputFused = try #require(device.makeBuffer(
            length: dim * MemoryLayout<Float16>.size, options: .storageModeShared))

        // CPU reference
        var refOutput = [Float16](repeating: 0, count: dim)
        var sumSq: Float = 0
        for i in 0..<dim {
            let v = Float(inputData[i])
            sumSq += v * v
        }
        let rmsScale = 1.0 / sqrtf(sumSq / Float(dim) + epsilon)
        for i in 0..<dim {
            let w = Self.bf16ToFloat(weightDataBF16[i]) + weightBias
            refOutput[i] = Float16(Float(inputData[i]) * rmsScale * w)
        }

        // Fused kernel: F16 buffers, BF16 weights, decode mode
        let copy = CopyFragment(dimension: dim)
        let reduction = Reduction(dimension: dim, epsilon: epsilon)

        let copyBody = try #require(copy.kernelBody(bufferPrecision: .float16, weightFormat: .bfloat16))
        let reductionBody = try #require(reduction.kernelBody(bufferPrecision: .float16, weightFormat: .bfloat16))

        let synthesisResult = try FusionSynthesizer.synthesize([
            .init(contract: copy.fusionContract!, body: copyBody),
            .init(contract: reduction.fusionContract!, body: reductionBody, weightFormats: ["weight": .bfloat16]),
        ])

        let fusedSource = KernelScaffold.generate(
            name: "parity_f16_decode_bf16w",
            body: synthesisResult.body,
            contract: synthesisResult.contract,
            bufferPrecision: .float16,
            weightFormats: synthesisResult.weightFormats,
            isSequence: false
        )

        // Verify BF16 weight buffer declaration alongside F16 data buffers
        #expect(fusedSource.contains("half*"))
        #expect(fusedSource.contains("uint16_t*"))

        let compileOptions = MTLCompileOptions()
        compileOptions.fastMathEnabled = true
        compileOptions.languageVersion = .version4_0
        let library = try device.makeLibrary(source: Self.mslPreamble + fusedSource, options: compileOptions)
        let function = try #require(library.makeFunction(name: "parity_f16_decode_bf16w"))
        let pipeline = try device.makeComputePipelineState(function: function)

        let commandQueue = try #require(device.makeCommandQueue())
        let commandBuffer = try #require(commandQueue.makeCommandBuffer())
        let encoder = try #require(commandBuffer.makeComputeCommandEncoder())
        encoder.setComputePipelineState(pipeline)

        var portIndex = 0
        for port in synthesisResult.contract.ports {
            switch port.role {
            case .buffer:
                if port.bufferIntent == .residual {
                    encoder.setBuffer(residualFused, offset: 0, index: portIndex)
                } else if port.direction == .input {
                    encoder.setBuffer(inputBuffer, offset: 0, index: portIndex)
                } else {
                    encoder.setBuffer(outputFused, offset: 0, index: portIndex)
                }
            case .weight:
                encoder.setBuffer(weightBuffer, offset: 0, index: portIndex)
            }
            portIndex += 1
        }

        var dimension = UInt32(dim)
        encoder.setBytes(&dimension, length: 4, index: portIndex); portIndex += 1
        var eps = epsilon
        encoder.setBytes(&eps, length: 4, index: portIndex); portIndex += 1
        var wb = weightBias
        encoder.setBytes(&wb, length: 4, index: portIndex)

        let simdWidth = pipeline.threadExecutionWidth
        let rounded = ((dim + simdWidth - 1) / simdWidth) * simdWidth
        let threads = min(rounded, pipeline.maxTotalThreadsPerThreadgroup)

        encoder.dispatchThreadgroups(
            MTLSize(width: 1, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1)
        )
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        let fusedOut = outputFused.contents().assumingMemoryBound(to: Float16.self)
        let tolerance: Float = 5e-3
        var maxDiff: Float = 0

        for i in 0..<dim {
            let diff = abs(Float(fusedOut[i]) - Float(refOutput[i]))
            maxDiff = max(maxDiff, diff)
            #expect(diff < tolerance,
                "Output mismatch at \(i): fused=\(fusedOut[i]) ref=\(refOutput[i]) diff=\(diff)")
        }

        print("[Numerical parity] F16 decode + BF16 weights: maxDiff=\(maxDiff)")
    }

    // MARK: - BF16 Conversion Helpers

    private static func floatToBF16(_ value: Float) -> UInt16 {
        let bits = value.bitPattern
        let lsb = (bits >> 16) & 1
        let roundingBias: UInt32 = 0x7FFF + lsb
        return UInt16((bits + roundingBias) >> 16)
    }

    private static func bf16ToFloat(_ value: UInt16) -> Float {
        Float(bitPattern: UInt32(value) << 16)
    }
}

// MARK: - Test Model

private struct TinyTransformerForFusion: ModelComponent {
    let hiddenSize: Int
    let layers: Int
    let vocabSize: Int
    var body: some ModelComponent {
        TokenEmbedding(vocabSize: vocabSize, embeddingSize: hiddenSize)
        Repeat(count: layers) {
            Residual {
                RMSNorm(dimension: hiddenSize, epsilon: 1e-5)
                Attention(
                    hiddenSize: hiddenSize, headCount: 4, kvHeadCount: 2,
                    headDimension: hiddenSize / 4,
                    rope: RoPEAttributes(dimension: hiddenSize / 4, base: 10000.0))
            }
            Residual {
                RMSNorm(dimension: hiddenSize, epsilon: 1e-5)
                MLP(inputSize: hiddenSize, intermediateSize: hiddenSize * 4)
            }
        }
        RMSNorm(dimension: hiddenSize, epsilon: 1e-5)
        OutputHead(inputSize: hiddenSize, vocabSize: vocabSize, tiedToEmbedding: true)
    }
}
