import Testing
import Metal
@testable import MetalCompiler
import LMArchitecture
import ModelDeclarations
import LMIR

/// Verify GPU-side F32→F16/BF16 hidden state conversion correctness.
@Suite("Hidden Conversion")
struct HiddenConversionTests {

    // MARK: - GPU Kernel Conversion

    @Test("hidden_copy_from_float kernel converts F32 to F16 correctly")
    func f32ToF16ConversionKernel() throws {
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device"); return
        }

        let count = 256
        let sourceValues: [Float32] = (0..<count).map { Float32($0) * 0.01 - 1.28 }

        // Allocate source (F32) and destination (F16) buffers
        let sourceBuffer = try #require(
            device.makeBuffer(length: count * MemoryLayout<Float32>.size, options: .storageModeShared))
        let destBuffer = try #require(
            device.makeBuffer(length: count * MemoryLayout<Float16>.size, options: .storageModeShared))

        // Fill source
        let srcPtr = sourceBuffer.contents().bindMemory(to: Float32.self, capacity: count)
        for i in 0..<count { srcPtr[i] = sourceValues[i] }

        // Compile kernel
        let kernelSource = """
        #include <metal_stdlib>
        using namespace metal;
        kernel void hidden_copy_from_float(
            device half* dst [[buffer(0)]],
            device const float* src [[buffer(1)]],
            constant uint& count [[buffer(2)]],
            uint gid [[thread_position_in_grid]]
        ) {
            if (gid < count) { dst[gid] = half(src[gid]); }
        }
        """
        let library = try device.makeLibrary(source: kernelSource, options: nil)
        let function = try #require(library.makeFunction(name: "hidden_copy_from_float"))
        let pipeline = try device.makeComputePipelineState(function: function)

        // Dispatch
        let queue = try #require(device.makeCommandQueue())
        let commandBuffer = try #require(queue.makeCommandBuffer())
        let encoder = try #require(commandBuffer.makeComputeCommandEncoder())

        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(destBuffer, offset: 0, index: 0)
        encoder.setBuffer(sourceBuffer, offset: 0, index: 1)
        var countValue = UInt32(count)
        withUnsafeBytes(of: &countValue) { bytes in
            encoder.setBytes(bytes.baseAddress!, length: bytes.count, index: 2)
        }

        let threadCount = min(count, pipeline.maxTotalThreadsPerThreadgroup)
        let gridSize = MTLSize(width: (count + threadCount - 1) / threadCount, height: 1, depth: 1)
        encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: MTLSize(width: threadCount, height: 1, depth: 1))
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        // Verify
        let dstPtr = destBuffer.contents().bindMemory(to: Float16.self, capacity: count)
        var maxError: Float = 0
        for i in 0..<count {
            let expected = Float16(sourceValues[i])
            let actual = dstPtr[i]
            let error = abs(Float(actual) - Float(expected))
            maxError = max(maxError, error)
            #expect(error < 0.001,
                    "Element \(i): expected \(expected) got \(actual) (source \(sourceValues[i]))")
        }
        print("[F32→F16] max error: \(maxError) over \(count) elements")
    }

    @Test("Conversion handles edge values: zero, subnormal, max F16, negative")
    func conversionEdgeCases() throws {
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device"); return
        }

        let edgeValues: [Float32] = [
            0.0,                    // zero
            -0.0,                   // negative zero
            1.0,                    // unity
            -1.0,                   // negative unity
            65504.0,                // max F16
            -65504.0,               // min F16
            0.000061035156,         // smallest normal F16
            0.00001,                // subnormal in F16
            3.14159,                // pi
        ]
        let count = edgeValues.count

        let sourceBuffer = try #require(
            device.makeBuffer(length: count * MemoryLayout<Float32>.size, options: .storageModeShared))
        let destBuffer = try #require(
            device.makeBuffer(length: count * MemoryLayout<Float16>.size, options: .storageModeShared))

        let srcPtr = sourceBuffer.contents().bindMemory(to: Float32.self, capacity: count)
        for i in 0..<count { srcPtr[i] = edgeValues[i] }

        let kernelSource = """
        #include <metal_stdlib>
        using namespace metal;
        kernel void hidden_copy_from_float(
            device half* dst [[buffer(0)]],
            device const float* src [[buffer(1)]],
            constant uint& count [[buffer(2)]],
            uint gid [[thread_position_in_grid]]
        ) {
            if (gid < count) { dst[gid] = half(src[gid]); }
        }
        """
        let library = try device.makeLibrary(source: kernelSource, options: nil)
        let function = try #require(library.makeFunction(name: "hidden_copy_from_float"))
        let pipeline = try device.makeComputePipelineState(function: function)

        let queue = try #require(device.makeCommandQueue())
        let commandBuffer = try #require(queue.makeCommandBuffer())
        let encoder = try #require(commandBuffer.makeComputeCommandEncoder())
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(destBuffer, offset: 0, index: 0)
        encoder.setBuffer(sourceBuffer, offset: 0, index: 1)
        var countValue = UInt32(count)
        withUnsafeBytes(of: &countValue) { bytes in
            encoder.setBytes(bytes.baseAddress!, length: bytes.count, index: 2)
        }
        encoder.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: count, height: 1, depth: 1))
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        let dstPtr = destBuffer.contents().bindMemory(to: Float16.self, capacity: count)
        for i in 0..<count {
            let expected = Float16(edgeValues[i])
            let actual = dstPtr[i]
            #expect(actual == expected,
                    "Edge case \(i) (\(edgeValues[i])): expected \(expected) got \(actual)")
        }
    }

    // MARK: - Conversion Pipeline Resolution

    @Test("Compiled model provides hidden_copy_from_float in auxiliary pipelines")
    func compiledModelHasConversionPipeline() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device"); return
        }

        let config = makeTestConfig()
        let graph = try ModelGraph(Transformer(config: config))
        let resolved = ParameterResolver().resolve(graph: graph, convention: .llamaFamily)
        let compiled = try MetalInferenceCompiler().compile(
            graph: resolved, hiddenSize: config.hiddenSize,
            intermediateSize: config.intermediateSize,
            vocabSize: config.vocabSize, device: device)

        // Decode precision is F16, so hidden_copy_from_float should be present
        if compiled.decodePlan.buffers.bufferPrecision == .float16 {
            #expect(compiled.auxiliaryPipelines["hidden_copy_from_float"] != nil,
                    "F16 decode should have hidden_copy_from_float auxiliary pipeline")
        } else if compiled.decodePlan.buffers.bufferPrecision == .bfloat16 {
            #expect(compiled.auxiliaryPipelines["hidden_copy_from_float_bf16"] != nil,
                    "BF16 decode should have hidden_copy_from_float_bf16 auxiliary pipeline")
        }
    }

    // MARK: - Conversion Precision Consistency

    @Test("F32 prefill hidden values survive roundtrip through F16 conversion")
    func f32ToF16RoundtripPrecision() throws {
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device"); return
        }

        // Simulate typical hidden state values (mean ~0, std ~1)
        let count = 2048
        var rng = SystemRandomNumberGenerator()
        let sourceValues: [Float32] = (0..<count).map { _ in
            // Box-Muller approximation: uniform → roughly normal
            let u = Float.random(in: 0.001...0.999, using: &rng)
            let v = Float.random(in: 0.001...0.999, using: &rng)
            return sqrt(-2.0 * log(u)) * cos(2.0 * .pi * v)
        }

        let sourceBuffer = try #require(
            device.makeBuffer(length: count * MemoryLayout<Float32>.size, options: .storageModeShared))
        let destBuffer = try #require(
            device.makeBuffer(length: count * MemoryLayout<Float16>.size, options: .storageModeShared))

        let srcPtr = sourceBuffer.contents().bindMemory(to: Float32.self, capacity: count)
        for i in 0..<count { srcPtr[i] = sourceValues[i] }

        let kernelSource = """
        #include <metal_stdlib>
        using namespace metal;
        kernel void hidden_copy_from_float(
            device half* dst [[buffer(0)]],
            device const float* src [[buffer(1)]],
            constant uint& count [[buffer(2)]],
            uint gid [[thread_position_in_grid]]
        ) {
            if (gid < count) { dst[gid] = half(src[gid]); }
        }
        """
        let library = try device.makeLibrary(source: kernelSource, options: nil)
        let function = try #require(library.makeFunction(name: "hidden_copy_from_float"))
        let pipeline = try device.makeComputePipelineState(function: function)

        let queue = try #require(device.makeCommandQueue())
        let commandBuffer = try #require(queue.makeCommandBuffer())
        let encoder = try #require(commandBuffer.makeComputeCommandEncoder())
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(destBuffer, offset: 0, index: 0)
        encoder.setBuffer(sourceBuffer, offset: 0, index: 1)
        var countValue = UInt32(count)
        withUnsafeBytes(of: &countValue) { bytes in
            encoder.setBytes(bytes.baseAddress!, length: bytes.count, index: 2)
        }
        let threadCount = min(count, pipeline.maxTotalThreadsPerThreadgroup)
        let gridSize = MTLSize(width: (count + threadCount - 1) / threadCount, height: 1, depth: 1)
        encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: MTLSize(width: threadCount, height: 1, depth: 1))
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        // Measure precision loss
        let dstPtr = destBuffer.contents().bindMemory(to: Float16.self, capacity: count)
        var totalError: Double = 0
        var maxError: Float = 0
        for i in 0..<count {
            let original = sourceValues[i]
            let converted = Float(dstPtr[i])
            let error = abs(original - converted)
            totalError += Double(error)
            maxError = max(maxError, error)
        }
        let meanError = totalError / Double(count)

        print("[F32→F16 precision] mean error: \(String(format: "%.6f", meanError)) max error: \(String(format: "%.6f", maxError))")

        // F16 has ~3.3 decimal digits of precision. For values in [-3, 3],
        // relative error should be < 0.1%
        #expect(maxError < 0.01,
                "Max conversion error \(maxError) exceeds threshold for typical hidden state values")
        #expect(meanError < 0.001,
                "Mean conversion error \(meanError) exceeds threshold")
    }

    // MARK: - Source Offset Correctness

    @Test("Hidden conversion reads from correct offset for last token in sequence")
    func conversionSourceOffsetForLastToken() throws {
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device"); return
        }

        let hiddenSize = 64
        let maxSeqLen = 16
        let seqLen = 10
        let f32Size = MemoryLayout<Float32>.size

        // Allocate prefill hidden (F32, [maxSeqLen × hiddenSize])
        let prefillHidden = try #require(
            device.makeBuffer(length: maxSeqLen * hiddenSize * f32Size, options: .storageModeShared))

        // Fill with position-dependent values: each position has unique pattern
        let srcPtr = prefillHidden.contents().bindMemory(to: Float32.self, capacity: maxSeqLen * hiddenSize)
        for pos in 0..<maxSeqLen {
            for dim in 0..<hiddenSize {
                srcPtr[pos * hiddenSize + dim] = Float32(pos * 1000 + dim)
            }
        }

        // The last token in sequence of 10 is at position 9
        let lastTokenOffset = (seqLen - 1) * hiddenSize * f32Size

        // Allocate decode hidden (F16, [hiddenSize])
        let decodeHidden = try #require(
            device.makeBuffer(length: hiddenSize * MemoryLayout<Float16>.size, options: .storageModeShared))

        let kernelSource = """
        #include <metal_stdlib>
        using namespace metal;
        kernel void hidden_copy_from_float(
            device half* dst [[buffer(0)]],
            device const float* src [[buffer(1)]],
            constant uint& count [[buffer(2)]],
            uint gid [[thread_position_in_grid]]
        ) {
            if (gid < count) { dst[gid] = half(src[gid]); }
        }
        """
        let library = try device.makeLibrary(source: kernelSource, options: nil)
        let function = try #require(library.makeFunction(name: "hidden_copy_from_float"))
        let pipeline = try device.makeComputePipelineState(function: function)

        let queue = try #require(device.makeCommandQueue())
        let commandBuffer = try #require(queue.makeCommandBuffer())
        let encoder = try #require(commandBuffer.makeComputeCommandEncoder())
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(decodeHidden, offset: 0, index: 0)
        encoder.setBuffer(prefillHidden, offset: lastTokenOffset, index: 1)
        var countValue = UInt32(hiddenSize)
        withUnsafeBytes(of: &countValue) { bytes in
            encoder.setBytes(bytes.baseAddress!, length: bytes.count, index: 2)
        }
        encoder.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: hiddenSize, height: 1, depth: 1))
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        // Verify: decode hidden should contain values from position 9
        let dstPtr = decodeHidden.contents().bindMemory(to: Float16.self, capacity: hiddenSize)
        for dim in 0..<hiddenSize {
            let expected = Float16(Float32(9 * 1000 + dim))
            let actual = dstPtr[dim]
            #expect(actual == expected,
                    "Dim \(dim): expected \(expected) (from position 9) got \(actual)")
        }
    }

    // MARK: - Helpers

    private func makeTestConfig() -> ModelConfig {
        ModelConfig(
            hiddenSize: 128, layerCount: 2, intermediateSize: 512,
            vocabSize: 1000, attentionHeads: 4, kvHeads: 4, headDim: 32,
            attentionBias: false, mlpBias: false, normEps: 1e-5,
            normKind: .rmsNorm, ropeTheta: 10000, ropeDimension: 32,
            ropeScaling: nil, tiedEmbeddings: true,
            expertCount: nil, expertsPerToken: nil, qkNorm: false,
            fullAttentionInterval: nil, ssmNumHeads: nil, ssmKeyHeadDim: nil,
            ssmValueHeadDim: nil, convKernelSize: nil,
            partialRotaryFactor: nil, slidingWindow: nil
        )
    }
}
