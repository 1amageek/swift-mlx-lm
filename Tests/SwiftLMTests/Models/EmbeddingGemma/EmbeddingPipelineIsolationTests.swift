import Foundation
import Metal
import Testing
@testable import MetalCompiler

/// Minimal isolation test for custom MSL pipeline + MTL4 argument table dispatch.
///
/// This test bypasses all embedding/prefill machinery and directly verifies that
/// a pipeline compiled from inline MSL source can write data through the
/// MTL4ComputeCommandEncoder + MTL4ArgumentTable dispatch pattern.
@Suite("Embedding Pipeline Isolation", .serialized)
struct EmbeddingPipelineIsolationTests {

    @Test("Custom MSL pipeline writes data via MTL4 dispatch", .timeLimit(.minutes(1)))
    func customPipelineWrites() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            print("[PipelineIsolation.Skip] No Metal device")
            return
        }

        // 1. Create a trivial MSL kernel that writes known values
        let msl = """
        #include <metal_stdlib>
        using namespace metal;

        kernel void test_write(
            device float* output       [[buffer(0)]],
            const device uint* params  [[buffer(1)]],
            uint tid                   [[thread_position_in_grid]]
        ) {
            const uint count = params[0];
            if (tid >= count) return;
            output[tid] = float(tid + 1) * 0.5f;
        }
        """

        let options = MTLCompileOptions()
        options.languageVersion = .version4_0
        let library = try device.makeLibrary(source: msl, options: options)
        guard let function = library.makeFunction(name: "test_write") else {
            throw MetalCompilerError.deviceSetupFailed("test_write not found")
        }
        let descriptor = MTLComputePipelineDescriptor()
        descriptor.computeFunction = function
        descriptor.threadGroupSizeIsMultipleOfThreadExecutionWidth = false
        let pipeline = try device.makeComputePipelineState(
            descriptor: descriptor, options: [], reflection: nil
        )

        // 2. Allocate buffers
        let count = 768
        let outputSize = count * MemoryLayout<Float>.stride
        let paramSize = MemoryLayout<UInt32>.stride
        guard let outputBuffer = device.makeBuffer(length: outputSize, options: .storageModeShared),
              let paramBuffer = device.makeBuffer(length: paramSize, options: .storageModeShared)
        else {
            throw MetalCompilerError.deviceSetupFailed("Failed to allocate test buffers")
        }

        // Write parameters
        paramBuffer.contents().bindMemory(to: UInt32.self, capacity: 1).pointee = UInt32(count)

        // Prefill output with sentinel value
        let outPtr = outputBuffer.contents().bindMemory(to: Float.self, capacity: count)
        for i in 0..<count { outPtr[i] = -999.0 }

        // 3. Dispatch via MetalSubmissionContext
        var submission = try MetalSubmissionContext(device: device)
        let residency = try MetalResidencyLease.required(
            device: device,
            label: "test.isolation",
            buffers: [outputBuffer, paramBuffer]
        )
        residency.add(to: submission.queue)

        try submission.withCompute(ephemeralResidency: residency) { encoder, argumentTable in
            argumentTable.setAddress(outputBuffer.gpuAddress, index: 0)
            argumentTable.setAddress(paramBuffer.gpuAddress, index: 1)
            encoder.setArgumentTable(argumentTable)
            encoder.setComputePipelineState(pipeline)

            let threads = min(count, pipeline.maxTotalThreadsPerThreadgroup)
            let groups = (count + threads - 1) / threads
            encoder.dispatchThreadgroups(
                threadgroupsPerGrid: MTLSize(width: groups, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1)
            )
        }

        // 4. Read back and verify
        let resultPtr = outputBuffer.contents().bindMemory(to: Float.self, capacity: count)
        let result0 = resultPtr[0]
        let result1 = resultPtr[1]
        let resultLast = resultPtr[count - 1]

        print("[PipelineIsolation] result[0]=\(result0) result[1]=\(result1) result[\(count-1)]=\(resultLast)")
        #expect(result0 == 0.5, "Expected 0.5, got \(result0)")
        #expect(result1 == 1.0, "Expected 1.0, got \(result1)")
        #expect(resultLast == Float(count) * 0.5, "Expected \(Float(count) * 0.5), got \(resultLast)")
    }

    @Test("Mean pool kernel reads input and writes output via MTL4", .timeLimit(.minutes(1)))
    func meanPoolKernelWorks() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            print("[PipelineIsolation.Skip] No Metal device")
            return
        }

        // Compile the actual embedding MSL
        let postProcessor = try MetalEmbeddingPostProcessor.compile(
            device: device,
            poolingStrategy: .mean,
            denseLayers: [],
            l2NormalizeEnabled: false
        )
        let workspace = try postProcessor.makeWorkspace(device: device)

        // Create a small hidden buffer with known data
        let seqLen = 4
        let hiddenDim = 8
        let hiddenSize = seqLen * hiddenDim * MemoryLayout<Float>.stride
        guard let hiddenBuffer = device.makeBuffer(length: hiddenSize, options: .storageModeShared) else {
            throw MetalCompilerError.deviceSetupFailed("Failed to allocate hidden buffer")
        }

        // Fill with known values: row i, col j = float(i * hiddenDim + j + 1)
        let hiddenPtr = hiddenBuffer.contents().bindMemory(to: Float.self, capacity: seqLen * hiddenDim)
        for row in 0..<seqLen {
            for col in 0..<hiddenDim {
                hiddenPtr[row * hiddenDim + col] = Float(row * hiddenDim + col + 1)
            }
        }

        // Dump input for verification
        for row in 0..<seqLen {
            let rowVals = (0..<hiddenDim).map { String(format: "%.1f", hiddenPtr[row * hiddenDim + $0]) }
            print("[PipelineIsolation.MeanPool] input row \(row): [\(rowVals.joined(separator: ", "))]")
        }

        var submission = try MetalSubmissionContext(device: device)
        let residency = try MetalResidencyLease.required(
            device: device,
            label: "test.meanpool",
            buffers: [hiddenBuffer] + workspace.residencyBuffers
        )
        residency.add(to: submission.queue)

        try submission.withCompute(ephemeralResidency: residency) { encoder, argumentTable in
            workspace.encode(
                encoder: encoder,
                argumentTable: argumentTable,
                hiddenBuffer: hiddenBuffer,
                hiddenBaseOffset: 0,
                hiddenRowStride: hiddenDim * MemoryLayout<Float>.stride,
                hiddenDimension: hiddenDim,
                sequenceLength: seqLen,
                promptTokenCount: 0
            )
        }

        // Read result
        let result = workspace.readResult(hiddenDimension: hiddenDim)
        print("[PipelineIsolation.MeanPool] result: \(result.map { String(format: "%.4f", $0) })")

        // Expected: mean of all rows. col j = mean of (j+1, j+9, j+17, j+25) = mean of rows
        // Row 0: 1..8, Row 1: 9..16, Row 2: 17..24, Row 3: 25..32
        // Mean col 0: (1+9+17+25)/4 = 13.0
        let expectedCol0: Float = Float(1 + 1 + hiddenDim + 1 + 2*hiddenDim + 1 + 3*hiddenDim) / 4.0
        print("[PipelineIsolation.MeanPool] expected col0 = \(expectedCol0)")

        let norm = sqrt(result.reduce(0) { $0 + $1 * $1 })
        print("[PipelineIsolation.MeanPool] resultNorm=\(String(format: "%.4f", norm))")
        #expect(norm > 0, "Result norm is zero — kernel produced no output")
        #expect(result.count == hiddenDim)
    }
}
