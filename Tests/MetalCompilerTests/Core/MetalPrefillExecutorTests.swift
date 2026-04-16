import Metal
import Testing
@testable import MetalCompiler

@Suite("Metal Prefill Executor")
struct MetalPrefillExecutorTests {
    @Test("Shared final hidden rows preserve per-token outputs")
    func sharedFinalHiddenRowsPreservePerTokenOutputs() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device")
            return
        }

        let hiddenSize = 4
        let sequenceLength = 3
        let maximumSequenceLength = 4
        let rowStride = hiddenSize * MemoryLayout<Float>.stride
        let hiddenByteCount = maximumSequenceLength * rowStride

        let hidden = try requiredSharedBuffer(device, length: hiddenByteCount)
        memset(hidden.contents(), 0, hidden.length)

        let buffers = PrefillBufferSet(
            bufferPrecision: .float32,
            hidden: hidden,
            residual: try requiredSharedBuffer(device, length: hiddenByteCount),
            scratch: try requiredSharedBuffer(device, length: hiddenByteCount),
            weights: [],
            kvCache: nil,
            convState: nil,
            recurrentState: nil,
            convStateDimension: 0,
            convStateKernelSize: 0,
            recurrentStateBytesPerLayer: 0,
            perLayerInputs: nil,
            perLayerInputDimension: 0,
            perLayerInputLayerCount: 0,
            logits: try requiredSharedBuffer(device, length: hiddenByteCount),
            tokenIDs: try requiredSharedBuffer(
                device,
                length: maximumSequenceLength * MemoryLayout<Int32>.stride
            ),
            positions: try requiredSharedBuffer(
                device,
                length: maximumSequenceLength * MemoryLayout<UInt32>.stride
            ),
            ropePositionAxes: try requiredSharedBuffer(
                device,
                length: maximumSequenceLength * 3 * MemoryLayout<UInt32>.stride
            ),
            tokenOut: try requiredSharedBuffer(device, length: MemoryLayout<Int32>.stride),
            dequantScratch: nil,
            runtimeConstantBuffer: try requiredSharedBuffer(
                device,
                length: PrefillBufferSet.runtimeConstantBufferSize(
                    maximumSequenceLength: maximumSequenceLength
                )
            )
        )

        let pipeline = try makeSharedHiddenWritePipeline(
            device: device,
            hiddenSize: hiddenSize,
            sequenceLength: sequenceLength
        )
        let step = MetalPrefillStep(
            pipeline: pipeline,
            gridSize: MTLSize(width: hiddenSize, height: sequenceLength, depth: 1),
            threadgroupSize: MTLSize(width: 1, height: 1, depth: 1),
            bufferBindings: [(index: 0, buffer: hidden, offset: 0)],
            bytesBindings: [],
            threadgroupMemoryLength: 0,
            sync: .none,
            mode: .batch,
            sequenceLengthPolicy: .none,
            positionBufferIndex: nil,
            perPositionStrides: [:]
        )
        let plan = MetalPrefillPlan(
            steps: [step],
            buffers: buffers,
            slotDimension: hiddenSize,
            maximumSequenceLength: maximumSequenceLength,
            stepCount: 1,
            usesMPP: false,
            finalHiddenBuffer: hidden,
            finalHiddenBaseOffset: 0,
            finalHiddenRowStride: rowStride,
            supplementalResidencyBuffers: []
        )

        var submission = try MetalSubmissionContext(device: device)
        let executor = MetalPrefillExecutor()
        let tokens: [Int32] = [11, 22, 33]

        let rows = try executor.captureFinalHiddenRows(
            prefillPlan: plan,
            submission: &submission,
            position: 0,
            tokens: tokens
        )
        #expect(
            rows == [
                [1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12],
            ]
        )

        let lastHidden = try executor.captureLastTokenFinalHidden(
            prefillPlan: plan,
            submission: &submission,
            position: 0,
            tokens: tokens
        )
        #expect(lastHidden == [9, 10, 11, 12])
    }
}

private func requiredSharedBuffer(
    _ device: MTLDevice,
    length: Int
) throws -> MTLBuffer {
    guard let buffer = device.makeBuffer(length: length, options: .storageModeShared) else {
        throw MetalCompilerError.deviceSetupFailed("Cannot allocate shared test buffer")
    }
    return buffer
}

private func makeSharedHiddenWritePipeline(
    device: MTLDevice,
    hiddenSize: Int,
    sequenceLength: Int
) throws -> MTLComputePipelineState {
    let source = """
    #include <metal_stdlib>
    using namespace metal;

    kernel void write_hidden_rows(
        device float* hidden [[buffer(0)]],
        uint2 gid [[thread_position_in_grid]]
    ) {
        if (gid.x >= \(hiddenSize) || gid.y >= \(sequenceLength)) {
            return;
        }
        hidden[gid.y * \(hiddenSize) + gid.x] = float(gid.y * \(hiddenSize) + gid.x + 1);
    }
    """
    let options = MTLCompileOptions()
    options.languageVersion = .version4_0
    let library = try device.makeLibrary(source: source, options: options)
    guard let function = library.makeFunction(name: "write_hidden_rows") else {
        throw MetalCompilerError.deviceSetupFailed("Cannot compile shared hidden write kernel")
    }
    return try device.makeComputePipelineState(function: function)
}
