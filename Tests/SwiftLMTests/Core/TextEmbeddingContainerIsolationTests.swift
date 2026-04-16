import Testing
import Metal
@testable import SwiftLM
@testable import MetalCompiler

@Suite("TextEmbeddingContainer Isolation", .serialized)
struct TextEmbeddingContainerIsolationTests {

    @Test("TextEmbeddingContext initializer returns isolated runtime buffers while sharing immutable weights")
    func contextInitializerReturnsIsolatedRuntimeState() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device")
            return
        }

        let weight = try requiredPrivateBuffer(device, length: 512)
        let buffers = try makePrefillBufferSet(device: device, weights: [weight])
        let plan = MetalPrefillPlan(
            steps: [],
            buffers: buffers,
            slotDimension: 128,
            maximumSequenceLength: 8,
            stepCount: 0,
            usesMPP: false,
            finalHiddenBuffer: buffers.hidden,
            finalHiddenBaseOffset: 0,
            finalHiddenRowStride: 128,
            supplementalResidencyBuffers: []
        )
        let container = try TextEmbeddingContainer(
            prefillPlan: plan,
            device: device,
            tokenizer: QwenVisionTestTokenizer(),
            runtime: makeRuntime(),
            configuration: ModelConfiguration(name: "embedding-test-model")
        )

        let first = try TextEmbeddingContext(container)
        let second = try TextEmbeddingContext(container)

        let firstBuffers = first.debugPrefillPlan.buffers
        let secondBuffers = second.debugPrefillPlan.buffers

        #expect(firstBuffers.hidden !== secondBuffers.hidden)
        #expect(firstBuffers.residual !== secondBuffers.residual)
        #expect(firstBuffers.scratch !== secondBuffers.scratch)
        #expect(firstBuffers.logits !== secondBuffers.logits)
        #expect(firstBuffers.tokenIDs !== secondBuffers.tokenIDs)
        #expect(firstBuffers.positions !== secondBuffers.positions)
        #expect(firstBuffers.tokenOut !== secondBuffers.tokenOut)
        #expect(firstBuffers.runtimeConstantBuffer !== secondBuffers.runtimeConstantBuffer)
        #expect(firstBuffers.weights[0] === secondBuffers.weights[0])
        #expect(firstBuffers.weights[0] === weight)
    }

    private func makeRuntime() throws -> SentenceTransformerTextEmbeddingRuntime {
        let metadata = SentenceTransformerMetadata(
            prompts: ["query": "query: "],
            defaultPromptName: "query",
            similarityFunctionName: "cosine",
            pooling: .init(strategy: .mean, includePrompt: true),
            denseLayers: [],
            postprocessors: []
        )
        return try SentenceTransformerTextEmbeddingRuntime(
            metadata: metadata,
            weightStore: CPUWeightStore(denseTensors: [:])
        )
    }

    private func makePrefillBufferSet(
        device: MTLDevice,
        weights: [MTLBuffer]
    ) throws -> PrefillBufferSet {
        PrefillBufferSet(
            bufferPrecision: .float32,
            hidden: try requiredSharedBuffer(device, length: 1024),
            residual: try requiredPrivateBuffer(device, length: 1024),
            scratch: try requiredPrivateBuffer(device, length: 2048),
            weights: weights,
            kvCache: nil,
            convState: nil,
            recurrentState: nil,
            convStateDimension: 0,
            convStateKernelSize: 0,
            recurrentStateBytesPerLayer: 0,
            perLayerInputs: nil,
            perLayerInputDimension: 0,
            perLayerInputLayerCount: 0,
            logits: try requiredPrivateBuffer(device, length: 1024),
            tokenIDs: try requiredSharedBuffer(device, length: 16),
            positions: try requiredSharedBuffer(device, length: 16),
            ropePositionAxes: try requiredSharedBuffer(device, length: 48),
            tokenOut: try requiredSharedBuffer(device, length: 4),
            dequantScratch: nil,
            runtimeConstantBuffer: try requiredSharedBuffer(device, length: 32)
        )
    }

    private func requiredSharedBuffer(_ device: MTLDevice, length: Int) throws -> MTLBuffer {
        try #require(device.makeBuffer(length: length, options: .storageModeShared))
    }

    private func requiredPrivateBuffer(_ device: MTLDevice, length: Int) throws -> MTLBuffer {
        try #require(device.makeBuffer(length: length, options: .storageModePrivate))
    }
}
