import Testing
import Metal
@testable import MetalCompiler
@testable import SwiftLM

@Suite("LanguageModelContainer Isolation", .serialized)
struct LanguageModelContainerIsolationTests {

    @Test("makeContext returns isolated runtime buffers while sharing immutable weights")
    func makeContextReturnsIsolatedRuntimeState() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device")
            return
        }

        let weight = try requiredPrivateBuffer(device, length: 512)
        let decodePlan = MetalDispatchPlan(
            steps: [],
            buffers: try makeDecodeBufferSet(device: device, weights: [weight]),
            unfusedEntryCount: 0,
            fusedEntryCount: 0,
            supplementalResidencyBuffers: []
        )
        let inferenceModel = try MetalInferenceModel(plan: decodePlan, device: device)
        let prototypeContext = LanguageModelContext(
            inferenceModel: inferenceModel,
            tokenizer: QwenVisionTestTokenizer(),
            configuration: ModelConfiguration(name: "test-model")
        )
        let container = LanguageModelContainer(prototypeContext: prototypeContext)

        let first = try container.makeContext()
        let second = try container.makeContext()

        let firstBuffers = first.debugCompiledModel.decodePlan.buffers
        let secondBuffers = second.debugCompiledModel.decodePlan.buffers

        #expect(firstBuffers.hidden !== secondBuffers.hidden)
        #expect(firstBuffers.residual !== secondBuffers.residual)
        #expect(firstBuffers.scratch !== secondBuffers.scratch)
        #expect(firstBuffers.logits !== secondBuffers.logits)
        #expect(firstBuffers.position !== secondBuffers.position)
        #expect(firstBuffers.weights[0] === secondBuffers.weights[0])
        #expect(firstBuffers.weights[0] === weight)
    }

    private func makeDecodeBufferSet(
        device: MTLDevice,
        weights: [MTLBuffer]
    ) throws -> MetalBufferSet {
        MetalBufferSet(
            bufferPrecision: .float16,
            hidden: try requiredPrivateBuffer(device, length: 256),
            residual: try requiredPrivateBuffer(device, length: 256),
            scratch: try requiredPrivateBuffer(device, length: 512),
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
            position: try requiredSharedBuffer(device, length: 4),
            ropePositionAxes: try requiredSharedBuffer(device, length: 12),
            tokenIn: try requiredSharedBuffer(device, length: 4),
            tokenOut: try requiredSharedBuffer(device, length: 4)
        )
    }

    private func requiredSharedBuffer(_ device: MTLDevice, length: Int) throws -> MTLBuffer {
        try #require(device.makeBuffer(length: length, options: .storageModeShared))
    }

    private func requiredPrivateBuffer(_ device: MTLDevice, length: Int) throws -> MTLBuffer {
        try #require(device.makeBuffer(length: length, options: .storageModePrivate))
    }
}
