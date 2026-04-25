import Testing
import Metal
@testable import MetalCompiler
import LMArchitecture
import LMIR
import ModelDeclarations

@Suite("Metal Allocator Regression", .serialized)
struct MetalAllocatorRegressionTests {

    @Test("Prefill scratch sizing uses unfused graph requirements for Qwen DeltaNet")
    func qwenDeltaNetScratchSizingTracksLargestProjection() throws {
        let gpuLock = try GPUTestExclusion.acquire()
        defer { gpuLock.release() }
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device")
            return
        }

        let config = ModelConfig(
            hiddenSize: 1024,
            layerCount: 1,
            intermediateSize: 3584,
            vocabSize: 32000,
            attentionHeads: 8,
            kvHeads: 2,
            headDim: 256,
            attentionBias: false,
            mlpBias: false,
            normEps: 1e-6,
            normKind: .rmsNorm,
            ropeTheta: 10_000_000,
            ropeDimension: 256,
            ropeScaling: nil,
            tiedEmbeddings: true,
            expertCount: nil,
            expertsPerToken: nil,
            qkNorm: false,
            fullAttentionInterval: 4,
            ssmNumHeads: 16,
            ssmGroupCount: 16,
            ssmKeyHeadDim: 128,
            ssmValueHeadDim: 128,
            convKernelSize: 4,
            convLCache: nil,
            partialRotaryFactor: 0.25,
            slidingWindow: nil,
            layerTypes: nil,
            hiddenSizePerLayerInput: nil,
            vocabSizePerLayerInput: nil,
            globalHeadDim: nil,
            globalKVHeads: nil,
            numKVSharedLayers: nil,
            useDoubleWideMLP: false,
            attentionKEqualsV: false,
            fullAttentionRopeTheta: nil,
            fullAttentionPartialRotaryFactor: nil,
            fullAttentionRoPEScaling: nil,
            numDenseLayers: 0,
            mropeAxes: MRoPEAxes(sections: [11, 11, 10], interleaved: true)
        )

        let graph = try ModelGraph(Qwen35(config: config))
        let resolved = ParameterResolver().resolve(graph: graph, convention: .qwen35Family)
        let store = try makeWeightStore(for: resolved, device: device)
        let compiler = MetalInferenceCompiler()
        let maxSequenceLength = 8
        let plan = try compiler.compilePrefill(
            graph: resolved,
            hiddenSize: config.hiddenSize,
            intermediateSize: config.intermediateSize,
            vocabSize: config.vocabSize,
            inferencePolicy: InferencePolicy(maximumSequenceLength: maxSequenceLength),
            stafWeightStore: store,
            device: device
        )

        let slotDimension = plan.buffers.scratch.length
            / (maxSequenceLength * MemoryLayout<Float>.size * 5)
        #expect(
            slotDimension >= 6144,
            "Qwen DeltaNet scratch slot must fit in_proj_qkv (6144), got \(slotDimension)"
        )
    }

    private func makeWeightStore(
        for graph: ModelGraph,
        device: MTLDevice
    ) throws -> STAFWeightStore {
        guard let buffer = device.makeBuffer(length: MemoryLayout<Float16>.size, options: .storageModeShared) else {
            throw MetalCompilerError.deviceSetupFailed("Cannot allocate dummy STAF weight buffer")
        }

        var entries: [String: STAFTensorEntry] = [:]
        for tensorName in tensorNames(in: graph.rootRegion) {
            entries[tensorName] = STAFTensorEntry(
                name: tensorName,
                payloadOffset: 0,
                payloadSize: buffer.length,
                schemeIdentifier: .passthrough,
                semanticRole: .unknown,
                shape: [1],
                blockSize: 0,
                groupSize: 0,
                bufferOffset: 0
            )
        }

        return STAFWeightStore(
            buffer: buffer,
            entries: entries,
            metadata: .empty,
            specializedBufferAccesses: [:]
        )
    }

    private func tensorNames(in region: Region) -> Set<String> {
        var names = Set(region.operations.flatMap { operation in
            operation.parameterBindings.map(\.tensorName)
        })

        for operation in region.operations {
            switch operation.kind {
            case .primitive:
                break
            case .residual(_, let body):
                names.formUnion(tensorNames(in: body))
            case .parallel(_, let branches):
                for branch in branches {
                    names.formUnion(tensorNames(in: branch))
                }
            case .repeating(_, let body):
                names.formUnion(tensorNames(in: body))
            case .conditional(_, let thenRegion, let elseRegion):
                names.formUnion(tensorNames(in: thenRegion))
                names.formUnion(tensorNames(in: elseRegion))
            }
        }

        return names
    }
}
