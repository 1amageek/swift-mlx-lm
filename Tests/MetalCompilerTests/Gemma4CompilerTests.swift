import Metal
import Testing
@testable import LMArchitecture
@testable import MetalCompiler
@testable import ModelDeclarations

@Suite("Gemma4 Compiler", .serialized)
struct Gemma4CompilerTests {
    @Test("Gemma4 compile allocates per-layer input buffers", .timeLimit(.minutes(2)))
    func compileAllocatesPerLayerInputs() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            print("[Skip] No Metal device available for Gemma4 compiler tests")
            return
        }

        let config = ModelConfig(
            hiddenSize: 64,
            layerCount: 2,
            intermediateSize: 128,
            vocabSize: 4096,
            attentionHeads: 4,
            kvHeads: 1,
            headDim: 16,
            attentionBias: false,
            mlpBias: false,
            normEps: 1e-6,
            normKind: .rmsNorm,
            ropeTheta: 10_000.0,
            ropeDimension: 16,
            ropeScaling: nil,
            tiedEmbeddings: true,
            expertCount: nil,
            expertsPerToken: nil,
            qkNorm: true,
            fullAttentionInterval: nil,
            ssmNumHeads: nil,
            ssmKeyHeadDim: nil,
            ssmValueHeadDim: nil,
            convKernelSize: nil,
            partialRotaryFactor: nil,
            slidingWindow: 32,
            layerTypes: ["sliding_attention", "full_attention"],
            hiddenSizePerLayerInput: 8,
            vocabSizePerLayerInput: 4096,
            globalHeadDim: 16,
            globalKVHeads: nil,
            numKVSharedLayers: 1,
            useDoubleWideMLP: false,
            attentionKEqualsV: false,
            fullAttentionRopeTheta: 1_000_000.0,
            fullAttentionPartialRotaryFactor: 0.25,
            fullAttentionRoPEScaling: RoPEScaling(kind: .custom("proportional"), factor: 1.0)
        )
        let graph = try Gemma4(config: config).makeModelGraph()
        let resolvedGraph = ParameterResolver().resolve(graph: graph, convention: .gemma4Family)
        let compiler = MetalInferenceCompiler(optimizer: AggressiveOptimizer())
        var compiled = try compiler.compile(
            graph: resolvedGraph,
            hiddenSize: config.hiddenSize,
            intermediateSize: config.intermediateSize,
            vocabSize: config.vocabSize,
            device: device
        )
        let prefillPlan = try compiler.compilePrefill(
            graph: resolvedGraph,
            hiddenSize: config.hiddenSize,
            intermediateSize: config.intermediateSize,
            vocabSize: config.vocabSize,
            maximumSequenceLength: 128,
            sharedKVCache: compiled.buffers.kvCache,
            sharedConvState: compiled.buffers.convState,
            sharedConvStateDimension: compiled.buffers.convStateDimension,
            sharedConvStateKernelSize: compiled.buffers.convStateKernelSize,
            sharedRecurrentState: compiled.buffers.recurrentState,
            sharedRecurrentStateBytesPerLayer: compiled.buffers.recurrentStateBytesPerLayer,
            device: device
        )
        compiled = compiled.withPrefillPlan(prefillPlan)

        #expect(compiled.buffers.perLayerInputs != nil)
        #expect(compiled.buffers.perLayerInputDimension == 8)
        #expect(compiled.buffers.perLayerInputLayerCount == 2)
        #expect(prefillPlan.buffers.perLayerInputs != nil)
        #expect(prefillPlan.buffers.perLayerInputDimension == 8)
        #expect(prefillPlan.buffers.perLayerInputLayerCount == 2)
    }
}
