import Metal
import Testing
@testable import MetalCompiler
@testable import LMArchitecture
@testable import ModelDeclarations
@testable import LMIR

@Suite("Gemma4 Kernel Dump", .serialized)
struct Gemma4KernelDumpTests {

    @Test("Dump synthesized kernels and detect collisions", .timeLimit(.minutes(2)))
    func dumpKernels() throws {
        let config = makeRealGemma4Config()
        let graph = try ModelGraph(Gemma4(config: config))
        let resolvedGraph = ParameterResolver().resolve(graph: graph, convention: .gemma4Family)

        let compiler = MetalInferenceCompiler()
        let report = compiler.dumpKernels(graph: resolvedGraph, hiddenSize: config.hiddenSize)

        print("\n=== Kernel Report: \(report.kernels.count) unique kernels ===\n")

        for kernel in report.kernels {
            let layerList = kernel.layers.map { $0.map(String.init) ?? "-" }.joined(separator: ", ")
            print("[\(kernel.name)]")
            print("  components: \(kernel.components.joined(separator: " -> "))")
            print("  layers: [\(layerList)]")
            print("  source (\(kernel.source.count) chars):")
            print(kernel.source)
            print()
        }

        if report.collisions.isEmpty {
            print("=== No collisions ===")
        } else {
            print("=== COLLISIONS: \(report.collisions.count) ===")
            for c in report.collisions {
                print("  \(c.kernelName):")
                print("    layer \(c.firstLayer.map(String.init) ?? "-"): \(c.firstComponents.joined(separator: " -> "))")
                print("    layer \(c.secondLayer.map(String.init) ?? "-"): \(c.secondComponents.joined(separator: " -> "))")
            }
        }
    }
}

private func makeRealGemma4Config() -> ModelConfig {
    ModelConfig(
        hiddenSize: 1536,
        layerCount: 35,
        intermediateSize: 6144,
        vocabSize: 262144,
        attentionHeads: 8,
        kvHeads: 1,
        headDim: 256,
        attentionBias: false,
        mlpBias: false,
        normEps: 1e-6,
        normKind: .rmsNorm,
        ropeTheta: 10_000.0,
        ropeDimension: 256,
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
        slidingWindow: 512,
        layerTypes: [
            "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "full_attention",
            "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "full_attention",
            "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "full_attention",
            "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "full_attention",
            "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "full_attention",
            "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "full_attention",
            "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "full_attention",
        ],
        hiddenSizePerLayerInput: 256,
        vocabSizePerLayerInput: 262144,
        globalHeadDim: 512,
        globalKVHeads: nil,
        numKVSharedLayers: 20,
        useDoubleWideMLP: true,
        attentionKEqualsV: false,
        fullAttentionRopeTheta: 1_000_000.0,
        fullAttentionPartialRotaryFactor: 0.25,
        fullAttentionRoPEScaling: RoPEScaling(kind: .custom("proportional"), factor: 1.0)
    )
}
