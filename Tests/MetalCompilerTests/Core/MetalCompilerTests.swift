import Testing
import Metal
@testable import MetalCompiler
import LMArchitecture
import LMIR

@Suite("Fragment Protocol")
struct FragmentProtocolTests {

    @Test
    func embeddingFragmentIsGather() {
        let a = TokenEmbeddingAttributes(vocabSize: 32000, embeddingSize: 2048)
        let frag = a.fragment(context: KernelContext(bufferPrecision: .float16, weightFormat: .float16))
        #expect(frag is GatherFragment)
        if case .gather(let count) = frag.dispatchDimension {
            #expect(count == 2048)
        }
    }

    @Test
    func rmsNormFragmentIsReduction() {
        let a = RMSNormAttributes(dimension: 2048, epsilon: 1e-5)
        let frag = a.fragment(context: KernelContext(bufferPrecision: .float16, weightFormat: .float16))
        #expect(frag is Reduction)
        if case .reduction(let dim) = frag.dispatchDimension {
            #expect(dim == 2048)
        }
    }

    @Test
    func structuralOpsAreNotPrimitive() {
        let r = OperationKind.residual(strategy: .add, body: Region())
        if case .primitive = r {
            Issue.record("Residual should not be primitive")
        }
    }
}

@Suite
struct DispatchPlanCompilationTests {

    @Test
    func tinyModelCompilesToMultiDispatch() throws {
        guard let device = MTLCreateSystemDefaultDevice() else { return }
        let graph = try ModelGraph(TinyTestModel(hiddenSize: 64, vocabSize: 100))
        let compiler = MetalInferenceCompiler()
        let plan = try compiler.compile(graph: graph, hiddenSize: 64, vocabSize: 100, device: device)

        // Multiple dispatch steps (not 1)
        #expect(plan.steps.count > 1, "Expected multiple dispatches, got \(plan.steps.count)")
    }

    @Test
    func transformerCompilesToManyDispatches() throws {
        guard let device = MTLCreateSystemDefaultDevice() else { return }
        let graph = try ModelGraph(TinyTransformer(hiddenSize: 64, layers: 2, vocabSize: 100))
        let compiler = MetalInferenceCompiler()
        let plan = try compiler.compile(graph: graph, hiddenSize: 64, vocabSize: 100, device: device)

        // 2 layers x (norm + 5 attn + residual + norm + 4 mlp + residual) + embed + final_norm + output + argmax
        #expect(plan.steps.count > 20, "Expected many dispatches for 2-layer transformer, got \(plan.steps.count)")
    }

    @Test
    func threadgroupSizesRespectPipelineLimits() throws {
        guard let device = MTLCreateSystemDefaultDevice() else { return }
        let graph = try ModelGraph(TinyTestModel(hiddenSize: 64, vocabSize: 100))
        let compiler = MetalInferenceCompiler()
        let plan = try compiler.compile(graph: graph, hiddenSize: 64, vocabSize: 100, device: device)

        for (i, step) in plan.steps.enumerated() {
            let maxThreads = step.pipeline.maxTotalThreadsPerThreadgroup
            let tgWidth = step.threadgroupSize.width * step.threadgroupSize.height * step.threadgroupSize.depth
            #expect(tgWidth <= maxThreads,
                "Step \(i) threadgroup \(tgWidth) exceeds pipeline max \(maxThreads)")

            let warpWidth = step.pipeline.threadExecutionWidth
            #expect(step.threadgroupSize.width % warpWidth == 0 || step.threadgroupSize.width < warpWidth,
                "Step \(i) threadgroup width \(step.threadgroupSize.width) not aligned to warp \(warpWidth)")
        }
    }

}

@Suite
struct SafetensorsTests {
    @Test
    func dtypeParsing() {
        #expect(SafetensorsDType(rawValue: "F16") == .float16)
        #expect(SafetensorsDType(rawValue: "F32") == .float32)
        #expect(SafetensorsDType.float16.elementSize == 2)
    }
}

// MARK: - Test Models

struct TinyTestModel: ModelComponent {
    let hiddenSize: Int
    let vocabSize: Int
    var body: some ModelComponent {
        TokenEmbedding(vocabSize: vocabSize, embeddingSize: hiddenSize)
        RMSNorm(dimension: hiddenSize, epsilon: 1e-5)
        OutputHead(inputSize: hiddenSize, vocabSize: vocabSize, tiedToEmbedding: true)
    }
}

struct TinyTransformer: ModelComponent {
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

// MARK: - Kernel Completeness Tests

@Suite("Kernel Completeness")
struct KernelCompletenessTests {

    @Test("All QuantizationFormat kernel names exist in MSL")
    func quantizationFormatKernelsExist() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device")
            return
        }
        let options = MTLCompileOptions()
        options.fastMathEnabled = true
        options.languageVersion = .version4_0
        let library = try device.makeLibrary(
            source: MetalSourceGenerator.generateCompleteLibrary(weightFormat: .bfloat16), options: options)
        let available = Set(library.functionNames)

        let formats: [any QuantizationFormat] = [
            Float16Format(),
            BFloat16Format(),
            AffineQ4Group64Format(),
            AffineQ4Group128Format(),
            AffineQ8Group32Format(),
            AffineQ8Group64Format(),
        ]
        for format in formats {
            #expect(available.contains(format.gemvKernelName),
                "Missing kernel '\(format.gemvKernelName)' for \(type(of: format))")
        }
    }

    @Test("All primitive fragment kernels can be generated")
    func primitiveFragmentKernelsGenerate() throws {
        let fragments: [any PrimitiveMetalKernelFragment] = [
            Reduction(dimension: 128, epsilon: 1e-6),
            ElementwiseFragment(count: 128),
            GatherFragment(vocabularySize: 1000, embeddingDimension: 128),
            ArgmaxFragment(vocabularySize: 1000),
            FlashAttentionFragment(headCount: 4, kvHeadCount: 4, headDimension: 64),
            RoPEFragment(headCount: 4, kvHeadCount: 4, headDimension: 64, ropeDimension: 64, base: 10000),
            QKNormFragment(headCount: 4, headDimension: 64, epsilon: 1e-6, weightRole: "q_layernorm"),
            Conv1dFragment(dimension: 128, kernelSize: 3),
            SigmoidGateFragment(dimension: 128),
        ]
        for frag in fragments {
            let src = frag.kernelSource(
                name: "test_\(type(of: frag))",
                bufferPrecision: .float16, weightFormat: .bfloat16)
            #expect(!src.isEmpty, "Failed to generate kernel for \(type(of: frag))")
        }
    }

    @Test("Structural kernels exist in MSL")
    func structuralKernelsExist() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device")
            return
        }
        let options = MTLCompileOptions()
        options.fastMathEnabled = true
        options.languageVersion = .version4_0
        let library = try device.makeLibrary(
            source: MetalSourceGenerator.generateCompleteLibrary(weightFormat: .bfloat16), options: options)
        let available = Set(library.functionNames)

        #expect(available.contains("copy_buffer"))
        #expect(available.contains("residual_add"))
        #expect(available.contains("quantize_kv_q8"))
        #expect(available.contains("dequantize_kv_q8"))
    }
}
