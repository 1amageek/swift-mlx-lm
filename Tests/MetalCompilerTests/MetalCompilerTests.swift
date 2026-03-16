import Testing
import Metal
import MetalCompiler
import LMArchitecture

@Suite
struct MetalComponentProtocolTests {

    @Test
    func embeddingDeclaresGatherDispatch() {
        let a = TokenEmbeddingAttributes(vocabSize: 32000, embeddingSize: 2048)
        #expect(a.dispatchDeclarations.count == 1)
        if case .compute(let op) = a.dispatchDeclarations[0] {
            #expect(op.kernelName == "embedding_lookup")
            if case .gather(let count) = op.dispatchDimension {
                #expect(count == 2048)
            } else {
                Issue.record("Expected .gather dispatch dimension")
            }
        } else {
            Issue.record("Expected .compute dispatch declaration")
        }
    }

    @Test
    func rmsNormDeclaresReductionDispatch() {
        let a = RMSNormAttributes(dimension: 2048, epsilon: 1e-5)
        #expect(a.dispatchDeclarations.count == 1)
        if case .compute(let op) = a.dispatchDeclarations[0] {
            #expect(op.kernelName == "rms_norm")
            if case .reduction(let dimension) = op.dispatchDimension {
                #expect(dimension == 2048)
            } else {
                Issue.record("Expected .reduction dispatch dimension")
            }
        } else {
            Issue.record("Expected .compute dispatch declaration")
        }
    }

    @Test
    func attentionDeclaresProjectionsAndCompute() {
        let a = AttentionAttributes(
            hiddenSize: 2048, headCount: 32, kvHeadCount: 8,
            headDimension: 64, bias: false, causal: true,
            rope: RoPEAttributes(dimension: 64, base: 500000.0))
        // Q, K, V projections + RoPE compute + flash_attn compute + O projection = 6
        #expect(a.dispatchDeclarations.count == 6)
        // First three are projections
        if case .projection(let qProj) = a.dispatchDeclarations[0] {
            #expect(qProj.field == "q_proj")
            #expect(qProj.inputDimension == 2048)
            #expect(qProj.outputDimension == 32 * 64)
        } else {
            Issue.record("Expected q_proj projection")
        }
        // RoPE compute
        if case .compute(let ropeOp) = a.dispatchDeclarations[3] {
            #expect(ropeOp.kernelName == "rope")
        } else {
            Issue.record("Expected RoPE compute")
        }
        // Flash attention compute
        if case .compute(let flashOp) = a.dispatchDeclarations[4] {
            #expect(flashOp.kernelName == "flash_attn_decode")
        } else {
            Issue.record("Expected flash_attn_decode compute")
        }
        // O projection
        if case .projection(let oProj) = a.dispatchDeclarations[5] {
            #expect(oProj.field == "o_proj")
        } else {
            Issue.record("Expected o_proj projection")
        }
    }

    @Test
    func mlpDeclaresProjectionsAndSwiGLU() {
        let a = MLPAttributes(
            inputSize: 2048, outputSize: 2048, intermediateSize: 8192,
            activation: .silu, gating: .swiglu, bias: false)
        // gate_proj + up_proj + swiglu compute + down_proj = 4
        #expect(a.dispatchDeclarations.count == 4)
        if case .compute(let swiOp) = a.dispatchDeclarations[2] {
            #expect(swiOp.kernelName == "swiglu")
        } else {
            Issue.record("Expected swiglu compute")
        }
    }

    @Test
    func outputHeadDeclaresProjectionAndArgmax() {
        let a = OutputHeadAttributes(inputSize: 2048, vocabSize: 32000, tiedToEmbedding: true, bias: false)
        #expect(a.dispatchDeclarations.count == 2)
        if case .projection(let proj) = a.dispatchDeclarations[0] {
            #expect(proj.field == "weight")
            #expect(proj.inputDimension == 2048)
            #expect(proj.outputDimension == 32000)
        } else {
            Issue.record("Expected weight projection")
        }
        if case .compute(let argOp) = a.dispatchDeclarations[1] {
            #expect(argOp.kernelName == "argmax")
        } else {
            Issue.record("Expected argmax compute")
        }
    }

    @Test
    func structuralOpsReturnNilMetalComponent() {
        let r = OperationKind.residual(strategy: .add, body: Region())
        let op = Operation(key: OperationKey(rawValue: 0), kind: r, operands: [], results: [])
        #expect(op.metalComponent == nil)
    }

    @Test
    func derivedProjectionsExtractsFromDeclarations() {
        let a = AttentionAttributes(
            hiddenSize: 2048, headCount: 32, kvHeadCount: 8,
            headDimension: 64, bias: false, causal: true,
            rope: RoPEAttributes(dimension: 64, base: 500000.0))
        let projs = a.projections
        #expect(projs.count == 4)
        #expect(projs[0].field == "q_proj")
        #expect(projs[1].field == "k_proj")
        #expect(projs[2].field == "v_proj")
        #expect(projs[3].field == "o_proj")
    }
}

@Suite
struct DispatchPlanCompilationTests {

    @Test
    func tinyModelCompilesToMultiDispatch() throws {
        guard let device = MTLCreateSystemDefaultDevice() else { return }
        let graph = try TinyTestModel(hiddenSize: 64, vocabSize: 100).makeModelGraph()
        let compiler = MetalInferenceCompiler()
        let plan = try compiler.compile(graph: graph, hiddenSize: 64, vocabSize: 100, device: device)

        // Multiple dispatch steps (not 1)
        #expect(plan.steps.count > 1, "Expected multiple dispatches, got \(plan.steps.count)")
    }

    @Test
    func transformerCompilesToManyDispatches() throws {
        guard let device = MTLCreateSystemDefaultDevice() else { return }
        let graph = try TinyTransformer(hiddenSize: 64, layers: 2, vocabSize: 100).makeModelGraph()
        let compiler = MetalInferenceCompiler()
        let plan = try compiler.compile(graph: graph, hiddenSize: 64, vocabSize: 100, device: device)

        // 2 layers x (norm + 5 attn + residual + norm + 4 mlp + residual) + embed + final_norm + output + argmax
        #expect(plan.steps.count > 20, "Expected many dispatches for 2-layer transformer, got \(plan.steps.count)")
    }

    @Test
    func threadgroupSizesRespectPipelineLimits() throws {
        guard let device = MTLCreateSystemDefaultDevice() else { return }
        let graph = try TinyTestModel(hiddenSize: 64, vocabSize: 100).makeModelGraph()
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
        options.languageVersion = .version3_0
        let library = try device.makeLibrary(
            source: MetalKernelSource.allKernelSource, options: options)
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

    @Test("All MetalComputeOperation kernel names exist in MSL")
    func computeOperationKernelsExist() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device")
            return
        }
        let options = MTLCompileOptions()
        options.fastMathEnabled = true
        options.languageVersion = .version3_0
        let library = try device.makeLibrary(
            source: MetalKernelSource.allKernelSource, options: options)
        let available = Set(library.functionNames)

        let operations: [any MetalComputeOperation] = [
            RMSNormOperation(dimension: 128, epsilon: 1e-6),
            LayerNormOperation(dimension: 128, epsilon: 1e-6, affine: true),
            SwiGLUOperation(dimension: 128),
            FlashAttentionDecodeOperation(headCount: 4, kvHeadCount: 4, headDimension: 64),
            RoPEOperation(headCount: 4, kvHeadCount: 4, headDimension: 64, ropeDimension: 64, base: 10000),
            EmbeddingLookupOperation(vocabularySize: 1000, embeddingDimension: 128),
            ArgmaxOperation(vocabularySize: 1000),
            Conv1dOperation(dimension: 128, kernelSize: 3),
            SSMRecurrenceOperation(headCount: 4, keyHeadDimension: 64, valueHeadDimension: 64),
            SigmoidGateOperation(dimension: 128),
            ResidualAddCopyRMSNormOperation(dimension: 128, epsilon: 1e-6),
            CopyRMSNormOperation(dimension: 128, epsilon: 1e-6),
        ]
        for operation in operations {
            #expect(available.contains(operation.kernelName),
                "Missing kernel '\(operation.kernelName)' for \(type(of: operation))")
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
        options.languageVersion = .version3_0
        let library = try device.makeLibrary(
            source: MetalKernelSource.allKernelSource, options: options)
        let available = Set(library.functionNames)

        #expect(available.contains("copy_buffer"))
        #expect(available.contains("residual_add"))
        #expect(available.contains("quantize_kv_q8"))
        #expect(available.contains("dequantize_kv_q8"))
    }
}
