import Testing
import Metal
@testable import MetalCompiler
import LMIR
import ModelDeclarations

@Suite("Generated Kernel Library")
struct GeneratedLibraryTests {

    @Test("Complete generated library compiles and contains all needed functions")
    func generatedLibraryCompiles() throws {
        guard let device = MTLCreateSystemDefaultDevice() else { return }

        let source = MetalSourceGenerator.generateCompleteLibrary(weightFormat: .bfloat16)
        let options = MTLCompileOptions()
        options.languageVersion = .version4_0

        let library = try device.makeLibrary(source: source, options: options)

        // Verify critical decode kernels
        for name in ["gemv_bf16", "rms_norm_bf16", "rms_norm_bf16_argbuf", "swiglu", "argmax", "argmax_argbuf",
                      "qk_rms_norm_bf16", "qk_rms_norm_bf16_argbuf", "rope", "conv_state_update_bf16",
                      "flash_attn_decode", "gemv_q4_g64", "sigmoid_gate", "ssm_recurrence_bf16"] {
            #expect(library.makeFunction(name: name) != nil, "Missing: \(name)")
        }

        // Verify critical prefill kernels
        for name in ["gemm_bf16_f32s", "rms_norm_seq_bf16_f32_inplace",
                      "swiglu_seq_f32", "embedding_lookup_seq_bf16_f32",
                      "qk_rms_norm_seq_f32", "rope_seq_f32",
                      "conv1d_causal_seq_f32", "argmax_f32", "ssm_recurrence_bf16_f32",
                      "ssm_recurrence_seq_bf16_f32",
                      "flash_attn_decode_f32"] {
            #expect(library.makeFunction(name: name) != nil, "Missing: \(name)")
        }

        print("[GenLib] Library: \(library.functionNames.count) functions, \(source.count) chars")
    }

    @Test("Kernel source catalog emits q4 prefill projection kernels for quantized weights")
    func sourceCatalogEmitsQ4PrefillProjectionKernel() throws {
        guard let device = MTLCreateSystemDefaultDevice() else { return }
        let buffer = try #require(device.makeBuffer(length: 1, options: .storageModeShared))
        let store = STAFWeightStore(
            buffer: buffer,
            entries: [
                "dense.0.weight": STAFTensorEntry(
                    name: "dense.0.weight",
                    payloadOffset: 0,
                    payloadSize: 0,
                    schemeIdentifier: .q4Group64ScaleF16,
                    semanticRole: .other,
                    shape: [3072, 96],
                    blockSize: 64,
                    groupSize: 64,
                    bufferOffset: 0
                )
            ],
            metadata: .empty,
            specializedBufferAccesses: [:]
        )
        let entry = DispatchEntry(
            index: 0,
            fragment: LinearFragment(
                field: "weight", inputDimension: 768, outputDimension: 3072
            ),
            parameterBindings: [
                .init(role: "weight", tensorName: "dense.0.weight")
            ]
        )
        let resolver = MetalKernelNameResolver(
            stafWeightStore: store,
            weightAccessPolicyOverride: nil
        )
        let kernelName = resolver.kernelName(
            for: entry,
            kernelContext: .init(
                bufferPrecision: .float32,
                weightFormat: WeightFormats.quantized4Bit(groupSize: 64)
            )
        )
        #expect(kernelName == "gemm_q4_g64_f32s")

        let catalog = MetalKernelSourceCatalog(
            stafWeightStore: store,
            modelWeightFormat: WeightFormats.quantized4Bit(groupSize: 64),
            bufferPrecision: .float32,
            accessPolicyResolver: ProjectionWeightAccessPolicyResolver(),
            kernelNameResolver: resolver
        )
        let generated = catalog.generateSources(entries: [entry])
        #expect(generated.baseSource.contains("kernel void gemm_q4_g64_f32s"))
    }

    #if ENABLE_METAL_PROBES
    @Test("Dump generated decode kernel library for LFM2")
    func dumpGeneratedDecodeKernelLibraryForLFM2() throws {
        guard let resources = try RealModelTestSupport.loadOrSkip(skipMessage: "STAF not found — skipping") else {
            return
        }
        defer { resources.release() }

        let store = resources.store
        let config = ModelConfig(
            hiddenSize: 2048, layerCount: 16, intermediateSize: 8192,
            vocabSize: 65536, attentionHeads: 32, kvHeads: 8, headDim: 64,
            attentionBias: false, mlpBias: false, normEps: 1e-5,
            normKind: .rmsNorm, ropeTheta: 1000000.0, ropeDimension: 64,
            ropeScaling: nil, tiedEmbeddings: true,
            expertCount: nil, expertsPerToken: nil, qkNorm: true,
            fullAttentionInterval: nil, ssmNumHeads: nil, ssmKeyHeadDim: nil,
            ssmValueHeadDim: nil, convKernelSize: nil, convLCache: 3,
            partialRotaryFactor: nil, slidingWindow: nil,
            layerTypes: ["conv", "conv", "full_attention", "conv", "conv", "full_attention",
                         "conv", "conv", "full_attention", "conv", "full_attention", "conv",
                         "full_attention", "conv", "full_attention", "conv"])
        let graph = try ModelGraph(LFM2(config: config))
        let resolved = ParameterResolver().resolve(graph: graph, convention: .lfm2Family)
        let compiler = MetalInferenceCompiler()
        let dump = compiler.dumpGeneratedDecodeKernelLibrary(
            graph: resolved,
            hiddenSize: 2048,
            stafWeightStore: store)

        print("=== GENERATED DECODE LIBRARY ===")
        print(String(dump.prefix(30000)))
    }
    #endif

    @Test("Compiled decode plan marks argument-table and resident-constant steps")
    func compiledDecodePlanRecordsBindingBackends() throws {
        guard let resources = try RealModelTestSupport.loadOrSkip(skipMessage: "STAF not found — skipping") else {
            return
        }
        defer { resources.release() }

        let device = resources.device
        let store = resources.store
        let config = ModelConfig(
            hiddenSize: 2048, layerCount: 16, intermediateSize: 8192,
            vocabSize: 65536, attentionHeads: 32, kvHeads: 8, headDim: 64,
            attentionBias: false, mlpBias: false, normEps: 1e-5,
            normKind: .rmsNorm, ropeTheta: 1000000.0, ropeDimension: 64,
            ropeScaling: nil, tiedEmbeddings: true,
            expertCount: nil, expertsPerToken: nil, qkNorm: true,
            fullAttentionInterval: nil, ssmNumHeads: nil, ssmKeyHeadDim: nil,
            ssmValueHeadDim: nil, convKernelSize: nil, convLCache: 3,
            partialRotaryFactor: nil, slidingWindow: nil,
            layerTypes: ["conv", "conv", "full_attention", "conv", "conv", "full_attention",
                         "conv", "conv", "full_attention", "conv", "full_attention", "conv",
                         "full_attention", "conv", "full_attention", "conv"])
        let graph = try ModelGraph(LFM2(config: config))
        let resolved = ParameterResolver().resolve(graph: graph, convention: .lfm2Family)
        let compiler = MetalInferenceCompiler()
        let plan = try compiler.compile(
            graph: resolved,
            hiddenSize: 2048,
            intermediateSize: 8192,
            vocabSize: 65536,
            stafWeightStore: store,
            device: device)

        let argumentTableSteps = plan.steps.filter { $0.bindings.argumentPolicy == .argumentTable }.count
        let preparedArgumentSteps = plan.steps.filter { step in
            guard case .argumentTable(let table) = step.bindings.bufferBindings else { return false }
            if case .prepared = table.encodingState { return true }
            return false
        }.count
        let encodedArgumentSteps = plan.steps.filter { step in
            guard case .argumentTable(let table) = step.bindings.bufferBindings else { return false }
            if case .encoded = table.encodingState { return true }
            return false
        }.count
        let plannedArgumentSteps = plan.steps.filter { step in
            guard case .argumentTable(let table) = step.bindings.bufferBindings else { return false }
            if case .planned = table.encodingState { return true }
            return false
        }.count
        let preparedKernels = plan.steps.compactMap { step -> String? in
            guard case .argumentTable(let table) = step.bindings.bufferBindings else { return nil }
            if case .prepared = table.encodingState {
                return step.pipeline.label ?? "(unlabeled)"
            }
            return nil
        }
        let residentConstantSteps = plan.steps.filter { $0.bindings.constantPolicy == .residentConstantBuffer }.count
        let argumentTableLayouts = Set(plan.steps.compactMap { step -> Int? in
            guard case .argumentTable(let table) = step.bindings.bufferBindings else { return nil }
            return table.layout.id
        })

        print("[BindingPlan] decode steps=\(plan.steps.count) argTable=\(argumentTableSteps) argPlanned=\(plannedArgumentSteps) argPrepared=\(preparedArgumentSteps) argEncoded=\(encodedArgumentSteps) residentConst=\(residentConstantSteps) layouts=\(argumentTableLayouts.count)")
        if !preparedKernels.isEmpty {
            print("[BindingPlan.PreparedKernels] \(preparedKernels.joined(separator: " | "))")
        }

        #expect(argumentTableSteps == plan.steps.count)
        #expect(preparedArgumentSteps == 1)
        #expect(preparedKernels == ["argmax"])
        #expect(encodedArgumentSteps == 0)
        #expect(plannedArgumentSteps + preparedArgumentSteps + encodedArgumentSteps == argumentTableSteps)
        #expect(residentConstantSteps > 0)
        #expect(argumentTableLayouts.count < argumentTableSteps)
    }

    @Test("Compiled decode plan reports dominant argument-table layouts")
    func compiledDecodePlanReportsDominantArgumentTableLayouts() throws {
        guard let resources = try RealModelTestSupport.loadOrSkip(skipMessage: "STAF not found — skipping") else {
            return
        }
        defer { resources.release() }

        let device = resources.device
        let store = resources.store
        let config = ModelConfig(
            hiddenSize: 2048, layerCount: 16, intermediateSize: 8192,
            vocabSize: 65536, attentionHeads: 32, kvHeads: 8, headDim: 64,
            attentionBias: false, mlpBias: false, normEps: 1e-5,
            normKind: .rmsNorm, ropeTheta: 1000000.0, ropeDimension: 64,
            ropeScaling: nil, tiedEmbeddings: true,
            expertCount: nil, expertsPerToken: nil, qkNorm: true,
            fullAttentionInterval: nil, ssmNumHeads: nil, ssmKeyHeadDim: nil,
            ssmValueHeadDim: nil, convKernelSize: nil, convLCache: 3,
            partialRotaryFactor: nil, slidingWindow: nil,
            layerTypes: ["conv", "conv", "full_attention", "conv", "conv", "full_attention",
                         "conv", "conv", "full_attention", "conv", "full_attention", "conv",
                         "full_attention", "conv", "full_attention", "conv"])
        let graph = try ModelGraph(LFM2(config: config))
        let resolved = ParameterResolver().resolve(graph: graph, convention: .lfm2Family)
        let compiler = MetalInferenceCompiler()
        let plan = try compiler.compile(
            graph: resolved,
            hiddenSize: 2048,
            intermediateSize: 8192,
            vocabSize: 65536,
            stafWeightStore: store,
            device: device)

        let layoutUsage = MetalArgumentBindingAllocator().summarizeUsage(
            in: plan.steps.map(\.bindings))
        let topLayout = try #require(layoutUsage.first)
        let topLayouts = layoutUsage.prefix(5).map { usage in
            "#\(usage.layout.id)x\(usage.useCount) indices=\(usage.layout.indices)"
        }.joined(separator: " | ")

        print("[BindingPlan.TopLayouts] \(topLayouts)")

        #expect(layoutUsage.count > 0)
        #expect(topLayout.useCount > 1)
        #expect(topLayout.layout.indices.count >= 3)
    }

    @Test("Compiled decode plan reports dominant layout kernel families")
    func compiledDecodePlanReportsDominantLayoutKernelFamilies() throws {
        guard let resources = try RealModelTestSupport.loadOrSkip(skipMessage: "STAF not found — skipping") else {
            return
        }
        defer { resources.release() }

        let device = resources.device
        let store = resources.store
        let config = ModelConfig(
            hiddenSize: 2048, layerCount: 16, intermediateSize: 8192,
            vocabSize: 65536, attentionHeads: 32, kvHeads: 8, headDim: 64,
            attentionBias: false, mlpBias: false, normEps: 1e-5,
            normKind: .rmsNorm, ropeTheta: 1000000.0, ropeDimension: 64,
            ropeScaling: nil, tiedEmbeddings: true,
            expertCount: nil, expertsPerToken: nil, qkNorm: true,
            fullAttentionInterval: nil, ssmNumHeads: nil, ssmKeyHeadDim: nil,
            ssmValueHeadDim: nil, convKernelSize: nil, convLCache: 3,
            partialRotaryFactor: nil, slidingWindow: nil,
            layerTypes: ["conv", "conv", "full_attention", "conv", "conv", "full_attention",
                         "conv", "conv", "full_attention", "conv", "full_attention", "conv",
                         "full_attention", "conv", "full_attention", "conv"])
        let graph = try ModelGraph(LFM2(config: config))
        let resolved = ParameterResolver().resolve(graph: graph, convention: .lfm2Family)
        let compiler = MetalInferenceCompiler()
        let plan = try compiler.compile(
            graph: resolved,
            hiddenSize: 2048,
            intermediateSize: 8192,
            vocabSize: 65536,
            stafWeightStore: store,
            device: device)

        let layoutUsage = MetalArgumentBindingAllocator().summarizeUsage(
            in: plan.steps.map(\.bindings))
        let topLayout = try #require(layoutUsage.first)

        var kernelCounts: [String: Int] = [:]
        for step in plan.steps {
            guard case .argumentTable(let table) = step.bindings.bufferBindings,
                  table.layout.id == topLayout.layout.id else {
                continue
            }
            let kernel = step.pipeline.label ?? "(unlabeled)"
            kernelCounts[kernel, default: 0] += 1
        }

        let topKernelFamilies = kernelCounts
            .sorted { lhs, rhs in
                if lhs.value != rhs.value {
                    return lhs.value > rhs.value
                }
                return lhs.key < rhs.key
            }
        let summary = topKernelFamilies.prefix(5)
            .map { "\($0.key)x\($0.value)" }
            .joined(separator: " | ")

        print("[BindingPlan.LayoutKernels] layout#\(topLayout.layout.id) indices=\(topLayout.layout.indices) \(summary)")

        let dominantKernel = try #require(topKernelFamilies.first)
        #expect(!kernelCounts.isEmpty)
        #expect(dominantKernel.value > 1)
    }

    @Test("Decode projection cost report highlights low-intensity hot families")
    func decodeProjectionCostReportHighlightsHotFamilies() throws {
        guard let resources = try RealModelTestSupport.loadOrSkip(skipMessage: "STAF not found — skipping") else {
            return
        }
        defer { resources.release() }

        let device = resources.device
        let store = resources.store
        let config = ModelConfig(
            hiddenSize: 2048, layerCount: 16, intermediateSize: 8192,
            vocabSize: 65536, attentionHeads: 32, kvHeads: 8, headDim: 64,
            attentionBias: false, mlpBias: false, normEps: 1e-5,
            normKind: .rmsNorm, ropeTheta: 1000000.0, ropeDimension: 64,
            ropeScaling: nil, tiedEmbeddings: true,
            expertCount: nil, expertsPerToken: nil, qkNorm: true,
            fullAttentionInterval: nil, ssmNumHeads: nil, ssmKeyHeadDim: nil,
            ssmValueHeadDim: nil, convKernelSize: nil, convLCache: 3,
            partialRotaryFactor: nil, slidingWindow: nil,
            layerTypes: ["conv", "conv", "full_attention", "conv", "conv", "full_attention",
                         "conv", "conv", "full_attention", "conv", "full_attention", "conv",
                         "full_attention", "conv", "full_attention", "conv"])
        let graph = try ModelGraph(LFM2(config: config))
        let resolved = ParameterResolver().resolve(graph: graph, convention: .lfm2Family)
        let compiler = MetalInferenceCompiler()
        let report = try compiler.analyzeDecodeProjectionCosts(
            graph: resolved,
            hiddenSize: 2048,
            intermediateSize: 8192,
            vocabSize: 65536,
            stafWeightStore: store,
            device: device
        )

        print(report.formatted(limit: 8))

        let hot6144 = try #require(report.families.first(where: { $0.kernelName.hasPrefix("gemv_2048_6144") }))
        let hotSquare = try #require(report.families.first(where: { $0.kernelName.hasPrefix("gemv_2048_sq") }))

        #expect(report.totalProjectionSteps > 0)
        #expect(hot6144.layouts == [.rowMajor])
        #expect(hotSquare.layouts == [.rowMajor])
        #expect(hot6144.arithmeticIntensity < 1.05)
        #expect(hotSquare.arithmeticIntensity < 1.05)
        #expect(hot6144.weightBytesPerStep > hot6144.inputBytesPerStep)
        #expect(hotSquare.weightBytesPerStep > hotSquare.inputBytesPerStep)
    }
}
