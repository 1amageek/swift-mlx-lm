import Metal
import Testing
import LMArchitecture
import LMIR
import ModelDeclarations
@testable import MetalCompiler

@Suite("Quantization Planning", .serialized)
struct QuantizationPlanningTests {

    @Test("q4 prefill projection records custom quantized kernel family")
    func q4PrefillProjectionUsesQuantizedKernelFamily() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device")
            return
        }

        let config = makeConfig()
        let graph = try resolvedGraph(config: config)
        let target = try firstPrefillProjection(in: graph, device: device)
        let store = try makeWeightStore(
            for: graph,
            device: device,
            overriding: target.tensorName,
            withShape: [target.outputDimension, target.inputDimension],
            schemeIdentifier: .q4Group64ScaleF16
        )

        let plan = try MetalInferenceCompiler().compilePrefill(
            graph: graph,
            hiddenSize: config.hiddenSize,
            intermediateSize: config.intermediateSize,
            vocabSize: config.vocabSize,
            inferencePolicy: InferencePolicy(maximumSequenceLength: 16),
            stafWeightStore: store,
            device: device
        )

        let quantizedEntry = try #require(
            plan.quantizationPlan.entries.first(where: { $0.tensorName == target.tensorName })
        )
        #expect(quantizedEntry.path == .prefillProjection)
        #expect(quantizedEntry.schemeIdentifier == .q4Group64ScaleF16)
        // Q4 prefill uses dequant→AMX matmul2d pipeline (commit 33beb7d)
        #expect(quantizedEntry.kernelFamily == .mppGEMM)
    }

    @Test("dense prefill projection records disabled environment fallback when MPP is off")
    func densePrefillProjectionRecordsDisabledEnvironmentFallback() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device")
            return
        }

        let config = makeConfig()
        let graph = try resolvedGraph(config: config)
        let target = try firstPrefillProjection(in: graph, device: device)
        let store = try makeWeightStore(
            for: graph,
            device: device,
            overriding: target.tensorName,
            withShape: [target.outputDimension, target.inputDimension],
            schemeIdentifier: .fp16RowMajor
        )

        setenv("SWIFTLM_DISABLE_MPP", "1", 1)
        defer { unsetenv("SWIFTLM_DISABLE_MPP") }

        let plan = try MetalInferenceCompiler().compilePrefill(
            graph: graph,
            hiddenSize: config.hiddenSize,
            intermediateSize: config.intermediateSize,
            vocabSize: config.vocabSize,
            inferencePolicy: InferencePolicy(maximumSequenceLength: 16),
            stafWeightStore: store,
            device: device
        )

        let denseEntry = try #require(
            plan.quantizationPlan.entries.first(where: { $0.tensorName == target.tensorName })
        )
        #expect(denseEntry.schemeIdentifier == .fp16RowMajor)
        #expect(denseEntry.kernelFamily == .naiveGEMM)
        #expect(denseEntry.usedFallback)
        #expect(denseEntry.fallbackReason == .disabledByEnvironment)
    }

    @Test("prefill diagnostics include quantization summary")
    func prefillDiagnosticsIncludeQuantizationSummary() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device")
            return
        }

        let config = makeConfig()
        let graph = try resolvedGraph(config: config)
        let store = try makeWeightStore(for: graph, device: device)
        let diagnostics = try MetalInferenceCompiler().dumpCompiledPrefillPlan(
            graph: graph,
            hiddenSize: config.hiddenSize,
            intermediateSize: config.intermediateSize,
            vocabSize: config.vocabSize,
            inferencePolicy: InferencePolicy(maximumSequenceLength: 16),
            stafWeightStore: store,
            device: device
        )

        #expect(diagnostics.contains("quantization: entries="))
        #expect(diagnostics.contains("prefillAccel="))
    }

    @Test("q4 prefill embedding lookup records quantized embedding kernel family")
    func q4PrefillEmbeddingLookupUsesQuantizedKernelFamily() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device")
            return
        }

        let config = makeConfig()
        let graph = try resolvedGraph(config: config)
        let embeddingBinding = try firstEmbeddingBinding(in: graph, device: device, phase: .prefill)
        let resolvedEmbeddingBinding = try #require(embeddingBinding)
        let store = try makeWeightStore(
            for: graph,
            device: device,
            overriding: resolvedEmbeddingBinding.tensorName,
            withShape: [config.vocabSize, config.hiddenSize],
            schemeIdentifier: .q4Group64ScaleF16
        )

        let plan = try MetalInferenceCompiler().compilePrefill(
            graph: graph,
            hiddenSize: config.hiddenSize,
            intermediateSize: config.intermediateSize,
            vocabSize: config.vocabSize,
            inferencePolicy: InferencePolicy(maximumSequenceLength: 16),
            stafWeightStore: store,
            device: device
        )

        let embeddingEntry = try #require(
            plan.quantizationPlan.entries.first {
                $0.tensorName == resolvedEmbeddingBinding.tensorName && $0.path == .embeddingLookup
            }
        )
        #expect(embeddingEntry.schemeIdentifier == .q4Group64ScaleF16)
        #expect(embeddingEntry.kernelFamily == .q4G64EmbeddingLookup)
        #expect(!embeddingEntry.usedFallback)
    }

    @Test("q3 prefill projection records kernel family and requires sequential ingestion")
    func q3PrefillProjectionRequiresSequentialPromptIngestion() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device")
            return
        }

        let config = makeConfig()
        let graph = try resolvedGraph(config: config)
        let target = try firstPrefillProjection(in: graph, device: device)
        let store = try makeWeightStore(
            for: graph,
            device: device,
            overriding: target.tensorName,
            withShape: [target.outputDimension, target.inputDimension],
            schemeIdentifier: .q3Group64ScaleF16
        )

        let plan = try MetalInferenceCompiler().compilePrefill(
            graph: graph,
            hiddenSize: config.hiddenSize,
            intermediateSize: config.intermediateSize,
            vocabSize: config.vocabSize,
            inferencePolicy: InferencePolicy(maximumSequenceLength: 16),
            stafWeightStore: store,
            device: device
        )

        let quantizedEntry = try #require(
            plan.quantizationPlan.entries.first(where: { $0.tensorName == target.tensorName })
        )
        #expect(quantizedEntry.path == .prefillProjection)
        #expect(quantizedEntry.schemeIdentifier == .q3Group64ScaleF16)
        #expect(quantizedEntry.kernelFamily == .mppGEMM)
        #expect(!quantizedEntry.usedFallback)
        #expect(plan.requiresSequentialPromptIngestion)
        #expect(plan.sequencePrefillFallbackReason == .unsupportedQ3Quantization)
    }

    @Test("q8 decode embedding lookup records quantized embedding kernel family")
    func q8DecodeEmbeddingLookupUsesQuantizedKernelFamily() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device")
            return
        }

        let config = makeConfig()
        let graph = try resolvedGraph(config: config)
        let embeddingBinding = try firstEmbeddingBinding(in: graph, device: device, phase: .decode)
        let resolvedEmbeddingBinding = try #require(embeddingBinding)
        let store = try makeWeightStore(
            for: graph,
            device: device,
            overriding: resolvedEmbeddingBinding.tensorName,
            withShape: [config.vocabSize, config.hiddenSize],
            schemeIdentifier: .q8Group32ScaleF16
        )

        let compiled = try MetalInferenceCompiler().compile(
            graph: graph,
            hiddenSize: config.hiddenSize,
            intermediateSize: config.intermediateSize,
            vocabSize: config.vocabSize,
            stafWeightStore: store,
            device: device
        )

        let embeddingEntry = try #require(
            compiled.decodePlan.quantizationPlan.entries.first {
                $0.tensorName == resolvedEmbeddingBinding.tensorName && $0.path == .embeddingLookup
            }
        )
        #expect(embeddingEntry.schemeIdentifier == .q8Group32ScaleF16)
        #expect(embeddingEntry.kernelFamily == .q8G32EmbeddingLookup)
        #expect(!embeddingEntry.usedFallback)
    }

    @Test("decode diagnostics include quantization summary")
    func decodeDiagnosticsIncludeQuantizationSummary() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device")
            return
        }

        let config = makeConfig()
        let graph = try resolvedGraph(config: config)
        let store = try makeWeightStore(for: graph, device: device)
        let diagnostics = try MetalInferenceCompiler().dumpCompiledDecodePlan(
            graph: graph,
            hiddenSize: config.hiddenSize,
            intermediateSize: config.intermediateSize,
            vocabSize: config.vocabSize,
            stafWeightStore: store,
            device: device
        )

        #expect(diagnostics.contains("quantization: entries="))
    }

    @Test("non-aligned quantized kernels are classified by concrete family")
    func nonAlignedQuantizedKernelFamiliesAreClassified() {
        #expect(MetalQuantizationKernelFamily.classify(
            kernelName: "embedding_lookup_seq_q3_g64",
            usesMPP: false
        ) == .q3G64EmbeddingLookup)
        #expect(MetalQuantizationKernelFamily.classify(
            kernelName: "gemv_q3_g64",
            usesMPP: false
        ) == .q3G64GEMV)
        #expect(MetalQuantizationKernelFamily.classify(
            kernelName: "gemm_q3_g64_f32s",
            usesMPP: false
        ) == .q3G64GEMM)
        #expect(MetalQuantizationKernelFamily.classify(
            kernelName: "embedding_lookup_seq_q5_g32",
            usesMPP: false
        ) == .q5G32EmbeddingLookup)
        #expect(MetalQuantizationKernelFamily.classify(
            kernelName: "gemv_q5_g64",
            usesMPP: false
        ) == .q5G64GEMV)
        #expect(MetalQuantizationKernelFamily.classify(
            kernelName: "gemm_q6_g16_f32s",
            usesMPP: false
        ) == .q6G16GEMM)
    }

    private func makeConfig() -> ModelConfig {
        ModelConfig(
            hiddenSize: 128,
            layerCount: 1,
            intermediateSize: 512,
            vocabSize: 1024,
            attentionHeads: 4,
            kvHeads: 4,
            headDim: 32,
            attentionBias: false,
            mlpBias: false,
            normEps: 1e-5,
            normKind: .rmsNorm,
            ropeTheta: 10000,
            ropeDimension: 32,
            ropeScaling: nil,
            tiedEmbeddings: true,
            expertCount: nil,
            expertsPerToken: nil,
            qkNorm: false,
            fullAttentionInterval: nil,
            ssmNumHeads: nil,
            ssmKeyHeadDim: nil,
            ssmValueHeadDim: nil,
            convKernelSize: nil,
            partialRotaryFactor: nil,
            slidingWindow: nil
        )
    }

    private func resolvedGraph(config: ModelConfig) throws -> ModelGraph {
        let graph = try ModelGraph(Transformer(config: config))
        return ParameterResolver().resolve(graph: graph, convention: .llamaFamily)
    }

    private struct ProjectionTarget {
        let entry: DispatchEntry
        let tensorName: String
        let inputDimension: Int
        let outputDimension: Int
    }

    private func firstPrefillProjection(
        in graph: ModelGraph,
        device: MTLDevice
    ) throws -> ProjectionTarget {
        let context = CompileContext(
            graph: graph,
            hiddenSize: 128,
            intermediateSize: 512,
            vocabSize: 1024,
            inferencePolicy: .default,
            stafWeightStore: nil,
            device: device,
            weightFormat: .float16,
            decodeBufferPrecision: .float16,
            accessPolicyResolver: ProjectionWeightAccessPolicyResolver()
        )
        let entries = MetalEntryCollector().collect(
            using: context,
            kernelContext: context.prefillKernelContext
        ).fusedEntries

        for entry in entries {
            // Find standalone LinearFragment (e.g. o_proj, down_proj)
            if let projection = entry.fragment as? LinearFragment,
               let binding = entry.parameterBindings.first(where: { $0.role == projection.field }) {
                return ProjectionTarget(
                    entry: entry,
                    tensorName: binding.tensorName,
                    inputDimension: projection.inputDimension,
                    outputDimension: projection.outputDimension
                )
            }
        }

        throw QuantizationPlanningError.missingProjection
    }

    private func firstEmbeddingBinding(
        in graph: ModelGraph,
        device: MTLDevice,
        phase: STAFWeightExecutionPhase
    ) throws -> ParameterBinding? {
        let context = CompileContext(
            graph: graph,
            hiddenSize: 128,
            intermediateSize: 512,
            vocabSize: 1024,
            inferencePolicy: .default,
            stafWeightStore: nil,
            device: device,
            weightFormat: .float16,
            decodeBufferPrecision: .float16,
            accessPolicyResolver: ProjectionWeightAccessPolicyResolver()
        )
        let kernelContext: KernelContext
        switch phase {
        case .decode:
            kernelContext = context.decodeKernelContext
        case .prefill:
            kernelContext = context.prefillKernelContext
        }
        let entries = MetalEntryCollector().collect(
            using: context,
            kernelContext: kernelContext
        ).fusedEntries

        for entry in entries {
            guard entry.fragment is GatherFragment else {
                continue
            }
            return entry.parameterBindings.first(where: { $0.role == "embedding_table" })
        }
        return nil
    }

    private func makeWeightStore(
        for graph: ModelGraph,
        device: MTLDevice,
        overriding tensorName: String? = nil,
        withShape shape: [Int] = [1],
        schemeIdentifier: QuantizationSchemeIdentifier = .passthrough
    ) throws -> STAFWeightStore {
        let overridePayloadSize = max(1, shape.reduce(1, *) * MemoryLayout<UInt16>.size)
        let buffer = try #require(device.makeBuffer(length: overridePayloadSize, options: .storageModeShared))
        var entries: [String: STAFTensorEntry] = [:]
        for name in tensorNames(in: graph.rootRegion) {
            let isOverride = name == tensorName
            let entryShape = isOverride ? shape : [1]
            entries[name] = STAFTensorEntry(
                name: name,
                payloadOffset: 0,
                payloadSize: isOverride ? overridePayloadSize : buffer.length,
                schemeIdentifier: isOverride ? schemeIdentifier : .passthrough,
                semanticRole: .other,
                shape: entryShape,
                blockSize: 64,
                groupSize: 64,
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

private enum QuantizationPlanningError: Error {
    case missingProjection
}
