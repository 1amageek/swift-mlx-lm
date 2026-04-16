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
            device: device,
            tensorName: target.tensorName,
            shape: [target.outputDimension, target.inputDimension],
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
            device: device,
            tensorName: target.tensorName,
            shape: [target.outputDimension, target.inputDimension],
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
        let diagnostics = try MetalInferenceCompiler().dumpCompiledPrefillPlan(
            graph: graph,
            hiddenSize: config.hiddenSize,
            intermediateSize: config.intermediateSize,
            vocabSize: config.vocabSize,
            inferencePolicy: InferencePolicy(maximumSequenceLength: 16),
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
            device: device,
            tensorName: resolvedEmbeddingBinding.tensorName,
            shape: [config.vocabSize, config.hiddenSize],
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
            device: device,
            tensorName: resolvedEmbeddingBinding.tensorName,
            shape: [config.vocabSize, config.hiddenSize],
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
        let diagnostics = try MetalInferenceCompiler().dumpCompiledDecodePlan(
            graph: graph,
            hiddenSize: config.hiddenSize,
            intermediateSize: config.intermediateSize,
            vocabSize: config.vocabSize,
            device: device
        )

        #expect(diagnostics.contains("quantization: entries="))
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
        device: MTLDevice,
        tensorName: String,
        shape: [Int],
        schemeIdentifier: QuantizationSchemeIdentifier
    ) throws -> STAFWeightStore {
        let buffer = try #require(device.makeBuffer(length: 1, options: .storageModeShared))
        return STAFWeightStore(
            buffer: buffer,
            entries: [
                tensorName: STAFTensorEntry(
                    name: tensorName,
                    payloadOffset: 0,
                    payloadSize: 0,
                    schemeIdentifier: schemeIdentifier,
                    semanticRole: .other,
                    shape: shape,
                    blockSize: 64,
                    groupSize: 64,
                    bufferOffset: 0
                )
            ],
            metadata: .empty,
            specializedBufferAccesses: [:]
        )
    }
}

private enum QuantizationPlanningError: Error {
    case missingProjection
}
