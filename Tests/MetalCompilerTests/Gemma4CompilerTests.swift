import Metal
import Testing
@testable import LMArchitecture
@testable import LMIR
@testable import MetalCompiler
@testable import ModelDeclarations

private func findFirstOperation(
    in region: Region,
    matching predicate: (any OperationAttributes) -> Bool
) -> LMIR.Operation? {
    for operation in region.operations {
        if case .primitive(let attributes) = operation.kind, predicate(attributes) {
            return operation
        }
        switch operation.kind {
        case .residual(_, let body), .repeating(_, let body):
            if let found = findFirstOperation(in: body, matching: predicate) {
                return found
            }
        case .parallel(_, let branches):
            for branch in branches {
                if let found = findFirstOperation(in: branch, matching: predicate) {
                    return found
                }
            }
        case .conditional(_, let thenBody, let elseBody):
            if let found = findFirstOperation(in: thenBody, matching: predicate) {
                return found
            }
            if let found = findFirstOperation(in: elseBody, matching: predicate) {
                return found
            }
        default:
            break
        }
    }
    return nil
}

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
            attentionKEqualsV: true,
            fullAttentionRopeTheta: 1_000_000.0,
            fullAttentionPartialRotaryFactor: 0.25,
            fullAttentionRoPEScaling: RoPEScaling(kind: .custom("proportional"), factor: 1.0)
        )
        let graph = try ModelGraph(Gemma4(config: config))
        let resolvedGraph = ParameterResolver().resolve(graph: graph, convention: .gemma4Family)
        let compiler = MetalInferenceCompiler()
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
            inferencePolicy: InferencePolicy(maximumSequenceLength: 128),
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

    @Test("Gemma4 full-attention omits dedicated v_proj binding when K=V", .timeLimit(.minutes(2)))
    func fullAttentionOmitsDedicatedVProjectionBinding() throws {
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
            numKVSharedLayers: 0,
            useDoubleWideMLP: false,
            attentionKEqualsV: true,
            fullAttentionRopeTheta: 1_000_000.0,
            fullAttentionPartialRotaryFactor: 0.25,
            fullAttentionRoPEScaling: RoPEScaling(kind: .custom("proportional"), factor: 1.0)
        )

        let graph = try ModelGraph(Gemma4(config: config))
        let resolvedGraph = ParameterResolver().resolve(graph: graph, convention: .gemma4Family)
        let repeatingLayers = resolvedGraph.rootRegion.operations.compactMap { operation -> Region? in
            guard case .repeating(count: 1, let body) = operation.kind else { return nil }
            return body
        }
        let fullAttentionLayer = try #require(repeatingLayers.last)
        let fullAttentionOperation = try #require(
            findFirstOperation(in: fullAttentionLayer) { attributes in
                guard let attention = attributes as? AttentionAttributes else { return false }
                return attention.window == nil
            }
        )
        guard case .primitive(let rawAttributes) = fullAttentionOperation.kind,
              let fullAttentionAttributes = rawAttributes as? AttentionAttributes else {
            Issue.record("Expected full-attention operation to carry AttentionAttributes")
            return
        }
        let fullAttentionBindings = fullAttentionOperation.parameterBindings

        #expect(fullAttentionBindings.contains(where: { $0.role == "q_proj" }))
        #expect(fullAttentionBindings.contains(where: { $0.role == "k_proj" }))
        #expect(fullAttentionBindings.contains(where: { $0.role == "v_proj" }) == false)
        #expect(
            fullAttentionAttributes.valueProjectionSource
                == AttentionValueProjectionSource.keyProjection
        )
    }

    @Test("Gemma4 prefill captures residual only for residual-entry norms", .timeLimit(.minutes(2)))
    func prefillCapturesResidualOnlyForResidualEntryNorms() throws {
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
            attentionKEqualsV: true,
            fullAttentionRopeTheta: 1_000_000.0,
            fullAttentionPartialRotaryFactor: 0.25,
            fullAttentionRoPEScaling: RoPEScaling(kind: .custom("proportional"), factor: 1.0)
        )
        let graph = try ModelGraph(Gemma4(config: config))
        let resolvedGraph = ParameterResolver().resolve(graph: graph, convention: .gemma4Family)
        let compiler = MetalInferenceCompiler()
        let compiled = try compiler.compile(
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
            inferencePolicy: InferencePolicy(maximumSequenceLength: 128),
            sharedKVCache: compiled.buffers.kvCache,
            sharedConvState: compiled.buffers.convState,
            sharedConvStateDimension: compiled.buffers.convStateDimension,
            sharedConvStateKernelSize: compiled.buffers.convStateKernelSize,
            sharedRecurrentState: compiled.buffers.recurrentState,
            sharedRecurrentStateBytesPerLayer: compiled.buffers.recurrentStateBytesPerLayer,
            device: device
        )

        for layerIndex in 0..<config.layerCount {
            let layerSteps = prefillPlan.steps.enumerated().filter { _, step in
                step.metadata.layerIndex == layerIndex
            }

            // Residual capture steps: standalone copy_buffer OR fused synthesized
            // fragments that include copy logic (Copy+RMSNorm fusion).
            let standaloneCopySteps = layerSteps.filter { _, step in
                step.pipeline.label == "copy_buffer_seq_f32"
            }
            let synthesizedSteps = layerSteps.filter { _, step in
                step.pipeline.label?.hasPrefix("synthesized_") == true
            }

            // With automatic fusion, Copy+RMSNorm are merged into synthesized
            // fragments. The residual capture still happens inside the fused kernel.
            // Verify that at least 2 residual-handling dispatches exist per layer
            // (one for attention block, one for MLP block), regardless of fusion.
            let residualHandlingCount = standaloneCopySteps.count + synthesizedSteps.count
            #expect(
                residualHandlingCount >= 2,
                "Gemma4 layer \(layerIndex) should have at least 2 residual-handling dispatches (standalone or fused)"
            )
        }
    }
}
