import Metal
import Testing
@testable import LMArchitecture
@testable import LMIR
@testable import MetalCompiler
@testable import ModelDeclarations

// MARK: - Shared Helpers

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

private func collectTensorNames(in region: Region, into names: inout [String]) {
    for operation in region.operations {
        for binding in operation.parameterBindings {
            names.append(binding.tensorName)
        }
        switch operation.kind {
        case .residual(_, let body):
            collectTensorNames(in: body, into: &names)
        case .repeating(let count, let body):
            _ = count
            collectTensorNames(in: body, into: &names)
        case .parallel(_, let branches):
            for branch in branches {
                collectTensorNames(in: branch, into: &names)
            }
        case .conditional(_, let thenBody, let elseBody):
            collectTensorNames(in: thenBody, into: &names)
            collectTensorNames(in: elseBody, into: &names)
        default:
            break
        }
    }
}

private func makeSyntheticWeightStore(
    tensorNames: [String],
    payloadSize: Int,
    device: MTLDevice
) throws -> STAFWeightStore {
    guard let buffer = device.makeBuffer(length: payloadSize, options: .storageModeShared) else {
        throw MetalCompilerError.deviceSetupFailed("Cannot allocate synthetic weight buffer")
    }

    var entries: [String: STAFTensorEntry] = [:]
    for tensorName in Set(tensorNames) {
        entries[tensorName] = STAFTensorEntry(
            name: tensorName,
            payloadOffset: 0,
            payloadSize: payloadSize,
            schemeIdentifier: .passthrough,
            semanticRole: .unknown,
            shape: [payloadSize / MemoryLayout<UInt16>.stride],
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

private func appendGemma4TextSyntheticTensorNames(config: ModelConfig, to names: inout [String]) {
    names.append(contentsOf: [
        "model.language_model.embed_tokens.weight",
        "model.language_model.norm.weight",
        "model.embed_tokens.weight",
        "model.norm.weight",
        "lm_head.weight"
    ])
    for layerIndex in 0..<config.layerCount {
        let prefixes = [
            "model.language_model.layers.\(layerIndex)",
            "model.layers.\(layerIndex)"
        ]
        for prefix in prefixes {
            names.append(contentsOf: [
                "\(prefix).input_layernorm.weight",
                "\(prefix).post_attention_layernorm.weight",
                "\(prefix).pre_feedforward_layernorm.weight",
                "\(prefix).post_feedforward_layernorm.weight",
                "\(prefix).self_attn.q_proj.weight",
                "\(prefix).self_attn.k_proj.weight",
                "\(prefix).self_attn.v_proj.weight",
                "\(prefix).self_attn.o_proj.weight",
                "\(prefix).self_attn.q_norm.weight",
                "\(prefix).self_attn.k_norm.weight",
                "\(prefix).mlp.gate_proj.weight",
                "\(prefix).mlp.up_proj.weight",
                "\(prefix).mlp.down_proj.weight",
                "\(prefix).per_layer_embedding_table.weight",
                "\(prefix).per_layer_model_projection.weight",
                "\(prefix).per_layer_projection_norm.weight",
                "\(prefix).per_layer_input_gate.weight",
                "\(prefix).per_layer_projection.weight",
                "\(prefix).post_per_layer_input_norm.weight",
                "\(prefix).layer_scalar"
            ])
        }
    }
}

private func appendGemma4VisionSyntheticTensorNames(layerCount: Int, to names: inout [String]) {
    names.append(contentsOf: [
        "model.vision_tower.patch_embedder.input_proj.weight",
        "model.vision_tower.patch_embedder.position_embedding_table",
        "model.vision_tower.std_bias",
        "model.vision_tower.std_scale",
        "model.embed_vision.embedding_projection.weight"
    ])
    for layerIndex in 0..<layerCount {
        let prefix = "model.vision_tower.encoder.layers.\(layerIndex)"
        names.append(contentsOf: [
            "\(prefix).input_layernorm.weight",
            "\(prefix).post_attention_layernorm.weight",
            "\(prefix).pre_feedforward_layernorm.weight",
            "\(prefix).post_feedforward_layernorm.weight",
            "\(prefix).self_attn.q_proj.linear.weight",
            "\(prefix).self_attn.k_proj.linear.weight",
            "\(prefix).self_attn.v_proj.linear.weight",
            "\(prefix).self_attn.o_proj.linear.weight",
            "\(prefix).self_attn.q_norm.weight",
            "\(prefix).self_attn.k_norm.weight",
            "\(prefix).mlp.gate_proj.linear.weight",
            "\(prefix).mlp.up_proj.linear.weight",
            "\(prefix).mlp.down_proj.linear.weight"
        ])
    }
}

private func makeGemma4TextConfig(numKVSharedLayers: Int = 1) -> ModelConfig {
    ModelConfig(
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
        numKVSharedLayers: numKVSharedLayers,
        useDoubleWideMLP: false,
        attentionKEqualsV: true,
        fullAttentionRopeTheta: 1_000_000.0,
        fullAttentionPartialRotaryFactor: 0.25,
        fullAttentionRoPEScaling: RoPEScaling(kind: .custom("proportional"), factor: 1.0)
    )
}

// MARK: - Gemma4 Text Decoder Compiler Tests

@Suite("Gemma4 Text Compiler", .serialized)
struct Gemma4TextCompilerTests {
    @Test("Allocates per-layer input buffers", .timeLimit(.minutes(2)))
    func compileAllocatesPerLayerInputs() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            print("[Skip] No Metal device available")
            return
        }

        let result = try autoreleasepool {
            let config = makeGemma4TextConfig()
            let graph = try ModelGraph(Gemma4(config: config))
            let resolvedGraph = ParameterResolver().resolve(graph: graph, convention: .gemma4Family)
            var tensorNames: [String] = []
            collectTensorNames(in: resolvedGraph.rootRegion, into: &tensorNames)
            appendGemma4TextSyntheticTensorNames(config: config, to: &tensorNames)
            let store = try makeSyntheticWeightStore(
                tensorNames: tensorNames,
                payloadSize: config.vocabSize * config.hiddenSize * MemoryLayout<UInt16>.stride,
                device: device
            )
            let compiler = MetalInferenceCompiler()
            var compiled = try compiler.compile(
                graph: resolvedGraph,
                hiddenSize: config.hiddenSize,
                intermediateSize: config.intermediateSize,
                vocabSize: config.vocabSize,
                stafWeightStore: store,
                device: device
            )
            let prefillPlan = try compiler.compilePrefill(
                graph: resolvedGraph,
                hiddenSize: config.hiddenSize,
                intermediateSize: config.intermediateSize,
                vocabSize: config.vocabSize,
                inferencePolicy: InferencePolicy(maximumSequenceLength: 128),
                stafWeightStore: store,
                sharedKVCache: compiled.buffers.kvCache,
                sharedConvState: compiled.buffers.convState,
                sharedConvStateDimension: compiled.buffers.convStateDimension,
                sharedConvStateKernelSize: compiled.buffers.convStateKernelSize,
                sharedRecurrentState: compiled.buffers.recurrentState,
                sharedRecurrentStateBytesPerLayer: compiled.buffers.recurrentStateBytesPerLayer,
                device: device
            )
            compiled = compiled.withPrefillPlan(prefillPlan)
            return (
                hasPerLayerInputs: compiled.buffers.perLayerInputs != nil,
                perLayerInputDimension: compiled.buffers.perLayerInputDimension,
                perLayerInputLayerCount: compiled.buffers.perLayerInputLayerCount,
                prefillHasPerLayerInputs: prefillPlan.buffers.perLayerInputs != nil,
                prefillPerLayerInputDimension: prefillPlan.buffers.perLayerInputDimension,
                prefillPerLayerInputLayerCount: prefillPlan.buffers.perLayerInputLayerCount
            )
        }

        #expect(result.hasPerLayerInputs)
        #expect(result.perLayerInputDimension == 8)
        #expect(result.perLayerInputLayerCount == 2)
        #expect(result.prefillHasPerLayerInputs)
        #expect(result.prefillPerLayerInputDimension == 8)
        #expect(result.prefillPerLayerInputLayerCount == 2)
    }

    @Test("Full-attention omits dedicated v_proj binding when K=V", .timeLimit(.minutes(2)))
    func fullAttentionOmitsDedicatedVProjectionBinding() throws {
        let config = makeGemma4TextConfig(numKVSharedLayers: 0)

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

    @Test("Prefill captures residual only for residual-entry norms", .timeLimit(.minutes(2)))
    func prefillCapturesResidualOnlyForResidualEntryNorms() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            print("[Skip] No Metal device available")
            return
        }

        let config = makeGemma4TextConfig()
        let layerCount = config.layerCount
        let residualCounts = try autoreleasepool {
            let graph = try ModelGraph(Gemma4(config: config))
            let resolvedGraph = ParameterResolver().resolve(graph: graph, convention: .gemma4Family)
            var tensorNames: [String] = []
            collectTensorNames(in: resolvedGraph.rootRegion, into: &tensorNames)
            appendGemma4TextSyntheticTensorNames(config: config, to: &tensorNames)
            let store = try makeSyntheticWeightStore(
                tensorNames: tensorNames,
                payloadSize: config.vocabSize * config.hiddenSize * MemoryLayout<UInt16>.stride,
                device: device
            )
            let compiler = MetalInferenceCompiler()
            let compiled = try compiler.compile(
                graph: resolvedGraph,
                hiddenSize: config.hiddenSize,
                intermediateSize: config.intermediateSize,
                vocabSize: config.vocabSize,
                stafWeightStore: store,
                device: device
            )
            let prefillPlan = try compiler.compilePrefill(
                graph: resolvedGraph,
                hiddenSize: config.hiddenSize,
                intermediateSize: config.intermediateSize,
                vocabSize: config.vocabSize,
                inferencePolicy: InferencePolicy(maximumSequenceLength: 128),
                stafWeightStore: store,
                sharedKVCache: compiled.buffers.kvCache,
                sharedConvState: compiled.buffers.convState,
                sharedConvStateDimension: compiled.buffers.convStateDimension,
                sharedConvStateKernelSize: compiled.buffers.convStateKernelSize,
                sharedRecurrentState: compiled.buffers.recurrentState,
                sharedRecurrentStateBytesPerLayer: compiled.buffers.recurrentStateBytesPerLayer,
                device: device
            )

            var counts: [Int: Int] = [:]
            for layerIndex in 0..<layerCount {
                let layerSteps = prefillPlan.steps.enumerated().filter { _, step in
                    step.metadata.layerIndex == layerIndex
                }

                let standaloneCopySteps = layerSteps.filter { _, step in
                    step.pipeline.label == "copy_buffer_seq_f32"
                }
                let synthesizedSteps = layerSteps.filter { _, step in
                    step.pipeline.label?.hasPrefix("synthesized_") == true
                }

                counts[layerIndex] = standaloneCopySteps.count + synthesizedSteps.count
            }
            return counts
        }

        for layerIndex in 0..<layerCount {
            let residualHandlingCount = residualCounts[layerIndex, default: 0]
            #expect(
                residualHandlingCount >= 2,
                "Gemma4 layer \(layerIndex) should have at least 2 residual-handling dispatches (standalone or fused)"
            )
        }
    }

    @Test("Embedding fragment carries scale and kernel name includes _scaled", .timeLimit(.minutes(1)))
    func embeddingFragmentCarriesScale() throws {
        let config = makeGemma4TextConfig()
        let graph = try ModelGraph(Gemma4(config: config))
        let resolvedGraph = ParameterResolver().resolve(graph: graph, convention: .gemma4Family)

        // Find TokenEmbeddingAttributes in the IR
        let embeddingOperation = findFirstOperation(in: resolvedGraph.rootRegion) {
            $0 is TokenEmbeddingAttributes
        }
        let embeddingAttributes = try #require(
            embeddingOperation.flatMap { op -> TokenEmbeddingAttributes? in
                if case .primitive(let attrs) = op.kind { return attrs as? TokenEmbeddingAttributes }
                return nil
            },
            "Graph must contain a TokenEmbeddingAttributes operation"
        )

        // Verify IR attributes carry embeddingScale
        let expectedScale = Float(config.hiddenSize).squareRoot()
        let irScale = try #require(embeddingAttributes.embeddingScale, "embeddingScale must not be nil")
        #expect(abs(irScale - expectedScale) < 0.01, "IR embeddingScale \(irScale) should equal sqrt(\(config.hiddenSize))=\(expectedScale)")

        // Verify MetalCompilable produces GatherFragment with embeddingScale
        let kernelContext = KernelContext(bufferPrecision: .float32, weightFormat: .float16)
        let fragment = embeddingAttributes.fragment(context: kernelContext)
        let fragmentScale = try #require(fragment.embeddingScale, "GatherFragment.embeddingScale must not be nil")
        #expect(abs(fragmentScale - expectedScale) < 0.01)

        // Verify kernel name includes _scaled
        let kernelName = fragment.kernelName(context: kernelContext)
        print("[Embedding] kernelName=\(kernelName) embeddingScale=\(fragmentScale)")
        #expect(kernelName.contains("_scaled"), "Kernel name '\(kernelName)' should include '_scaled'")

        // Verify kernel source includes scale parameter at buffer(5) for sequence mode
        let kernelSource = fragment.kernelSource(
            name: kernelName,
            bufferPrecision: .float32,
            weightFormat: .float16
        )
        #expect(kernelSource.contains("buffer(5)"), "Kernel source should have scale at buffer(5)")
        #expect(kernelSource.contains("* scale"), "Kernel source should apply '* scale'")
    }
}

// MARK: - Gemma4 Vision Compiler Tests


@Suite("Gemma4 Vision Compiler", .serialized)
struct Gemma4VisionCompilerTests {
    @Test("Resolves weight bindings with vision tower paths", .timeLimit(.minutes(1)))
    func visionWeightBindings() throws {
        let model = Gemma4Vision(
            hiddenSize: 64,
            intermediateSize: 128,
            headCount: 4,
            layerCount: 2,
            patchSize: 14,
            inChannels: 3,
            poolingKernelSize: 4,
            positionEmbeddingSize: 8,
            gridWidth: 4,
            ropeTheta: 100.0,
            hiddenAct: "gelu_pytorch_tanh",
            textHiddenSize: 96,
            standardize: true
        )
        let graph = try ModelGraph(model)
        let resolved = ParameterResolver().resolve(graph: graph, convention: .gemma4VisionFamily)

        var allTensorNames: [String] = []
        collectTensorNames(in: resolved.rootRegion, into: &allTensorNames)

        // Patch embedding
        #expect(allTensorNames.contains("model.vision_tower.patch_embedder.input_proj.weight"))

        // Position embedding table
        #expect(allTensorNames.contains("model.vision_tower.patch_embedder.position_embedding_table"))

        // Layer 0 sandwich norms
        #expect(allTensorNames.contains("model.vision_tower.encoder.layers.0.input_layernorm.weight"))
        #expect(allTensorNames.contains("model.vision_tower.encoder.layers.0.post_attention_layernorm.weight"))
        #expect(allTensorNames.contains("model.vision_tower.encoder.layers.0.pre_feedforward_layernorm.weight"))
        #expect(allTensorNames.contains("model.vision_tower.encoder.layers.0.post_feedforward_layernorm.weight"))

        // Layer 0 attention with .linear.weight suffix
        #expect(allTensorNames.contains("model.vision_tower.encoder.layers.0.self_attn.q_proj.linear.weight"))
        #expect(allTensorNames.contains("model.vision_tower.encoder.layers.0.self_attn.k_proj.linear.weight"))
        #expect(allTensorNames.contains("model.vision_tower.encoder.layers.0.self_attn.v_proj.linear.weight"))
        #expect(allTensorNames.contains("model.vision_tower.encoder.layers.0.self_attn.o_proj.linear.weight"))
        #expect(allTensorNames.contains("model.vision_tower.encoder.layers.0.self_attn.q_norm.weight"))
        #expect(allTensorNames.contains("model.vision_tower.encoder.layers.0.self_attn.k_norm.weight"))

        // Layer 0 MLP with .linear.weight suffix
        #expect(allTensorNames.contains("model.vision_tower.encoder.layers.0.mlp.gate_proj.linear.weight"))
        #expect(allTensorNames.contains("model.vision_tower.encoder.layers.0.mlp.up_proj.linear.weight"))
        #expect(allTensorNames.contains("model.vision_tower.encoder.layers.0.mlp.down_proj.linear.weight"))

        // Standardize weights
        #expect(allTensorNames.contains("model.vision_tower.std_bias"))
        #expect(allTensorNames.contains("model.vision_tower.std_scale"))

        // Root-level norm (post-pooling) has withScale=false, so no weight binding.
        #expect(allTensorNames.contains("model.vision_tower.norm.weight") == false)

        // Vision-to-text projection
        #expect(allTensorNames.contains("model.embed_vision.embedding_projection.weight"))
    }

    @Test("compilePrefill succeeds for embedding-only graph", .timeLimit(.minutes(2)))
    func visionCompilePrefill() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            print("[Skip] No Metal device available")
            return
        }

        let hasSteps = try autoreleasepool {
            let model = Gemma4Vision(
                hiddenSize: 64,
                intermediateSize: 128,
                headCount: 4,
                layerCount: 2,
                patchSize: 14,
                inChannels: 3,
                poolingKernelSize: 4,
                positionEmbeddingSize: 8,
                gridWidth: 4,
                ropeTheta: 100.0,
                hiddenAct: "gelu_pytorch_tanh",
                textHiddenSize: 96
            )
            let graph = try ModelGraph(model)
            let resolved = ParameterResolver().resolve(graph: graph, convention: .gemma4VisionFamily)
            var tensorNames: [String] = []
            collectTensorNames(in: resolved.rootRegion, into: &tensorNames)
            appendGemma4VisionSyntheticTensorNames(layerCount: 2, to: &tensorNames)
            let store = try makeSyntheticWeightStore(
                tensorNames: tensorNames,
                payloadSize: 4 * 1024 * 1024,
                device: device
            )
            let compiler = MetalInferenceCompiler()
            let prefillPlan = try compiler.compilePrefill(
                graph: resolved,
                hiddenSize: 64,
                intermediateSize: 128,
                vocabSize: 0,
                inferencePolicy: InferencePolicy(maximumSequenceLength: 256),
                stafWeightStore: store,
                sharedKVCache: nil,
                sharedConvState: nil,
                sharedConvStateDimension: 0,
                sharedConvStateKernelSize: 0,
                sharedRecurrentState: nil,
                sharedRecurrentStateBytesPerLayer: 0,
                device: device
            )
            return !prefillPlan.steps.isEmpty
        }
        #expect(hasSteps)
    }
}
