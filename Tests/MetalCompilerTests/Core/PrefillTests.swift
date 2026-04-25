import Testing
import Metal
@testable import MetalCompiler
import LMArchitecture
import ModelDeclarations

@Suite("Prefill Performance")
struct PrefillTests {

    @Test("Single token decode completes without hanging")
    func singleTokenDecode() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device")
            return
        }

        let config = makeTestConfig()
        let graph = try ModelGraph(Transformer(config: config))
        let resolved = ParameterResolver().resolve(graph: graph, convention: .llamaFamily)
        let plan = try compileDecodePlan(config, resolved, device)
        var model = try MetalInferenceModel(plan: plan, device: device)

        let start = CFAbsoluteTimeGetCurrent()
        let result = model.decodeSync(tokenID: 42)
        let elapsed = CFAbsoluteTimeGetCurrent() - start

        print("[Test] decodeSync: \(String(format: "%.3f", elapsed))s, result=\(result)")
        #expect(elapsed < 1.0, "Single decode should complete in <1s, took \(elapsed)s")
    }

    @Test("Prefill fallback to sequential decode")
    func prefillFallback() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device")
            return
        }

        let config = makeTestConfig()
        let graph = try ModelGraph(Transformer(config: config))
        let resolved = ParameterResolver().resolve(graph: graph, convention: .llamaFamily)
        let plan = try compileDecodePlan(config, resolved, device)
        var model = try MetalInferenceModel(plan: plan, device: device)

        let tokens: [Int32] = Array(1...10)
        model.prefill(tokens: tokens)
        #expect(model.position == 10)
    }

    @Test("Prefill plan step count is independent of token count")
    func prefillStepCountIndependent() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device")
            return
        }

        let config = makeTestConfig()
        let graph = try ModelGraph(Transformer(config: config))
        let resolved = ParameterResolver().resolve(graph: graph, convention: .llamaFamily)
        let prefillPlan = try compilePrefillPlan(config, resolved, device)

        // Step count should NOT scale with maximumSequenceLength.
        // It should be O(layers × ops_per_layer), not O(tokens × layers × ops_per_layer).
        let stepCount = prefillPlan.stepCount
        print("[Test] Prefill plan: \(stepCount) steps (sequence graph)")

        // For a 2-layer transformer, expect roughly:
        // embedding + 2 * (copy+norm + 3 GEMM + rope + attn + GEMM + add + copy+norm + 3 GEMM + swiglu + GEMM + add) + norm + GEMM + argmax
        // The key assertion: step count < 100 (not 180 × seqLen)
        #expect(stepCount < 200, "Step count \(stepCount) should be constant, not proportional to token count")
        #expect(stepCount > 10, "Step count \(stepCount) too low — something is wrong")
    }

    @Test("Prefill with sequence graph completes and advances position")
    func prefillSequenceGraph() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device")
            return
        }

        let config = makeTestConfig()
        let graph = try ModelGraph(Transformer(config: config))
        let resolved = ParameterResolver().resolve(graph: graph, convention: .llamaFamily)
        let plan = try compileDecodePlan(config, resolved, device)
        let prefillPlan = try compilePrefillPlan(config, resolved, device)

        var model = try MetalInferenceModel(plan: plan, device: device)
        model.prefillPlan = prefillPlan

        let tokens: [Int32] = (0..<100).map { Int32($0 % 1000) }
        let start = CFAbsoluteTimeGetCurrent()
        model.prefill(tokens: tokens)
        let elapsed = CFAbsoluteTimeGetCurrent() - start

        print("[Test] prefill 100 tokens: \(String(format: "%.3f", elapsed))s (\(String(format: "%.1f", 100.0 / elapsed)) tok/s)")
        #expect(model.position == 100, "Position should be 100 after prefilling 100 tokens")
        #expect(elapsed < 5.0, "Prefill should complete in <5s, took \(elapsed)s")
    }

    @Test("Prefill step bindings are consistent with execution mode")
    func prefillStepModes() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device")
            return
        }

        let config = makeTestConfig()
        let graph = try ModelGraph(Transformer(config: config))
        let resolved = ParameterResolver().resolve(graph: graph, convention: .llamaFamily)
        let prefillPlan = try compilePrefillPlan(config, resolved, device)

        let batchSteps = prefillPlan.steps.filter { $0.mode == .batch }
        let perPosSteps = prefillPlan.steps.filter { $0.mode == .perPosition }

        #expect(!batchSteps.isEmpty, "Should have batch steps (GEMM, norm, etc.)")

        // All batch steps with grid.y > 1 should have seqLen binding
        for step in batchSteps where step.gridSize.height > 1 {
            #expect(step.sequenceLengthPolicy.bindingIndex != nil,
                    "Batch step with grid.y > 1 should have seqLen binding")
        }

        // Per-position steps are optional. If present, they must have a position binding.
        for step in perPosSteps {
            #expect(step.positionBufferIndex != nil,
                    "PerPosition step should have position binding")
        }

        print("[Test] \(batchSteps.count) batch + \(perPosSteps.count) perPosition = \(prefillPlan.stepCount) total")
    }

    @Test("Prefill GEMM batch steps always receive runtime seqLen")
    func prefillGEMMBatchStepsUseRuntimeSequenceLength() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device")
            return
        }

        let config = makeTestConfig()
        let graph = try ModelGraph(Transformer(config: config))
        let resolved = ParameterResolver().resolve(graph: graph, convention: .llamaFamily)
        let prefillPlan = try compilePrefillPlan(config, resolved, device)

        let gemmBatchSteps = prefillPlan.steps.filter {
            $0.mode == .batch && (($0.pipeline.label ?? "").hasPrefix("gemm"))
        }

        #expect(!gemmBatchSteps.isEmpty, "Expected at least one prefill GEMM batch step")

        for step in gemmBatchSteps {
            #expect(
                step.sequenceLengthPolicy.bindingIndex != nil,
                "Prefill GEMM batch step must override buffer(seqLen) at runtime"
            )
        }
    }

    // MARK: - Helpers

    private func makeTestConfig() -> ModelConfig {
        ModelConfig(
            hiddenSize: 128, layerCount: 2, intermediateSize: 512,
            vocabSize: 1000, attentionHeads: 4, kvHeads: 4, headDim: 32,
            attentionBias: false, mlpBias: false, normEps: 1e-5,
            normKind: .rmsNorm, ropeTheta: 10000, ropeDimension: 32,
            ropeScaling: nil, tiedEmbeddings: true,
            expertCount: nil, expertsPerToken: nil, qkNorm: false,
            fullAttentionInterval: nil, ssmNumHeads: nil, ssmKeyHeadDim: nil,
            ssmValueHeadDim: nil, convKernelSize: nil,
            partialRotaryFactor: nil, slidingWindow: nil
        )
    }

    private func compileDecodePlan(
        _ config: ModelConfig,
        _ graph: ModelGraph,
        _ device: MTLDevice
    ) throws -> MetalCompiledModel {
        let store = try makeSyntheticWeightStore(config: config, device: device)
        return try MetalInferenceCompiler().compile(
            graph: graph, hiddenSize: config.hiddenSize,
            intermediateSize: config.intermediateSize,
            vocabSize: config.vocabSize,
            stafWeightStore: store,
            device: device)
    }

    private func compilePrefillPlan(
        _ config: ModelConfig,
        _ graph: ModelGraph,
        _ device: MTLDevice
    ) throws -> MetalPrefillPlan {
        let store = try makeSyntheticWeightStore(config: config, device: device)
        return try MetalInferenceCompiler().compilePrefill(
            graph: graph, hiddenSize: config.hiddenSize,
            intermediateSize: config.intermediateSize,
            vocabSize: config.vocabSize,
            inferencePolicy: InferencePolicy(maximumSequenceLength: 1024),
            stafWeightStore: store,
            device: device)
    }

    private func makeSyntheticWeightStore(config: ModelConfig, device: MTLDevice) throws -> STAFWeightStore {
        let maxElements = max(
            config.vocabSize * config.hiddenSize,
            config.hiddenSize * config.intermediateSize,
            config.hiddenSize * config.hiddenSize
        )
        let payloadSize = max(1, maxElements * MemoryLayout<UInt16>.stride)
        guard let buffer = device.makeBuffer(length: payloadSize, options: .storageModeShared) else {
            throw MetalCompilerError.deviceSetupFailed("Cannot allocate synthetic weight buffer")
        }

        var tensorNames: Set<String> = [
            "model.embed_tokens.weight",
            "model.norm.weight",
            "lm_head.weight"
        ]
        for layerIndex in 0..<config.layerCount {
            let prefix = "model.layers.\(layerIndex)"
            tensorNames.formUnion([
                "\(prefix).input_layernorm.weight",
                "\(prefix).self_attn.q_proj.weight",
                "\(prefix).self_attn.k_proj.weight",
                "\(prefix).self_attn.v_proj.weight",
                "\(prefix).self_attn.o_proj.weight",
                "\(prefix).post_attention_layernorm.weight",
                "\(prefix).mlp.gate_proj.weight",
                "\(prefix).mlp.up_proj.weight",
                "\(prefix).mlp.down_proj.weight"
            ])
        }

        var entries: [String: STAFTensorEntry] = [:]
        for tensorName in tensorNames {
            entries[tensorName] = STAFTensorEntry(
                name: tensorName,
                payloadOffset: 0,
                payloadSize: payloadSize,
                schemeIdentifier: .passthrough,
                semanticRole: .unknown,
                shape: [maxElements],
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
}
