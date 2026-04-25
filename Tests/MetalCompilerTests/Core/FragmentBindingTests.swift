import Testing
import Metal
@testable import MetalCompiler
import LMArchitecture
import ModelDeclarations
import LMIR

/// Verify fragment binding generation and buffer access declarations.
@Suite("Fragment Bindings")
struct FragmentBindingTests {

    // MARK: - Dispatch Entry Buffer Access Declarations

    @Test("All dispatch entries have non-empty buffer accesses")
    func allEntriesHaveBufferAccesses() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device"); return
        }

        let config = makeTestConfig()
        let compiled = try compileModel(config: config, device: device)

        for (index, step) in compiled.steps.enumerated() {
            let reads = step.bufferAccesses.reads
            let writes = step.bufferAccesses.writes

            #expect(!reads.isEmpty || !writes.isEmpty,
                    "Step \(index) (\(step.metadata.kernelName ?? "unknown")) has empty buffer accesses")
        }
    }

    @Test("Fragment steps have explicit write indices (not conservative)")
    func fragmentStepsHaveExplicitWriteIndices() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device"); return
        }

        let config = makeTestConfig()
        let compiled = try compileModel(config: config, device: device)

        var conservativeCount = 0
        var explicitCount = 0

        for step in compiled.steps {
            let reads = step.bufferAccesses.reads
            let writes = step.bufferAccesses.writes

            if reads == writes && reads.count > 1 {
                conservativeCount += 1
            } else {
                explicitCount += 1
            }
        }

        print("[Fragment binding test] explicit=\(explicitCount) conservative=\(conservativeCount) total=\(compiled.steps.count)")

        // Most steps should have explicit (non-conservative) access patterns
        let explicitRatio = Double(explicitCount) / Double(compiled.steps.count)
        #expect(explicitRatio > 0.5,
                "Majority of steps should have explicit write indices, got \(String(format: "%.0f%%", explicitRatio * 100))")
    }

    @Test("Projection steps write only to output buffer")
    func projectionStepsWriteOnlyOutput() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device"); return
        }

        let config = makeTestConfig()
        let compiled = try compileModel(config: config, device: device)

        let projectionSteps = compiled.steps.filter {
            $0.metadata.kernelName?.hasPrefix("gemv") ?? false
        }

        for step in projectionSteps {
            let writes = step.bufferAccesses.writes
            #expect(writes.count <= 1,
                    "Projection kernel '\(step.metadata.kernelName ?? "")' should write to at most 1 region, writes to \(writes.count)")
        }
    }

    @Test("Synthetic weights use dedicated buffers instead of runtime state")
    func syntheticWeightsUseDedicatedBuffers() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device"); return
        }

        let config = ModelConfig(
            hiddenSize: 64, layerCount: 2, intermediateSize: 128,
            vocabSize: 248_320, attentionHeads: 4, kvHeads: 2, headDim: 16,
            attentionBias: false, mlpBias: false, normEps: 1e-5,
            normKind: .rmsNorm, ropeTheta: 10_000, ropeDimension: 16,
            ropeScaling: nil, tiedEmbeddings: false,
            expertCount: nil, expertsPerToken: nil, qkNorm: false,
            fullAttentionInterval: nil, ssmNumHeads: nil, ssmKeyHeadDim: nil,
            ssmValueHeadDim: nil, convKernelSize: nil,
            partialRotaryFactor: nil, slidingWindow: nil
        )
        let compiled = try compileModel(config: config, device: device)

        let embeddingStep = try #require(compiled.steps.first(where: {
            $0.metadata.kernelName?.hasPrefix("embedding_lookup") == true
        }))
        let embeddingWeight = try #require(embeddingStep.bindings.buffers.first(where: { $0.index == 1 }))
        #expect(embeddingWeight.buffer !== compiled.buffers.hidden)
        #expect(embeddingWeight.buffer.length >= config.vocabSize * config.hiddenSize * MemoryLayout<UInt16>.stride)

        let outputHeadStep = try #require(compiled.steps.first(where: { step in
            guard step.metadata.kernelName?.hasPrefix("gemv") == true else { return false }
            guard let outputBinding = step.bindings.buffers.first(where: { $0.index == 2 }) else { return false }
            return outputBinding.buffer === compiled.buffers.logits
        }))
        let outputWeight = try #require(outputHeadStep.bindings.buffers.first(where: { $0.index == 1 }))
        #expect(outputWeight.buffer !== compiled.buffers.hidden)
        #expect(outputWeight.buffer.length >= config.vocabSize * config.hiddenSize * MemoryLayout<UInt16>.stride)
    }

    // MARK: - Barrier Policy Consistency

    @Test("Steps with barriers have genuine data dependencies")
    func barrierStepsHaveDataDependencies() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device"); return
        }

        let config = makeTestConfig()
        let compiled = try compileModel(config: config, device: device)

        // Track pending writes to verify barriers are justified
        var pendingWrites = Set<BufferRegion>()
        var unjustifiedBarriers = 0

        for step in compiled.steps {
            let needsBarrier = step.bufferAccesses.requiresBarrier(after: pendingWrites)

            if step.barrierPolicy.isBarrier && !needsBarrier && !pendingWrites.isEmpty {
                unjustifiedBarriers += 1
            }

            if step.barrierPolicy.isBarrier {
                pendingWrites = step.bufferAccesses.writes
            } else {
                pendingWrites.formUnion(step.bufferAccesses.writes)
            }
        }

        #expect(unjustifiedBarriers == 0,
                "Found \(unjustifiedBarriers) barriers without data dependencies — optimizer missed these")
    }

    @Test("Steps without barriers have no pending write conflicts")
    func noBarrierStepsAreSafe() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device"); return
        }

        let config = makeTestConfig()
        let compiled = try compileModel(config: config, device: device)

        var pendingWrites = Set<BufferRegion>()
        var unsafeElisions = 0

        for step in compiled.steps {
            let needsBarrier = step.bufferAccesses.requiresBarrier(after: pendingWrites)

            if step.barrierPolicy == .none && needsBarrier {
                unsafeElisions += 1
                print("[UNSAFE] Step '\(step.metadata.kernelName ?? "unknown")' elides barrier but has conflict with pending writes")
            }

            if step.barrierPolicy.isBarrier {
                pendingWrites = step.bufferAccesses.writes
            } else {
                pendingWrites.formUnion(step.bufferAccesses.writes)
            }
        }

        #expect(unsafeElisions == 0,
                "Found \(unsafeElisions) steps that elide barriers but have write conflicts — data corruption risk")
    }

    // MARK: - Step Metadata

    @Test("All steps have kernel names in metadata")
    func allStepsHaveKernelNames() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device"); return
        }

        let config = makeTestConfig()
        let compiled = try compileModel(config: config, device: device)

        for (index, step) in compiled.steps.enumerated() {
            #expect(step.metadata.kernelName != nil,
                    "Step \(index) should have a kernel name in metadata")
        }
    }

    @Test("Layer indices are monotonically assigned in repeating blocks")
    func layerIndicesMonotonic() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device"); return
        }

        let config = ModelConfig(
            hiddenSize: 128, layerCount: 4, intermediateSize: 512,
            vocabSize: 1000, attentionHeads: 4, kvHeads: 4, headDim: 32,
            attentionBias: false, mlpBias: false, normEps: 1e-5,
            normKind: .rmsNorm, ropeTheta: 10000, ropeDimension: 32,
            ropeScaling: nil, tiedEmbeddings: true,
            expertCount: nil, expertsPerToken: nil, qkNorm: false,
            fullAttentionInterval: nil, ssmNumHeads: nil, ssmKeyHeadDim: nil,
            ssmValueHeadDim: nil, convKernelSize: nil,
            partialRotaryFactor: nil, slidingWindow: nil
        )
        let compiled = try compileModel(config: config, device: device)

        var lastLayerIndex = -1
        for step in compiled.steps {
            if let layerIndex = step.metadata.layerIndex {
                #expect(layerIndex >= lastLayerIndex,
                        "Layer index should be monotonically non-decreasing, got \(layerIndex) after \(lastLayerIndex)")
                lastLayerIndex = layerIndex
            }
        }

        // Should see all 4 layers
        let uniqueLayers = Set(compiled.steps.compactMap(\.metadata.layerIndex))
        #expect(uniqueLayers.count == 4,
                "4-layer model should have steps for all 4 layers, got \(uniqueLayers.sorted())")
    }

    // MARK: - Fusion Statistics

    @Test("Fusion reduces entry count from unfused baseline")
    func fusionReducesEntryCount() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device"); return
        }

        let config = makeTestConfig()
        let compiled = try compileModel(config: config, device: device)

        print("[Fusion test] unfused=\(compiled.unfusedEntryCount) fused=\(compiled.fusedEntryCount) steps=\(compiled.steps.count)")

        #expect(compiled.fusedEntryCount <= compiled.unfusedEntryCount,
                "Fused entry count (\(compiled.fusedEntryCount)) should not exceed unfused (\(compiled.unfusedEntryCount))")

        #expect(compiled.fusedEntryCount < compiled.unfusedEntryCount,
                "Fusion should reduce at least one entry (unfused=\(compiled.unfusedEntryCount), fused=\(compiled.fusedEntryCount))")
    }

    // MARK: - Prefill Step Modes

    @Test("Prefill attention steps use perPosition mode for KV cache fill")
    func prefillAttentionUsesPerPosition() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device"); return
        }

        let config = makeTestConfig()
        let graph = try ModelGraph(Transformer(config: config))
        let resolved = ParameterResolver().resolve(graph: graph, convention: .llamaFamily)
        let store = try makeSyntheticWeightStore(config: config, device: device)
        let prefillPlan = try MetalInferenceCompiler().compilePrefill(
            graph: resolved, hiddenSize: config.hiddenSize,
            intermediateSize: config.intermediateSize,
            vocabSize: config.vocabSize,
            inferencePolicy: InferencePolicy(maximumSequenceLength: 64),
            stafWeightStore: store,
            device: device)

        let perPositionSteps = prefillPlan.steps.filter { $0.mode == .perPosition }
        let batchSteps = prefillPlan.steps.filter { $0.mode == .batch }
        let lastTokenSteps = prefillPlan.steps.filter { $0.mode == .lastToken }

        print("[Prefill modes] batch=\(batchSteps.count) perPosition=\(perPositionSteps.count) lastToken=\(lastTokenSteps.count)")

        #expect(!batchSteps.isEmpty, "Should have batch steps (GEMM, norm, etc.)")
        // Attention writes to KV cache per-position
        #expect(!perPositionSteps.isEmpty || !lastTokenSteps.isEmpty,
                "Should have per-position or last-token steps (attention/argmax)")
    }

    @Test("Prefill last-token steps exist for output head")
    func prefillLastTokenStepsForOutput() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Issue.record("No Metal device"); return
        }

        let config = makeTestConfig()
        let graph = try ModelGraph(Transformer(config: config))
        let resolved = ParameterResolver().resolve(graph: graph, convention: .llamaFamily)
        let store = try makeSyntheticWeightStore(config: config, device: device)
        let prefillPlan = try MetalInferenceCompiler().compilePrefill(
            graph: resolved, hiddenSize: config.hiddenSize,
            intermediateSize: config.intermediateSize,
            vocabSize: config.vocabSize,
            inferencePolicy: InferencePolicy(maximumSequenceLength: 64),
            stafWeightStore: store,
            device: device)

        let lastTokenSteps = prefillPlan.steps.filter { $0.mode == .lastToken }

        #expect(!lastTokenSteps.isEmpty,
                "Output head should use lastToken mode (only need logits for final position)")
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

    private func compileModel(config: ModelConfig, device: MTLDevice) throws -> MetalCompiledModel {
        let graph = try ModelGraph(Transformer(config: config))
        let resolved = ParameterResolver().resolve(graph: graph, convention: .llamaFamily)
        let store = try makeSyntheticWeightStore(config: config, device: device)
        return try MetalInferenceCompiler().compile(
            graph: resolved, hiddenSize: config.hiddenSize,
            intermediateSize: config.intermediateSize,
            vocabSize: config.vocabSize,
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
