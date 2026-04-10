import Testing
@testable import LMArchitecture
@testable import LMIR
@testable import ModelDeclarations

@Suite("Model Declaration Tests", .tags(.unit))
struct ModelDeclarationTests {

    // MARK: - Transformer

    @Test("Transformer produces valid ModelGraph")
    func transformerGraph() throws {
        let model = Transformer(config: TestConfigs.transformer7B)

        let graph = try model.makeModelGraph()
        #expect(graph.rootRegion.operations.isEmpty == false)
        #expect(graph.rootRegion.results.count == 1)
    }

    @Test("Transformer with MoE produces valid ModelGraph")
    func transformerMoEGraph() throws {
        let model = Transformer(config: TestConfigs.transformerMoE)

        let graph = try model.makeModelGraph()
        #expect(graph.rootRegion.operations.isEmpty == false)
        #expect(graph.rootRegion.results.count == 1)
    }

    @Test("Transformer declaration is deterministic")
    func transformerDeterministic() throws {
        let config = TestConfigs.transformerSmall

        let graph1 = try Transformer(config: config).makeModelGraph()
        let graph2 = try Transformer(config: config).makeModelGraph()

        let canonical1 = canonicalize(graph1)
        let canonical2 = canonicalize(graph2)
        #expect(canonical1.dump() == canonical2.dump())
    }

    // MARK: - Qwen 3.5

    @Test("Qwen35 produces valid ModelGraph")
    func qwen35Graph() throws {
        let model = Qwen35(config: TestConfigs.qwen35_0_8B)

        let graph = try model.makeModelGraph()
        #expect(graph.rootRegion.operations.isEmpty == false)
        #expect(graph.rootRegion.results.count == 1)
    }

    @Test("Qwen35 hybrid group structure")
    func qwen35HybridGroups() throws {
        let config = TestConfigs.qwen35_0_8B
        // fullAttentionInterval=4, 24 layers -> 6 hybrid groups
        #expect(config.layerCount / config.fullAttentionInterval! == 6)
        // deltaNetLayersPerGroup = interval - 1 = 3
        #expect(config.fullAttentionInterval! - 1 == 3)
        // ropePartialDim = headDim * partialRotaryFactor = 256 * 0.25 = 64
        #expect(Int(Float(config.headDim) * config.partialRotaryFactor!) == 64)
    }

    @Test("Qwen35 text-only produces valid ModelGraph without VisionEncoder")
    func qwen35TextOnlyGraph() throws {
        let model = Qwen35(config: TestConfigs.qwen35TextOnly)

        let graph = try model.makeModelGraph()
        #expect(graph.rootRegion.operations.isEmpty == false)
        #expect(graph.rootRegion.results.count == 1)

        // First op should be tokenEmbedding, NOT parallel
        let firstOp = graph.rootRegion.operations[0]
        if case .primitive(let attrs) = firstOp.kind, attrs is TokenEmbeddingAttributes {
            // OK
        } else {
            Issue.record("Expected tokenEmbedding as first op for text-only, got \(firstOp.kind)")
        }
    }

    @Test("Qwen35 attention uses M-RoPE")
    func qwen35MRoPE() throws {
        let graph = try Qwen35(config: TestConfigs.qwen35_0_8B).makeModelGraph()

        // Find the first attention operation (inside repeating > residual)
        let attnOp = findFirstOperation(in: graph.rootRegion) { kind in
            if kind is AttentionAttributes { return true }
            return false
        }
        guard case .primitive(let rawAttrs) = attnOp?.kind,
              let attrs = rawAttrs as? AttentionAttributes else {
            Issue.record("No attention operation found")
            return
        }

        // Verify M-RoPE is configured
        #expect(attrs.rope != nil)
        #expect(attrs.rope?.mropeAxes != nil)
        #expect(attrs.rope?.mropeAxes?.sections == [11, 11, 10])
        #expect(attrs.rope?.mropeAxes?.interleaved == true)
        #expect(attrs.qkNorm == .rmsNormUnitOffset)
    }

    @Test("Qwen35 declaration is deterministic")
    func qwen35Deterministic() throws {
        let graph1 = try Qwen35(config: TestConfigs.qwen35_0_8B).makeModelGraph()
        let graph2 = try Qwen35(config: TestConfigs.qwen35_0_8B).makeModelGraph()

        let canonical1 = canonicalize(graph1)
        let canonical2 = canonicalize(graph2)
        #expect(canonical1.dump() == canonical2.dump())
    }

    // MARK: - Gemma 4

    @Test("Gemma4 produces valid ModelGraph")
    func gemma4Graph() throws {
        let model = try Gemma4(config: TestConfigs.gemma4E2B)

        let graph = try model.makeModelGraph()
        #expect(graph.rootRegion.operations.isEmpty == false)
        #expect(graph.rootRegion.results.count == 1)
    }

    @Test("Gemma4 sliding/full attention schedule is preserved")
    func gemma4LayerSchedule() throws {
        let graph = try Gemma4(config: TestConfigs.gemma4E2B).makeModelGraph()

        var slidingAttentionCount = 0
        var fullAttentionCount = 0
        var validatedSlidingWindow = false
        traverse(graph.rootRegion) { operation in
            guard case .primitive(let rawAttributes) = operation.kind,
                  let attributes = rawAttributes as? AttentionAttributes else {
                return
            }
            if attributes.window == nil {
                fullAttentionCount += 1
            } else {
                slidingAttentionCount += 1
                if validatedSlidingWindow == false {
                    #expect(attributes.window?.left == 512)
                    #expect(attributes.window?.right == 0)
                    validatedSlidingWindow = true
                }
            }
        }

        #expect(slidingAttentionCount == 28, "Expected 28 sliding attention layers")
        #expect(fullAttentionCount == 7, "Expected 7 full attention layers")
        #expect(validatedSlidingWindow)
    }

    @Test("Gemma4 full-attention keeps proportional RoPE metadata")
    func gemma4FullAttentionRopeDimension() throws {
        let graph = try Gemma4(config: TestConfigs.gemma4E2B).makeModelGraph()

        var fullAttentionAttributes: [AttentionAttributes] = []
        traverse(graph.rootRegion) { operation in
            guard case .primitive(let rawAttributes) = operation.kind,
                  let attributes = rawAttributes as? AttentionAttributes,
                  attributes.window == nil else {
                return
            }
            fullAttentionAttributes.append(attributes)
        }

        #expect(fullAttentionAttributes.isEmpty == false)
        for attributes in fullAttentionAttributes {
            #expect(attributes.headDimension == 512)
            #expect(attributes.rope?.dimension == 128)
            #expect(attributes.rope?.scaling?.kind == .custom("proportional"))
            #expect(attributes.valueProjectionSource == .keyProjection)
        }
    }

    @Test("Gemma4 decoder RMSNorms keep explicit checkpoint scales")
    func gemma4DecoderNormsKeepExplicitCheckpointScales() throws {
        let graph = try Gemma4(config: TestConfigs.gemma4E2B).makeModelGraph()

        var regularNormCount = 0
        var attentionQKNormCount = 0
        traverse(graph.rootRegion) { operation in
            guard case .primitive(let rawAttributes) = operation.kind else {
                return
            }
            if let attributes = rawAttributes as? RMSNormAttributes {
                #expect(attributes.weightBias == 0)
                regularNormCount += 1
            }
            if let attributes = rawAttributes as? AttentionAttributes,
               attributes.qkNorm == .rmsNorm {
                attentionQKNormCount += 1
            }
        }

        #expect(regularNormCount == 141, "Expected 4 decoder RMSNorms per layer plus one final RMSNorm")
        #expect(attentionQKNormCount == 35, "Expected Gemma4 q/k norm to use explicit checkpoint scales in every layer")
    }

    @Test("Gemma4 emits one per-layer input residual per decoder layer")
    func gemma4PerLayerInput() throws {
        let graph = try Gemma4(config: TestConfigs.gemma4E2B).makeModelGraph()

        let perLayerInputCount = countOperations(in: graph.rootRegion) { kind in
            if kind is PerLayerInputAttributes { return true }
            return false
        }

        #expect(perLayerInputCount == 35, "Expected 35 per-layer input residuals")
    }

    @Test("Gemma4 preserves per-layer templates as repeating blocks")
    func gemma4RepeatingLayerTemplates() throws {
        let graph = try Gemma4(config: TestConfigs.gemma4E2B).makeModelGraph()
        let repeatingOps = graph.rootRegion.operations.filter {
            if case .repeating(count: 1, _) = $0.kind { return true }
            return false
        }
        #expect(repeatingOps.count == 35, "Expected 35 repeating(count: 1) layer templates")
    }

    @Test("Gemma4 doubles MLP width on KV-shared layers")
    func gemma4DoubleWideMLP() throws {
        let graph = try Gemma4(config: TestConfigs.gemma4E2B).makeModelGraph()

        var regularMLPCount = 0
        var doubleWideMLPCount = 0
        traverse(graph.rootRegion) { operation in
            guard case .primitive(let rawAttributes) = operation.kind,
                  let attributes = rawAttributes as? MLPAttributes else {
                return
            }
            if attributes.intermediateSize == 6144 {
                regularMLPCount += 1
            } else if attributes.intermediateSize == 12_288 {
                doubleWideMLPCount += 1
            }
        }

        #expect(regularMLPCount == 15, "Expected 15 regular-width MLP layers")
        #expect(doubleWideMLPCount == 20, "Expected 20 double-width MLP layers")
    }

    @Test("Gemma4 declaration is deterministic")
    func gemma4Deterministic() throws {
        let graph1 = try Gemma4(config: TestConfigs.gemma4E2B).makeModelGraph()
        let graph2 = try Gemma4(config: TestConfigs.gemma4E2B).makeModelGraph()

        let canonical1 = canonicalize(graph1)
        let canonical2 = canonicalize(graph2)
        #expect(canonical1.dump() == canonical2.dump())
    }

    @Test("Gemma4 validate rejects missing per-layer input metadata")
    func gemma4ValidateMissingMetadata() throws {
        let invalidConfig = ModelConfig(
            hiddenSize: 1536, layerCount: 35, intermediateSize: 6144, vocabSize: 262144,
            attentionHeads: 8, kvHeads: 1, headDim: 256,
            attentionBias: false, mlpBias: false,
            normEps: 1e-6, normKind: .rmsNorm,
            ropeTheta: 10_000.0, ropeDimension: 256, ropeScaling: nil,
            tiedEmbeddings: true,
            expertCount: nil, expertsPerToken: nil,
            qkNorm: true,
            fullAttentionInterval: nil,
            ssmNumHeads: nil, ssmKeyHeadDim: nil, ssmValueHeadDim: nil,
            convKernelSize: nil, partialRotaryFactor: nil, slidingWindow: 512,
            layerTypes: TestConfigs.gemma4E2B.layerTypes,
            globalHeadDim: 512,
            numKVSharedLayers: 20,
            useDoubleWideMLP: true,
            fullAttentionRopeTheta: 1_000_000.0,
            fullAttentionPartialRotaryFactor: 0.25,
            fullAttentionRoPEScaling: RoPEScaling(kind: .custom("proportional"), factor: 1.0)
        )

        #expect(throws: ModelGraphBuildError.self) {
            try Gemma4.validate(invalidConfig)
        }
    }

    // MARK: - Cohere

    @Test("Cohere produces valid ModelGraph")
    func cohereGraph() throws {
        let model = Cohere(config: TestConfigs.cohereCommandR)

        let graph = try model.makeModelGraph()
        #expect(graph.rootRegion.operations.isEmpty == false)
        #expect(graph.rootRegion.results.count == 1)
    }

    // MARK: - LFM2

    @Test("LFM2.5 1.2B produces valid ModelGraph")
    func lfm25_1_2B() throws {
        let graph = try LFM2(config: TestConfigs.lfm25_1_2B).makeModelGraph()
        #expect(graph.rootRegion.operations.isEmpty == false)
        #expect(graph.rootRegion.results.count == 1)
    }

    @Test("LFM2 350M produces valid ModelGraph")
    func lfm2_350M() throws {
        let graph = try LFM2(config: TestConfigs.lfm2_350M).makeModelGraph()
        #expect(graph.rootRegion.operations.isEmpty == false)
        #expect(graph.rootRegion.results.count == 1)
    }

    @Test("LFM2 2.6B produces valid ModelGraph")
    func lfm2_2_6B() throws {
        let graph = try LFM2(config: TestConfigs.lfm2_2_6B).makeModelGraph()
        #expect(graph.rootRegion.operations.isEmpty == false)
        #expect(graph.rootRegion.results.count == 1)
    }

    @Test("LFM2 8B-A1B MoE produces valid ModelGraph")
    func lfm2_8B_A1B() throws {
        let graph = try LFM2(config: TestConfigs.lfm2_8B_A1B).makeModelGraph()
        #expect(graph.rootRegion.operations.isEmpty == false)
        #expect(graph.rootRegion.results.count == 1)
    }

    @Test("LFM2 24B-A2B MoE produces valid ModelGraph")
    func lfm2_24B_A2B() throws {
        let graph = try LFM2(config: TestConfigs.lfm2_24B_A2B).makeModelGraph()
        #expect(graph.rootRegion.operations.isEmpty == false)
        #expect(graph.rootRegion.results.count == 1)
    }

    @Test("LFM2 section pattern config")
    func lfm2SectionPattern() throws {
        let config = TestConfigs.lfm25_1_2B
        #expect(config.layerCount == 16)
        #expect(config.layerTypes?.count == 16)
        // resolvedHeadDim = 2048 / 32 = 64
        #expect(config.hiddenSize / config.attentionHeads == 64)
        // convKernelSize = convLCache + 1 = 3 + 1 = 4
        #expect((config.convLCache ?? 0) + 1 == 4)
        // Not MoE
        #expect(config.expertCount == nil)

        let moeConfig = TestConfigs.lfm2_8B_A1B
        #expect(moeConfig.expertCount == 32)
        #expect(moeConfig.expertsPerToken == 4)
    }

    @Test("LFM2 declaration is deterministic")
    func lfm2Deterministic() throws {
        let graph1 = try LFM2(config: TestConfigs.lfm25_1_2B).makeModelGraph()
        let graph2 = try LFM2(config: TestConfigs.lfm25_1_2B).makeModelGraph()

        let canonical1 = canonicalize(graph1)
        let canonical2 = canonicalize(graph2)
        #expect(canonical1.dump() == canonical2.dump())
    }

    @Test("LFM2 MoE deterministic")
    func lfm2MoeDeterministic() throws {
        let graph1 = try LFM2(config: TestConfigs.lfm2_8B_A1B).makeModelGraph()
        let graph2 = try LFM2(config: TestConfigs.lfm2_8B_A1B).makeModelGraph()

        let canonical1 = canonicalize(graph1)
        let canonical2 = canonicalize(graph2)
        #expect(canonical1.dump() == canonical2.dump())
    }

    @Test("LFM2 24B-A2B numDenseLayers splits dense and MoE layers")
    func lfm2NumDenseLayers() throws {
        let model = try LFM2(config: TestConfigs.lfm2_24B_A2B)
        let graph = try model.makeModelGraph()

        // Count MLP and MoE ops in the graph
        let mlpCount = countOperations(in: graph.rootRegion) { kind in
            if kind is MLPAttributes { return true }
            return false
        }
        let moeCount = countOperations(in: graph.rootRegion) { kind in
            if kind is MoEAttributes { return true }
            return false
        }

        // numDenseLayers=2 -> 2 layers with MLP, 38 layers with MoE
        #expect(mlpCount == 2, "Expected 2 dense MLP layers, got \(mlpCount)")
        #expect(moeCount == 38, "Expected 38 MoE layers, got \(moeCount)")
    }

    @Test("LFM2 8B-A1B graph has 2 MLP and 22 MoE (numDenseLayers=2)")
    func lfm2_8B_A1B_DenseLayersInGraph() throws {
        let graph = try LFM2(config: TestConfigs.lfm2_8B_A1B).makeModelGraph()

        let mlpCount = countOperations(in: graph.rootRegion) { kind in
            if kind is MLPAttributes { return true }
            return false
        }
        let moeCount = countOperations(in: graph.rootRegion) { kind in
            if kind is MoEAttributes { return true }
            return false
        }

        // numDenseLayers=2 -> 2 layers with MLP, 22 layers with MoE
        #expect(mlpCount == 2, "Expected 2 MLP layers, got \(mlpCount)")
        #expect(moeCount == 22, "Expected 22 MoE layers, got \(moeCount)")
    }

    @Test("LFM2 non-MoE graph has all MLP, no MoE")
    func lfm2AllMLPInGraph() throws {
        let graph = try LFM2(config: TestConfigs.lfm25_1_2B).makeModelGraph()

        let mlpCount = countOperations(in: graph.rootRegion) { kind in
            if kind is MLPAttributes { return true }
            return false
        }
        let moeCount = countOperations(in: graph.rootRegion) { kind in
            if kind is MoEAttributes { return true }
            return false
        }

        #expect(mlpCount == 16, "Expected 16 MLP layers, got \(mlpCount)")
        #expect(moeCount == 0, "Expected 0 MoE layers, got \(moeCount)")
    }

    @Test("LFM2 custom MoE with numDenseLayers boundary within section")
    func lfm2CustomNumDenseLayers() throws {
        // 3 layers: conv, conv, attn -> numDenseLayers=1, layer 0 is dense
        let config = TestConfigs.lfm2CustomMoE

        let model = try LFM2(config: config)
        let graph = try model.makeModelGraph()
        let mlpCount = countOperations(in: graph.rootRegion) { kind in
            if kind is MLPAttributes { return true }
            return false
        }
        let moeCount = countOperations(in: graph.rootRegion) { kind in
            if kind is MoEAttributes { return true }
            return false
        }
        #expect(mlpCount == 1)
        #expect(moeCount == 2)
    }

    // MARK: - Graph-level Layer Type Verification

    @Test("LFM2 graph conv/attn op count matches sections")
    func lfm2GraphLayerTypes() throws {
        let graph = try LFM2(config: TestConfigs.lfm2_350M).makeModelGraph()

        let convCount = countOperations(in: graph.rootRegion) { kind in
            if kind is ShortConvAttributes { return true }
            return false
        }
        let attnCount = countOperations(in: graph.rootRegion) { kind in
            if kind is AttentionAttributes { return true }
            return false
        }

        // lfm2_350M: (conv*2+attn)*3, (conv+attn)*3, conv*1
        // 10 conv layers, 6 attn layers
        #expect(convCount == 10, "Expected 10 conv layers, got \(convCount)")
        #expect(attnCount == 6, "Expected 6 attn layers, got \(attnCount)")
    }

    @Test("Qwen35 graph has correct DeltaNet and Attention layer counts")
    func qwen35GraphLayerCounts() throws {
        let config = TestConfigs.qwen35_0_8B
        let graph = try Qwen35(config: config).makeModelGraph()

        // Qwen35 uses LayerStack with flat layer schedule.
        // Count stateSpace and attention ops across all layers.
        let deltaNetCount = countOperations(in: graph.rootRegion) { kind in
            if kind is StateSpaceAttributes { return true }
            return false
        }
        let attnCount = countOperations(in: graph.rootRegion) { kind in
            if kind is AttentionAttributes { return true }
            return false
        }

        // 24 layers: 18 DeltaNet + 6 Attention (fullAttentionInterval=4)
        #expect(deltaNetCount == 18, "Expected 18 DeltaNet ops, got \(deltaNetCount)")
        #expect(attnCount == 6, "Expected 6 Attention ops, got \(attnCount)")
    }

    // MARK: - Cross-model Comparison

    @Test("Different architectures produce different graphs")
    func crossModelComparison() throws {
        let transformer = try Transformer(config: TestConfigs.transformerSmall2).makeModelGraph()
        let qwen35 = try Qwen35(config: TestConfigs.qwen35_0_8B).makeModelGraph()

        let canonTransformer = canonicalize(transformer)
        let canonQwen35 = canonicalize(qwen35)
        #expect(canonTransformer.dump() != canonQwen35.dump())
    }

    @Test("LFM2 graph differs from Transformer and Qwen35")
    func lfm2CrossComparison() throws {
        let lfm2 = try LFM2(config: TestConfigs.lfm25_1_2B).makeModelGraph()
        let transformer = try Transformer(config: TestConfigs.transformerMatchLFM2).makeModelGraph()

        let canonLFM2 = canonicalize(lfm2)
        let canonTransformer = canonicalize(transformer)
        #expect(canonLFM2.dump() != canonTransformer.dump())
    }

    @Test("LFM2 dense vs MoE produce different graphs")
    func lfm2DenseVsMoE() throws {
        let dense = try LFM2(config: TestConfigs.lfm25_1_2B).makeModelGraph()
        let moe = try LFM2(config: TestConfigs.lfm2_8B_A1B).makeModelGraph()

        let canonDense = canonicalize(dense)
        let canonMoE = canonicalize(moe)
        #expect(canonDense.dump() != canonMoE.dump())
    }

    @Test("LFM2 custom config with arbitrary sections")
    func lfm2CustomSections() throws {
        // 2*(3+1) + 1*2 = 10 layers
        let config = TestConfigs.lfm2Custom10L

        let graph = try LFM2(config: config).makeModelGraph()
        #expect(graph.rootRegion.operations.isEmpty == false)
        #expect(graph.rootRegion.results.count == 1)
    }
}

// MARK: - Test Configurations

/// Preset ModelConfig values for tests.
///
/// These replicate the previous model-specific Config presets
/// using ModelConfig directly.
private enum TestConfigs {

    // MARK: - Transformer Presets

    static let transformer7B = ModelConfig(
        hiddenSize: 4096, layerCount: 32, intermediateSize: 11008, vocabSize: 32000,
        attentionHeads: 32, kvHeads: 8, headDim: 128,
        attentionBias: false, mlpBias: false,
        normEps: 1e-5, normKind: .rmsNorm,
        ropeTheta: 10000.0, ropeDimension: 128, ropeScaling: nil,
        tiedEmbeddings: false,
        expertCount: nil, expertsPerToken: nil,
        qkNorm: false,
        fullAttentionInterval: nil,
        ssmNumHeads: nil, ssmKeyHeadDim: nil, ssmValueHeadDim: nil,
        convKernelSize: nil, partialRotaryFactor: nil, slidingWindow: nil
    )

    static let transformerMoE = ModelConfig(
        hiddenSize: 4096, layerCount: 32, intermediateSize: 14336, vocabSize: 32000,
        attentionHeads: 32, kvHeads: 8, headDim: 128,
        attentionBias: false, mlpBias: false,
        normEps: 1e-5, normKind: .rmsNorm,
        ropeTheta: 10000.0, ropeDimension: 128, ropeScaling: nil,
        tiedEmbeddings: false,
        expertCount: 8, expertsPerToken: 2,
        qkNorm: false,
        fullAttentionInterval: nil,
        ssmNumHeads: nil, ssmKeyHeadDim: nil, ssmValueHeadDim: nil,
        convKernelSize: nil, partialRotaryFactor: nil, slidingWindow: nil
    )

    static let transformerSmall = ModelConfig(
        hiddenSize: 2048, layerCount: 16, intermediateSize: 5632, vocabSize: 32000,
        attentionHeads: 16, kvHeads: 4, headDim: 128,
        attentionBias: false, mlpBias: false,
        normEps: 1e-5, normKind: .rmsNorm,
        ropeTheta: 10000.0, ropeDimension: 128, ropeScaling: nil,
        tiedEmbeddings: false,
        expertCount: nil, expertsPerToken: nil,
        qkNorm: false,
        fullAttentionInterval: nil,
        ssmNumHeads: nil, ssmKeyHeadDim: nil, ssmValueHeadDim: nil,
        convKernelSize: nil, partialRotaryFactor: nil, slidingWindow: nil
    )

    static let transformerSmall2 = ModelConfig(
        hiddenSize: 1024, layerCount: 12, intermediateSize: 2816, vocabSize: 32000,
        attentionHeads: 8, kvHeads: 4, headDim: 128,
        attentionBias: false, mlpBias: false,
        normEps: 1e-5, normKind: .rmsNorm,
        ropeTheta: 10000.0, ropeDimension: 128, ropeScaling: nil,
        tiedEmbeddings: false,
        expertCount: nil, expertsPerToken: nil,
        qkNorm: false,
        fullAttentionInterval: nil,
        ssmNumHeads: nil, ssmKeyHeadDim: nil, ssmValueHeadDim: nil,
        convKernelSize: nil, partialRotaryFactor: nil, slidingWindow: nil
    )

    /// Transformer with same dimensions as LFM2.5 1.2B for cross-model comparison.
    static let transformerMatchLFM2 = ModelConfig(
        hiddenSize: 2048, layerCount: 16, intermediateSize: 5632, vocabSize: 65536,
        attentionHeads: 32, kvHeads: 8, headDim: 64,
        attentionBias: false, mlpBias: false,
        normEps: 1e-5, normKind: .rmsNorm,
        ropeTheta: 10000.0, ropeDimension: 64, ropeScaling: nil,
        tiedEmbeddings: false,
        expertCount: nil, expertsPerToken: nil,
        qkNorm: false,
        fullAttentionInterval: nil,
        ssmNumHeads: nil, ssmKeyHeadDim: nil, ssmValueHeadDim: nil,
        convKernelSize: nil, partialRotaryFactor: nil, slidingWindow: nil
    )

    // MARK: - Cohere Presets

    static let cohereCommandR = ModelConfig(
        hiddenSize: 8192, layerCount: 40, intermediateSize: 22528, vocabSize: 256000,
        attentionHeads: 64, kvHeads: 8, headDim: 128,
        attentionBias: false, mlpBias: false,
        normEps: 1e-5, normKind: .layerNorm,
        ropeTheta: 10000.0, ropeDimension: 128, ropeScaling: nil,
        tiedEmbeddings: true,
        expertCount: nil, expertsPerToken: nil,
        qkNorm: true,
        fullAttentionInterval: nil,
        ssmNumHeads: nil, ssmKeyHeadDim: nil, ssmValueHeadDim: nil,
        convKernelSize: nil, partialRotaryFactor: nil, slidingWindow: nil
    )

    // MARK: - Qwen 3.5 Presets

    /// Qwen 3.5 0.8B VLM preset (with M-RoPE axes for VLM).
    static let qwen35_0_8B = ModelConfig(
        hiddenSize: 1024, layerCount: 24, intermediateSize: 3584, vocabSize: 248320,
        attentionHeads: 8, kvHeads: 2, headDim: 256,
        attentionBias: false, mlpBias: false,
        normEps: 1e-6, normKind: .rmsNorm,
        ropeTheta: 10_000_000.0, ropeDimension: 64, ropeScaling: nil,
        tiedEmbeddings: true,
        expertCount: nil, expertsPerToken: nil,
        qkNorm: true,
        fullAttentionInterval: 4,
        ssmNumHeads: 16, ssmKeyHeadDim: 128, ssmValueHeadDim: 128,
        convKernelSize: 4, partialRotaryFactor: 0.25, slidingWindow: nil,
        mropeAxes: MRoPEAxes(sections: [11, 11, 10], interleaved: true)
    )

    /// Qwen 3.5 text-only (no M-RoPE).
    static let qwen35TextOnly = ModelConfig(
        hiddenSize: 1024, layerCount: 24, intermediateSize: 3584, vocabSize: 248320,
        attentionHeads: 8, kvHeads: 2, headDim: 256,
        attentionBias: false, mlpBias: false,
        normEps: 1e-6, normKind: .rmsNorm,
        ropeTheta: 10_000_000.0, ropeDimension: 64, ropeScaling: nil,
        tiedEmbeddings: true,
        expertCount: nil, expertsPerToken: nil,
        qkNorm: true,
        fullAttentionInterval: 4,
        ssmNumHeads: 16, ssmKeyHeadDim: 128, ssmValueHeadDim: 128,
        convKernelSize: 4, partialRotaryFactor: 0.25, slidingWindow: nil
    )

    // MARK: - Gemma 4 Presets

    /// Gemma 4 E2B-it text backbone preset from the public HF config.
    static let gemma4E2B = ModelConfig(
        hiddenSize: 1536, layerCount: 35, intermediateSize: 6144, vocabSize: 262144,
        attentionHeads: 8, kvHeads: 1, headDim: 256,
        attentionBias: false, mlpBias: false,
        normEps: 1e-6, normKind: .rmsNorm,
        ropeTheta: 10_000.0, ropeDimension: 256, ropeScaling: nil,
        tiedEmbeddings: true,
        expertCount: nil, expertsPerToken: nil,
        qkNorm: true,
        fullAttentionInterval: nil,
        ssmNumHeads: nil, ssmKeyHeadDim: nil, ssmValueHeadDim: nil,
        convKernelSize: nil, partialRotaryFactor: nil, slidingWindow: 512,
        layerTypes: [
            "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention",
            "full_attention",
            "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention",
            "full_attention",
            "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention",
            "full_attention",
            "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention",
            "full_attention",
            "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention",
            "full_attention",
            "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention",
            "full_attention",
            "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention",
            "full_attention",
        ],
        hiddenSizePerLayerInput: 256,
        vocabSizePerLayerInput: 262144,
        globalHeadDim: 512,
        globalKVHeads: nil,
        numKVSharedLayers: 20,
        useDoubleWideMLP: true,
        attentionKEqualsV: true,
        fullAttentionRopeTheta: 1_000_000.0,
        fullAttentionPartialRotaryFactor: 0.25,
        fullAttentionRoPEScaling: RoPEScaling(kind: .custom("proportional"), factor: 1.0)
    )

    // MARK: - LFM2 Presets

    /// LFM2 350M: (conv*2+attn)*3, (conv+attn)*3, conv*1 = 16 layers
    static let lfm2_350M = ModelConfig(
        hiddenSize: 1024, layerCount: 16, intermediateSize: 6656, vocabSize: 65536,
        attentionHeads: 16, kvHeads: 8, headDim: 64,
        attentionBias: false, mlpBias: false,
        normEps: 1e-5, normKind: .rmsNorm,
        ropeTheta: 1_000_000.0, ropeDimension: 64, ropeScaling: nil,
        tiedEmbeddings: true,
        expertCount: nil, expertsPerToken: nil,
        qkNorm: true,
        fullAttentionInterval: nil,
        ssmNumHeads: nil, ssmKeyHeadDim: nil, ssmValueHeadDim: nil,
        convKernelSize: nil, convLCache: 3, partialRotaryFactor: nil, slidingWindow: nil,
        layerTypes: expandSections([
            (groupCount: 3, convsPerGroup: 2, hasAttention: true),
            (groupCount: 3, convsPerGroup: 1, hasAttention: true),
            (groupCount: 1, convsPerGroup: 1, hasAttention: false),
        ])
    )

    /// LFM 2.5 1.2B: (conv*2+attn)*3, (conv+attn)*3, conv*1 = 16 layers
    static let lfm25_1_2B = ModelConfig(
        hiddenSize: 2048, layerCount: 16, intermediateSize: 12288, vocabSize: 65536,
        attentionHeads: 32, kvHeads: 8, headDim: 64,
        attentionBias: false, mlpBias: false,
        normEps: 1e-5, normKind: .rmsNorm,
        ropeTheta: 1_000_000.0, ropeDimension: 64, ropeScaling: nil,
        tiedEmbeddings: true,
        expertCount: nil, expertsPerToken: nil,
        qkNorm: true,
        fullAttentionInterval: nil,
        ssmNumHeads: nil, ssmKeyHeadDim: nil, ssmValueHeadDim: nil,
        convKernelSize: nil, convLCache: 3, partialRotaryFactor: nil, slidingWindow: nil,
        layerTypes: expandSections([
            (groupCount: 3, convsPerGroup: 2, hasAttention: true),
            (groupCount: 3, convsPerGroup: 1, hasAttention: true),
            (groupCount: 1, convsPerGroup: 1, hasAttention: false),
        ])
    )

    /// LFM2 2.6B: (conv*2+attn)*2, (conv*3+attn)*4, (conv*2+attn)*2, conv*2 = 30 layers
    static let lfm2_2_6B = ModelConfig(
        hiddenSize: 2048, layerCount: 30, intermediateSize: 10752, vocabSize: 65536,
        attentionHeads: 32, kvHeads: 8, headDim: 64,
        attentionBias: false, mlpBias: false,
        normEps: 1e-5, normKind: .rmsNorm,
        ropeTheta: 1_000_000.0, ropeDimension: 64, ropeScaling: nil,
        tiedEmbeddings: true,
        expertCount: nil, expertsPerToken: nil,
        qkNorm: true,
        fullAttentionInterval: nil,
        ssmNumHeads: nil, ssmKeyHeadDim: nil, ssmValueHeadDim: nil,
        convKernelSize: nil, convLCache: 3, partialRotaryFactor: nil, slidingWindow: nil,
        layerTypes: expandSections([
            (groupCount: 2, convsPerGroup: 2, hasAttention: true),
            (groupCount: 4, convsPerGroup: 3, hasAttention: true),
            (groupCount: 2, convsPerGroup: 2, hasAttention: true),
            (groupCount: 1, convsPerGroup: 2, hasAttention: false),
        ])
    )

    /// LFM2 8B-A1B MoE: (conv*2+attn)*1, (conv*3+attn)*4, (conv*2+attn)*1, conv*2 = 24 layers
    static let lfm2_8B_A1B = ModelConfig(
        hiddenSize: 2048, layerCount: 24, intermediateSize: 7168, vocabSize: 65536,
        attentionHeads: 32, kvHeads: 8, headDim: 64,
        attentionBias: false, mlpBias: true,
        normEps: 1e-5, normKind: .rmsNorm,
        ropeTheta: 1_000_000.0, ropeDimension: 64, ropeScaling: nil,
        tiedEmbeddings: true,
        expertCount: 32, expertsPerToken: 4,
        moeIntermediateSize: 1792,
        qkNorm: true,
        fullAttentionInterval: nil,
        ssmNumHeads: nil, ssmKeyHeadDim: nil, ssmValueHeadDim: nil,
        convKernelSize: nil, convLCache: 3, partialRotaryFactor: nil, slidingWindow: nil,
        layerTypes: expandSections([
            (groupCount: 1, convsPerGroup: 2, hasAttention: true),
            (groupCount: 4, convsPerGroup: 3, hasAttention: true),
            (groupCount: 1, convsPerGroup: 2, hasAttention: true),
            (groupCount: 1, convsPerGroup: 2, hasAttention: false),
        ]),
        numDenseLayers: 2
    )

    /// LFM2 24B-A2B MoE: (conv*2+attn)*1, (conv*3+attn)*9, conv*1 = 40 layers
    static let lfm2_24B_A2B = ModelConfig(
        hiddenSize: 2048, layerCount: 40, intermediateSize: 11776, vocabSize: 65536,
        attentionHeads: 32, kvHeads: 8, headDim: 64,
        attentionBias: false, mlpBias: true,
        normEps: 1e-5, normKind: .rmsNorm,
        ropeTheta: 1_000_000.0, ropeDimension: 64, ropeScaling: nil,
        tiedEmbeddings: true,
        expertCount: 64, expertsPerToken: 4,
        moeIntermediateSize: 1536,
        qkNorm: true,
        fullAttentionInterval: nil,
        ssmNumHeads: nil, ssmKeyHeadDim: nil, ssmValueHeadDim: nil,
        convKernelSize: nil, convLCache: 3, partialRotaryFactor: nil, slidingWindow: nil,
        layerTypes: expandSections([
            (groupCount: 1, convsPerGroup: 2, hasAttention: true),
            (groupCount: 9, convsPerGroup: 3, hasAttention: true),
            (groupCount: 1, convsPerGroup: 1, hasAttention: false),
        ]),
        numDenseLayers: 2
    )

    /// LFM2 custom: 2*(3+1) + 1*2 = 10 layers
    static let lfm2Custom10L = ModelConfig(
        hiddenSize: 1024, layerCount: 10, intermediateSize: 4096, vocabSize: 32000,
        attentionHeads: 16, kvHeads: 4, headDim: 64,
        attentionBias: false, mlpBias: false,
        normEps: 1e-5, normKind: .rmsNorm,
        ropeTheta: 1_000_000.0, ropeDimension: 64, ropeScaling: nil,
        tiedEmbeddings: true,
        expertCount: nil, expertsPerToken: nil,
        qkNorm: true,
        fullAttentionInterval: nil,
        ssmNumHeads: nil, ssmKeyHeadDim: nil, ssmValueHeadDim: nil,
        convKernelSize: nil, convLCache: 3, partialRotaryFactor: nil, slidingWindow: nil,
        layerTypes: expandSections([
            (groupCount: 2, convsPerGroup: 3, hasAttention: true),
            (groupCount: 1, convsPerGroup: 2, hasAttention: false),
        ])
    )

    /// LFM2 custom MoE with numDenseLayers boundary: 3 layers: conv, conv, attn
    static let lfm2CustomMoE = ModelConfig(
        hiddenSize: 1024, layerCount: 3, intermediateSize: 4096, vocabSize: 32000,
        attentionHeads: 16, kvHeads: 4, headDim: 64,
        attentionBias: false, mlpBias: false,
        normEps: 1e-5, normKind: .rmsNorm,
        ropeTheta: 1_000_000.0, ropeDimension: 64, ropeScaling: nil,
        tiedEmbeddings: true,
        expertCount: 8, expertsPerToken: 2,
        moeIntermediateSize: 1024,
        qkNorm: true,
        fullAttentionInterval: nil,
        ssmNumHeads: nil, ssmKeyHeadDim: nil, ssmValueHeadDim: nil,
        convKernelSize: nil, convLCache: 3, partialRotaryFactor: nil, slidingWindow: nil,
        layerTypes: ["conv", "conv", "full_attention"],
        numDenseLayers: 1
    )

    // MARK: - Section Expansion Helper

    /// Expand section definitions to a flat layer_types array.
    private static func expandSections(
        _ sections: [(groupCount: Int, convsPerGroup: Int, hasAttention: Bool)]
    ) -> [String] {
        var result: [String] = []
        for section in sections {
            for _ in 0..<section.groupCount {
                for _ in 0..<section.convsPerGroup {
                    result.append("conv")
                }
                if section.hasAttention {
                    result.append("full_attention")
                }
            }
        }
        return result
    }
}

// MARK: - Helpers

/// Extract primitive attributes from an operation, returning nil for structural operations.
private func primitiveAttributes(_ kind: OperationKind) -> (any OperationAttributes)? {
    guard case .primitive(let attrs) = kind else { return nil }
    return attrs
}

/// Depth-first search for the first operation whose primitive attributes match the predicate.
private func findFirstOperation(
    in region: Region,
    matching predicate: (any OperationAttributes) -> Bool
) -> Operation? {
    for op in region.operations {
        if let attrs = primitiveAttributes(op.kind), predicate(attrs) { return op }
        switch op.kind {
        case .residual(_, let body):
            if let found = findFirstOperation(in: body, matching: predicate) {
                return found
            }
        case .parallel(_, let branches):
            for branch in branches {
                if let found = findFirstOperation(in: branch, matching: predicate) {
                    return found
                }
            }
        case .repeating(_, let body):
            if let found = findFirstOperation(in: body, matching: predicate) {
                return found
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

/// Count all primitive operations matching a predicate, recursing into all nested regions.
private func countOperations(
    in region: Region,
    matching predicate: (any OperationAttributes) -> Bool
) -> Int {
    var count = 0
    for op in region.operations {
        if let attrs = primitiveAttributes(op.kind), predicate(attrs) { count += 1 }
        switch op.kind {
        case .residual(_, let body):
            count += countOperations(in: body, matching: predicate)
        case .parallel(_, let branches):
            for branch in branches {
                count += countOperations(in: branch, matching: predicate)
            }
        case .repeating(_, let body):
            count += countOperations(in: body, matching: predicate)
        case .conditional(_, let thenBody, let elseBody):
            count += countOperations(in: thenBody, matching: predicate)
            count += countOperations(in: elseBody, matching: predicate)
        default:
            break
        }
    }
    return count
}

/// Count operations inside the first repeating body found (for Repeat-based models).
private func countOperations(
    inRepeatingBody region: Region,
    matching predicate: (any OperationAttributes) -> Bool
) -> Int {
    for op in region.operations {
        switch op.kind {
        case .repeating(_, let body):
            return countOperations(in: body, matching: predicate)
        case .residual(_, let body):
            let result = countOperations(inRepeatingBody: body, matching: predicate)
            if result > 0 { return result }
        case .parallel(_, let branches):
            for branch in branches {
                let result = countOperations(inRepeatingBody: branch, matching: predicate)
                if result > 0 { return result }
            }
        default:
            break
        }
    }
    return 0
}

/// Traverse all operations depth-first, including nested structural regions.
private func traverse(_ region: Region, visit: (Operation) -> Void) {
    for operation in region.operations {
        visit(operation)
        switch operation.kind {
        case .residual(_, let body):
            traverse(body, visit: visit)
        case .parallel(_, let branches):
            for branch in branches {
                traverse(branch, visit: visit)
            }
        case .repeating(_, let body):
            traverse(body, visit: visit)
        case .conditional(_, let thenBody, let elseBody):
            traverse(thenBody, visit: visit)
            traverse(elseBody, visit: visit)
        default:
            break
        }
    }
}
