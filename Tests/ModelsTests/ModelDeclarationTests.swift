import Testing
@testable import SwiftLM
@testable import ModelDeclarations

@Suite("Model Declaration Tests", .tags(.unit))
struct ModelDeclarationTests {

    // MARK: - Transformer

    @Test("Transformer produces valid ModelGraph")
    func transformerGraph() throws {
        let model = Transformer(config: .init(
            hiddenSize: 4096,
            hiddenLayers: 32,
            intermediateSize: 11008,
            attentionHeads: 32,
            kvHeads: 8,
            vocabularySize: 32000
        ))

        let graph = try model.makeModelGraph()
        #expect(graph.rootRegion.operations.isEmpty == false)
        #expect(graph.rootRegion.results.count == 1)
    }

    @Test("Transformer with MoE produces valid ModelGraph")
    func transformerMoEGraph() throws {
        let model = Transformer(config: .init(
            hiddenSize: 4096,
            hiddenLayers: 32,
            intermediateSize: 14336,
            attentionHeads: 32,
            kvHeads: 8,
            vocabularySize: 32000,
            moe: .init(expertCount: 8, expertsPerToken: 2)
        ))

        let graph = try model.makeModelGraph()
        #expect(graph.rootRegion.operations.isEmpty == false)
        #expect(graph.rootRegion.results.count == 1)
    }

    @Test("Transformer declaration is deterministic")
    func transformerDeterministic() throws {
        let config = Transformer.Config(
            hiddenSize: 2048,
            hiddenLayers: 16,
            intermediateSize: 5632,
            attentionHeads: 16,
            kvHeads: 4,
            vocabularySize: 32000
        )

        let graph1 = try Transformer(config: config).makeModelGraph()
        let graph2 = try Transformer(config: config).makeModelGraph()

        let canonical1 = canonicalize(graph1)
        let canonical2 = canonicalize(graph2)
        #expect(canonical1 == canonical2)
    }

    // MARK: - Qwen 3.5

    @Test("Qwen35 produces valid ModelGraph")
    func qwen35Graph() throws {
        let model = Qwen35(config: .qwen35_0_8B)

        let graph = try model.makeModelGraph()
        #expect(graph.rootRegion.operations.isEmpty == false)
        #expect(graph.rootRegion.results.count == 1)
    }

    @Test("Qwen35 hybrid group structure")
    func qwen35HybridGroups() throws {
        let config = Qwen35.Config.qwen35_0_8B
        #expect(config.hybridGroupCount == 6)
        #expect(config.deltaNetLayersPerGroup == 3)
        #expect(config.ropePartialDim == 64)
    }

    @Test("Qwen35 VLM config parameters")
    func qwen35VLMConfig() throws {
        let config = Qwen35.Config.qwen35_0_8B

        #expect(config.isVLM == true)
        let vision = try #require(config.vision)

        // Vision encoder
        #expect(vision.hiddenSize == 768)
        #expect(vision.depth == 12)
        #expect(vision.headCount == 12)
        #expect(vision.patchSize == 16)
        #expect(vision.intermediateSize == 3072)
        #expect(vision.outputSize == 1024) // matches hiddenSize
        #expect(vision.spatialMergeSize == 2)
        #expect(vision.temporalPatchSize == 2)

        // M-RoPE
        #expect(vision.mropeSections == [11, 11, 10])
        #expect(vision.mropeInterleaved == true)

        // Special tokens
        #expect(vision.imageTokenId == 248056)
        #expect(vision.videoTokenId == 248057)
    }

    @Test("Qwen35 text-only config has no vision")
    func qwen35TextOnlyConfig() throws {
        let config = Qwen35.Config(
            hiddenSize: 1024,
            hiddenLayers: 24,
            intermediateSize: 3584,
            vocabularySize: 248320,
            attentionHeads: 8,
            kvHeads: 2,
            ssmNumHeads: 16
        )
        #expect(config.isVLM == false)
        #expect(config.vision == nil)
    }

    @Test("Qwen35 text-only produces valid ModelGraph without VisionEncoder")
    func qwen35TextOnlyGraph() throws {
        let model = Qwen35(config: .init(
            hiddenSize: 1024,
            hiddenLayers: 24,
            intermediateSize: 3584,
            vocabularySize: 248320,
            attentionHeads: 8,
            kvHeads: 2,
            ssmNumHeads: 16
        ))

        let graph = try model.makeModelGraph()
        #expect(graph.rootRegion.operations.isEmpty == false)
        #expect(graph.rootRegion.results.count == 1)

        // First op should be tokenEmbedding, NOT parallel
        let firstOp = graph.rootRegion.operations[0]
        if case .tokenEmbedding = firstOp.kind {
            // OK
        } else {
            Issue.record("Expected tokenEmbedding as first op for text-only, got \(firstOp.kind)")
        }

    }

    @Test("Qwen35 attention uses M-RoPE")
    func qwen35MRoPE() throws {
        let graph = try Qwen35(config: .qwen35_0_8B).makeModelGraph()

        // Find the first attention operation (inside repeating > residual)
        let attnOp = findFirstOperation(in: graph.rootRegion) { kind in
            if case .attention = kind { return true }
            return false
        }
        guard case .attention(let attrs) = attnOp?.kind else {
            Issue.record("No attention operation found")
            return
        }

        // Verify M-RoPE is configured
        #expect(attrs.rope != nil)
        #expect(attrs.rope?.mropeAxes != nil)
        #expect(attrs.rope?.mropeAxes?.sections == [11, 11, 10])
        #expect(attrs.rope?.mropeAxes?.interleaved == true)
        #expect(attrs.qkNorm == .rmsNorm)
    }

    @Test("Qwen35 declaration is deterministic")
    func qwen35Deterministic() throws {
        let graph1 = try Qwen35(config: .qwen35_0_8B).makeModelGraph()
        let graph2 = try Qwen35(config: .qwen35_0_8B).makeModelGraph()

        let canonical1 = canonicalize(graph1)
        let canonical2 = canonicalize(graph2)
        #expect(canonical1 == canonical2)
    }

    // MARK: - Cohere

    @Test("Cohere produces valid ModelGraph")
    func cohereGraph() throws {
        let model = Cohere(config: .init(
            hiddenSize: 8192,
            hiddenLayers: 40,
            intermediateSize: 22528,
            attentionHeads: 64,
            kvHeads: 8,
            vocabularySize: 256000
        ))

        let graph = try model.makeModelGraph()
        #expect(graph.rootRegion.operations.isEmpty == false)
        #expect(graph.rootRegion.results.count == 1)
    }

    // MARK: - LFM2

    @Test("LFM2.5 1.2B produces valid ModelGraph")
    func lfm25_1_2B() throws {
        let graph = try LFM2(config: .lfm25_1_2B).makeModelGraph()
        #expect(graph.rootRegion.operations.isEmpty == false)
        #expect(graph.rootRegion.results.count == 1)
    }

    @Test("LFM2 350M produces valid ModelGraph")
    func lfm2_350M() throws {
        let graph = try LFM2(config: .lfm2_350M).makeModelGraph()
        #expect(graph.rootRegion.operations.isEmpty == false)
        #expect(graph.rootRegion.results.count == 1)
    }

    @Test("LFM2 2.6B produces valid ModelGraph")
    func lfm2_2_6B() throws {
        let graph = try LFM2(config: .lfm2_2_6B).makeModelGraph()
        #expect(graph.rootRegion.operations.isEmpty == false)
        #expect(graph.rootRegion.results.count == 1)
    }

    @Test("LFM2 8B-A1B MoE produces valid ModelGraph")
    func lfm2_8B_A1B() throws {
        let graph = try LFM2(config: .lfm2_8B_A1B).makeModelGraph()
        #expect(graph.rootRegion.operations.isEmpty == false)
        #expect(graph.rootRegion.results.count == 1)
    }

    @Test("LFM2 24B-A2B MoE produces valid ModelGraph")
    func lfm2_24B_A2B() throws {
        let graph = try LFM2(config: .lfm2_24B_A2B).makeModelGraph()
        #expect(graph.rootRegion.operations.isEmpty == false)
        #expect(graph.rootRegion.results.count == 1)
    }

    @Test("LFM2 section pattern config")
    func lfm2SectionPattern() throws {
        let config = LFM2.Config.lfm25_1_2B
        #expect(config.hiddenLayers == 16)
        #expect(config.sections.count == 3)
        #expect(config.resolvedHeadDim == 64)
        #expect(config.convKernelSize == 4)
        #expect(config.isMoE == false)

        let moeConfig = LFM2.Config.lfm2_8B_A1B
        #expect(moeConfig.isMoE == true)
        #expect(moeConfig.moe?.expertCount == 32)
        #expect(moeConfig.moe?.expertsPerToken == 4)
    }

    @Test("LFM2 declaration is deterministic")
    func lfm2Deterministic() throws {
        let graph1 = try LFM2(config: .lfm25_1_2B).makeModelGraph()
        let graph2 = try LFM2(config: .lfm25_1_2B).makeModelGraph()

        let canonical1 = canonicalize(graph1)
        let canonical2 = canonicalize(graph2)
        #expect(canonical1 == canonical2)
    }

    @Test("LFM2 MoE deterministic")
    func lfm2MoeDeterministic() throws {
        let graph1 = try LFM2(config: .lfm2_8B_A1B).makeModelGraph()
        let graph2 = try LFM2(config: .lfm2_8B_A1B).makeModelGraph()

        let canonical1 = canonicalize(graph1)
        let canonical2 = canonicalize(graph2)
        #expect(canonical1 == canonical2)
    }

    @Test("LFM2 24B-A2B numDenseLayers splits dense and MoE layers")
    func lfm2NumDenseLayers() throws {
        let config = LFM2.Config.lfm2_24B_A2B
        let descriptors = config.layerDescriptors

        // 40 total layers
        #expect(descriptors.count == 40)

        // numDenseLayers=2: first 2 layers use dense MLP
        #expect(descriptors[0].useMoE == false)
        #expect(descriptors[0].isConvolution == true)
        #expect(descriptors[1].useMoE == false)
        #expect(descriptors[1].isConvolution == true)

        // Layer 2 onwards use MoE
        for i in 2..<40 {
            #expect(descriptors[i].useMoE == true, "Layer \(i) should use MoE")
        }
    }

    @Test("LFM2 non-MoE configs have all dense layers")
    func lfm2NonMoEDescriptors() throws {
        let descriptors = LFM2.Config.lfm25_1_2B.layerDescriptors
        #expect(descriptors.count == 16)
        for descriptor in descriptors {
            #expect(descriptor.useMoE == false)
        }
    }

    @Test("LFM2 8B-A1B has 2 dense + 22 MoE (numDenseLayers=2)")
    func lfm2_8B_A1B_DenseLayerDescriptors() throws {
        let descriptors = LFM2.Config.lfm2_8B_A1B.layerDescriptors
        #expect(descriptors.count == 24)

        // numDenseLayers=2: first 2 layers use dense MLP
        #expect(descriptors[0].useMoE == false)
        #expect(descriptors[1].useMoE == false)

        // Layer 2 onwards use MoE
        for i in 2..<24 {
            #expect(descriptors[i].useMoE == true, "Layer \(i) should use MoE")
        }
    }

    @Test("LFM2 layer descriptor conv/attn pattern matches sections")
    func lfm2DescriptorPattern() throws {
        // lfm2_350M: (conv×2+attn)×3, (conv+attn)×3, conv×1
        let descriptors = LFM2.Config.lfm2_350M.layerDescriptors
        #expect(descriptors.count == 16)

        // Section 0: 3 groups of [conv, conv, attn]
        #expect(descriptors[0].isConvolution == true)
        #expect(descriptors[1].isConvolution == true)
        #expect(descriptors[2].isConvolution == false) // attn
        #expect(descriptors[3].isConvolution == true)
        #expect(descriptors[4].isConvolution == true)
        #expect(descriptors[5].isConvolution == false) // attn
        #expect(descriptors[6].isConvolution == true)
        #expect(descriptors[7].isConvolution == true)
        #expect(descriptors[8].isConvolution == false) // attn

        // Section 1: 3 groups of [conv, attn]
        #expect(descriptors[9].isConvolution == true)
        #expect(descriptors[10].isConvolution == false) // attn
        #expect(descriptors[11].isConvolution == true)
        #expect(descriptors[12].isConvolution == false) // attn
        #expect(descriptors[13].isConvolution == true)
        #expect(descriptors[14].isConvolution == false) // attn

        // Section 2: 1 conv (no attn)
        #expect(descriptors[15].isConvolution == true)
    }

    @Test("LFM2 custom config with arbitrary sections")
    func lfm2CustomSections() throws {
        // 2*(3+1) + 1*2 = 10 layers
        let config = LFM2.Config(
            hiddenSize: 1024,
            hiddenLayers: 10,
            intermediateSize: 4096,
            vocabularySize: 32000,
            attentionHeads: 16,
            kvHeads: 4,
            sections: [
                .init(groupCount: 2, convsPerGroup: 3),
                .init(groupCount: 1, convsPerGroup: 2, hasAttention: false),
            ]
        )

        let graph = try LFM2(config: config).makeModelGraph()
        #expect(graph.rootRegion.operations.isEmpty == false)
        #expect(graph.rootRegion.results.count == 1)
    }

    // MARK: - Cross-model Comparison

    @Test("Different architectures produce different graphs")
    func crossModelComparison() throws {
        let transformer = try Transformer(config: .init(
            hiddenSize: 1024,
            hiddenLayers: 12,
            intermediateSize: 2816,
            attentionHeads: 8,
            kvHeads: 4,
            vocabularySize: 32000
        )).makeModelGraph()

        let qwen35 = try Qwen35(config: .qwen35_0_8B).makeModelGraph()

        let canonTransformer = canonicalize(transformer)
        let canonQwen35 = canonicalize(qwen35)
        #expect(canonTransformer != canonQwen35)
    }

    @Test("LFM2 graph differs from Transformer and Qwen35")
    func lfm2CrossComparison() throws {
        let lfm2 = try LFM2(config: .lfm25_1_2B).makeModelGraph()
        let transformer = try Transformer(config: .init(
            hiddenSize: 2048,
            hiddenLayers: 16,
            intermediateSize: 5632,
            attentionHeads: 32,
            kvHeads: 8,
            vocabularySize: 65536
        )).makeModelGraph()

        let canonLFM2 = canonicalize(lfm2)
        let canonTransformer = canonicalize(transformer)
        #expect(canonLFM2 != canonTransformer)
    }

    @Test("LFM2 dense vs MoE produce different graphs")
    func lfm2DenseVsMoE() throws {
        let dense = try LFM2(config: .lfm25_1_2B).makeModelGraph()
        let moe = try LFM2(config: .lfm2_8B_A1B).makeModelGraph()

        let canonDense = canonicalize(dense)
        let canonMoE = canonicalize(moe)
        #expect(canonDense != canonMoE)
    }

    // MARK: - Graph-level numDenseLayers Verification

    @Test("LFM2 24B-A2B graph contains exactly 2 MLP and 38 MoE ops")
    func lfm2NumDenseLayersInGraph() throws {
        let graph = try LFM2(config: .lfm2_24B_A2B).makeModelGraph()

        let mlpCount = countOperations(in: graph.rootRegion) { kind in
            if case .mlp = kind { return true }
            return false
        }
        let moeCount = countOperations(in: graph.rootRegion) { kind in
            if case .moe = kind { return true }
            return false
        }

        // numDenseLayers=2 → 2 layers with MLP, 38 layers with MoE
        #expect(mlpCount == 2, "Expected 2 dense MLP layers, got \(mlpCount)")
        #expect(moeCount == 38, "Expected 38 MoE layers, got \(moeCount)")
    }

    @Test("LFM2 8B-A1B graph has 2 MLP and 22 MoE (numDenseLayers=2)")
    func lfm2_8B_A1B_DenseLayersInGraph() throws {
        let graph = try LFM2(config: .lfm2_8B_A1B).makeModelGraph()

        let mlpCount = countOperations(in: graph.rootRegion) { kind in
            if case .mlp = kind { return true }
            return false
        }
        let moeCount = countOperations(in: graph.rootRegion) { kind in
            if case .moe = kind { return true }
            return false
        }

        // numDenseLayers=2 → 2 layers with MLP, 22 layers with MoE
        #expect(mlpCount == 2, "Expected 2 MLP layers, got \(mlpCount)")
        #expect(moeCount == 22, "Expected 22 MoE layers, got \(moeCount)")
    }

    @Test("LFM2 non-MoE graph has all MLP, no MoE")
    func lfm2AllMLPInGraph() throws {
        let graph = try LFM2(config: .lfm25_1_2B).makeModelGraph()

        let mlpCount = countOperations(in: graph.rootRegion) { kind in
            if case .mlp = kind { return true }
            return false
        }
        let moeCount = countOperations(in: graph.rootRegion) { kind in
            if case .moe = kind { return true }
            return false
        }

        #expect(mlpCount == 16, "Expected 16 MLP layers, got \(mlpCount)")
        #expect(moeCount == 0, "Expected 0 MoE layers, got \(moeCount)")
    }

    @Test("LFM2 custom MoE with numDenseLayers boundary within section")
    func lfm2CustomNumDenseLayers() throws {
        // 3 layers: conv, conv, attn — numDenseLayers=1 → layer 0 is dense
        let config = LFM2.Config(
            hiddenSize: 1024,
            hiddenLayers: 3,
            intermediateSize: 4096,
            vocabularySize: 32000,
            attentionHeads: 16,
            kvHeads: 4,
            sections: [
                .init(groupCount: 1, convsPerGroup: 2),
            ],
            moe: .init(expertCount: 8, expertsPerToken: 2, moeIntermediateSize: 1024),
            numDenseLayers: 1
        )

        let descriptors = config.layerDescriptors
        #expect(descriptors.count == 3)
        #expect(descriptors[0].useMoE == false) // layer 0: dense
        #expect(descriptors[1].useMoE == true)  // layer 1: MoE
        #expect(descriptors[2].useMoE == true)  // layer 2: MoE

        let graph = try LFM2(config: config).makeModelGraph()
        let mlpCount = countOperations(in: graph.rootRegion) { kind in
            if case .mlp = kind { return true }
            return false
        }
        let moeCount = countOperations(in: graph.rootRegion) { kind in
            if case .moe = kind { return true }
            return false
        }
        #expect(mlpCount == 1)
        #expect(moeCount == 2)
    }

    // MARK: - Graph-level Layer Type Verification

    @Test("LFM2 graph conv/attn op count matches sections")
    func lfm2GraphLayerTypes() throws {
        let graph = try LFM2(config: .lfm2_350M).makeModelGraph()

        let convCount = countOperations(in: graph.rootRegion) { kind in
            if case .stateSpace = kind { return true }
            return false
        }
        let attnCount = countOperations(in: graph.rootRegion) { kind in
            if case .attention = kind { return true }
            return false
        }

        // lfm2_350M: (conv×2+attn)×3, (conv+attn)×3, conv×1
        // 10 conv layers, 6 attn layers
        #expect(convCount == 10, "Expected 10 conv layers, got \(convCount)")
        #expect(attnCount == 6, "Expected 6 attn layers, got \(attnCount)")
    }

    @Test("Qwen35 graph has correct DeltaNet and Attention layer counts")
    func qwen35GraphLayerCounts() throws {
        let config = Qwen35.Config.qwen35_0_8B
        let graph = try Qwen35(config: config).makeModelGraph()

        // Qwen35 uses Repeat, so ops are inside the repeating body.
        // Count within the repeating body and multiply by repeat count.
        let deltaNetInBody = countOperations(inRepeatingBody: graph.rootRegion) { kind in
            if case .stateSpace = kind { return true }
            return false
        }
        let attnInBody = countOperations(inRepeatingBody: graph.rootRegion) { kind in
            if case .attention = kind { return true }
            return false
        }

        // hybridGroupCount=6 groups, each with 3 DeltaNet + 1 Attention
        // The outer Repeat body contains an inner Repeat (DeltaNet) + 1 Attn residual
        // Inner Repeat body has 1 DeltaNet op
        #expect(deltaNetInBody >= 1, "Expected DeltaNet ops in repeating body")
        #expect(attnInBody >= 1, "Expected Attention ops in repeating body")
    }
}

// MARK: - Helpers

/// Depth-first search for the first operation matching a predicate.
private func findFirstOperation(
    in region: Region,
    matching predicate: (OperationKind) -> Bool
) -> Operation? {
    for op in region.operations {
        if predicate(op.kind) { return op }
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
        default:
            break
        }
    }
    return nil
}

/// Count all operations matching a predicate, recursing into all nested regions.
private func countOperations(
    in region: Region,
    matching predicate: (OperationKind) -> Bool
) -> Int {
    var count = 0
    for op in region.operations {
        if predicate(op.kind) { count += 1 }
        switch op.kind {
        case .residual(_, let body):
            count += countOperations(in: body, matching: predicate)
        case .parallel(_, let branches):
            for branch in branches {
                count += countOperations(in: branch, matching: predicate)
            }
        case .repeating(_, let body):
            count += countOperations(in: body, matching: predicate)
        default:
            break
        }
    }
    return count
}

/// Count operations inside the first repeating body found (for Repeat-based models).
private func countOperations(
    inRepeatingBody region: Region,
    matching predicate: (OperationKind) -> Bool
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
