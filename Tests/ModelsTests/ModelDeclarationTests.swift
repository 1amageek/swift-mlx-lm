import Testing
@testable import SwiftLM
@testable import Models

@Suite("Model Declaration Tests")
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

        // Vision encoder defaults
        #expect(config.visionHiddenSize == 768)
        #expect(config.visionDepth == 12)
        #expect(config.visionHeadCount == 12)
        #expect(config.visionPatchSize == 16)
        #expect(config.visionIntermediateSize == 3072)
        #expect(config.visionOutputSize == 1024) // matches hiddenSize
        #expect(config.spatialMergeSize == 2)
        #expect(config.temporalPatchSize == 2)

        // M-RoPE
        #expect(config.mropeSections == [11, 11, 10])
        #expect(config.mropeInterleaved == true)

        // Special tokens
        #expect(config.imageTokenId == 248056)
        #expect(config.videoTokenId == 248057)
    }

    @Test("Qwen35 graph contains vision encoder")
    func qwen35VisionEncoder() throws {
        let graph = try Qwen35(config: .qwen35_0_8B).makeModelGraph()

        // First op should be parallel (visionMerge) containing both
        // token embedding and vision encoder
        let firstOp = graph.rootRegion.operations[0]
        guard case .parallel(let merge, let branches) = firstOp.kind else {
            Issue.record("Expected parallel as first operation, got \(firstOp.kind)")
            return
        }

        // Merge strategy should be visionMerge
        guard case .visionMerge(let config) = merge else {
            Issue.record("Expected visionMerge strategy, got \(merge)")
            return
        }
        #expect(config.imageTokenId == 248056)
        #expect(config.videoTokenId == 248057)

        // Should have 2 branches: text embedding + vision encoder
        #expect(branches.count == 2)

        // Branch 0: token embedding
        let textBranch = branches[0]
        #expect(textBranch.operations.count == 1)
        if case .tokenEmbedding = textBranch.operations[0].kind {
            // OK
        } else {
            Issue.record("Expected tokenEmbedding in branch 0")
        }

        // Branch 1: vision encoder
        let visionBranch = branches[1]
        #expect(visionBranch.operations.count == 1)
        if case .visionEncoder(let attrs) = visionBranch.operations[0].kind {
            #expect(attrs.hiddenSize == 768)
            #expect(attrs.depth == 12)
            #expect(attrs.outputSize == 1024)
            #expect(attrs.temporalPatchSize == 2)
        } else {
            Issue.record("Expected visionEncoder in branch 1")
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
