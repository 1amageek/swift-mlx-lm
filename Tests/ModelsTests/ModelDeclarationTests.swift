import Testing
import SwiftLM
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
}
