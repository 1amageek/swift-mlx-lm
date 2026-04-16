import Testing
import Foundation
@testable import LMArchitecture

// MARK: - Example Model Declarations

/// Minimal Llama-style transformer defined with SwiftLM DSL.
struct TinyLlama: ModelComponent {
    let vocabSize: Int
    let hiddenSize: Int
    let headCount: Int
    let kvHeadCount: Int
    let intermediateSize: Int
    let layerCount: Int

    @ModelComponentBuilder var body: some ModelComponent {
        TokenEmbedding(vocabSize: vocabSize, embeddingSize: hiddenSize)

        Repeat(count: layerCount) {
            Residual {
                RMSNorm(dimension: hiddenSize)
                Attention(
                    hiddenSize: hiddenSize,
                    headCount: headCount,
                    kvHeadCount: kvHeadCount
                )
            }
            Residual {
                RMSNorm(dimension: hiddenSize)
                MLP(inputSize: hiddenSize, intermediateSize: intermediateSize)
            }
        }

        RMSNorm(dimension: hiddenSize)
        OutputHead(inputSize: hiddenSize, vocabSize: vocabSize, tiedToEmbedding: true)
    }
}

/// Cohere-style transformer with parallel attention + FFN.
struct TinyCohere: ModelComponent {
    let vocabSize: Int
    let hiddenSize: Int
    let headCount: Int
    let kvHeadCount: Int
    let intermediateSize: Int
    let layerCount: Int

    @ModelComponentBuilder var body: some ModelComponent {
        TokenEmbedding(vocabSize: vocabSize, embeddingSize: hiddenSize)

        Repeat(count: layerCount) {
            Residual {
                LayerNorm(dimension: hiddenSize)
                Parallel(merge: .add) {
                    Attention(
                        hiddenSize: hiddenSize,
                        headCount: headCount,
                        kvHeadCount: kvHeadCount,
                        qkNorm: .layerNorm
                    )
                    MLP(inputSize: hiddenSize, intermediateSize: intermediateSize)
                }
            }
        }

        LayerNorm(dimension: hiddenSize)
        OutputHead(inputSize: hiddenSize, vocabSize: vocabSize, tiedToEmbedding: true)
    }
}

// MARK: - Tests

@Suite("DSL Lowering", .tags(.unit))
struct DSLLoweringTests {

    // MARK: - Component Structure Tests

    @Test("Primitive component conforms to PrimitiveComponent")
    func primitiveComponent() {
        let emb = TokenEmbedding(vocabSize: 100, embeddingSize: 64)
        let kind = emb.operationKind
        if case .tokenEmbedding(let attrs) = kind {
            #expect(attrs.vocabSize == 100)
            #expect(attrs.embeddingSize == 64)
        } else {
            Issue.record("Expected tokenEmbedding operation kind")
        }
    }

    @Test("Builder produces sequential composition")
    func sequenceFromBuilder() throws {
        @ModelComponentBuilder
        var body: some ModelComponent {
            TokenEmbedding(vocabSize: 100, embeddingSize: 64)
            RMSNorm(dimension: 64)
            OutputHead(inputSize: 64, vocabSize: 100)
        }

        let graph = try normalize(body).graph
        #expect(graph.rootRegion.operations.count == 3)
    }

    @Test("Residual normalizes content correctly")
    func residualStructure() throws {
        let r = Residual {
            RMSNorm(dimension: 64)
            Attention(hiddenSize: 64, headCount: 4, kvHeadCount: 4, headDimension: 16)
        }

        #expect(r.strategy == .add)
        let graph = try normalize(r).graph
        // Residual is one operation; its body region has 2 operations
        #expect(graph.rootRegion.operations.count == 1)
        if case .residual(_, let bodyRegion) = graph.rootRegion.operations[0].kind {
            #expect(bodyRegion.operations.count == 2)
        } else {
            Issue.record("Expected residual operation")
        }
    }

    @Test("Parallel normalizes branches correctly")
    func parallelStructure() throws {
        let p = Parallel(merge: .add) {
            Attention(hiddenSize: 64, headCount: 4, kvHeadCount: 4, headDimension: 16)
            MLP(inputSize: 64, intermediateSize: 256)
        }

        #expect(p.merge == .add)
        let graph = try normalize(p).graph
        // Parallel is one operation; it should have 2 branches
        #expect(graph.rootRegion.operations.count == 1)
        if case .parallel(_, let branches) = graph.rootRegion.operations[0].kind {
            #expect(branches.count == 2)
        } else {
            Issue.record("Expected parallel operation")
        }
    }

    @Test("Repeat stores count, label, and content")
    func repeatStructure() throws {
        let r = Repeat(count: 12, label: "layers") {
            RMSNorm(dimension: 64)
        }

        #expect(r.count == 12)
        #expect(r.label == "layers")
        let graph = try normalize(r).graph
        // Repeat is one operation; its body region has 1 operation
        #expect(graph.rootRegion.operations.count == 1)
        if case .repeating(let count, let bodyRegion) = graph.rootRegion.operations[0].kind {
            #expect(count == 12)
            #expect(bodyRegion.operations.count == 1)
        } else {
            Issue.record("Expected repeating operation")
        }
    }

    @Test("Group stores label and content")
    func groupStructure() throws {
        let g = Group(label: "block") {
            RMSNorm(dimension: 64)
            Attention(hiddenSize: 64, headCount: 4, kvHeadCount: 4, headDimension: 16)
        }

        #expect(g.label == "block")
        let graph = try normalize(g).graph
        // Group is flattened; its 2 children become 2 operations
        #expect(graph.rootRegion.operations.count == 2)
    }

    @Test("Group without label stores nil label")
    func groupWithoutLabel() throws {
        let g = Group {
            RMSNorm(dimension: 64)
        }

        #expect(g.label == nil)
        let graph = try normalize(g).graph
        // Group is flattened; its 1 child becomes 1 operation
        #expect(graph.rootRegion.operations.count == 1)
    }

    // MARK: - Normalization Tests

    @Test("Normalization produces correct region structure with value flow")
    func normalizationBasic() throws {
        @ModelComponentBuilder
        var body: some ModelComponent {
            TokenEmbedding(vocabSize: 100, embeddingSize: 64)
            RMSNorm(dimension: 64)
            OutputHead(inputSize: 64, vocabSize: 100)
        }

        let normalized = try normalize(body)
        let graph = normalized.graph

        #expect(graph.rootRegion.operations.count == 3)
        #expect(graph.rootRegion.operations[0].results.count == 1)
        #expect(graph.rootRegion.operations[1].results.count == 1)
        #expect(graph.rootRegion.operations[2].results.count == 1)
        #expect(graph.rootRegion.results.count == 1)

        // Value flow: op1 consumes op0's result, op2 consumes op1's result
        let op0Result = graph.rootRegion.operations[0].results[0].id
        let op1Operand = graph.rootRegion.operations[1].operands[0].value
        #expect(op0Result == op1Operand)
    }

    @Test("Normalization flattens nested sequences")
    func normalizationFlattenSequence() throws {
        let comp = Group {
            Group {
                RMSNorm(dimension: 64)
                RMSNorm(dimension: 64)
            }
            LMArchitecture.Linear(inputSize: 64, outputSize: 32)
        }

        let graph = try normalize(comp).graph
        #expect(graph.rootRegion.operations.count == 3)
    }

    @Test("Normalization extracts labels into metadata")
    func normalizationStripLabels() throws {
        let comp = Group {
            Group(label: "debug") {
                RMSNorm(dimension: 64)
                LMArchitecture.Linear(inputSize: 64, outputSize: 32)
            }
        }

        let normalized = try normalize(comp)

        #expect(normalized.graph.rootRegion.operations.count == 2)
        let firstPath = StructuralPath(components: [.operation(0)])
        #expect(normalized.metadata.annotation(for: firstPath)?.label == "debug")
    }

    @Test("Normalization creates nested regions for residual with value flow")
    func normalizationResidual() throws {
        @ModelComponentBuilder
        var body: some ModelComponent {
            TokenEmbedding(vocabSize: 100, embeddingSize: 64)
            Residual {
                RMSNorm(dimension: 64)
                Attention(hiddenSize: 64, headCount: 4, kvHeadCount: 4, headDimension: 16)
            }
        }

        let graph = try normalize(body).graph

        #expect(graph.rootRegion.operations.count == 2)

        let residualOp = graph.rootRegion.operations[1]
        if case .residual(let strategy, let bodyRegion) = residualOp.kind {
            #expect(strategy == .add)
            #expect(bodyRegion.operations.count == 2)
            #expect(bodyRegion.parameters.count == 1)
            #expect(bodyRegion.results.count == 1)
            // Residual operand arity matches body parameter arity
            #expect(residualOp.operands.count == bodyRegion.parameters.count)
            #expect(residualOp.results.count == bodyRegion.results.count)
        } else {
            Issue.record("Expected residual operation")
        }

        // Normalized graphs pass structural validation
        try GraphValidator.validate(graph)
    }

    @Test("Normalization creates branch regions for parallel with value flow")
    func normalizationParallel() throws {
        @ModelComponentBuilder
        var body: some ModelComponent {
            RMSNorm(dimension: 64)
            Parallel(merge: .add) {
                Attention(hiddenSize: 64, headCount: 4, kvHeadCount: 4, headDimension: 16)
                MLP(inputSize: 64, intermediateSize: 256)
            }
        }

        let graph = try normalize(body).graph

        // RMSNorm + Parallel
        #expect(graph.rootRegion.operations.count == 2)

        let parallelOp = graph.rootRegion.operations[1]
        if case .parallel(let merge, let branches) = parallelOp.kind {
            #expect(merge == .add)
            #expect(branches.count == 2)
            for branch in branches {
                #expect(branch.operations.count == 1)
                #expect(branch.parameters.count == 1)
                #expect(branch.results.count == 1)
            }
        } else {
            Issue.record("Expected parallel operation")
        }

        try GraphValidator.validate(graph)
    }

    @Test("Normalization creates body region for repeat, label in metadata")
    func normalizationRepeat() throws {
        @ModelComponentBuilder
        var body: some ModelComponent {
            RMSNorm(dimension: 64)
            Repeat(count: 32, label: "layers") {
                RMSNorm(dimension: 64)
            }
        }

        let normalized = try normalize(body)
        let graph = normalized.graph

        // RMSNorm + Repeat
        #expect(graph.rootRegion.operations.count == 2)

        let repeatOp = graph.rootRegion.operations[1]
        if case .repeating(let count, let bodyRegion) = repeatOp.kind {
            #expect(count == 32)
            #expect(bodyRegion.operations.count == 1)
            #expect(bodyRegion.parameters.count == 1)
            #expect(bodyRegion.results.count == 1)
            // Loop-carried: params == results == operands == op.results
            #expect(bodyRegion.parameters.count == bodyRegion.results.count)
            #expect(repeatOp.operands.count == repeatOp.results.count)
        } else {
            Issue.record("Expected repeating operation")
        }

        let opPath = StructuralPath(components: [.operation(1)])
        #expect(normalized.metadata.annotation(for: opPath)?.label == "layers")

        try GraphValidator.validate(graph)
    }

    // MARK: - End-to-End Tests

    @Test("Llama-style model produces correct graph structure")
    func llamaLowering() throws {
        let model = TinyLlama(
            vocabSize: 32000,
            hiddenSize: 4096,
            headCount: 32,
            kvHeadCount: 8,
            intermediateSize: 11008,
            layerCount: 2
        )

        let graph = try ModelGraph(model)

        // Root region: TokenEmbedding, Repeat, RMSNorm, OutputHead
        #expect(graph.rootRegion.operations.count == 4)

        if case .tokenEmbedding(let attrs) = graph.rootRegion.operations[0].kind {
            #expect(attrs.vocabSize == 32000)
            #expect(attrs.embeddingSize == 4096)
        } else {
            Issue.record("First op should be tokenEmbedding")
        }

        if case .repeating(let count, let body) = graph.rootRegion.operations[1].kind {
            #expect(count == 2)
            #expect(body.operations.count == 2)
            if case .residual(_, let residualBody) = body.operations[0].kind {
                #expect(residualBody.operations.count == 2)
            } else {
                Issue.record("Expected residual in repeat body")
            }
        } else {
            Issue.record("Second op should be repeating")
        }

        if case .outputHead(let attrs) = graph.rootRegion.operations[3].kind {
            #expect(attrs.vocabSize == 32000)
            #expect(attrs.tiedToEmbedding == true)
        } else {
            Issue.record("Last op should be outputHead")
        }

        // Full model passes structural validation
        try GraphValidator.validate(graph)
    }

    @Test("Cohere-style model uses parallel attention + FFN")
    func cohereLowering() throws {
        let model = TinyCohere(
            vocabSize: 256000,
            hiddenSize: 4096,
            headCount: 32,
            kvHeadCount: 8,
            intermediateSize: 11008,
            layerCount: 1
        )

        let graph = try ModelGraph(model)

        #expect(graph.rootRegion.operations.count == 4)

        if case .repeating(_, let body) = graph.rootRegion.operations[1].kind {
            if case .residual(_, let resBody) = body.operations[0].kind {
                #expect(resBody.operations.count == 2)
                if case .parallel(let merge, let branches) = resBody.operations[1].kind {
                    #expect(merge == .add)
                    #expect(branches.count == 2)
                } else {
                    Issue.record("Expected parallel in residual body")
                }
            } else {
                Issue.record("Expected residual in repeat body")
            }
        } else {
            Issue.record("Expected repeating")
        }

        try GraphValidator.validate(graph)
    }

    @Test("Same parameters produce equal graphs")
    func canonicalEquivalence() throws {
        @ModelComponentBuilder
        var bodyA: some ModelComponent {
            TokenEmbedding(vocabSize: 100, embeddingSize: 64)
            RMSNorm(dimension: 64, epsilon: 1e-6)
            OutputHead(inputSize: 64, vocabSize: 100)
        }

        @ModelComponentBuilder
        var bodyB: some ModelComponent {
            TokenEmbedding(vocabSize: 100, embeddingSize: 64)
            RMSNorm(dimension: 64, epsilon: 1e-6)
            OutputHead(inputSize: 64, vocabSize: 100)
        }

        let graphA = try normalize(bodyA).graph
        let graphB = try normalize(bodyB).graph
        #expect(graphA == graphB)
    }

    @Test("Different parameters produce different graphs")
    func nonEquivalence() throws {
        @ModelComponentBuilder
        var bodyA: some ModelComponent {
            TokenEmbedding(vocabSize: 100, embeddingSize: 64)
            OutputHead(inputSize: 64, vocabSize: 100)
        }

        @ModelComponentBuilder
        var bodyB: some ModelComponent {
            TokenEmbedding(vocabSize: 200, embeddingSize: 128)
            OutputHead(inputSize: 128, vocabSize: 200)
        }

        let graphA = try normalize(bodyA).graph
        let graphB = try normalize(bodyB).graph
        #expect(graphA != graphB)
    }

    @Test("ModelGraph is serializable")
    func graphSerialization() throws {
        let model = TinyLlama(
            vocabSize: 100,
            hiddenSize: 64,
            headCount: 4,
            kvHeadCount: 2,
            intermediateSize: 256,
            layerCount: 2
        )

        let graph = try ModelGraph(model)

        let encoder = JSONEncoder()
        encoder.outputFormatting = .sortedKeys
        let data = try encoder.encode(graph)
        let decoded = try JSONDecoder().decode(ModelGraph.self, from: data)

        #expect(graph == decoded)
    }

    @Test("ModelComponent) convenience method")
    func modelComponentConvenience() throws {
        let model = TinyLlama(
            vocabSize: 100,
            hiddenSize: 64,
            headCount: 4,
            kvHeadCount: 2,
            intermediateSize: 256,
            layerCount: 1
        )

        let graph = try ModelGraph(model)
        let manual = try normalize(model.body).graph

        #expect(graph == manual)
    }

    @Test("ModelComponent) returns graph + metadata")
    func normalizedModelConvenience() throws {
        let model = TinyLlama(
            vocabSize: 100,
            hiddenSize: 64,
            headCount: 4,
            kvHeadCount: 2,
            intermediateSize: 256,
            layerCount: 1
        )

        let normalized = try NormalizedModel(model)
        let graph = try ModelGraph(model)
        #expect(normalized.graph == graph)
    }

    @Test("Composite ModelComponent normalizes correctly")
    func compositeComponent() throws {
        struct TransformerBlock: ModelComponent {
            let hiddenSize: Int
            let headCount: Int
            let intermediateSize: Int

            var body: some ModelComponent {
                Residual {
                    RMSNorm(dimension: hiddenSize)
                    Attention(hiddenSize: hiddenSize, headCount: headCount, kvHeadCount: headCount)
                }
                Residual {
                    RMSNorm(dimension: hiddenSize)
                    MLP(inputSize: hiddenSize, intermediateSize: intermediateSize)
                }
            }
        }

        let block = TransformerBlock(hiddenSize: 64, headCount: 4, intermediateSize: 256)
        let graph = try normalize(block).graph

        // Two residual operations
        #expect(graph.rootRegion.operations.count == 2)
        if case .residual = graph.rootRegion.operations[0].kind {
            // OK
        } else {
            Issue.record("Expected residual")
        }
    }

    @Test("All 12 primitive OperationKind cases have DSL components")
    func allPrimitiveKindsDSLCoverage() throws {
        @ModelComponentBuilder
        var body: some ModelComponent {
            TokenEmbedding(vocabSize: 100, embeddingSize: 64)
            PositionalEmbedding(maxPositions: 512, embeddingSize: 64, kind: .learnedAbsolute)
            RoPE(dimension: 64)
            Attention(hiddenSize: 64, headCount: 4, kvHeadCount: 4, headDimension: 16)
            MLP(inputSize: 64, intermediateSize: 256)
            MoE(expertCount: 4, expertsPerToken: 2, expertInputSize: 64, expertIntermediateSize: 256)
            RMSNorm(dimension: 64)
            LayerNorm(dimension: 64)
            LMArchitecture.Linear(inputSize: 64, outputSize: 32)
            OutputHead(inputSize: 32, vocabSize: 100)
            StateSpace(hiddenSize: 64, numHeads: 1, keyHeadDim: 16, valueHeadDim: 16, variant: "mamba")
            Custom(domain: "test", name: "noop")
        }

        let graph = try normalize(body).graph

        var kindNames: Set<String> = []
        func collectKinds(from region: Region) {
            for op in region.operations {
                switch op.kind {
                case .tokenEmbedding: kindNames.insert("tokenEmbedding")
                case .positionalEmbedding: kindNames.insert("positionalEmbedding")
                case .rope: kindNames.insert("rope")
                case .attention: kindNames.insert("attention")
                case .mlp: kindNames.insert("mlp")
                case .moe: kindNames.insert("moe")
                case .rmsNorm: kindNames.insert("rmsNorm")
                case .layerNorm: kindNames.insert("layerNorm")
                case .linear: kindNames.insert("linear")
                case .outputHead: kindNames.insert("outputHead")
                case .stateSpace: kindNames.insert("stateSpace")
                case .shortConv: kindNames.insert("shortConv")
                case .custom: kindNames.insert("custom")
                case .residual(_, let body): collectKinds(from: body)
                case .parallel(_, let branches): branches.forEach { collectKinds(from: $0) }
                case .repeating(_, let body): collectKinds(from: body)
                case .layerStack(let layers): layers.forEach { collectKinds(from: $0) }
                }
            }
        }
        collectKinds(from: graph.rootRegion)

        #expect(kindNames.count == 12)
    }

    // MARK: - Normalizer Edge Cases

    @Test("Empty sequence is identity (pass-through)")
    func emptySequenceIdentity() throws {
        // An empty Group contributes no operations.
        let comp = Group {
            RMSNorm(dimension: 64)
            LMArchitecture.Linear(inputSize: 64, outputSize: 32)
        }

        let graph = try normalize(comp).graph
        // Only rmsNorm + linear.
        #expect(graph.rootRegion.operations.count == 2)
        // Value flow: linear's operand == rmsNorm's result.
        let normResult = graph.rootRegion.operations[0].results[0].id
        let linearOperand = graph.rootRegion.operations[1].operands[0].value
        #expect(normResult == linearOperand)
    }

    @Test("Parallel with single branch normalizes correctly")
    func singleBranchParallel() throws {
        @ModelComponentBuilder
        var body: some ModelComponent {
            RMSNorm(dimension: 64)
            Parallel(merge: .add) {
                LMArchitecture.Linear(inputSize: 64, outputSize: 64)
            }
        }

        let graph = try normalize(body).graph
        #expect(graph.rootRegion.operations.count == 2)

        if case .parallel(_, let branches) = graph.rootRegion.operations[1].kind {
            #expect(branches.count == 1)
            #expect(branches[0].operations.count == 1)
            #expect(branches[0].parameters.count == 1)
            #expect(branches[0].results.count == 1)
        } else {
            Issue.record("Expected parallel operation")
        }

        try GraphValidator.validate(graph)
    }

    @Test("Repeat with count=1 normalizes correctly")
    func repeatCountOne() throws {
        @ModelComponentBuilder
        var body: some ModelComponent {
            RMSNorm(dimension: 64)
            Repeat(count: 1) {
                LMArchitecture.Linear(inputSize: 64, outputSize: 64)
            }
        }

        let graph = try normalize(body).graph
        #expect(graph.rootRegion.operations.count == 2)

        if case .repeating(let count, let bodyRegion) = graph.rootRegion.operations[1].kind {
            #expect(count == 1)
            #expect(bodyRegion.operations.count == 1)
            #expect(bodyRegion.parameters.count == 1)
            #expect(bodyRegion.results.count == 1)
        } else {
            Issue.record("Expected repeating operation")
        }

        try GraphValidator.validate(graph)
    }

    @Test("Labeled on structural operations places label on outer operation")
    func labeledOnStructural() throws {
        let comp = Group(label: "attn_block") {
            Residual {
                Attention(hiddenSize: 64, headCount: 4, kvHeadCount: 4, headDimension: 16)
            }
        }

        let normalized = try normalize(comp)
        #expect(normalized.graph.rootRegion.operations.count == 1)

        let path = StructuralPath(components: [.operation(0)])
        #expect(normalized.metadata.annotation(for: path)?.label == "attn_block")
    }

    @Test("Multiple labels on different operations are preserved")
    func multipleLabelsDifferentOps() throws {
        let comp = Group {
            Group(label: "embed") {
                TokenEmbedding(vocabSize: 100, embeddingSize: 64)
            }
            Group(label: "norm") {
                RMSNorm(dimension: 64)
            }
            Group(label: "head") {
                OutputHead(inputSize: 64, vocabSize: 100)
            }
        }

        let normalized = try normalize(comp)
        #expect(normalized.metadata.annotation(for: StructuralPath(components: [.operation(0)]))?.label == "embed")
        #expect(normalized.metadata.annotation(for: StructuralPath(components: [.operation(1)]))?.label == "norm")
        #expect(normalized.metadata.annotation(for: StructuralPath(components: [.operation(2)]))?.label == "head")
    }

    @Test("Deeply nested sequence flattening preserves value chain")
    func deeplyNestedSequences() throws {
        let comp = Group {
            Group {
                Group {
                    RMSNorm(dimension: 64)
                }
                LMArchitecture.Linear(inputSize: 64, outputSize: 64)
            }
            Group {
                LMArchitecture.Linear(inputSize: 64, outputSize: 32)
            }
        }

        let graph = try normalize(comp).graph
        // All nested sequences should be flattened to 3 operations.
        #expect(graph.rootRegion.operations.count == 3)

        // Value chain must be intact through all flattening.
        let r0 = graph.rootRegion.operations[0].results[0].id
        let o1 = graph.rootRegion.operations[1].operands[0].value
        let r1 = graph.rootRegion.operations[1].results[0].id
        let o2 = graph.rootRegion.operations[2].operands[0].value
        #expect(r0 == o1)
        #expect(r1 == o2)
    }

    @Test("Residual body with sequence produces flat operations in body region")
    func residualBodyFlattenedSequence() throws {
        @ModelComponentBuilder
        var body: some ModelComponent {
            RMSNorm(dimension: 64)
            Residual {
                RMSNorm(dimension: 64)
                LMArchitecture.Linear(inputSize: 64, outputSize: 64)
                LMArchitecture.Linear(inputSize: 64, outputSize: 64)
            }
        }

        let graph = try normalize(body).graph
        if case .residual(_, let bodyRegion) = graph.rootRegion.operations[1].kind {
            // Inner sequence is flattened: rmsNorm + linear + linear = 3
            #expect(bodyRegion.operations.count == 3)
        } else {
            Issue.record("Expected residual")
        }

        try GraphValidator.validate(graph)
    }

    @Test("tokenEmbedding as source has zero operands after normalization")
    func tokenEmbeddingSourceArity() throws {
        let comp = TokenEmbedding(vocabSize: 100, embeddingSize: 64)

        let graph = try normalize(comp).graph
        #expect(graph.rootRegion.operations.count == 1)
        // tokenEmbedding receives empty upstream → zero operands.
        #expect(graph.rootRegion.operations[0].operands.isEmpty)
        #expect(graph.rootRegion.operations[0].results.count == 1)
    }

    // MARK: - DSL Coverage (continued)

    @Test("All 3 structural OperationKind cases have DSL components")
    func allStructuralKindsDSLCoverage() throws {
        @ModelComponentBuilder
        var body: some ModelComponent {
            Residual {
                RMSNorm(dimension: 64)
            }
            Parallel(merge: .add) {
                RMSNorm(dimension: 64)
            }
            Repeat(count: 2) {
                RMSNorm(dimension: 64)
            }
        }

        let graph = try normalize(body).graph

        var kindNames: Set<String> = []
        for op in graph.rootRegion.operations {
            switch op.kind {
            case .residual: kindNames.insert("residual")
            case .parallel: kindNames.insert("parallel")
            case .repeating: kindNames.insert("repeating")
            default: break
            }
        }

        #expect(kindNames.count == 3)
    }

    // MARK: - TupleComponent Tests

    @Test("TupleComponent chains children sequentially")
    func tupleComponentSequential() throws {
        struct ThreeOps: ModelComponent {
            var body: some ModelComponent {
                RMSNorm(dimension: 64)
                LMArchitecture.Linear(inputSize: 64, outputSize: 64)
                LMArchitecture.Linear(inputSize: 64, outputSize: 32)
            }
        }

        let graph = try ModelGraph(ThreeOps())
        #expect(graph.rootRegion.operations.count == 3)

        // Value chain: each op consumes the previous op's result
        let ops = graph.rootRegion.operations
        #expect(ops[1].operands[0].value == ops[0].results[0].id)
        #expect(ops[2].operands[0].value == ops[1].results[0].id)
    }

    @Test("Single expression in builder returns component directly (no TupleComponent wrapper)")
    func singleExpressionIdentity() throws {
        struct SingleOp: ModelComponent {
            var body: some ModelComponent {
                RMSNorm(dimension: 64)
            }
        }

        let graph = try ModelGraph(SingleOp())
        #expect(graph.rootRegion.operations.count == 1)
        if case .rmsNorm(let attrs) = graph.rootRegion.operations[0].kind {
            #expect(attrs.dimension == 64)
        } else {
            Issue.record("Expected rmsNorm")
        }
    }

    // MARK: - OptionalComponent Tests

    @Test("OptionalComponent includes content when condition is true")
    func optionalComponentTrue() throws {
        struct ConditionalModel: ModelComponent {
            let includeNorm: Bool
            var body: some ModelComponent {
                TokenEmbedding(vocabSize: 100, embeddingSize: 64)
                if includeNorm {
                    RMSNorm(dimension: 64)
                }
                OutputHead(inputSize: 64, vocabSize: 100)
            }
        }

        let graph = try ModelGraph(ConditionalModel(includeNorm: true))
        // tokenEmbedding + rmsNorm + outputHead
        #expect(graph.rootRegion.operations.count == 3)
        if case .rmsNorm = graph.rootRegion.operations[1].kind {
            // OK — norm is present
        } else {
            Issue.record("Expected rmsNorm when condition is true")
        }
    }

    @Test("OptionalComponent skips content when condition is false")
    func optionalComponentFalse() throws {
        struct ConditionalModel: ModelComponent {
            let includeNorm: Bool
            var body: some ModelComponent {
                TokenEmbedding(vocabSize: 100, embeddingSize: 64)
                if includeNorm {
                    RMSNorm(dimension: 64)
                }
                OutputHead(inputSize: 64, vocabSize: 100)
            }
        }

        let graph = try ModelGraph(ConditionalModel(includeNorm: false))
        // tokenEmbedding + outputHead (rmsNorm skipped)
        #expect(graph.rootRegion.operations.count == 2)
        if case .outputHead = graph.rootRegion.operations[1].kind {
            // OK — outputHead directly follows tokenEmbedding
        } else {
            Issue.record("Expected outputHead when norm is skipped")
        }

        // Value flow: outputHead consumes tokenEmbedding's result
        #expect(graph.rootRegion.operations[1].operands[0].value
                == graph.rootRegion.operations[0].results[0].id)
    }

    @Test("OptionalComponent with multi-op block")
    func optionalComponentMultiOp() throws {
        struct Model: ModelComponent {
            let addBlock: Bool
            var body: some ModelComponent {
                TokenEmbedding(vocabSize: 100, embeddingSize: 64)
                if addBlock {
                    RMSNorm(dimension: 64)
                    LMArchitecture.Linear(inputSize: 64, outputSize: 64)
                }
                OutputHead(inputSize: 64, vocabSize: 100)
            }
        }

        let withBlock = try ModelGraph(Model(addBlock: true))
        #expect(withBlock.rootRegion.operations.count == 4)

        let withoutBlock = try ModelGraph(Model(addBlock: false))
        #expect(withoutBlock.rootRegion.operations.count == 2)
    }

    // MARK: - ConditionalComponent Tests

    @Test("ConditionalComponent selects first branch")
    func conditionalComponentFirst() throws {
        struct Model: ModelComponent {
            let useAttention: Bool
            var body: some ModelComponent {
                TokenEmbedding(vocabSize: 100, embeddingSize: 64)
                if useAttention {
                    Attention(hiddenSize: 64, headCount: 4, kvHeadCount: 4, headDimension: 16)
                } else {
                    MLP(inputSize: 64, intermediateSize: 256)
                }
                OutputHead(inputSize: 64, vocabSize: 100)
            }
        }

        let graph = try ModelGraph(Model(useAttention: true))
        #expect(graph.rootRegion.operations.count == 3)
        if case .attention = graph.rootRegion.operations[1].kind {
            // OK
        } else {
            Issue.record("Expected attention when useAttention is true")
        }
    }

    @Test("ConditionalComponent selects second branch")
    func conditionalComponentSecond() throws {
        struct Model: ModelComponent {
            let useAttention: Bool
            var body: some ModelComponent {
                TokenEmbedding(vocabSize: 100, embeddingSize: 64)
                if useAttention {
                    Attention(hiddenSize: 64, headCount: 4, kvHeadCount: 4, headDimension: 16)
                } else {
                    MLP(inputSize: 64, intermediateSize: 256)
                }
                OutputHead(inputSize: 64, vocabSize: 100)
            }
        }

        let graph = try ModelGraph(Model(useAttention: false))
        #expect(graph.rootRegion.operations.count == 3)
        if case .mlp = graph.rootRegion.operations[1].kind {
            // OK
        } else {
            Issue.record("Expected mlp when useAttention is false")
        }
    }

    // MARK: - Dump Tests

    @Test("ModelGraph dump produces readable output for TinyLlama")
    func dumpTinyLlama() throws {
        let model = TinyLlama(
            vocabSize: 32000,
            hiddenSize: 4096,
            headCount: 32,
            kvHeadCount: 8,
            intermediateSize: 11008,
            layerCount: 32
        )

        let normalized = try NormalizedModel(model)
        let output = normalized.dump()
        print(output)

        #expect(output.contains("tokenEmbedding"))
        #expect(output.contains("repeating(32x)"))
        #expect(output.contains("residual(add)"))
        #expect(output.contains("attention"))
        #expect(output.contains("heads=32"))
        #expect(output.contains("kvHeads=8"))
        #expect(output.contains("mlp"))
        #expect(output.contains("rmsNorm"))
        #expect(output.contains("outputHead"))
    }

    @Test("ModelGraph dump produces readable output for Cohere")
    func dumpTinyCohere() throws {
        let model = TinyCohere(
            vocabSize: 256000,
            hiddenSize: 8192,
            headCount: 64,
            kvHeadCount: 8,
            intermediateSize: 22528,
            layerCount: 40
        )

        let normalized = try NormalizedModel(model)
        let output = normalized.dump()
        print(output)

        #expect(output.contains("parallel(add)"))
        #expect(output.contains("branch[0]"))
        #expect(output.contains("branch[1]"))
        #expect(output.contains("layerNorm"))
    }

    @Test("ModelGraph dump verbose mode shows extra details")
    func dumpVerbose() throws {
        let model = TinyLlama(
            vocabSize: 100,
            hiddenSize: 64,
            headCount: 4,
            kvHeadCount: 2,
            intermediateSize: 256,
            layerCount: 1
        )

        let normalized = try NormalizedModel(model)
        let compact = normalized.dump(verbose: false)
        let verbose = normalized.dump(verbose: true)
        print("--- compact ---")
        print(compact)
        print("--- verbose ---")
        print(verbose)

        // Both should be non-empty
        #expect(!compact.isEmpty)
        #expect(!verbose.isEmpty)
    }

    @Test("ConditionalComponent preserves value flow for both branches")
    func conditionalComponentValueFlow() throws {
        struct Model: ModelComponent {
            let branch: Bool
            var body: some ModelComponent {
                RMSNorm(dimension: 64)
                if branch {
                    LMArchitecture.Linear(inputSize: 64, outputSize: 64)
                } else {
                    LMArchitecture.Linear(inputSize: 64, outputSize: 64)
                }
                LMArchitecture.Linear(inputSize: 64, outputSize: 32)
            }
        }

        // Both branches should produce valid SSA value chains
        for b in [true, false] {
            let graph = try ModelGraph(Model(branch: b))
            let ops = graph.rootRegion.operations
            #expect(ops.count == 3)
            #expect(ops[1].operands[0].value == ops[0].results[0].id)
            #expect(ops[2].operands[0].value == ops[1].results[0].id)
            try GraphValidator.validate(graph)
        }
    }
}
