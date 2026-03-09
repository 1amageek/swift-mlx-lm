import Testing
import Foundation
@testable import SwiftLM

// MARK: - Example Model Declarations

/// Minimal Llama-style transformer defined with SwiftLM DSL.
struct TinyLlama: LanguageModel {
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
struct TinyCohere: LanguageModel {
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

@Suite("DSL Lowering")
struct DSLLoweringTests {

    // MARK: - Declaration Tree Tests

    @Test("Primitive component produces primitive declaration")
    func primitiveDeclaration() {
        let emb = TokenEmbedding(vocabSize: 100, embeddingSize: 64)
        let decl = emb.makeDeclaration()

        if case .primitive(.tokenEmbedding(let attrs)) = decl {
            #expect(attrs.vocabSize == 100)
            #expect(attrs.embeddingSize == 64)
        } else {
            Issue.record("Expected primitive(.tokenEmbedding)")
        }
    }

    @Test("Builder produces sequence declaration")
    func sequenceDeclaration() {
        @ModelComponentBuilder
        var body: ModelDeclaration {
            TokenEmbedding(vocabSize: 100, embeddingSize: 64)
            RMSNorm(dimension: 64)
            OutputHead(inputSize: 64, vocabSize: 100)
        }

        if case .sequence(let children) = body {
            #expect(children.count == 3)
        } else {
            Issue.record("Expected sequence")
        }
    }

    @Test("Residual wraps body in declaration tree")
    func residualDeclaration() {
        let r = Residual {
            RMSNorm(dimension: 64)
            Attention(hiddenSize: 64, headCount: 4, kvHeadCount: 4, headDimension: 16)
        }
        let decl = r.makeDeclaration()

        if case .residual(let strategy, let body) = decl {
            #expect(strategy == .add)
            if case .sequence(let children) = body {
                #expect(children.count == 2)
            } else {
                Issue.record("Expected sequence body in residual")
            }
        } else {
            Issue.record("Expected residual declaration")
        }
    }

    @Test("Parallel separates branches in declaration tree")
    func parallelDeclaration() {
        let p = Parallel(merge: .add) {
            Attention(hiddenSize: 64, headCount: 4, kvHeadCount: 4, headDimension: 16)
            MLP(inputSize: 64, intermediateSize: 256)
        }
        let decl = p.makeDeclaration()

        if case .parallel(let merge, let branches) = decl {
            #expect(merge == .add)
            #expect(branches.count == 2)
        } else {
            Issue.record("Expected parallel declaration")
        }
    }

    @Test("Repeat wraps body in declaration tree")
    func repeatDeclaration() {
        let r = Repeat(count: 12, label: "layers") {
            RMSNorm(dimension: 64)
        }
        let decl = r.makeDeclaration()

        if case .repeating(let count, let label, _) = decl {
            #expect(count == 12)
            #expect(label == "layers")
        } else {
            Issue.record("Expected repeating declaration")
        }
    }

    @Test("Group with label produces labeled declaration")
    func groupDeclaration() {
        let g = Group(label: "block") {
            RMSNorm(dimension: 64)
            Attention(hiddenSize: 64, headCount: 4, kvHeadCount: 4, headDimension: 16)
        }
        let decl = g.makeDeclaration()

        if case .labeled(let label, _) = decl {
            #expect(label == "block")
        } else {
            Issue.record("Expected labeled declaration")
        }
    }

    @Test("Group without label produces raw declaration")
    func groupWithoutLabel() {
        let g = Group {
            RMSNorm(dimension: 64)
        }
        let decl = g.makeDeclaration()

        if case .primitive(.rmsNorm) = decl {
            // OK
        } else {
            Issue.record("Expected primitive declaration (no wrapper)")
        }
    }

    // MARK: - Normalization Tests

    @Test("Normalization produces correct region structure with value flow")
    func normalizationBasic() throws {
        @ModelComponentBuilder
        var body: ModelDeclaration {
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
        let decl: ModelDeclaration = .sequence([
            .sequence([
                .primitive(.rmsNorm(RMSNormAttributes(dimension: 64))),
                .primitive(.rmsNorm(RMSNormAttributes(dimension: 64))),
            ]),
            .primitive(.linear(LinearAttributes(inputSize: 64, outputSize: 32))),
        ])

        let graph = try normalize(decl).graph
        #expect(graph.rootRegion.operations.count == 3)
    }

    @Test("Normalization extracts labels into metadata")
    func normalizationStripLabels() throws {
        let decl: ModelDeclaration = .labeled("debug", .sequence([
            .primitive(.rmsNorm(RMSNormAttributes(dimension: 64))),
            .labeled("inner", .primitive(.linear(LinearAttributes(inputSize: 64, outputSize: 32)))),
        ]))

        let normalized = try normalize(decl)

        #expect(normalized.graph.rootRegion.operations.count == 2)
        let firstPath = StructuralPath(components: [.operation(0)])
        let secondPath = StructuralPath(components: [.operation(1)])
        #expect(normalized.metadata.annotation(for: firstPath)?.label == "debug")
        #expect(normalized.metadata.annotation(for: secondPath)?.label == "inner")
    }

    @Test("Normalization creates nested regions for residual with value flow")
    func normalizationResidual() throws {
        @ModelComponentBuilder
        var body: ModelDeclaration {
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
        var body: ModelDeclaration {
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
        var body: ModelDeclaration {
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

        let graph = try model.makeModelGraph()

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

        let graph = try model.makeModelGraph()

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
        var bodyA: ModelDeclaration {
            TokenEmbedding(vocabSize: 100, embeddingSize: 64)
            RMSNorm(dimension: 64, epsilon: 1e-6)
            OutputHead(inputSize: 64, vocabSize: 100)
        }

        @ModelComponentBuilder
        var bodyB: ModelDeclaration {
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
        var bodyA: ModelDeclaration {
            TokenEmbedding(vocabSize: 100, embeddingSize: 64)
            OutputHead(inputSize: 64, vocabSize: 100)
        }

        @ModelComponentBuilder
        var bodyB: ModelDeclaration {
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

        let graph = try model.makeModelGraph()

        let encoder = JSONEncoder()
        encoder.outputFormatting = .sortedKeys
        let data = try encoder.encode(graph)
        let decoded = try JSONDecoder().decode(ModelGraph.self, from: data)

        #expect(graph == decoded)
    }

    @Test("LanguageModel.makeModelGraph() convenience method")
    func languageModelConvenience() throws {
        let model = TinyLlama(
            vocabSize: 100,
            hiddenSize: 64,
            headCount: 4,
            kvHeadCount: 2,
            intermediateSize: 256,
            layerCount: 1
        )

        let graph = try model.makeModelGraph()
        let manual = try normalize(model.body.makeDeclaration()).graph

        #expect(graph == manual)
    }

    @Test("LanguageModel.makeNormalizedModel() returns graph + metadata")
    func normalizedModelConvenience() throws {
        let model = TinyLlama(
            vocabSize: 100,
            hiddenSize: 64,
            headCount: 4,
            kvHeadCount: 2,
            intermediateSize: 256,
            layerCount: 1
        )

        let normalized = try model.makeNormalizedModel()
        let graph = try model.makeModelGraph()
        #expect(normalized.graph == graph)
    }

    @Test("CompositeModelComponent works as expected")
    func compositeComponent() {
        struct TransformerBlock: CompositeModelComponent {
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
        let decl = block.makeDeclaration()

        if case .sequence(let children) = decl {
            #expect(children.count == 2)
            if case .residual = children[0] {
                // OK
            } else {
                Issue.record("Expected residual")
            }
        } else {
            Issue.record("Expected sequence from composite component")
        }
    }

    @Test("All 12 primitive OperationKind cases have DSL components")
    func allPrimitiveKindsDSLCoverage() throws {
        @ModelComponentBuilder
        var body: ModelDeclaration {
            TokenEmbedding(vocabSize: 100, embeddingSize: 64)
            PositionalEmbedding(maxPositions: 512, embeddingSize: 64, kind: .learnedAbsolute)
            RoPE(dimension: 64)
            Attention(hiddenSize: 64, headCount: 4, kvHeadCount: 4, headDimension: 16)
            MLP(inputSize: 64, intermediateSize: 256)
            MoE(expertCount: 4, expertsPerToken: 2, expertInputSize: 64, expertIntermediateSize: 256)
            RMSNorm(dimension: 64)
            LayerNorm(dimension: 64)
            Linear(inputSize: 64, outputSize: 32)
            OutputHead(inputSize: 32, vocabSize: 100)
            StateSpace(hiddenSize: 64, stateSize: 16, variant: "mamba")
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
                case .custom: kindNames.insert("custom")
                case .residual(_, let body): collectKinds(from: body)
                case .parallel(_, let branches): branches.forEach { collectKinds(from: $0) }
                case .repeating(_, let body): collectKinds(from: body)
                }
            }
        }
        collectKinds(from: graph.rootRegion)

        #expect(kindNames.count == 12)
    }

    // MARK: - Normalizer Edge Cases

    @Test("Empty sequence is identity (pass-through)")
    func emptySequenceIdentity() throws {
        // Inner empty sequence should not produce operations; upstream flows through.
        let decl: ModelDeclaration = .sequence([
            .primitive(.rmsNorm(RMSNormAttributes(dimension: 64))),
            .sequence([]),
            .primitive(.linear(LinearAttributes(inputSize: 64, outputSize: 32))),
        ])

        let graph = try normalize(decl).graph
        // Empty sequence contributes no operations; only rmsNorm + linear.
        #expect(graph.rootRegion.operations.count == 2)
        // Value flow: linear's operand == rmsNorm's result.
        let normResult = graph.rootRegion.operations[0].results[0].id
        let linearOperand = graph.rootRegion.operations[1].operands[0].value
        #expect(normResult == linearOperand)
    }

    @Test("Parallel with single branch normalizes correctly")
    func singleBranchParallel() throws {
        let decl: ModelDeclaration = .sequence([
            .primitive(.rmsNorm(RMSNormAttributes(dimension: 64))),
            .parallel(merge: .add, branches: [
                .primitive(.linear(LinearAttributes(inputSize: 64, outputSize: 64))),
            ]),
        ])

        let graph = try normalize(decl).graph
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
        let decl: ModelDeclaration = .sequence([
            .primitive(.rmsNorm(RMSNormAttributes(dimension: 64))),
            .repeating(count: 1, label: nil, body:
                .primitive(.linear(LinearAttributes(inputSize: 64, outputSize: 64)))
            ),
        ])

        let graph = try normalize(decl).graph
        #expect(graph.rootRegion.operations.count == 2)

        if case .repeating(let count, let body) = graph.rootRegion.operations[1].kind {
            #expect(count == 1)
            #expect(body.operations.count == 1)
            #expect(body.parameters.count == 1)
            #expect(body.results.count == 1)
        } else {
            Issue.record("Expected repeating operation")
        }

        try GraphValidator.validate(graph)
    }

    @Test("Labeled on structural operations places label on outer operation")
    func labeledOnStructural() throws {
        let decl: ModelDeclaration = .labeled("attn_block",
            .residual(strategy: .add, body:
                .primitive(.attention(AttentionAttributes(
                    hiddenSize: 64, headCount: 4, kvHeadCount: 4, headDimension: 16
                )))
            )
        )

        let normalized = try normalize(decl)
        #expect(normalized.graph.rootRegion.operations.count == 1)

        let path = StructuralPath(components: [.operation(0)])
        #expect(normalized.metadata.annotation(for: path)?.label == "attn_block")
    }

    @Test("Multiple labels on different operations are preserved")
    func multipleLabelsDifferentOps() throws {
        let decl: ModelDeclaration = .sequence([
            .labeled("embed", .primitive(.tokenEmbedding(TokenEmbeddingAttributes(vocabSize: 100, embeddingSize: 64)))),
            .labeled("norm", .primitive(.rmsNorm(RMSNormAttributes(dimension: 64)))),
            .labeled("head", .primitive(.outputHead(OutputHeadAttributes(inputSize: 64, vocabSize: 100)))),
        ])

        let normalized = try normalize(decl)
        #expect(normalized.metadata.annotation(for: StructuralPath(components: [.operation(0)]))?.label == "embed")
        #expect(normalized.metadata.annotation(for: StructuralPath(components: [.operation(1)]))?.label == "norm")
        #expect(normalized.metadata.annotation(for: StructuralPath(components: [.operation(2)]))?.label == "head")
    }

    @Test("Deeply nested sequence flattening preserves value chain")
    func deeplyNestedSequences() throws {
        let decl: ModelDeclaration = .sequence([
            .sequence([
                .sequence([
                    .primitive(.rmsNorm(RMSNormAttributes(dimension: 64))),
                ]),
                .primitive(.linear(LinearAttributes(inputSize: 64, outputSize: 64))),
            ]),
            .sequence([
                .primitive(.linear(LinearAttributes(inputSize: 64, outputSize: 32))),
            ]),
        ])

        let graph = try normalize(decl).graph
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
        let decl: ModelDeclaration = .sequence([
            .primitive(.rmsNorm(RMSNormAttributes(dimension: 64))),
            .residual(strategy: .add, body: .sequence([
                .primitive(.rmsNorm(RMSNormAttributes(dimension: 64))),
                .sequence([
                    .primitive(.linear(LinearAttributes(inputSize: 64, outputSize: 64))),
                    .primitive(.linear(LinearAttributes(inputSize: 64, outputSize: 64))),
                ]),
            ])),
        ])

        let graph = try normalize(decl).graph
        if case .residual(_, let body) = graph.rootRegion.operations[1].kind {
            // Inner sequence is flattened: rmsNorm + linear + linear = 3
            #expect(body.operations.count == 3)
        } else {
            Issue.record("Expected residual")
        }

        try GraphValidator.validate(graph)
    }

    @Test("tokenEmbedding as source has zero operands after normalization")
    func tokenEmbeddingSourceArity() throws {
        let decl: ModelDeclaration = .primitive(
            .tokenEmbedding(TokenEmbeddingAttributes(vocabSize: 100, embeddingSize: 64))
        )

        let graph = try normalize(decl).graph
        #expect(graph.rootRegion.operations.count == 1)
        // tokenEmbedding receives empty upstream → zero operands.
        #expect(graph.rootRegion.operations[0].operands.isEmpty)
        #expect(graph.rootRegion.operations[0].results.count == 1)
    }

    // MARK: - DSL Coverage (continued)

    @Test("All 3 structural OperationKind cases have DSL components")
    func allStructuralKindsDSLCoverage() throws {
        @ModelComponentBuilder
        var body: ModelDeclaration {
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
}
