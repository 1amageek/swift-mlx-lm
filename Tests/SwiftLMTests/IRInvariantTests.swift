import Testing
import Foundation
@testable import LMArchitecture

// MARK: - Helpers

/// Collect all ValueIDs defined (produced) in a region, recursively.
private func allDefinedValues(in region: Region) -> Set<ValueID> {
    var ids = Set<ValueID>()
    for param in region.parameters {
        ids.insert(param.id)
    }
    for op in region.operations {
        for result in op.results {
            ids.insert(result.id)
        }
        switch op.kind {
        case .residual(_, let body):
            ids.formUnion(allDefinedValues(in: body))
        case .parallel(_, let branches):
            for branch in branches {
                ids.formUnion(allDefinedValues(in: branch))
            }
        case .repeating(_, let body):
            ids.formUnion(allDefinedValues(in: body))
        default:
            break
        }
    }
    return ids
}

/// Collect all ValueIDs used (consumed) in a region, recursively.
private func allUsedValues(in region: Region) -> Set<ValueID> {
    var ids = Set<ValueID>()
    for result in region.results {
        ids.insert(result.value)
    }
    for op in region.operations {
        for operand in op.operands {
            ids.insert(operand.value)
        }
        switch op.kind {
        case .residual(_, let body):
            ids.formUnion(allUsedValues(in: body))
        case .parallel(_, let branches):
            for branch in branches {
                ids.formUnion(allUsedValues(in: branch))
            }
        case .repeating(_, let body):
            ids.formUnion(allUsedValues(in: body))
        default:
            break
        }
    }
    return ids
}

/// Verify the SSA def-use chain within a region (non-recursive).
/// Returns true if every operand and region result in this region
/// references a value defined by a parameter or a prior operation's result.
private func verifyLocalSSA(_ region: Region) -> Bool {
    var scope = Set<ValueID>()
    for param in region.parameters {
        scope.insert(param.id)
    }
    for op in region.operations {
        for operand in op.operands {
            if !scope.contains(operand.value) { return false }
        }
        for result in op.results {
            scope.insert(result.id)
        }
    }
    for result in region.results {
        if !scope.contains(result.value) { return false }
    }
    return true
}

/// Recursively verify SSA for all regions in a graph.
private func verifySSARecursive(_ region: Region) -> Bool {
    guard verifyLocalSSA(region) else { return false }
    for op in region.operations {
        switch op.kind {
        case .residual(_, let body):
            if !verifySSARecursive(body) { return false }
        case .parallel(_, let branches):
            for branch in branches {
                if !verifySSARecursive(branch) { return false }
            }
        case .repeating(_, let body):
            if !verifySSARecursive(body) { return false }
        default:
            break
        }
    }
    return true
}

/// All ValueIDs in a graph are unique (no two definitions share an ID).
private func allValueIDsUnique(in region: Region) -> Bool {
    var seen = Set<ValueID>()
    return checkUniqueness(region, seen: &seen)
}

private func checkUniqueness(_ region: Region, seen: inout Set<ValueID>) -> Bool {
    for param in region.parameters {
        if !seen.insert(param.id).inserted { return false }
    }
    for op in region.operations {
        for result in op.results {
            if !seen.insert(result.id).inserted { return false }
        }
        switch op.kind {
        case .residual(_, let body):
            if !checkUniqueness(body, seen: &seen) { return false }
        case .parallel(_, let branches):
            for branch in branches {
                if !checkUniqueness(branch, seen: &seen) { return false }
            }
        case .repeating(_, let body):
            if !checkUniqueness(body, seen: &seen) { return false }
        default:
            break
        }
    }
    return true
}

// MARK: - Architecture Declarations

/// Mixtral-style MoE model.
struct TinyMixtral: ModelComponent {
    let vocabSize: Int
    let hiddenSize: Int
    let headCount: Int
    let kvHeadCount: Int
    let intermediateSize: Int
    let expertCount: Int
    let expertsPerToken: Int
    let layerCount: Int

    @ModelComponentBuilder var body: some ModelComponent {
        TokenEmbedding(vocabSize: vocabSize, embeddingSize: hiddenSize)

        Repeat(count: layerCount, label: "layers") {
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
                MoE(
                    expertCount: expertCount,
                    expertsPerToken: expertsPerToken,
                    expertInputSize: hiddenSize,
                    expertIntermediateSize: intermediateSize
                )
            }
        }

        RMSNorm(dimension: hiddenSize)
        OutputHead(inputSize: hiddenSize, vocabSize: vocabSize)
    }
}

/// Jamba-style hybrid model (Mamba + Attention interleaved).
struct TinyJamba: ModelComponent {
    let vocabSize: Int
    let hiddenSize: Int
    let headCount: Int
    let kvHeadCount: Int
    let intermediateSize: Int
    let stateSize: Int

    @ModelComponentBuilder var body: some ModelComponent {
        TokenEmbedding(vocabSize: vocabSize, embeddingSize: hiddenSize)

        // Layer 1: Mamba block
        Residual {
            RMSNorm(dimension: hiddenSize)
            StateSpace(hiddenSize: hiddenSize, numHeads: 1, keyHeadDim: stateSize, valueHeadDim: stateSize, variant: "mamba")
        }
        Residual {
            RMSNorm(dimension: hiddenSize)
            MLP(inputSize: hiddenSize, intermediateSize: intermediateSize)
        }

        // Layer 2: Attention block
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

        RMSNorm(dimension: hiddenSize)
        OutputHead(inputSize: hiddenSize, vocabSize: vocabSize)
    }
}

// MARK: - Invariant Tests

@Suite("IR Invariants", .tags(.unit))
struct IRInvariantTests {

    // MARK: - Fundamental: normalize always produces valid graphs

    @Test("Invariant: normalize → GraphValidator.validate always succeeds")
    func normalizeAlwaysValid() throws {
        // Every well-formed ModelComponent, when normalized,
        // must produce a graph that passes structural validation.
        let components: [any ModelComponent] = [
            // Single primitive
            RMSNorm(dimension: 64),

            // Sequence of primitives
            Group {
                TokenEmbedding(vocabSize: 100, embeddingSize: 64)
                RMSNorm(dimension: 64)
                OutputHead(inputSize: 64, vocabSize: 100)
            },

            // Residual
            Group {
                RMSNorm(dimension: 64)
                Residual {
                    LMArchitecture.Linear(inputSize: 64, outputSize: 64)
                }
            },

            // Parallel
            Group {
                RMSNorm(dimension: 64)
                Parallel(merge: .add) {
                    LMArchitecture.Linear(inputSize: 64, outputSize: 64)
                    LMArchitecture.Linear(inputSize: 64, outputSize: 64)
                }
            },

            // Repeating
            Group {
                RMSNorm(dimension: 64)
                Repeat(count: 3) {
                    LMArchitecture.Linear(inputSize: 64, outputSize: 64)
                }
            },

            // Deeply nested
            Group {
                TokenEmbedding(vocabSize: 100, embeddingSize: 64)
                Repeat(count: 2) {
                    Residual {
                        RMSNorm(dimension: 64)
                        Parallel(merge: .add) {
                            Attention(hiddenSize: 64, headCount: 4, kvHeadCount: 4, headDimension: 16)
                            MLP(inputSize: 64, intermediateSize: 256)
                        }
                    }
                }
                OutputHead(inputSize: 64, vocabSize: 100)
            },

            // Empty sequence (identity)
            Group {
                RMSNorm(dimension: 64)
                Group { }
                LMArchitecture.Linear(inputSize: 64, outputSize: 64)
            },

            // Single-branch parallel
            Group {
                RMSNorm(dimension: 64)
                Parallel(merge: .concat) {
                    LMArchitecture.Linear(inputSize: 64, outputSize: 64)
                }
            },

            // Labeled wrapping
            Group(label: "model") {
                Group(label: "embed") {
                    TokenEmbedding(vocabSize: 100, embeddingSize: 64)
                }
                OutputHead(inputSize: 64, vocabSize: 100)
            },
        ]

        for (i, component) in components.enumerated() {
            let graph = try normalize(component).graph
            do {
                try GraphValidator.validate(graph)
            } catch {
                Issue.record("Component \(i) failed validation: \(error)")
            }
        }
    }

    @Test("Invariant: normalize → canonicalize → GraphValidator.validate always succeeds")
    func normalizeCanonicalizeAlwaysValid() throws {
        let models: [any ModelComponent] = [
            TinyLlama(vocabSize: 100, hiddenSize: 64, headCount: 4, kvHeadCount: 2, intermediateSize: 256, layerCount: 2),
            TinyCohere(vocabSize: 100, hiddenSize: 64, headCount: 4, kvHeadCount: 2, intermediateSize: 256, layerCount: 2),
            TinyMixtral(vocabSize: 100, hiddenSize: 64, headCount: 4, kvHeadCount: 2, intermediateSize: 256, expertCount: 4, expertsPerToken: 2, layerCount: 1),
            TinyJamba(vocabSize: 100, hiddenSize: 64, headCount: 4, kvHeadCount: 2, intermediateSize: 256, stateSize: 16),
        ]

        for model in models {
            let graph = try model.makeModelGraph()
            try GraphValidator.validate(graph)

            let canon = canonicalize(graph)
            try GraphValidator.validate(canon)
        }
    }

    // MARK: - SSA Invariants

    @Test("Invariant: normalized graph has globally unique ValueIDs")
    func normalizedGraphUniqueIDs() throws {
        let models: [any ModelComponent] = [
            TinyLlama(vocabSize: 100, hiddenSize: 64, headCount: 4, kvHeadCount: 2, intermediateSize: 256, layerCount: 2),
            TinyCohere(vocabSize: 100, hiddenSize: 64, headCount: 4, kvHeadCount: 2, intermediateSize: 256, layerCount: 2),
            TinyMixtral(vocabSize: 100, hiddenSize: 64, headCount: 4, kvHeadCount: 2, intermediateSize: 256, expertCount: 4, expertsPerToken: 2, layerCount: 1),
        ]

        for model in models {
            let graph = try model.makeModelGraph()
            #expect(allValueIDsUnique(in: graph.rootRegion))
        }
    }

    @Test("Invariant: normalized graph satisfies SSA def-use in all regions")
    func normalizedGraphSSA() throws {
        let models: [any ModelComponent] = [
            TinyLlama(vocabSize: 100, hiddenSize: 64, headCount: 4, kvHeadCount: 2, intermediateSize: 256, layerCount: 3),
            TinyCohere(vocabSize: 100, hiddenSize: 64, headCount: 4, kvHeadCount: 2, intermediateSize: 256, layerCount: 2),
            TinyMixtral(vocabSize: 100, hiddenSize: 64, headCount: 4, kvHeadCount: 2, intermediateSize: 256, expertCount: 8, expertsPerToken: 2, layerCount: 2),
            TinyJamba(vocabSize: 100, hiddenSize: 64, headCount: 4, kvHeadCount: 2, intermediateSize: 256, stateSize: 16),
        ]

        for model in models {
            let graph = try model.makeModelGraph()
            #expect(verifySSARecursive(graph.rootRegion))
        }
    }

    @Test("Invariant: canonicalized graph preserves SSA and unique IDs")
    func canonicalizedGraphSSA() throws {
        let model = TinyMixtral(
            vocabSize: 100, hiddenSize: 64, headCount: 4, kvHeadCount: 2,
            intermediateSize: 256, expertCount: 4, expertsPerToken: 2, layerCount: 2
        )
        let graph = try model.makeModelGraph()
        let canon = canonicalize(graph)

        #expect(verifySSARecursive(canon.rootRegion))
        #expect(allValueIDsUnique(in: canon.rootRegion))
    }

    // MARK: - Value Flow Topology

    @Test("Value flow: linear chain threads each output to next input")
    func linearChainValueFlow() throws {
        let component = Group {
            TokenEmbedding(vocabSize: 100, embeddingSize: 64)
            RMSNorm(dimension: 64)
            LMArchitecture.Linear(inputSize: 64, outputSize: 64)
            OutputHead(inputSize: 64, vocabSize: 100)
        }

        let graph = try normalize(component).graph
        let ops = graph.rootRegion.operations

        // tokenEmbedding is source: no operands
        #expect(ops[0].operands.isEmpty)

        // Each subsequent op consumes the previous op's result
        for i in 1..<ops.count {
            #expect(ops[i].operands.count == 1)
            #expect(ops[i].operands[0].value == ops[i - 1].results[0].id)
        }

        // Region result is the last op's result
        #expect(graph.rootRegion.results[0].value == ops.last!.results[0].id)
    }

    @Test("Value flow: residual operation's operand becomes body parameter")
    func residualValueFlow() throws {
        let component = Group {
            RMSNorm(dimension: 64)
            Residual {
                LMArchitecture.Linear(inputSize: 64, outputSize: 64)
                LMArchitecture.Linear(inputSize: 64, outputSize: 64)
            }
        }

        let graph = try normalize(component).graph
        let normOp = graph.rootRegion.operations[0]
        let residualOp = graph.rootRegion.operations[1]

        // Residual consumes norm's result
        #expect(residualOp.operands[0].value == normOp.results[0].id)

        // Body parameter is a FRESH value (not the same as the operand)
        guard case .residual(_, let body) = residualOp.kind else {
            Issue.record("Expected residual")
            return
        }
        let bodyParam = body.parameters[0].id
        #expect(bodyParam != residualOp.operands[0].value)

        // Body's first op consumes the body parameter
        #expect(body.operations[0].operands[0].value == bodyParam)

        // Body chain: linear0 → linear1
        #expect(body.operations[1].operands[0].value == body.operations[0].results[0].id)

        // Body result is the last body op's result
        #expect(body.results[0].value == body.operations[1].results[0].id)

        // Residual produces a new result (distinct from both operand and body result)
        #expect(residualOp.results[0].id != residualOp.operands[0].value)
        #expect(residualOp.results[0].id != body.results[0].value)
    }

    @Test("Value flow: parallel forks input to each branch via separate parameters")
    func parallelValueFlow() throws {
        let component = Group {
            RMSNorm(dimension: 64)
            Parallel(merge: .add) {
                Attention(hiddenSize: 64, headCount: 4, kvHeadCount: 4, headDimension: 16)
                MLP(inputSize: 64, intermediateSize: 256)
            }
        }

        let graph = try normalize(component).graph
        let normOp = graph.rootRegion.operations[0]
        let parallelOp = graph.rootRegion.operations[1]

        // Parallel consumes norm's result
        #expect(parallelOp.operands[0].value == normOp.results[0].id)

        guard case .parallel(_, let branches) = parallelOp.kind else {
            Issue.record("Expected parallel")
            return
        }

        // Each branch gets its OWN parameter (scope isolation)
        let param0 = branches[0].parameters[0].id
        let param1 = branches[1].parameters[0].id
        #expect(param0 != param1)

        // Neither branch parameter is the same as the parallel's operand
        #expect(param0 != parallelOp.operands[0].value)
        #expect(param1 != parallelOp.operands[0].value)

        // Each branch's first op consumes its own parameter
        #expect(branches[0].operations[0].operands[0].value == param0)
        #expect(branches[1].operations[0].operands[0].value == param1)

        // Each branch yields its own result
        #expect(branches[0].results[0].value == branches[0].operations[0].results[0].id)
        #expect(branches[1].results[0].value == branches[1].operations[0].results[0].id)

        // Parallel produces a single merged result (distinct from all branch values)
        #expect(parallelOp.results.count == 1)
        let allBranchValues = allDefinedValues(in: branches[0]).union(allDefinedValues(in: branches[1]))
        #expect(!allBranchValues.contains(parallelOp.results[0].id))
    }

    @Test("Value flow: repeating body results feed back as next iteration parameters")
    func repeatingValueFlow() throws {
        let component = Group {
            RMSNorm(dimension: 64)
            Repeat(count: 4) {
                LMArchitecture.Linear(inputSize: 64, outputSize: 64)
                LMArchitecture.Linear(inputSize: 64, outputSize: 64)
            }
        }

        let graph = try normalize(component).graph
        let repeatOp = graph.rootRegion.operations[1]

        guard case .repeating(let count, let body) = repeatOp.kind else {
            Issue.record("Expected repeating")
            return
        }
        #expect(count == 4)

        // Loop-carried: body params == body results in count
        #expect(body.parameters.count == body.results.count)
        // Also: operands == results in count
        #expect(repeatOp.operands.count == repeatOp.results.count)
        // And: params == operands in count
        #expect(body.parameters.count == repeatOp.operands.count)

        // Body chain is intact
        #expect(body.operations[0].operands[0].value == body.parameters[0].id)
        #expect(body.operations[1].operands[0].value == body.operations[0].results[0].id)
        #expect(body.results[0].value == body.operations[1].results[0].id)
    }

    // MARK: - Scope Isolation

    @Test("Scope isolation: nested region values do not leak to parent")
    func scopeIsolationNoLeak() throws {
        let model = TinyLlama(
            vocabSize: 100, hiddenSize: 64, headCount: 4,
            kvHeadCount: 2, intermediateSize: 256, layerCount: 2
        )
        let graph = try model.makeModelGraph()

        // Collect values defined inside nested regions
        var nestedValues = Set<ValueID>()
        for op in graph.rootRegion.operations {
            switch op.kind {
            case .residual(_, let body):
                nestedValues.formUnion(allDefinedValues(in: body))
            case .parallel(_, let branches):
                for branch in branches {
                    nestedValues.formUnion(allDefinedValues(in: branch))
                }
            case .repeating(_, let body):
                nestedValues.formUnion(allDefinedValues(in: body))
            default:
                break
            }
        }

        // Root-level operands and results must NOT reference nested values
        for op in graph.rootRegion.operations {
            for operand in op.operands {
                #expect(!nestedValues.contains(operand.value),
                    "Root operand \(operand.value) leaks from nested region")
            }
        }
        for result in graph.rootRegion.results {
            #expect(!nestedValues.contains(result.value),
                "Root result \(result.value) leaks from nested region")
        }
    }

    // MARK: - Normalization Determinism

    @Test("Normalization is deterministic: same input always produces same output")
    func normalizationDeterminism() throws {
        let model = TinyMixtral(
            vocabSize: 100, hiddenSize: 64, headCount: 4, kvHeadCount: 2,
            intermediateSize: 256, expertCount: 4, expertsPerToken: 2, layerCount: 2
        )

        let graph1 = try normalize(model).graph
        let graph2 = try normalize(model).graph

        #expect(graph1 == graph2)
    }

    // MARK: - Canonicalization Semantic Preservation

    @Test("Canonicalization preserves operation count and kind order")
    func canonicalizationPreservesStructure() throws {
        let model = TinyCohere(
            vocabSize: 100, hiddenSize: 64, headCount: 4,
            kvHeadCount: 2, intermediateSize: 256, layerCount: 2
        )
        let graph = try model.makeModelGraph()
        let canon = canonicalize(graph)

        // Same number of root operations
        #expect(graph.rootRegion.operations.count == canon.rootRegion.operations.count)

        // Same kind order (ignoring attribute detail changes like hint stripping)
        for (orig, can) in zip(graph.rootRegion.operations, canon.rootRegion.operations) {
            switch (orig.kind, can.kind) {
            case (.tokenEmbedding, .tokenEmbedding),
                 (.rmsNorm, .rmsNorm),
                 (.layerNorm, .layerNorm),
                 (.attention, .attention),
                 (.mlp, .mlp),
                 (.linear, .linear),
                 (.outputHead, .outputHead),
                 (.residual, .residual),
                 (.parallel, .parallel),
                 (.repeating, .repeating):
                break // same kind
            default:
                Issue.record("Kind mismatch: \(orig.kind) vs \(can.kind)")
            }
        }
    }

    @Test("Canonicalization preserves value flow topology")
    func canonicalizationPreservesTopology() throws {
        let component = Group {
            TokenEmbedding(vocabSize: 100, embeddingSize: 64)
            RMSNorm(dimension: 64)
            Residual {
                LMArchitecture.Linear(inputSize: 64, outputSize: 64)
            }
            OutputHead(inputSize: 64, vocabSize: 100)
        }

        let graph = try normalize(component).graph
        let canon = canonicalize(graph)

        // Both have same operation count
        #expect(graph.rootRegion.operations.count == canon.rootRegion.operations.count)

        // Both have same operand count per operation (topology is preserved)
        for (orig, can) in zip(graph.rootRegion.operations, canon.rootRegion.operations) {
            #expect(orig.operands.count == can.operands.count)
            #expect(orig.results.count == can.results.count)
        }

        // Canonicalized graph also has intact SSA chain
        let ops = canon.rootRegion.operations
        #expect(ops[0].operands.isEmpty) // tokenEmbedding: source
        #expect(ops[1].operands[0].value == ops[0].results[0].id) // norm ← embed
        #expect(ops[2].operands[0].value == ops[1].results[0].id) // residual ← norm
        #expect(ops[3].operands[0].value == ops[2].results[0].id) // head ← residual
    }

    // MARK: - Real Architecture Patterns

    @Test("Mixtral-style MoE model produces valid graph")
    func mixtralArchitecture() throws {
        let model = TinyMixtral(
            vocabSize: 32000, hiddenSize: 4096, headCount: 32,
            kvHeadCount: 8, intermediateSize: 14336,
            expertCount: 8, expertsPerToken: 2, layerCount: 2
        )

        let graph = try model.makeModelGraph()
        try GraphValidator.validate(graph)
        try LLMProfileValidator.validate(graph)

        // Root: tokenEmbedding, repeat, rmsNorm, outputHead
        #expect(graph.rootRegion.operations.count == 4)

        // Repeat body: 2 residuals (attn + moe)
        if case .repeating(let count, let body) = graph.rootRegion.operations[1].kind {
            #expect(count == 2)
            #expect(body.operations.count == 2)

            // Second residual body contains MoE
            if case .residual(_, let resBody) = body.operations[1].kind {
                #expect(resBody.operations.count == 2) // rmsNorm + moe
                if case .moe(let attrs) = resBody.operations[1].kind {
                    #expect(attrs.expertCount == 8)
                    #expect(attrs.expertsPerToken == 2)
                } else {
                    Issue.record("Expected moe in second residual")
                }
            } else {
                Issue.record("Expected residual")
            }
        } else {
            Issue.record("Expected repeating")
        }
    }

    @Test("Jamba-style hybrid model (Mamba + Attention) produces valid graph")
    func jambaArchitecture() throws {
        let model = TinyJamba(
            vocabSize: 65536, hiddenSize: 4096, headCount: 32,
            kvHeadCount: 8, intermediateSize: 14336, stateSize: 16
        )

        let graph = try model.makeModelGraph()
        try GraphValidator.validate(graph)
        try LLMProfileValidator.validate(graph)

        // Root: tokenEmbedding, residual(mamba), residual(mlp),
        //       residual(attn), residual(mlp), rmsNorm, outputHead
        #expect(graph.rootRegion.operations.count == 7)

        // First residual body contains stateSpace (Mamba)
        if case .residual(_, let body) = graph.rootRegion.operations[1].kind {
            if case .stateSpace(let attrs) = body.operations[1].kind {
                #expect(attrs.variant == "mamba")
                #expect(attrs.keyHeadDim == 16)
            } else {
                Issue.record("Expected stateSpace in first residual")
            }
        } else {
            Issue.record("Expected residual")
        }

        // Third residual body contains attention
        if case .residual(_, let body) = graph.rootRegion.operations[3].kind {
            if case .attention(let attrs) = body.operations[1].kind {
                #expect(attrs.headCount == 32)
                #expect(attrs.kvHeadCount == 8)
            } else {
                Issue.record("Expected attention in third residual")
            }
        } else {
            Issue.record("Expected residual")
        }
    }

    @Test("Mixtral and Jamba produce canonically distinct graphs")
    func differentArchitecturesNotEqual() throws {
        let mixtral = TinyMixtral(
            vocabSize: 100, hiddenSize: 64, headCount: 4, kvHeadCount: 2,
            intermediateSize: 256, expertCount: 4, expertsPerToken: 2, layerCount: 1
        )
        let jamba = TinyJamba(
            vocabSize: 100, hiddenSize: 64, headCount: 4, kvHeadCount: 2,
            intermediateSize: 256, stateSize: 16
        )

        let g1 = canonicalize(try mixtral.makeModelGraph())
        let g2 = canonicalize(try jamba.makeModelGraph())

        #expect(g1 != g2)
    }

    // MARK: - Multi-value Flow

    @Test("Custom operation with variadic arity supports multi-value flow")
    func multiValueCustomOp() throws {
        // Build a graph manually with a custom op that produces 2 results.
        let customOp = Operation(
            key: OperationKey(rawValue: 0),
            kind: .custom(CustomNodeAttributes(domain: "test", name: "split")),
            operands: [],
            results: [
                OperationResult(id: ValueID(rawValue: 0)),
                OperationResult(id: ValueID(rawValue: 1)),
            ]
        )
        // Two downstream ops each consume one of the custom op's results.
        let consumer0 = Operation(
            key: OperationKey(rawValue: 1),
            kind: .rmsNorm(RMSNormAttributes(dimension: 64)),
            operands: [Operand(value: ValueID(rawValue: 0))],
            results: [OperationResult(id: ValueID(rawValue: 2))]
        )
        let consumer1 = Operation(
            key: OperationKey(rawValue: 2),
            kind: .rmsNorm(RMSNormAttributes(dimension: 64)),
            operands: [Operand(value: ValueID(rawValue: 1))],
            results: [OperationResult(id: ValueID(rawValue: 3))]
        )

        let graph = ModelGraph(rootRegion: Region(
            operations: [customOp, consumer0, consumer1],
            results: [
                ValueUse(value: ValueID(rawValue: 2)),
                ValueUse(value: ValueID(rawValue: 3)),
            ]
        ))

        try GraphValidator.validate(graph)
        #expect(verifySSARecursive(graph.rootRegion))
        #expect(allValueIDsUnique(in: graph.rootRegion))
    }

    @Test("Multi-value flow through residual body")
    func multiValueResidual() throws {
        // Residual with 2-wide value tuple
        let body = Region(
            parameters: [
                RegionParameter(id: ValueID(rawValue: 10)),
                RegionParameter(id: ValueID(rawValue: 11)),
            ],
            operations: [
                Operation(
                    key: OperationKey(rawValue: 0),
                    kind: .linear(LinearAttributes(inputSize: 64, outputSize: 64)),
                    operands: [Operand(value: ValueID(rawValue: 10))],
                    results: [OperationResult(id: ValueID(rawValue: 12))]
                ),
                Operation(
                    key: OperationKey(rawValue: 1),
                    kind: .linear(LinearAttributes(inputSize: 64, outputSize: 64)),
                    operands: [Operand(value: ValueID(rawValue: 11))],
                    results: [OperationResult(id: ValueID(rawValue: 13))]
                ),
            ],
            results: [
                ValueUse(value: ValueID(rawValue: 12)),
                ValueUse(value: ValueID(rawValue: 13)),
            ]
        )

        let graph = ModelGraph(rootRegion: Region(
            operations: [
                // Source: custom op producing 2 values
                Operation(
                    key: OperationKey(rawValue: 2),
                    kind: .custom(CustomNodeAttributes(domain: "test", name: "source")),
                    results: [
                        OperationResult(id: ValueID(rawValue: 0)),
                        OperationResult(id: ValueID(rawValue: 1)),
                    ]
                ),
                // Residual with 2-wide operand/result tuple
                Operation(
                    key: OperationKey(rawValue: 3),
                    kind: .residual(strategy: .add, body: body),
                    operands: [
                        Operand(value: ValueID(rawValue: 0)),
                        Operand(value: ValueID(rawValue: 1)),
                    ],
                    results: [
                        OperationResult(id: ValueID(rawValue: 2)),
                        OperationResult(id: ValueID(rawValue: 3)),
                    ]
                ),
            ],
            results: [
                ValueUse(value: ValueID(rawValue: 2)),
                ValueUse(value: ValueID(rawValue: 3)),
            ]
        ))

        try GraphValidator.validate(graph)
        #expect(graph.rootRegion.operations[1].operands.count == 2)
        #expect(graph.rootRegion.operations[1].results.count == 2)
    }

    // MARK: - Codable Roundtrip Invariant

    @Test("Invariant: Codable roundtrip preserves graph equality for all architectures")
    func codableRoundtripAllArchitectures() throws {
        let models: [any ModelComponent] = [
            TinyLlama(vocabSize: 100, hiddenSize: 64, headCount: 4, kvHeadCount: 2, intermediateSize: 256, layerCount: 2),
            TinyCohere(vocabSize: 100, hiddenSize: 64, headCount: 4, kvHeadCount: 2, intermediateSize: 256, layerCount: 1),
            TinyMixtral(vocabSize: 100, hiddenSize: 64, headCount: 4, kvHeadCount: 2, intermediateSize: 256, expertCount: 4, expertsPerToken: 2, layerCount: 1),
            TinyJamba(vocabSize: 100, hiddenSize: 64, headCount: 4, kvHeadCount: 2, intermediateSize: 256, stateSize: 16),
        ]

        let encoder = JSONEncoder()
        encoder.outputFormatting = .sortedKeys
        let decoder = JSONDecoder()

        for model in models {
            let graph = try model.makeModelGraph()
            let data = try encoder.encode(graph)
            let decoded = try decoder.decode(ModelGraph.self, from: data)
            #expect(graph == decoded)
        }
    }

    // MARK: - LLMProfileValidator on Real Architectures

    @Test("LLMProfileValidator accepts all real architecture patterns")
    func profileValidatorAllArchitectures() throws {
        let models: [any ModelComponent] = [
            TinyLlama(vocabSize: 100, hiddenSize: 64, headCount: 4, kvHeadCount: 2, intermediateSize: 256, layerCount: 2),
            TinyCohere(vocabSize: 100, hiddenSize: 64, headCount: 4, kvHeadCount: 2, intermediateSize: 256, layerCount: 1),
            TinyMixtral(vocabSize: 100, hiddenSize: 64, headCount: 4, kvHeadCount: 2, intermediateSize: 256, expertCount: 4, expertsPerToken: 2, layerCount: 1),
            TinyJamba(vocabSize: 100, hiddenSize: 64, headCount: 4, kvHeadCount: 2, intermediateSize: 256, stateSize: 16),
        ]

        for model in models {
            let graph = try model.makeModelGraph()
            try GraphValidator.validate(graph)
            try LLMProfileValidator.validate(graph)
        }
    }
}
