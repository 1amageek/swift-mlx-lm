import Testing
import Foundation
@testable import SwiftLM

@Suite("ModelGraph")
struct ModelGraphTests {

    @Test("Empty graph has empty root region")
    func emptyGraph() {
        let graph = ModelGraph(rootRegion: Region())
        #expect(graph.rootRegion.operations.isEmpty)
        #expect(graph.rootRegion.parameters.isEmpty)
        #expect(graph.rootRegion.results.isEmpty)
    }

    @Test("Region preserves operation order")
    func regionOrder() {
        let region = Region(operations: [
            Operation(
                key: OperationKey(rawValue: 0),
                kind: .rmsNorm(RMSNormAttributes(dimension: 64)),
                results: [OperationResult(id: ValueID(rawValue: 0))]
            ),
            Operation(
                key: OperationKey(rawValue: 1),
                kind: .linear(LinearAttributes(inputSize: 64, outputSize: 32)),
                operands: [Operand(value: ValueID(rawValue: 0))],
                results: [OperationResult(id: ValueID(rawValue: 1))]
            ),
        ])
        #expect(region.operations.count == 2)
        #expect(region.operations[0].key == OperationKey(rawValue: 0))
        #expect(region.operations[1].key == OperationKey(rawValue: 1))
    }

    @Test("Region-bearing operation contains nested region with value flow")
    func regionBearingOperation() {
        let bodyParam = ValueID(rawValue: 0)
        let bodyRegion = Region(
            parameters: [RegionParameter(id: bodyParam)],
            operations: [
                Operation(
                    key: OperationKey(rawValue: 0),
                    kind: .rmsNorm(RMSNormAttributes(dimension: 64)),
                    operands: [Operand(value: bodyParam)],
                    results: [OperationResult(id: ValueID(rawValue: 1))]
                ),
                Operation(
                    key: OperationKey(rawValue: 1),
                    kind: .attention(AttentionAttributes(
                        hiddenSize: 64, headCount: 4, kvHeadCount: 4, headDimension: 16
                    )),
                    operands: [Operand(value: ValueID(rawValue: 1))],
                    results: [OperationResult(id: ValueID(rawValue: 2))]
                ),
            ],
            results: [ValueUse(value: ValueID(rawValue: 2))]
        )

        let op = Operation(
            key: OperationKey(rawValue: 2),
            kind: .residual(strategy: .add, body: bodyRegion),
            results: [OperationResult(id: ValueID(rawValue: 3))]
        )

        if case .residual(let strategy, let body) = op.kind {
            #expect(strategy == .add)
            #expect(body.operations.count == 2)
            #expect(body.parameters.count == 1)
            #expect(body.results.count == 1)
        } else {
            Issue.record("Expected residual operation")
        }
    }

    @Test("Multi-input / multi-result operation is expressible")
    func multiValueOperation() {
        // IR supports multi-value even though current DSL is unary.
        let op = Operation(
            key: OperationKey(rawValue: 0),
            kind: .custom(CustomNodeAttributes(domain: "test", name: "multi_io")),
            operands: [
                Operand(value: ValueID(rawValue: 0)),
                Operand(value: ValueID(rawValue: 1)),
                Operand(value: ValueID(rawValue: 2)),
            ],
            results: [
                OperationResult(id: ValueID(rawValue: 3)),
                OperationResult(id: ValueID(rawValue: 4)),
            ]
        )

        #expect(op.operands.count == 3)
        #expect(op.results.count == 2)
    }

    @Test("Multi-parameter / multi-result region is expressible")
    func multiValueRegion() {
        let region = Region(
            parameters: [
                RegionParameter(id: ValueID(rawValue: 0)),
                RegionParameter(id: ValueID(rawValue: 1)),
            ],
            operations: [
                Operation(
                    key: OperationKey(rawValue: 0),
                    kind: .custom(CustomNodeAttributes(domain: "test", name: "fuse")),
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
        )

        #expect(region.parameters.count == 2)
        #expect(region.results.count == 2)
    }

    @Test("ModelGraph Codable roundtrip")
    func codableRoundtrip() throws {
        let graph = ModelGraph(rootRegion: Region(operations: [
            Operation(
                key: OperationKey(rawValue: 0),
                kind: .tokenEmbedding(
                    TokenEmbeddingAttributes(vocabSize: 32000, embeddingSize: 4096)
                ),
                results: [OperationResult(id: ValueID(rawValue: 0))]
            ),
            Operation(
                key: OperationKey(rawValue: 1),
                kind: .residual(strategy: .add, body: Region(
                    parameters: [RegionParameter(id: ValueID(rawValue: 1))],
                    operations: [
                        Operation(
                            key: OperationKey(rawValue: 2),
                            kind: .rmsNorm(RMSNormAttributes(dimension: 4096)),
                            operands: [Operand(value: ValueID(rawValue: 1))],
                            results: [OperationResult(id: ValueID(rawValue: 2))]
                        ),
                        Operation(
                            key: OperationKey(rawValue: 3),
                            kind: .attention(AttentionAttributes(
                                hiddenSize: 4096, headCount: 32, kvHeadCount: 8, headDimension: 128,
                                bias: false, causal: true,
                                rope: RoPEAttributes(dimension: 128, base: 10_000.0)
                            )),
                            operands: [Operand(value: ValueID(rawValue: 2))],
                            results: [OperationResult(id: ValueID(rawValue: 3))]
                        ),
                    ],
                    results: [ValueUse(value: ValueID(rawValue: 3))]
                )),
                operands: [Operand(value: ValueID(rawValue: 0))],
                results: [OperationResult(id: ValueID(rawValue: 4))]
            ),
            Operation(
                key: OperationKey(rawValue: 4),
                kind: .rmsNorm(RMSNormAttributes(dimension: 4096)),
                operands: [Operand(value: ValueID(rawValue: 4))],
                results: [OperationResult(id: ValueID(rawValue: 5))]
            ),
            Operation(
                key: OperationKey(rawValue: 5),
                kind: .outputHead(OutputHeadAttributes(
                    inputSize: 4096, vocabSize: 32000, tiedToEmbedding: true
                )),
                operands: [Operand(value: ValueID(rawValue: 5))],
                results: [OperationResult(id: ValueID(rawValue: 6))]
            ),
        ], results: [ValueUse(value: ValueID(rawValue: 6))]))

        let encoder = JSONEncoder()
        encoder.outputFormatting = .sortedKeys
        let data = try encoder.encode(graph)
        let decoded = try JSONDecoder().decode(ModelGraph.self, from: data)

        #expect(graph == decoded)
    }

    @Test("ModelGraph Equatable for structural equivalence")
    func equatable() {
        let makeGraph = { (dim: Int) in
            ModelGraph(rootRegion: Region(operations: [
                Operation(
                    key: OperationKey(rawValue: 0),
                    kind: .rmsNorm(RMSNormAttributes(dimension: dim, epsilon: 1e-6)),
                    results: [OperationResult(id: ValueID(rawValue: 0))]
                ),
                Operation(
                    key: OperationKey(rawValue: 1),
                    kind: .linear(LinearAttributes(inputSize: dim, outputSize: 32000)),
                    operands: [Operand(value: ValueID(rawValue: 0))],
                    results: [OperationResult(id: ValueID(rawValue: 1))]
                ),
            ], results: [ValueUse(value: ValueID(rawValue: 1))]))
        }

        let graphA = makeGraph(4096)
        let graphB = makeGraph(4096)
        let graphC = makeGraph(2048)

        #expect(graphA == graphB)
        #expect(graphA != graphC)
    }

    @Test("All OperationKind primitive cases can be constructed")
    func allPrimitiveKinds() {
        let kinds: [OperationKind] = [
            .tokenEmbedding(TokenEmbeddingAttributes(vocabSize: 100, embeddingSize: 64)),
            .positionalEmbedding(PositionalEmbeddingAttributes(maxPositions: 512, embeddingSize: 64, kind: .sinusoidal)),
            .rope(RoPEAttributes(dimension: 64)),
            .attention(AttentionAttributes(hiddenSize: 64, headCount: 4, kvHeadCount: 4, headDimension: 16)),
            .mlp(MLPAttributes(inputSize: 64, outputSize: 64, intermediateSize: 256)),
            .moe(MoEAttributes(
                expertCount: 8, expertsPerToken: 2,
                expertMLP: MLPAttributes(inputSize: 64, outputSize: 64, intermediateSize: 256)
            )),
            .rmsNorm(RMSNormAttributes(dimension: 64)),
            .layerNorm(LayerNormAttributes(dimension: 64)),
            .linear(LinearAttributes(inputSize: 64, outputSize: 32)),
            .outputHead(OutputHeadAttributes(inputSize: 64, vocabSize: 100)),
            .stateSpace(StateSpaceAttributes(hiddenSize: 64, stateSize: 16, variant: "mamba")),
            .custom(CustomNodeAttributes(domain: "test", name: "noop")),
        ]

        #expect(kinds.count == 12)
    }

    @Test("Region-bearing OperationKind cases can be constructed")
    func regionBearingKinds() {
        let body = Region(
            parameters: [RegionParameter(id: ValueID(rawValue: 0))],
            operations: [
                Operation(
                    key: OperationKey(rawValue: 0),
                    kind: .rmsNorm(RMSNormAttributes(dimension: 64)),
                    operands: [Operand(value: ValueID(rawValue: 0))],
                    results: [OperationResult(id: ValueID(rawValue: 1))]
                )
            ],
            results: [ValueUse(value: ValueID(rawValue: 1))]
        )

        let kinds: [OperationKind] = [
            .residual(strategy: .add, body: body),
            .parallel(merge: .add, branches: [body, body]),
            .repeating(count: 12, body: body),
        ]

        #expect(kinds.count == 3)
    }

    @Test("LoweredGraph Codable roundtrip")
    func loweredGraphCodableRoundtrip() throws {
        let lowered = LoweredGraph(
            nodes: [
                LoweredNode(id: 0, op: .gather),
                LoweredNode(id: 1, op: .matmul, inputs: [0]),
                LoweredNode(id: 2, op: .ropeApply, inputs: [1]),
                LoweredNode(id: 3, op: .sdpa, inputs: [1, 2]),
                LoweredNode(id: 4, op: .activation(.silu), inputs: [3]),
                LoweredNode(id: 5, op: .rmsNorm, inputs: [4]),
                LoweredNode(id: 6, op: .add, inputs: [0, 5]),
                LoweredNode(id: 7, op: .softmax, inputs: [6]),
                LoweredNode(id: 8, op: .reshape, inputs: [7]),
                LoweredNode(id: 9, op: .transpose, inputs: [8]),
                LoweredNode(id: 10, op: .split, inputs: [9]),
                LoweredNode(id: 11, op: .concat, inputs: [10]),
                LoweredNode(id: 12, op: .mul, inputs: [11]),
                LoweredNode(id: 13, op: .layerNorm, inputs: [12]),
                LoweredNode(id: 14, op: .custom("experimental_op"), inputs: [13]),
            ],
            outputs: [14]
        )

        let data = try JSONEncoder().encode(lowered)
        let decoded = try JSONDecoder().decode(LoweredGraph.self, from: data)

        #expect(lowered == decoded)
        #expect(decoded.nodes.count == 15)
    }

    @Test("ParameterSlot path construction with StructuralPath")
    func parameterSlotPath() {
        let slot = ParameterSlot(
            path: StructuralPath(components: [.operation(0), .field("q_proj"), .field("weight")]),
            role: .weight
        )

        #expect(slot.path.components.count == 3)
        #expect(slot.role == .weight)

        let slotB = ParameterSlot(
            path: StructuralPath(components: [.operation(0), .field("q_proj"), .field("weight")]),
            role: .weight
        )
        #expect(slot == slotB)

        let slotC = ParameterSlot(
            path: StructuralPath(components: [.operation(0), .field("q_proj"), .field("bias")]),
            role: .bias
        )
        #expect(slot != slotC)
    }

    @Test("Canonicalization normalizes implementation hints")
    func canonicalizationHints() throws {
        let graphA = try normalize(
            .primitive(.attention(AttentionAttributes(
                hiddenSize: 64, headCount: 4, kvHeadCount: 4, headDimension: 16,
                implementationHint: .fused
            )))
        ).graph

        let graphB = try normalize(
            .primitive(.attention(AttentionAttributes(
                hiddenSize: 64, headCount: 4, kvHeadCount: 4, headDimension: 16,
                implementationHint: .unfused
            )))
        ).graph

        #expect(graphA != graphB)
        #expect(canonicalize(graphA) == canonicalize(graphB))
    }

    @Test("Canonicalization produces stable keys and values")
    func canonicalizationIDs() throws {
        let declA: ModelDeclaration = .sequence([
            .primitive(.rmsNorm(RMSNormAttributes(dimension: 64))),
            .primitive(.linear(LinearAttributes(inputSize: 64, outputSize: 32))),
        ])

        let declB: ModelDeclaration = .sequence([
            .primitive(.rmsNorm(RMSNormAttributes(dimension: 64))),
            .primitive(.linear(LinearAttributes(inputSize: 64, outputSize: 32))),
        ])

        let graphA = try normalize(declA).graph
        let graphB = try normalize(declB).graph

        #expect(graphA == graphB)
        #expect(canonicalize(graphA) == canonicalize(graphB))
    }

    @Test("Canonicalization normalizes nested regions")
    func canonicalizationNestedRegions() throws {
        let declA: ModelDeclaration = .residual(strategy: .add, body: .sequence([
            .primitive(.rmsNorm(RMSNormAttributes(dimension: 64))),
            .primitive(.attention(AttentionAttributes(
                hiddenSize: 64, headCount: 4, kvHeadCount: 4, headDimension: 16,
                implementationHint: .fused
            ))),
        ]))

        let declB: ModelDeclaration = .residual(strategy: .add, body: .sequence([
            .primitive(.rmsNorm(RMSNormAttributes(dimension: 64))),
            .primitive(.attention(AttentionAttributes(
                hiddenSize: 64, headCount: 4, kvHeadCount: 4, headDimension: 16,
                implementationHint: .unfused
            ))),
        ]))

        let graphA = try normalize(declA).graph
        let graphB = try normalize(declB).graph

        #expect(graphA != graphB)
        #expect(canonicalize(graphA) == canonicalize(graphB))
    }

    @Test("StructuralPath supports all component types including parameter and operand")
    func structuralPath() {
        let root = StructuralPath()
        let child = root.appending(.operation(0)).appending(.regionBody).appending(.operation(1))

        #expect(child.components.count == 3)
        #expect(child.components[0] == .operation(0))
        #expect(child.components[1] == .regionBody)
        #expect(child.components[2] == .operation(1))

        let same = StructuralPath(components: [.operation(0), .regionBody, .operation(1)])
        #expect(child == same)

        // Parameter and operand path components
        let paramPath = StructuralPath(components: [.parameter(0)])
        let operandPath = StructuralPath(components: [.operation(0), .operand(1)])
        #expect(paramPath.components[0] == .parameter(0))
        #expect(operandPath.components[1] == .operand(1))
    }

    @Test("ModelGraphMetadata annotation lookup")
    func metadataLookup() {
        let path = StructuralPath(components: [.operation(0)])
        let meta = ModelGraphMetadata(annotations: [
            AnnotationEntry(path: path, annotation: OperationAnnotation(label: "embedding")),
        ])

        #expect(meta.annotation(for: path)?.label == "embedding")
        #expect(meta.annotation(for: StructuralPath(components: [.operation(1)])) == nil)
    }

    @Test("NormalizedModel bundles graph and metadata")
    func normalizedModel() throws {
        let decl: ModelDeclaration = .repeating(count: 2, label: "layers", body:
            .primitive(.rmsNorm(RMSNormAttributes(dimension: 64)))
        )
        let normalized = try normalize(decl)

        #expect(normalized.graph.rootRegion.operations.count == 1)
        if case .repeating(let count, _) = normalized.graph.rootRegion.operations[0].kind {
            #expect(count == 2)
        } else {
            Issue.record("Expected repeating")
        }

        let opPath = StructuralPath(components: [.operation(0)])
        #expect(normalized.metadata.annotation(for: opPath)?.label == "layers")
    }

    @Test("WeightsDeclaration.empty case")
    func weightsEmpty() {
        let empty = WeightsDeclaration.empty
        #expect(empty == .empty)
        #expect(canonicalize(empty) == .empty)
    }

    @Test("WeightsDeclaration canonicalization simplifies merge")
    func weightsCanonicalization() {
        #expect(canonicalize(.merge([])) == .empty)

        let single = WeightsDeclaration.gguf(location: "model.gguf")
        #expect(canonicalize(.merge([single])) == single)

        #expect(canonicalize(.override(base: single, with: .empty)) == single)

        let merged = WeightsDeclaration.merge([.empty, single, .empty])
        #expect(canonicalize(merged) == single)
    }

    // MARK: - GraphValidator Tests

    @Test("GraphValidator accepts well-formed normalized graph")
    func validatorAcceptsNormalized() throws {
        let decl: ModelDeclaration = .sequence([
            .primitive(.rmsNorm(RMSNormAttributes(dimension: 64))),
            .residual(strategy: .add, body:
                .primitive(.attention(AttentionAttributes(
                    hiddenSize: 64, headCount: 4, kvHeadCount: 4, headDimension: 16
                )))
            ),
            .primitive(.outputHead(OutputHeadAttributes(inputSize: 64, vocabSize: 100))),
        ])

        let graph = try normalize(decl).graph
        try GraphValidator.validate(graph)
    }

    @Test("GraphValidator accepts graph with parallel branches")
    func validatorAcceptsParallel() throws {
        let decl: ModelDeclaration = .parallel(merge: .add, branches: [
            .primitive(.attention(AttentionAttributes(
                hiddenSize: 64, headCount: 4, kvHeadCount: 4, headDimension: 16
            ))),
            .primitive(.mlp(MLPAttributes(inputSize: 64, outputSize: 64, intermediateSize: 256))),
        ])

        let graph = try normalize(decl).graph
        try GraphValidator.validate(graph)
    }

    @Test("GraphValidator accepts graph with repeating")
    func validatorAcceptsRepeating() throws {
        let decl: ModelDeclaration = .sequence([
            .primitive(.rmsNorm(RMSNormAttributes(dimension: 64))),
            .repeating(count: 4, label: nil, body:
                .primitive(.rmsNorm(RMSNormAttributes(dimension: 64)))
            ),
        ])

        let graph = try normalize(decl).graph
        try GraphValidator.validate(graph)
    }

    @Test("GraphValidator detects residual arity mismatch")
    func validatorResidualMismatch() {
        // Manually construct a malformed graph: residual with 1 operand but 0 body params
        let graph = ModelGraph(rootRegion: Region(operations: [
            Operation(
                key: OperationKey(rawValue: 0),
                kind: .residual(strategy: .add, body: Region(
                    parameters: [], // WRONG: should have 1 param to match 1 operand
                    operations: [],
                    results: []
                )),
                operands: [Operand(value: ValueID(rawValue: 0))],
                results: [OperationResult(id: ValueID(rawValue: 1))]
            ),
        ]))

        #expect(throws: GraphValidationError.self) {
            try GraphValidator.validate(graph)
        }
    }

    @Test("GraphValidator detects repeating loop arity mismatch")
    func validatorRepeatingMismatch() {
        // body.parameters.count != body.results.count
        let graph = ModelGraph(rootRegion: Region(operations: [
            Operation(
                key: OperationKey(rawValue: 0),
                kind: .repeating(count: 2, body: Region(
                    parameters: [RegionParameter(id: ValueID(rawValue: 10))],
                    operations: [
                        Operation(
                            key: OperationKey(rawValue: 1),
                            kind: .rmsNorm(RMSNormAttributes(dimension: 64)),
                            operands: [Operand(value: ValueID(rawValue: 10))],
                            results: [
                                OperationResult(id: ValueID(rawValue: 11)),
                                OperationResult(id: ValueID(rawValue: 12)),
                            ]
                        ),
                    ],
                    results: [
                        ValueUse(value: ValueID(rawValue: 11)),
                        ValueUse(value: ValueID(rawValue: 12)),
                    ] // 2 results but 1 parameter → loop arity mismatch
                )),
                operands: [Operand(value: ValueID(rawValue: 0))],
                results: [OperationResult(id: ValueID(rawValue: 1))]
            ),
        ]))

        #expect(throws: GraphValidationError.self) {
            try GraphValidator.validate(graph)
        }
    }

    // MARK: - Scope Isolation Tests

    @Test("GraphValidator rejects nested region body referencing parent-only value")
    func validatorScopeIsolation() {
        // Parent region defines value 0. Residual body tries to use value 0
        // directly instead of through its own parameter — should fail.
        let graph = ModelGraph(rootRegion: Region(
            parameters: [],
            operations: [
                Operation(
                    key: OperationKey(rawValue: 0),
                    kind: .rmsNorm(RMSNormAttributes(dimension: 64)),
                    operands: [],
                    results: [OperationResult(id: ValueID(rawValue: 0))]
                ),
                Operation(
                    key: OperationKey(rawValue: 1),
                    kind: .residual(strategy: .add, body: Region(
                        parameters: [RegionParameter(id: ValueID(rawValue: 10))],
                        operations: [
                            Operation(
                                key: OperationKey(rawValue: 2),
                                kind: .rmsNorm(RMSNormAttributes(dimension: 64)),
                                // BUG: references parent value 0, not body param 10
                                operands: [Operand(value: ValueID(rawValue: 0))],
                                results: [OperationResult(id: ValueID(rawValue: 11))]
                            ),
                        ],
                        results: [ValueUse(value: ValueID(rawValue: 11))]
                    )),
                    operands: [Operand(value: ValueID(rawValue: 0))],
                    results: [OperationResult(id: ValueID(rawValue: 1))]
                ),
            ],
            results: [ValueUse(value: ValueID(rawValue: 1))]
        ))

        #expect(throws: GraphValidationError.self) {
            try GraphValidator.validate(graph)
        }
    }

    @Test("GraphValidator rejects deeply nested region referencing grandparent value")
    func validatorDeepScopeIsolation() {
        // Repeating body contains residual body that tries to use a
        // value from the root region — two levels of scope violation.
        let graph = ModelGraph(rootRegion: Region(
            parameters: [],
            operations: [
                Operation(
                    key: OperationKey(rawValue: 0),
                    kind: .rmsNorm(RMSNormAttributes(dimension: 64)),
                    operands: [],
                    results: [OperationResult(id: ValueID(rawValue: 0))]
                ),
                Operation(
                    key: OperationKey(rawValue: 1),
                    kind: .repeating(count: 2, body: Region(
                        parameters: [RegionParameter(id: ValueID(rawValue: 20))],
                        operations: [
                            Operation(
                                key: OperationKey(rawValue: 2),
                                kind: .residual(strategy: .add, body: Region(
                                    parameters: [RegionParameter(id: ValueID(rawValue: 30))],
                                    operations: [
                                        Operation(
                                            key: OperationKey(rawValue: 3),
                                            kind: .rmsNorm(RMSNormAttributes(dimension: 64)),
                                            // BUG: references root value 0
                                            operands: [Operand(value: ValueID(rawValue: 0))],
                                            results: [OperationResult(id: ValueID(rawValue: 31))]
                                        ),
                                    ],
                                    results: [ValueUse(value: ValueID(rawValue: 31))]
                                )),
                                operands: [Operand(value: ValueID(rawValue: 20))],
                                results: [OperationResult(id: ValueID(rawValue: 21))]
                            ),
                        ],
                        results: [ValueUse(value: ValueID(rawValue: 21))]
                    )),
                    operands: [Operand(value: ValueID(rawValue: 0))],
                    results: [OperationResult(id: ValueID(rawValue: 1))]
                ),
            ],
            results: [ValueUse(value: ValueID(rawValue: 1))]
        ))

        #expect(throws: GraphValidationError.self) {
            try GraphValidator.validate(graph)
        }
    }

    // MARK: - Parallel Merge Arity Tests

    @Test("GraphValidator rejects parallel .add with mismatched branch result arities")
    func validatorParallelAddMismatch() {
        // .add requires all branches to have equal result arity
        let graph = ModelGraph(rootRegion: Region(
            parameters: [],
            operations: [
                Operation(
                    key: OperationKey(rawValue: 0),
                    kind: .parallel(merge: .add, branches: [
                        Region(
                            parameters: [],
                            operations: [
                                Operation(
                                    key: OperationKey(rawValue: 1),
                                    kind: .rmsNorm(RMSNormAttributes(dimension: 64)),
                                    operands: [],
                                    results: [OperationResult(id: ValueID(rawValue: 10))]
                                ),
                            ],
                            results: [ValueUse(value: ValueID(rawValue: 10))] // 1 result
                        ),
                        Region(
                            parameters: [],
                            operations: [
                                Operation(
                                    key: OperationKey(rawValue: 2),
                                    kind: .rmsNorm(RMSNormAttributes(dimension: 64)),
                                    operands: [],
                                    results: [
                                        OperationResult(id: ValueID(rawValue: 20)),
                                        OperationResult(id: ValueID(rawValue: 21)),
                                    ]
                                ),
                            ],
                            results: [
                                ValueUse(value: ValueID(rawValue: 20)),
                                ValueUse(value: ValueID(rawValue: 21)),
                            ] // 2 results — mismatch with branch 0
                        ),
                    ]),
                    operands: [],
                    results: [OperationResult(id: ValueID(rawValue: 1))]
                ),
            ]
        ))

        #expect(throws: GraphValidationError.self) {
            try GraphValidator.validate(graph)
        }
    }

    @Test("GraphValidator rejects parallel .concat with mismatched op result arity")
    func validatorParallelConcatMismatch() {
        // .concat is tensor-level: same arity contract as .add.
        // 2 branches × 1 result each → op needs 1 result (not 2).
        // Here we give op 2 results → mismatch.
        let graph = ModelGraph(rootRegion: Region(
            parameters: [],
            operations: [
                Operation(
                    key: OperationKey(rawValue: 0),
                    kind: .parallel(merge: .concat, branches: [
                        Region(
                            parameters: [],
                            operations: [
                                Operation(
                                    key: OperationKey(rawValue: 1),
                                    kind: .rmsNorm(RMSNormAttributes(dimension: 64)),
                                    operands: [],
                                    results: [OperationResult(id: ValueID(rawValue: 10))]
                                ),
                            ],
                            results: [ValueUse(value: ValueID(rawValue: 10))] // 1 result
                        ),
                        Region(
                            parameters: [],
                            operations: [
                                Operation(
                                    key: OperationKey(rawValue: 2),
                                    kind: .rmsNorm(RMSNormAttributes(dimension: 64)),
                                    operands: [],
                                    results: [OperationResult(id: ValueID(rawValue: 20))]
                                ),
                            ],
                            results: [ValueUse(value: ValueID(rawValue: 20))] // 1 result
                        ),
                    ]),
                    operands: [],
                    // WRONG: branch arity is 1, but op has 2 results
                    results: [
                        OperationResult(id: ValueID(rawValue: 1)),
                        OperationResult(id: ValueID(rawValue: 2)),
                    ]
                ),
            ]
        ))

        #expect(throws: GraphValidationError.self) {
            try GraphValidator.validate(graph)
        }
    }

    @Test("GraphValidator accepts parallel .concat with matching arity")
    func validatorParallelConcatValid() throws {
        // .concat is tensor-level: branches each produce 1 result,
        // op produces 1 result (tensor concat happens at execution level).
        let graph = ModelGraph(rootRegion: Region(
            parameters: [],
            operations: [
                Operation(
                    key: OperationKey(rawValue: 0),
                    kind: .parallel(merge: .concat, branches: [
                        Region(
                            parameters: [],
                            operations: [
                                Operation(
                                    key: OperationKey(rawValue: 1),
                                    kind: .rmsNorm(RMSNormAttributes(dimension: 64)),
                                    operands: [],
                                    results: [OperationResult(id: ValueID(rawValue: 10))]
                                ),
                            ],
                            results: [ValueUse(value: ValueID(rawValue: 10))]
                        ),
                        Region(
                            parameters: [],
                            operations: [
                                Operation(
                                    key: OperationKey(rawValue: 2),
                                    kind: .rmsNorm(RMSNormAttributes(dimension: 64)),
                                    operands: [],
                                    results: [OperationResult(id: ValueID(rawValue: 20))]
                                ),
                            ],
                            results: [ValueUse(value: ValueID(rawValue: 20))]
                        ),
                    ]),
                    operands: [],
                    results: [OperationResult(id: ValueID(rawValue: 1))]
                ),
            ],
            results: [ValueUse(value: ValueID(rawValue: 1))]
        ))

        try GraphValidator.validate(graph)
    }

    @Test("GraphValidator accepts parallel .custom with any arity")
    func validatorParallelCustomNoConstraint() throws {
        // .custom imposes no arity constraints
        let graph = ModelGraph(rootRegion: Region(
            parameters: [],
            operations: [
                Operation(
                    key: OperationKey(rawValue: 0),
                    kind: .parallel(merge: .custom("whatever"), branches: [
                        Region(
                            parameters: [],
                            operations: [
                                Operation(
                                    key: OperationKey(rawValue: 1),
                                    kind: .rmsNorm(RMSNormAttributes(dimension: 64)),
                                    operands: [],
                                    results: [OperationResult(id: ValueID(rawValue: 10))]
                                ),
                            ],
                            results: [ValueUse(value: ValueID(rawValue: 10))] // 1 result
                        ),
                        Region(
                            parameters: [],
                            operations: [],
                            results: [] // 0 results — different arity
                        ),
                    ]),
                    operands: [],
                    results: [
                        OperationResult(id: ValueID(rawValue: 1)),
                        OperationResult(id: ValueID(rawValue: 2)),
                        OperationResult(id: ValueID(rawValue: 3)),
                    ] // 3 results — arbitrary
                ),
            ],
            results: [
                ValueUse(value: ValueID(rawValue: 1)),
                ValueUse(value: ValueID(rawValue: 2)),
                ValueUse(value: ValueID(rawValue: 3)),
            ]
        ))

        try GraphValidator.validate(graph)
    }

    // MARK: - WeightsDeclaration Canonicalization Tests

    @Test("WeightsDeclaration canonicalization flattens nested merges")
    func weightsNestedMergeFlatten() {
        let a = WeightsDeclaration.gguf(location: "a.gguf")
        let b = WeightsDeclaration.safetensors(directory: "b/", indexFile: nil)
        let c = WeightsDeclaration.gguf(location: "c.gguf")

        // merge([merge([a, b]), c]) should flatten to merge([a, b, c])
        let nested = WeightsDeclaration.merge([.merge([a, b]), c])
        let result = canonicalize(nested)
        #expect(result == .merge([a, b, c]))
    }

    @Test("WeightsDeclaration canonicalization removes empty from nested merges")
    func weightsNestedMergeEmpty() {
        let a = WeightsDeclaration.gguf(location: "a.gguf")

        // merge([merge([empty, a]), empty]) → a
        let nested = WeightsDeclaration.merge([.merge([.empty, a]), .empty])
        let result = canonicalize(nested)
        #expect(result == a)
    }

    @Test("WeightsDeclaration canonicalization handles deeply nested merges")
    func weightsDeeplyNestedMerge() {
        let a = WeightsDeclaration.gguf(location: "a.gguf")
        let b = WeightsDeclaration.gguf(location: "b.gguf")

        // merge([merge([merge([a]), b])]) → merge([a, b])
        let deep = WeightsDeclaration.merge([.merge([.merge([a]), b])])
        let result = canonicalize(deep)
        #expect(result == .merge([a, b]))
    }

    // MARK: - OperationSignature Tests

    @Test("primitiveInfo returns per-primitive signatures")
    func primitiveSignatures() {
        // tokenEmbedding is a source: exact(0) operands, exact(1) result
        let (_, embSig) = primitiveInfo(from: .tokenEmbedding(
            TokenEmbeddingAttributes(vocabSize: 100, embeddingSize: 64)))
        #expect(embSig.operandArity == .exact(0))
        #expect(embSig.resultArity == .exact(1))

        // All transform primitives: exact(1) operand, exact(1) result
        let transforms: [PrimitiveDeclaration] = [
            .positionalEmbedding(PositionalEmbeddingAttributes(maxPositions: 512, embeddingSize: 64, kind: .sinusoidal)),
            .rope(RoPEAttributes(dimension: 64)),
            .attention(AttentionAttributes(hiddenSize: 64, headCount: 4, kvHeadCount: 4, headDimension: 16)),
            .mlp(MLPAttributes(inputSize: 64, outputSize: 64, intermediateSize: 256)),
            .moe(MoEAttributes(expertCount: 4, expertsPerToken: 2, expertMLP: MLPAttributes(inputSize: 64, outputSize: 64, intermediateSize: 256))),
            .rmsNorm(RMSNormAttributes(dimension: 64)),
            .layerNorm(LayerNormAttributes(dimension: 64)),
            .linear(LinearAttributes(inputSize: 64, outputSize: 32)),
            .outputHead(OutputHeadAttributes(inputSize: 64, vocabSize: 100)),
            .stateSpace(StateSpaceAttributes(hiddenSize: 64, stateSize: 16, variant: "mamba")),
        ]
        for prim in transforms {
            let (_, sig) = primitiveInfo(from: prim)
            #expect(sig.operandArity == .exact(1))
            #expect(sig.resultArity == .exact(1))
        }

        // custom is escape hatch: variadic operands, variadic results
        let (_, customSig) = primitiveInfo(from: .custom(
            CustomNodeAttributes(domain: "test", name: "noop")))
        #expect(customSig.operandArity == .variadic)
        #expect(customSig.resultArity == .variadic)
    }

    @Test("resolveArity resolves exact and variadic correctly")
    func resolveArityTest() {
        #expect(resolveArity(.exact(3), fallback: 10) == 3)
        #expect(resolveArity(.variadic, fallback: 5) == 5)
        #expect(resolveArity(.exact(0), fallback: 99) == 0)
        #expect(resolveArity(.variadic, fallback: 0) == 0)
    }

    // MARK: - LLMProfileValidator Tests

    @Test("LLMProfileValidator accepts well-formed Llama graph")
    func profileValidatorAcceptsLlama() throws {
        let decl: ModelDeclaration = .sequence([
            .primitive(.tokenEmbedding(TokenEmbeddingAttributes(vocabSize: 100, embeddingSize: 64))),
            .primitive(.rmsNorm(RMSNormAttributes(dimension: 64))),
            .primitive(.outputHead(OutputHeadAttributes(inputSize: 64, vocabSize: 100))),
        ])
        let graph = try normalize(decl).graph
        try GraphValidator.validate(graph)
        try LLMProfileValidator.validate(graph)
    }

    @Test("LLMProfileValidator rejects empty model")
    func profileValidatorRejectsEmpty() throws {
        let graph = ModelGraph(rootRegion: Region())
        #expect(throws: LLMProfileError.self) {
            try LLMProfileValidator.validate(graph)
        }
    }

    @Test("LLMProfileValidator rejects multi-result root")
    func profileValidatorRejectsMultiResult() throws {
        let graph = ModelGraph(rootRegion: Region(
            operations: [
                Operation(
                    key: OperationKey(rawValue: 0),
                    kind: .rmsNorm(RMSNormAttributes(dimension: 64)),
                    operands: [],
                    results: [OperationResult(id: ValueID(rawValue: 0))]
                ),
                Operation(
                    key: OperationKey(rawValue: 1),
                    kind: .rmsNorm(RMSNormAttributes(dimension: 64)),
                    operands: [],
                    results: [OperationResult(id: ValueID(rawValue: 1))]
                ),
            ],
            results: [
                ValueUse(value: ValueID(rawValue: 0)),
                ValueUse(value: ValueID(rawValue: 1)),
            ]
        ))

        #expect(throws: LLMProfileError.self) {
            try LLMProfileValidator.validate(graph)
        }
    }

    @Test("LLMProfileValidator detects operand arity mismatch")
    func profileValidatorOperandArity() {
        // tokenEmbedding has signature exact(0) operands,
        // but here we give it 1 operand — profile violation.
        let graph = ModelGraph(rootRegion: Region(
            operations: [
                Operation(
                    key: OperationKey(rawValue: 0),
                    kind: .rmsNorm(RMSNormAttributes(dimension: 64)),
                    operands: [],
                    results: [OperationResult(id: ValueID(rawValue: 0))]
                ),
                Operation(
                    key: OperationKey(rawValue: 1),
                    kind: .tokenEmbedding(TokenEmbeddingAttributes(vocabSize: 100, embeddingSize: 64)),
                    // WRONG: tokenEmbedding is a source, should have 0 operands
                    operands: [Operand(value: ValueID(rawValue: 0))],
                    results: [OperationResult(id: ValueID(rawValue: 1))]
                ),
            ],
            results: [ValueUse(value: ValueID(rawValue: 1))]
        ))

        #expect(throws: LLMProfileError.self) {
            try LLMProfileValidator.validate(graph)
        }
    }

    @Test("LLMProfileValidator checks nested region primitives")
    func profileValidatorNestedArity() throws {
        // Well-formed Cohere-style model passes both validators.
        let decl: ModelDeclaration = .sequence([
            .primitive(.tokenEmbedding(TokenEmbeddingAttributes(vocabSize: 100, embeddingSize: 64))),
            .repeating(count: 2, label: nil, body:
                .residual(strategy: .add, body: .sequence([
                    .primitive(.layerNorm(LayerNormAttributes(dimension: 64))),
                    .parallel(merge: .add, branches: [
                        .primitive(.attention(AttentionAttributes(
                            hiddenSize: 64, headCount: 4, kvHeadCount: 4, headDimension: 16
                        ))),
                        .primitive(.mlp(MLPAttributes(inputSize: 64, outputSize: 64, intermediateSize: 256))),
                    ]),
                ]))
            ),
            .primitive(.layerNorm(LayerNormAttributes(dimension: 64))),
            .primitive(.outputHead(OutputHeadAttributes(inputSize: 64, vocabSize: 100))),
        ])

        let graph = try normalize(decl).graph
        try GraphValidator.validate(graph)
        try LLMProfileValidator.validate(graph)
    }
}
