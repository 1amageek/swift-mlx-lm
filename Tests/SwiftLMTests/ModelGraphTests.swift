import Testing
import Foundation
@testable import LMArchitecture

@Suite("ModelGraph", .tags(.unit))
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
            .stateSpace(StateSpaceAttributes(hiddenSize: 64, numHeads: 1, keyHeadDim: 16, valueHeadDim: 16, variant: "mamba")),
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
            Attention(hiddenSize: 64, headCount: 4, kvHeadCount: 4, headDimension: 16,
                      implementationHint: .fused)
        ).graph

        let graphB = try normalize(
            Attention(hiddenSize: 64, headCount: 4, kvHeadCount: 4, headDimension: 16,
                      implementationHint: .unfused)
        ).graph

        #expect(graphA != graphB)
        #expect(canonicalize(graphA) == canonicalize(graphB))
    }

    @Test("Canonicalization produces stable keys and values")
    func canonicalizationIDs() throws {
        let compA = Group {
            RMSNorm(dimension: 64)
            LMArchitecture.Linear(inputSize: 64, outputSize: 32)
        }

        let compB = Group {
            RMSNorm(dimension: 64)
            LMArchitecture.Linear(inputSize: 64, outputSize: 32)
        }

        let graphA = try normalize(compA).graph
        let graphB = try normalize(compB).graph

        #expect(graphA == graphB)
        #expect(canonicalize(graphA) == canonicalize(graphB))
    }

    @Test("Canonicalization normalizes nested regions")
    func canonicalizationNestedRegions() throws {
        let compA = Residual {
            RMSNorm(dimension: 64)
            Attention(hiddenSize: 64, headCount: 4, kvHeadCount: 4, headDimension: 16,
                      implementationHint: .fused)
        }

        let compB = Residual {
            RMSNorm(dimension: 64)
            Attention(hiddenSize: 64, headCount: 4, kvHeadCount: 4, headDimension: 16,
                      implementationHint: .unfused)
        }

        let graphA = try normalize(compA).graph
        let graphB = try normalize(compB).graph

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
        let comp = Repeat(count: 2, label: "layers") {
            RMSNorm(dimension: 64)
        }
        let normalized = try normalize(comp)

        #expect(normalized.graph.rootRegion.operations.count == 1)
        if case .repeating(let count, _) = normalized.graph.rootRegion.operations[0].kind {
            #expect(count == 2)
        } else {
            Issue.record("Expected repeating")
        }

        let opPath = StructuralPath(components: [.operation(0)])
        #expect(normalized.metadata.annotation(for: opPath)?.label == "layers")
    }

    // MARK: - GraphValidator Tests

    @Test("GraphValidator accepts well-formed normalized graph")
    func validatorAcceptsNormalized() throws {
        let comp = Group {
            RMSNorm(dimension: 64)
            Residual {
                Attention(hiddenSize: 64, headCount: 4, kvHeadCount: 4, headDimension: 16)
            }
            OutputHead(inputSize: 64, vocabSize: 100)
        }

        let graph = try normalize(comp).graph
        try GraphValidator.validate(graph)
    }

    @Test("GraphValidator accepts graph with parallel branches")
    func validatorAcceptsParallel() throws {
        let comp = Parallel(merge: .add) {
            Attention(hiddenSize: 64, headCount: 4, kvHeadCount: 4, headDimension: 16)
            MLP(inputSize: 64, intermediateSize: 256)
        }

        let graph = try normalize(comp).graph
        try GraphValidator.validate(graph)
    }

    @Test("GraphValidator accepts graph with repeating")
    func validatorAcceptsRepeating() throws {
        let comp = Group {
            RMSNorm(dimension: 64)
            Repeat(count: 4) {
                RMSNorm(dimension: 64)
            }
        }

        let graph = try normalize(comp).graph
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

    // MARK: - OperationSignature Tests

    @Test("primitiveSignature returns per-primitive signatures")
    func primitiveSignatures() {
        // tokenEmbedding is a source: exact(0) operands, exact(1) result
        let embSig = primitiveSignature(from: .tokenEmbedding(
            TokenEmbeddingAttributes(vocabSize: 100, embeddingSize: 64)))!
        #expect(embSig.operandArity == .exact(0))
        #expect(embSig.resultArity == .exact(1))

        // All transform primitives: exact(1) operand, exact(1) result
        let transforms: [OperationKind] = [
            .positionalEmbedding(PositionalEmbeddingAttributes(maxPositions: 512, embeddingSize: 64, kind: .sinusoidal)),
            .rope(RoPEAttributes(dimension: 64)),
            .attention(AttentionAttributes(hiddenSize: 64, headCount: 4, kvHeadCount: 4, headDimension: 16)),
            .mlp(MLPAttributes(inputSize: 64, outputSize: 64, intermediateSize: 256)),
            .moe(MoEAttributes(expertCount: 4, expertsPerToken: 2, expertMLP: MLPAttributes(inputSize: 64, outputSize: 64, intermediateSize: 256))),
            .rmsNorm(RMSNormAttributes(dimension: 64)),
            .layerNorm(LayerNormAttributes(dimension: 64)),
            .linear(LinearAttributes(inputSize: 64, outputSize: 32)),
            .outputHead(OutputHeadAttributes(inputSize: 64, vocabSize: 100)),
            .stateSpace(StateSpaceAttributes(hiddenSize: 64, numHeads: 1, keyHeadDim: 16, valueHeadDim: 16, variant: "mamba")),
        ]
        for kind in transforms {
            let sig = primitiveSignature(from: kind)!
            #expect(sig.operandArity == .exact(1))
            #expect(sig.resultArity == .exact(1))
        }

        // custom is escape hatch: variadic operands, variadic results
        let customSig = primitiveSignature(from: .custom(
            CustomNodeAttributes(domain: "test", name: "noop")))!
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
        let comp = Group {
            TokenEmbedding(vocabSize: 100, embeddingSize: 64)
            RMSNorm(dimension: 64)
            OutputHead(inputSize: 64, vocabSize: 100)
        }
        let graph = try normalize(comp).graph
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
        let comp = Group {
            TokenEmbedding(vocabSize: 100, embeddingSize: 64)
            Repeat(count: 2) {
                Residual {
                    LayerNorm(dimension: 64)
                    Parallel(merge: .add) {
                        Attention(hiddenSize: 64, headCount: 4, kvHeadCount: 4, headDimension: 16)
                        MLP(inputSize: 64, intermediateSize: 256)
                    }
                }
            }
            LayerNorm(dimension: 64)
            OutputHead(inputSize: 64, vocabSize: 100)
        }

        let graph = try normalize(comp).graph
        try GraphValidator.validate(graph)
        try LLMProfileValidator.validate(graph)
    }

    // MARK: - Canonicalizer Idempotence

    @Test("Canonicalization is idempotent")
    func canonicalizationIdempotence() throws {
        let model = TinyLlama(
            vocabSize: 100, hiddenSize: 64, headCount: 4,
            kvHeadCount: 2, intermediateSize: 256, layerCount: 2
        )
        let graph = try model.makeModelGraph()
        let once = canonicalize(graph)
        let twice = canonicalize(once)
        #expect(once == twice)
    }

    @Test("Canonicalization is idempotent for Cohere-style parallel model")
    func canonicalizationIdempotenceCohere() throws {
        let model = TinyCohere(
            vocabSize: 256000, hiddenSize: 4096, headCount: 32,
            kvHeadCount: 8, intermediateSize: 11008, layerCount: 2
        )
        let graph = try model.makeModelGraph()
        let once = canonicalize(graph)
        let twice = canonicalize(once)
        #expect(once == twice)
    }

    @Test("Canonicalization strips attention implementation hint")
    func canonicalizationStripsHint() throws {
        let comp = Group {
            TokenEmbedding(vocabSize: 100, embeddingSize: 64)
            Attention(hiddenSize: 64, headCount: 4, kvHeadCount: 4, headDimension: 16,
                      implementationHint: .fused)
            OutputHead(inputSize: 64, vocabSize: 100)
        }

        let graph = try normalize(comp).graph
        let canon = canonicalize(graph)

        if case .attention(let attrs) = canon.rootRegion.operations[1].kind {
            #expect(attrs.implementationHint == nil)
        } else {
            Issue.record("Expected attention operation")
        }
    }

    @Test("Canonicalization produces equal graphs from different ID assignments")
    func canonicalizationEqualizesIDs() throws {
        // Two graphs with same structure but different ValueID/OperationKey numbering.
        let graphA = ModelGraph(rootRegion: Region(
            operations: [
                Operation(
                    key: OperationKey(rawValue: 42),
                    kind: .rmsNorm(RMSNormAttributes(dimension: 64)),
                    results: [OperationResult(id: ValueID(rawValue: 99))]
                ),
            ],
            results: [ValueUse(value: ValueID(rawValue: 99))]
        ))
        let graphB = ModelGraph(rootRegion: Region(
            operations: [
                Operation(
                    key: OperationKey(rawValue: 7),
                    kind: .rmsNorm(RMSNormAttributes(dimension: 64)),
                    results: [OperationResult(id: ValueID(rawValue: 3))]
                ),
            ],
            results: [ValueUse(value: ValueID(rawValue: 3))]
        ))

        #expect(graphA != graphB)
        #expect(canonicalize(graphA) == canonicalize(graphB))
    }

    // MARK: - Attribute Codable Roundtrips

    @Test("RoPEAttributes with scaling roundtrips through JSON")
    func ropeScalingCodable() throws {
        let attrs = RoPEAttributes(
            dimension: 128,
            base: 500_000.0,
            scaling: RoPEScaling(
                kind: .yarn,
                factor: 4.0,
                originalMaxPositions: 8192
            )
        )
        let data = try JSONEncoder().encode(attrs)
        let decoded = try JSONDecoder().decode(RoPEAttributes.self, from: data)
        #expect(attrs == decoded)
    }

    @Test("RoPEScalingKind custom variant roundtrips through JSON")
    func ropeScalingKindCustomCodable() throws {
        let scaling = RoPEScaling(kind: .custom("ntk-aware"), factor: 2.0)
        let data = try JSONEncoder().encode(scaling)
        let decoded = try JSONDecoder().decode(RoPEScaling.self, from: data)
        #expect(scaling == decoded)
    }

    @Test("AttentionAttributes with all optional fields roundtrips through JSON")
    func attentionFullCodable() throws {
        let attrs = AttentionAttributes(
            hiddenSize: 4096,
            headCount: 32,
            kvHeadCount: 8,
            headDimension: 128,
            bias: true,
            causal: true,
            rope: RoPEAttributes(dimension: 128, base: 10_000.0),
            qkNorm: .rmsNorm,
            window: AttentionWindow(left: 4096, right: 0),
            implementationHint: .pagedKVPreferred
        )
        let data = try JSONEncoder().encode(attrs)
        let decoded = try JSONDecoder().decode(AttentionAttributes.self, from: data)
        #expect(attrs == decoded)
    }

    @Test("QKNormKind all cases roundtrip through JSON")
    func qkNormKindCodable() throws {
        let cases: [QKNormKind] = [.none, .rmsNorm, .rmsNormUnitOffset, .layerNorm, .custom("my-norm")]
        for kind in cases {
            let data = try JSONEncoder().encode(kind)
            let decoded = try JSONDecoder().decode(QKNormKind.self, from: data)
            #expect(kind == decoded)
        }
    }

    @Test("AttentionImplementationHint all cases roundtrip through JSON")
    func attentionHintCodable() throws {
        let cases: [AttentionImplementationHint] = [
            .unspecified, .fused, .unfused, .pagedKVPreferred, .custom("flash-attn-3")
        ]
        for hint in cases {
            let data = try JSONEncoder().encode(hint)
            let decoded = try JSONDecoder().decode(AttentionImplementationHint.self, from: data)
            #expect(hint == decoded)
        }
    }

    @Test("MoEAttributes with nested MLPAttributes roundtrips through JSON")
    func moeCodable() throws {
        let attrs = MoEAttributes(
            expertCount: 8,
            expertsPerToken: 2,
            gateKind: .sigmoidTopK,
            expertMLP: MLPAttributes(
                inputSize: 4096, outputSize: 4096,
                intermediateSize: 14336,
                activation: .gelu, gating: .geglu, bias: true
            )
        )
        let data = try JSONEncoder().encode(attrs)
        let decoded = try JSONDecoder().decode(MoEAttributes.self, from: data)
        #expect(attrs == decoded)
    }

    @Test("ActivationKind and GatingKind custom variants roundtrip through JSON")
    func activationGatingCustomCodable() throws {
        let activation = ActivationKind.custom("mish")
        let gating = GatingKind.custom("reglu")
        let data1 = try JSONEncoder().encode(activation)
        let data2 = try JSONEncoder().encode(gating)
        #expect(try JSONDecoder().decode(ActivationKind.self, from: data1) == activation)
        #expect(try JSONDecoder().decode(GatingKind.self, from: data2) == gating)
    }

    @Test("ResidualStrategy all cases roundtrip through JSON")
    func residualStrategyCodable() throws {
        let cases: [ResidualStrategy] = [.add, .weighted, .gated, .custom("learnable")]
        for strategy in cases {
            let data = try JSONEncoder().encode(strategy)
            let decoded = try JSONDecoder().decode(ResidualStrategy.self, from: data)
            #expect(strategy == decoded)
        }
    }

    @Test("ParallelMergeStrategy all cases roundtrip through JSON")
    func parallelMergeStrategyCodable() throws {
        let cases: [ParallelMergeStrategy] = [.add, .concat, .stack, .custom("mean")]
        for strategy in cases {
            let data = try JSONEncoder().encode(strategy)
            let decoded = try JSONDecoder().decode(ParallelMergeStrategy.self, from: data)
            #expect(strategy == decoded)
        }
    }

    @Test("CustomNodeAttributes with JSON value roundtrips through JSON")
    func customNodeCodable() throws {
        let attrs = CustomNodeAttributes(
            domain: "research.lab",
            name: "gated-linear-unit",
            attributes: .object([
                "width": .int(1024),
                "scale": .double(0.5),
                "enabled": .bool(true),
                "mode": .string("fast"),
                "dims": .array([.int(1), .int(2), .int(3)]),
            ])
        )
        let data = try JSONEncoder().encode(attrs)
        let decoded = try JSONDecoder().decode(CustomNodeAttributes.self, from: data)
        #expect(attrs == decoded)
    }

    @Test("StateSpaceAttributes roundtrips through JSON")
    func stateSpaceCodable() throws {
        let attrs = StateSpaceAttributes(hiddenSize: 2048, numHeads: 1, keyHeadDim: 64, valueHeadDim: 64, variant: "mamba2")
        let data = try JSONEncoder().encode(attrs)
        let decoded = try JSONDecoder().decode(StateSpaceAttributes.self, from: data)
        #expect(attrs == decoded)
    }

    @Test("PositionalEmbeddingKind all cases roundtrip through JSON")
    func positionalEmbeddingKindCodable() throws {
        let cases: [PositionalEmbeddingKind] = [.learnedAbsolute, .sinusoidal]
        for kind in cases {
            let data = try JSONEncoder().encode(kind)
            let decoded = try JSONDecoder().decode(PositionalEmbeddingKind.self, from: data)
            #expect(kind == decoded)
        }
    }

    @Test("OutputHeadAttributes roundtrips through JSON")
    func outputHeadCodable() throws {
        let attrs = OutputHeadAttributes(inputSize: 4096, vocabSize: 128256, tiedToEmbedding: false, bias: true)
        let data = try JSONEncoder().encode(attrs)
        let decoded = try JSONDecoder().decode(OutputHeadAttributes.self, from: data)
        #expect(attrs == decoded)
    }

    // MARK: - GraphValidator Edge Cases

    @Test("GraphValidator accepts parallel within parallel")
    func nestedParallel() throws {
        let comp = Group {
            RMSNorm(dimension: 64)
            Parallel(merge: .add) {
                Parallel(merge: .add) {
                    LMArchitecture.Linear(inputSize: 64, outputSize: 64)
                    LMArchitecture.Linear(inputSize: 64, outputSize: 64)
                }
                LMArchitecture.Linear(inputSize: 64, outputSize: 64)
            }
        }

        let graph = try normalize(comp).graph
        try GraphValidator.validate(graph)
    }

    @Test("GraphValidator accepts residual within residual")
    func nestedResidual() throws {
        let comp = Group {
            RMSNorm(dimension: 64)
            Residual {
                Residual {
                    LMArchitecture.Linear(inputSize: 64, outputSize: 64)
                }
            }
        }

        let graph = try normalize(comp).graph
        try GraphValidator.validate(graph)
    }

    @Test("GraphValidator accepts repeating within repeating")
    func nestedRepeating() throws {
        let comp = Group {
            RMSNorm(dimension: 64)
            Repeat(count: 2) {
                Repeat(count: 3) {
                    LMArchitecture.Linear(inputSize: 64, outputSize: 64)
                }
            }
        }

        let graph = try normalize(comp).graph
        try GraphValidator.validate(graph)
    }

    // MARK: - GraphValidator Error Detection

    @Test("GraphValidator detects forward reference (use before definition)")
    func forwardReference() {
        // Operation references a value produced by a later operation.
        let graph = ModelGraph(rootRegion: Region(
            operations: [
                Operation(
                    key: OperationKey(rawValue: 0),
                    kind: .rmsNorm(RMSNormAttributes(dimension: 64)),
                    operands: [Operand(value: ValueID(rawValue: 99))],
                    results: [OperationResult(id: ValueID(rawValue: 0))]
                ),
                Operation(
                    key: OperationKey(rawValue: 1),
                    kind: .linear(LinearAttributes(inputSize: 64, outputSize: 64)),
                    results: [OperationResult(id: ValueID(rawValue: 99))]
                ),
            ],
            results: [ValueUse(value: ValueID(rawValue: 99))]
        ))

        #expect(throws: GraphValidationError.self) {
            try GraphValidator.validate(graph)
        }
    }

    @Test("GraphValidator detects undefined region result")
    func undefinedRegionResult() {
        // Region result references a value that was never produced.
        let graph = ModelGraph(rootRegion: Region(
            operations: [
                Operation(
                    key: OperationKey(rawValue: 0),
                    kind: .rmsNorm(RMSNormAttributes(dimension: 64)),
                    results: [OperationResult(id: ValueID(rawValue: 0))]
                ),
            ],
            results: [ValueUse(value: ValueID(rawValue: 42))]
        ))

        #expect(throws: GraphValidationError.self) {
            try GraphValidator.validate(graph)
        }
    }

    @Test("GraphValidator detects residual body result arity mismatch")
    func residualBodyResultArityMismatch() {
        // body.results.count (2) != op.results.count (1)
        let body = Region(
            parameters: [RegionParameter(id: ValueID(rawValue: 10))],
            operations: [
                Operation(
                    key: OperationKey(rawValue: 0),
                    kind: .linear(LinearAttributes(inputSize: 64, outputSize: 64)),
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
            ]
        )
        let graph = ModelGraph(rootRegion: Region(
            operations: [
                Operation(
                    key: OperationKey(rawValue: 1),
                    kind: .rmsNorm(RMSNormAttributes(dimension: 64)),
                    results: [OperationResult(id: ValueID(rawValue: 0))]
                ),
                Operation(
                    key: OperationKey(rawValue: 2),
                    kind: .residual(strategy: .add, body: body),
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

    @Test("GraphValidator detects parallel branch parameter arity mismatch")
    func parallelBranchParameterArityMismatch() {
        // Branch has 2 parameters but operation has 1 operand.
        let branch = Region(
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
            ],
            results: [ValueUse(value: ValueID(rawValue: 12))]
        )
        let graph = ModelGraph(rootRegion: Region(
            operations: [
                Operation(
                    key: OperationKey(rawValue: 1),
                    kind: .rmsNorm(RMSNormAttributes(dimension: 64)),
                    results: [OperationResult(id: ValueID(rawValue: 0))]
                ),
                Operation(
                    key: OperationKey(rawValue: 2),
                    kind: .parallel(merge: .add, branches: [branch]),
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

    @Test("GraphValidator detects repeating result arity mismatch")
    func repeatingResultArityMismatch() {
        // op.results.count (2) != op.operands.count (1)
        let body = Region(
            parameters: [RegionParameter(id: ValueID(rawValue: 10))],
            operations: [
                Operation(
                    key: OperationKey(rawValue: 0),
                    kind: .linear(LinearAttributes(inputSize: 64, outputSize: 64)),
                    operands: [Operand(value: ValueID(rawValue: 10))],
                    results: [OperationResult(id: ValueID(rawValue: 11))]
                ),
            ],
            results: [ValueUse(value: ValueID(rawValue: 11))]
        )
        let graph = ModelGraph(rootRegion: Region(
            operations: [
                Operation(
                    key: OperationKey(rawValue: 1),
                    kind: .rmsNorm(RMSNormAttributes(dimension: 64)),
                    results: [OperationResult(id: ValueID(rawValue: 0))]
                ),
                Operation(
                    key: OperationKey(rawValue: 2),
                    kind: .repeating(count: 3, body: body),
                    operands: [Operand(value: ValueID(rawValue: 0))],
                    results: [
                        OperationResult(id: ValueID(rawValue: 1)),
                        OperationResult(id: ValueID(rawValue: 2)),
                    ]
                ),
            ],
            results: [ValueUse(value: ValueID(rawValue: 1))]
        ))

        #expect(throws: GraphValidationError.self) {
            try GraphValidator.validate(graph)
        }
    }

    @Test("GraphValidator detects value leak between sibling regions")
    func valueBetweenSiblingRegions() {
        // Branch 1 references a value produced inside Branch 0.
        let branch0 = Region(
            parameters: [RegionParameter(id: ValueID(rawValue: 10))],
            operations: [
                Operation(
                    key: OperationKey(rawValue: 0),
                    kind: .linear(LinearAttributes(inputSize: 64, outputSize: 64)),
                    operands: [Operand(value: ValueID(rawValue: 10))],
                    results: [OperationResult(id: ValueID(rawValue: 50))]
                ),
            ],
            results: [ValueUse(value: ValueID(rawValue: 50))]
        )
        let branch1 = Region(
            parameters: [RegionParameter(id: ValueID(rawValue: 11))],
            operations: [
                Operation(
                    key: OperationKey(rawValue: 1),
                    kind: .linear(LinearAttributes(inputSize: 64, outputSize: 64)),
                    // Illegally references ValueID 50 from branch0
                    operands: [Operand(value: ValueID(rawValue: 50))],
                    results: [OperationResult(id: ValueID(rawValue: 51))]
                ),
            ],
            results: [ValueUse(value: ValueID(rawValue: 51))]
        )
        let graph = ModelGraph(rootRegion: Region(
            operations: [
                Operation(
                    key: OperationKey(rawValue: 2),
                    kind: .rmsNorm(RMSNormAttributes(dimension: 64)),
                    results: [OperationResult(id: ValueID(rawValue: 0))]
                ),
                Operation(
                    key: OperationKey(rawValue: 3),
                    kind: .parallel(merge: .add, branches: [branch0, branch1]),
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

    // MARK: - GraphValidator Edge Cases (continued)

    @Test("GraphValidator accepts parallel with custom merge (no arity constraint)")
    func parallelCustomMergeNoConstraint() throws {
        // Custom merge: branches can have different result counts
        // Build manually since normalizer always uses equal arity from branches.
        let param = ValueID(rawValue: 0)
        let branch0 = Region(
            parameters: [RegionParameter(id: ValueID(rawValue: 1))],
            operations: [
                Operation(
                    key: OperationKey(rawValue: 0),
                    kind: .linear(LinearAttributes(inputSize: 64, outputSize: 64)),
                    operands: [Operand(value: ValueID(rawValue: 1))],
                    results: [OperationResult(id: ValueID(rawValue: 2))]
                ),
            ],
            results: [ValueUse(value: ValueID(rawValue: 2))]
        )
        let branch1 = Region(
            parameters: [RegionParameter(id: ValueID(rawValue: 3))],
            operations: [
                Operation(
                    key: OperationKey(rawValue: 1),
                    kind: .linear(LinearAttributes(inputSize: 64, outputSize: 64)),
                    operands: [Operand(value: ValueID(rawValue: 3))],
                    results: [
                        OperationResult(id: ValueID(rawValue: 4)),
                        OperationResult(id: ValueID(rawValue: 5)),
                    ]
                ),
            ],
            results: [
                ValueUse(value: ValueID(rawValue: 4)),
                ValueUse(value: ValueID(rawValue: 5)),
            ]
        )
        let graph = ModelGraph(rootRegion: Region(
            operations: [
                Operation(
                    key: OperationKey(rawValue: 2),
                    kind: .rmsNorm(RMSNormAttributes(dimension: 64)),
                    results: [OperationResult(id: param)]
                ),
                Operation(
                    key: OperationKey(rawValue: 3),
                    kind: .parallel(merge: .custom("my-merge"), branches: [branch0, branch1]),
                    operands: [Operand(value: param)],
                    results: [OperationResult(id: ValueID(rawValue: 6))]
                ),
            ],
            results: [ValueUse(value: ValueID(rawValue: 6))]
        ))

        // Custom merge allows mismatched branch arities — no error.
        try GraphValidator.validate(graph)
    }

    // MARK: - Canonicalizer Branch Coverage

    @Test("Canonicalization preserves RoPE attributes")
    func canonicalizationRoPE() throws {
        let comp = RoPE(
            dimension: 128, base: 500_000.0,
            scaling: RoPEScaling(kind: .yarn, factor: 4.0, originalMaxPositions: 8192)
        )

        let graph = try normalize(comp).graph
        let canon = canonicalize(graph)

        if case .rope(let attrs) = canon.rootRegion.operations[0].kind {
            #expect(attrs.dimension == 128)
            #expect(attrs.base == 500_000.0)
            #expect(attrs.scaling?.kind == .yarn)
            #expect(attrs.scaling?.factor == 4.0)
            #expect(attrs.scaling?.originalMaxPositions == 8192)
        } else {
            Issue.record("Expected rope operation")
        }
    }

    @Test("Canonicalization preserves MLP attributes")
    func canonicalizationMLP() throws {
        let comp = MLP(
            inputSize: 4096, intermediateSize: 11008,
            activation: .gelu, gating: .geglu, bias: true
        )

        let graph = try normalize(comp).graph
        let canon = canonicalize(graph)

        if case .mlp(let attrs) = canon.rootRegion.operations[0].kind {
            #expect(attrs.inputSize == 4096)
            #expect(attrs.intermediateSize == 11008)
            #expect(attrs.activation == .gelu)
            #expect(attrs.gating == .geglu)
            #expect(attrs.bias == true)
        } else {
            Issue.record("Expected mlp operation")
        }
    }

    @Test("Canonicalization preserves LayerNorm attributes")
    func canonicalizationLayerNorm() throws {
        let comp = LayerNorm(dimension: 2048, epsilon: 1e-5, affine: false)

        let graph = try normalize(comp).graph
        let canon = canonicalize(graph)

        if case .layerNorm(let attrs) = canon.rootRegion.operations[0].kind {
            #expect(attrs.dimension == 2048)
            #expect(attrs.epsilon == 1e-5)
            #expect(attrs.affine == false)
        } else {
            Issue.record("Expected layerNorm operation")
        }
    }

    @Test("Canonicalization passes through unhandled kinds unchanged")
    func canonicalizationPassthrough() throws {
        let comp = Group {
            TokenEmbedding(vocabSize: 100, embeddingSize: 64)
            StateSpace(hiddenSize: 64, numHeads: 1, keyHeadDim: 16, valueHeadDim: 16, variant: "mamba")
            Custom(domain: "test", name: "noop")
            OutputHead(inputSize: 64, vocabSize: 100)
        }

        let graph = try normalize(comp).graph
        let canon = canonicalize(graph)

        // Pass-through kinds: tokenEmbedding, stateSpace, custom, outputHead
        if case .tokenEmbedding(let a) = canon.rootRegion.operations[0].kind {
            #expect(a.vocabSize == 100)
        } else {
            Issue.record("Expected tokenEmbedding")
        }
        if case .stateSpace(let a) = canon.rootRegion.operations[1].kind {
            #expect(a.variant == "mamba")
        } else {
            Issue.record("Expected stateSpace")
        }
        if case .custom(let a) = canon.rootRegion.operations[2].kind {
            #expect(a.domain == "test")
        } else {
            Issue.record("Expected custom")
        }
    }

    @Test("Canonicalization recurses into residual body region")
    func canonicalizationResidualBody() throws {
        // Attention with hint inside residual — hint should be stripped.
        let comp = Residual {
            Attention(hiddenSize: 64, headCount: 4, kvHeadCount: 4, headDimension: 16,
                      implementationHint: .fused)
        }

        let graph = try normalize(comp).graph
        let canon = canonicalize(graph)

        if case .residual(_, let body) = canon.rootRegion.operations[0].kind {
            if case .attention(let attrs) = body.operations[0].kind {
                #expect(attrs.implementationHint == nil)
            } else {
                Issue.record("Expected attention inside residual body")
            }
        } else {
            Issue.record("Expected residual")
        }
    }

    @Test("Canonicalization recurses into parallel branch regions")
    func canonicalizationParallelBranches() throws {
        let comp = Parallel(merge: .add) {
            Attention(hiddenSize: 64, headCount: 4, kvHeadCount: 4, headDimension: 16,
                      implementationHint: .unfused)
            MLP(inputSize: 64, intermediateSize: 256,
                activation: .silu, gating: .swiglu)
        }

        let graph = try normalize(comp).graph
        let canon = canonicalize(graph)

        if case .parallel(_, let branches) = canon.rootRegion.operations[0].kind {
            if case .attention(let attrs) = branches[0].operations[0].kind {
                #expect(attrs.implementationHint == nil)
            } else {
                Issue.record("Expected attention in branch 0")
            }
            if case .mlp(let attrs) = branches[1].operations[0].kind {
                #expect(attrs.activation == .silu)
            } else {
                Issue.record("Expected mlp in branch 1")
            }
        } else {
            Issue.record("Expected parallel")
        }
    }

    @Test("Canonicalization recurses into repeating body region")
    func canonicalizationRepeatingBody() throws {
        let comp = Group {
            RMSNorm(dimension: 64)
            Repeat(count: 4) {
                Attention(hiddenSize: 64, headCount: 4, kvHeadCount: 4, headDimension: 16,
                          implementationHint: .pagedKVPreferred)
            }
        }

        let graph = try normalize(comp).graph
        let canon = canonicalize(graph)

        if case .repeating(let count, let body) = canon.rootRegion.operations[1].kind {
            #expect(count == 4)
            if case .attention(let attrs) = body.operations[0].kind {
                #expect(attrs.implementationHint == nil)
            } else {
                Issue.record("Expected attention inside repeating body")
            }
        } else {
            Issue.record("Expected repeating")
        }
    }

    // MARK: - Metadata and StructuralPath Codable

    @Test("ModelGraphMetadata roundtrips through JSON")
    func metadataCodable() throws {
        let metadata = ModelGraphMetadata(annotations: [
            AnnotationEntry(
                path: StructuralPath(components: [.operation(0)]),
                annotation: OperationAnnotation(label: "embed")
            ),
            AnnotationEntry(
                path: StructuralPath(components: [.operation(1), .regionBody, .operation(0)]),
                annotation: OperationAnnotation(label: "attn_norm")
            ),
        ])

        let data = try JSONEncoder().encode(metadata)
        let decoded = try JSONDecoder().decode(ModelGraphMetadata.self, from: data)
        #expect(metadata == decoded)
        #expect(decoded.annotation(for: StructuralPath(components: [.operation(0)]))?.label == "embed")
    }

    @Test("StructuralPath with all component types roundtrips through JSON")
    func structuralPathCodable() throws {
        let path = StructuralPath(components: [
            .operation(2),
            .regionBody,
            .regionBranch(1),
            .parameter(0),
            .operand(3),
            .result(1),
            .field("q_proj"),
            .index(5),
        ])

        let data = try JSONEncoder().encode(path)
        let decoded = try JSONDecoder().decode(StructuralPath.self, from: data)
        #expect(path == decoded)
    }

    @Test("ModelGraphMetadata returns nil for absent path")
    func metadataAbsentPath() {
        let metadata = ModelGraphMetadata(annotations: [
            AnnotationEntry(
                path: StructuralPath(components: [.operation(0)]),
                annotation: OperationAnnotation(label: "embed")
            ),
        ])

        #expect(metadata.annotation(for: StructuralPath(components: [.operation(99)])) == nil)
    }

    // MARK: - OperationSignature Edge Cases

    @Test("Custom primitive has variadic arity for both operands and results")
    func customPrimitiveVariadicArity() {
        let sig = primitiveSignature(from: .custom(CustomNodeAttributes(domain: "test", name: "multi-out")))!
        #expect(sig.operandArity == .variadic)
        #expect(sig.resultArity == .variadic)

        // Variadic resolves to fallback
        #expect(resolveArity(sig.operandArity, fallback: 3) == 3)
        #expect(resolveArity(sig.resultArity, fallback: 0) == 0)
    }

    @Test("Exact arity ignores fallback")
    func exactArityIgnoresFallback() {
        #expect(resolveArity(.exact(1), fallback: 99) == 1)
        #expect(resolveArity(.exact(0), fallback: 5) == 0)
    }

    // MARK: - OperationKind Codable Roundtrip with Regions

    @Test("OperationKind with nested region roundtrips through JSON")
    func operationKindWithRegionCodable() throws {
        let body = Region(
            parameters: [RegionParameter(id: ValueID(rawValue: 0))],
            operations: [
                Operation(
                    key: OperationKey(rawValue: 0),
                    kind: .rmsNorm(RMSNormAttributes(dimension: 64)),
                    operands: [Operand(value: ValueID(rawValue: 0))],
                    results: [OperationResult(id: ValueID(rawValue: 1))]
                ),
            ],
            results: [ValueUse(value: ValueID(rawValue: 1))]
        )
        let kinds: [OperationKind] = [
            .residual(strategy: .add, body: body),
            .parallel(merge: .concat, branches: [body, body]),
            .repeating(count: 12, body: body),
        ]

        for kind in kinds {
            let data = try JSONEncoder().encode(kind)
            let decoded = try JSONDecoder().decode(OperationKind.self, from: data)
            #expect(kind == decoded)
        }
    }

    // MARK: - NormalizedModel Metadata Integration

    @Test("Repeat label appears at correct structural path in metadata")
    func repeatLabelMetadataPath() throws {
        let comp = Group {
            TokenEmbedding(vocabSize: 100, embeddingSize: 64)
            Repeat(count: 2, label: "layers") {
                RMSNorm(dimension: 64)
            }
            OutputHead(inputSize: 64, vocabSize: 100)
        }

        let normalized = try normalize(comp)

        // Repeat is the 2nd operation in root → operation(1)
        let repeatPath = StructuralPath(components: [.operation(1)])
        #expect(normalized.metadata.annotation(for: repeatPath)?.label == "layers")
    }

    @Test("Labels inside repeating body use correct structural path")
    func labelsInsideRepeatingBody() throws {
        let comp = Repeat(count: 2, label: "layers") {
            Group(label: "block") {
                RMSNorm(dimension: 64)
            }
        }

        let normalized = try normalize(comp)

        // Repeat is operation(0), label "layers"
        let repeatPath = StructuralPath(components: [.operation(0)])
        #expect(normalized.metadata.annotation(for: repeatPath)?.label == "layers")

        // Inner label is operation(0) → regionBody → operation(0), label "block"
        let innerPath = StructuralPath(components: [.operation(0), .regionBody, .operation(0)])
        #expect(normalized.metadata.annotation(for: innerPath)?.label == "block")
    }

    // MARK: - Graph Equatable Semantics

    @Test("Graphs with same structure but different OperationKeys are not equal")
    func graphDifferentKeysNotEqual() {
        let graphA = ModelGraph(rootRegion: Region(
            operations: [
                Operation(
                    key: OperationKey(rawValue: 0),
                    kind: .rmsNorm(RMSNormAttributes(dimension: 64)),
                    results: [OperationResult(id: ValueID(rawValue: 0))]
                ),
            ],
            results: [ValueUse(value: ValueID(rawValue: 0))]
        ))
        let graphB = ModelGraph(rootRegion: Region(
            operations: [
                Operation(
                    key: OperationKey(rawValue: 999),
                    kind: .rmsNorm(RMSNormAttributes(dimension: 64)),
                    results: [OperationResult(id: ValueID(rawValue: 0))]
                ),
            ],
            results: [ValueUse(value: ValueID(rawValue: 0))]
        ))

        // Raw Equatable uses OperationKey, so they differ.
        #expect(graphA != graphB)
        // After canonicalization, they should be equal.
        #expect(canonicalize(graphA) == canonicalize(graphB))
    }

    @Test("Graphs with different attributes on same kind are not equal")
    func graphDifferentAttributesNotEqual() throws {
        let compA = RMSNorm(dimension: 64, epsilon: 1e-5)
        let compB = RMSNorm(dimension: 64, epsilon: 1e-6)

        let graphA = try normalize(compA).graph
        let graphB = try normalize(compB).graph

        #expect(graphA != graphB)
        #expect(canonicalize(graphA) != canonicalize(graphB))
    }
}
