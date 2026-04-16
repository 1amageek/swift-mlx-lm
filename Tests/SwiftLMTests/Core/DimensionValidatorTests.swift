import Testing
import Foundation
@testable import LMArchitecture

// MARK: - Helpers

/// Build a minimal well-formed graph with a single primitive operation
/// preceded by a tokenEmbedding source.
private func graphWith(
    embeddingSize: Int = 64,
    vocabSize: Int = 100,
    operation: OperationKind
) -> ModelGraph {
    var nextID = 0
    func freshVal() -> ValueID { defer { nextID += 1 }; return ValueID(rawValue: nextID) }
    func freshKey() -> OperationKey { defer { nextID += 1 }; return OperationKey(rawValue: nextID) }

    let embedResult = freshVal()
    let embedOp = Operation(
        key: freshKey(),
        kind: .tokenEmbedding(TokenEmbeddingAttributes(vocabSize: vocabSize, embeddingSize: embeddingSize)),
        operands: [],
        results: [OperationResult(id: embedResult)]
    )

    let opResult = freshVal()
    let mainOp = Operation(
        key: freshKey(),
        kind: operation,
        operands: [Operand(value: embedResult)],
        results: [OperationResult(id: opResult)]
    )

    return ModelGraph(rootRegion: Region(
        operations: [embedOp, mainOp],
        results: [ValueUse(value: opResult)]
    ))
}

/// Build a graph from a DSL model component.
private func graphFrom(_ component: some ModelComponent) throws -> ModelGraph {
    try component)
}

// MARK: - Attribute Invariant Tests

@Suite("DimensionValidator — Attribute Invariants", .tags(.unit))
struct AttributeInvariantTests {

    // MARK: Attention

    @Test("Attention: headCount * headDimension != hiddenSize is valid (rectangular output projection)")
    func attentionAsymmetricHeadDimProduct() throws {
        let graph = graphWith(embeddingSize: 64, operation: .attention(
            AttentionAttributes(
                hiddenSize: 64,
                headCount: 4,
                kvHeadCount: 4,
                headDimension: 32  // 4 * 32 = 128 != 64 — valid
            )
        ))
        try DimensionValidator.validate(graph)
    }

    @Test("Attention: kvHeadCount must not exceed headCount")
    func attentionKVHeadCountExceeds() {
        let graph = graphWith(embeddingSize: 64, operation: .attention(
            AttentionAttributes(
                hiddenSize: 64,
                headCount: 4,
                kvHeadCount: 8,  // 8 > 4
                headDimension: 16
            )
        ))
        #expect(throws: DimensionValidationError.self) {
            try DimensionValidator.validate(graph)
        }
    }

    @Test("Attention: headCount must be divisible by kvHeadCount")
    func attentionGQADivisibility() {
        let graph = graphWith(embeddingSize: 64, operation: .attention(
            AttentionAttributes(
                hiddenSize: 64,
                headCount: 4,
                kvHeadCount: 3,  // 4 % 3 != 0
                headDimension: 16
            )
        ))
        #expect(throws: DimensionValidationError.self) {
            try DimensionValidator.validate(graph)
        }
    }

    @Test("Attention: valid GQA configuration passes")
    func attentionValidGQA() throws {
        let graph = graphWith(embeddingSize: 64, operation: .attention(
            AttentionAttributes(
                hiddenSize: 64,
                headCount: 8,
                kvHeadCount: 2,  // 8 / 2 = 4 groups
                headDimension: 8 // 8 * 8 = 64
            )
        ))
        try DimensionValidator.validate(graph)
    }

    @Test("Attention: RoPE dimension must not exceed headDimension")
    func attentionRoPEDimensionExceeds() {
        let graph = graphWith(embeddingSize: 64, operation: .attention(
            AttentionAttributes(
                hiddenSize: 64,
                headCount: 4,
                kvHeadCount: 4,
                headDimension: 16,
                rope: RoPEAttributes(dimension: 20)  // 20 > 16
            )
        ))
        #expect(throws: DimensionValidationError.self) {
            try DimensionValidator.validate(graph)
        }
    }

    @Test("Attention: partial RoPE (dimension < headDimension) passes")
    func attentionPartialRoPE() throws {
        let graph = graphWith(embeddingSize: 64, operation: .attention(
            AttentionAttributes(
                hiddenSize: 64,
                headCount: 4,
                kvHeadCount: 4,
                headDimension: 16,
                rope: RoPEAttributes(dimension: 8)  // partial RoPE
            )
        ))
        try DimensionValidator.validate(graph)
    }

    @Test("Attention: RoPE dimension must be even")
    func attentionRoPEOddDimension() {
        let graph = graphWith(embeddingSize: 64, operation: .attention(
            AttentionAttributes(
                hiddenSize: 64,
                headCount: 4,
                kvHeadCount: 4,
                headDimension: 16,
                rope: RoPEAttributes(dimension: 15)  // odd
            )
        ))
        #expect(throws: DimensionValidationError.self) {
            try DimensionValidator.validate(graph)
        }
    }

    // MARK: StateSpace

    @Test("StateSpace: asymmetric DeltaNet (numHeads * valueHeadDim != hiddenSize) is valid")
    func stateSpaceAsymmetricValueDimProduct() throws {
        let graph = graphWith(embeddingSize: 64, operation: .stateSpace(
            StateSpaceAttributes(
                hiddenSize: 64,
                numHeads: 4,
                keyHeadDim: 16,
                valueHeadDim: 32,  // 4 * 32 = 128 != 64 — valid for asymmetric DeltaNet
                variant: "deltanet"
            )
        ))
        try DimensionValidator.validate(graph)
    }

    @Test("StateSpace: groupCount must not exceed numHeads")
    func stateSpaceGroupCountExceeds() {
        let graph = graphWith(embeddingSize: 64, operation: .stateSpace(
            StateSpaceAttributes(
                hiddenSize: 64,
                numHeads: 4,
                groupCount: 8,  // 8 > 4
                keyHeadDim: 16,
                valueHeadDim: 16,
                variant: "deltanet"
            )
        ))
        #expect(throws: DimensionValidationError.self) {
            try DimensionValidator.validate(graph)
        }
    }

    @Test("StateSpace: numHeads must be divisible by groupCount")
    func stateSpaceGroupDivisibility() {
        let graph = graphWith(embeddingSize: 64, operation: .stateSpace(
            StateSpaceAttributes(
                hiddenSize: 64,
                numHeads: 4,
                groupCount: 3,  // 4 % 3 != 0
                keyHeadDim: 16,
                valueHeadDim: 16,
                variant: "deltanet"
            )
        ))
        #expect(throws: DimensionValidationError.self) {
            try DimensionValidator.validate(graph)
        }
    }

    @Test("StateSpace: valid asymmetric DeltaNet passes")
    func stateSpaceValidAsymmetric() throws {
        // Qwen3.5-4B style: numHeads=32, groupCount=16, dk=128, dv=128
        let hiddenSize = 32 * 128  // 4096
        let graph = graphWith(embeddingSize: hiddenSize, operation: .stateSpace(
            StateSpaceAttributes(
                hiddenSize: hiddenSize,
                numHeads: 32,
                groupCount: 16,  // asymmetric: 32 / 16 = 2 expansion
                keyHeadDim: 128,
                valueHeadDim: 128,
                variant: "deltanet"
            )
        ))
        try DimensionValidator.validate(graph)
    }

    // MARK: MoE

    @Test("MoE: expertsPerToken must not exceed expertCount")
    func moeExpertsPerTokenExceeds() {
        let graph = graphWith(embeddingSize: 64, operation: .moe(
            MoEAttributes(
                expertCount: 4,
                expertsPerToken: 8,  // 8 > 4
                expertMLP: MLPAttributes(inputSize: 64, outputSize: 64, intermediateSize: 256)
            )
        ))
        #expect(throws: DimensionValidationError.self) {
            try DimensionValidator.validate(graph)
        }
    }

    // MARK: Non-positive dimensions

    @Test("Non-positive dimensions are rejected")
    func nonPositiveDimension() {
        let cases: [(String, OperationKind)] = [
            ("attention hiddenSize=0", .attention(
                AttentionAttributes(hiddenSize: 0, headCount: 4, kvHeadCount: 4, headDimension: 0)
            )),
            ("mlp inputSize=0", .mlp(
                MLPAttributes(inputSize: 0, outputSize: 64, intermediateSize: 256)
            )),
            ("rmsNorm dimension=0", .rmsNorm(
                RMSNormAttributes(dimension: 0)
            )),
        ]

        for (label, kind) in cases {
            let graph = graphWith(operation: kind)
            #expect(throws: DimensionValidationError.self, "\(label)") {
                try DimensionValidator.validate(graph)
            }
        }
    }
}

// MARK: - Hidden Dimension Propagation Tests

@Suite("DimensionValidator — Dimension Propagation", .tags(.unit))
struct DimensionPropagationTests {

    @Test("Norm dimension must match embedding dimension")
    func normDimensionMismatch() {
        let graph = graphWith(embeddingSize: 64, operation: .rmsNorm(
            RMSNormAttributes(dimension: 128)  // 128 != 64
        ))
        #expect(throws: DimensionValidationError.self) {
            try DimensionValidator.validate(graph)
        }
    }

    @Test("Attention hiddenSize must match preceding dimension")
    func attentionDimensionMismatch() {
        let graph = graphWith(embeddingSize: 64, operation: .attention(
            AttentionAttributes(
                hiddenSize: 128,  // 128 != 64
                headCount: 8,
                kvHeadCount: 8,
                headDimension: 16
            )
        ))
        #expect(throws: DimensionValidationError.self) {
            try DimensionValidator.validate(graph)
        }
    }

    @Test("MLP inputSize must match preceding dimension")
    func mlpInputMismatch() {
        let graph = graphWith(embeddingSize: 64, operation: .mlp(
            MLPAttributes(inputSize: 128, outputSize: 64, intermediateSize: 256)  // 128 != 64
        ))
        #expect(throws: DimensionValidationError.self) {
            try DimensionValidator.validate(graph)
        }
    }

    @Test("OutputHead inputSize must match preceding dimension")
    func outputHeadInputMismatch() throws {
        // Build manually: embed(64) → outputHead(inputSize: 128)
        var nextID = 0
        func freshVal() -> ValueID { defer { nextID += 1 }; return ValueID(rawValue: nextID) }
        func freshKey() -> OperationKey { defer { nextID += 1 }; return OperationKey(rawValue: nextID) }

        let embedResult = freshVal()
        let headResult = freshVal()

        let graph = ModelGraph(rootRegion: Region(
            operations: [
                Operation(
                    key: freshKey(),
                    kind: .tokenEmbedding(TokenEmbeddingAttributes(vocabSize: 100, embeddingSize: 64)),
                    operands: [],
                    results: [OperationResult(id: embedResult)]
                ),
                Operation(
                    key: freshKey(),
                    kind: .outputHead(OutputHeadAttributes(inputSize: 128, vocabSize: 100)),  // 128 != 64
                    operands: [Operand(value: embedResult)],
                    results: [OperationResult(id: headResult)]
                ),
            ],
            results: [ValueUse(value: headResult)]
        ))

        #expect(throws: DimensionValidationError.self) {
            try DimensionValidator.validate(graph)
        }
    }

    @Test("Valid linear chain passes propagation")
    func validLinearChain() throws {
        let component = Group {
            TokenEmbedding(vocabSize: 100, embeddingSize: 64)
            RMSNorm(dimension: 64)
            Attention(hiddenSize: 64, headCount: 4, kvHeadCount: 4, headDimension: 16)
            RMSNorm(dimension: 64)
            MLP(inputSize: 64, intermediateSize: 256)
            RMSNorm(dimension: 64)
            OutputHead(inputSize: 64, vocabSize: 100)
        }

        let graph = try graphFrom(component)
        try DimensionValidator.validate(graph)
    }
}

// MARK: - Structural Operation Tests

@Suite("DimensionValidator — Structural Operations", .tags(.unit))
struct StructuralDimensionTests {

    @Test("Residual(.add) body must preserve dimension")
    func residualAddPreservesDimension() throws {
        // Valid: body outputs same dimension
        let validComponent = Group {
            TokenEmbedding(vocabSize: 100, embeddingSize: 64)
            Residual {
                RMSNorm(dimension: 64)
                Attention(hiddenSize: 64, headCount: 4, kvHeadCount: 4, headDimension: 16)
            }
        }
        let validGraph = try graphFrom(validComponent)
        try DimensionValidator.validate(validGraph)
    }

    @Test("Residual(.add) body with dimension change fails")
    func residualAddDimensionChange() {
        // Build manually: residual body contains mlp that changes dimension
        var nextID = 0
        func freshVal() -> ValueID { defer { nextID += 1 }; return ValueID(rawValue: nextID) }
        func freshKey() -> OperationKey { defer { nextID += 1 }; return OperationKey(rawValue: nextID) }

        let embedResult = freshVal()
        let bodyParam = freshVal()
        let mlpResult = freshVal()
        let residualResult = freshVal()

        let body = Region(
            parameters: [RegionParameter(id: bodyParam)],
            operations: [
                Operation(
                    key: freshKey(),
                    kind: .mlp(MLPAttributes(inputSize: 64, outputSize: 128, intermediateSize: 256)),
                    operands: [Operand(value: bodyParam)],
                    results: [OperationResult(id: mlpResult)]
                ),
            ],
            results: [ValueUse(value: mlpResult)]
        )

        let graph = ModelGraph(rootRegion: Region(
            operations: [
                Operation(
                    key: freshKey(),
                    kind: .tokenEmbedding(TokenEmbeddingAttributes(vocabSize: 100, embeddingSize: 64)),
                    operands: [],
                    results: [OperationResult(id: embedResult)]
                ),
                Operation(
                    key: freshKey(),
                    kind: .residual(strategy: .add, body: body),
                    operands: [Operand(value: embedResult)],
                    results: [OperationResult(id: residualResult)]
                ),
            ],
            results: [ValueUse(value: residualResult)]
        ))

        #expect(throws: DimensionValidationError.self) {
            try DimensionValidator.validate(graph)
        }
    }

    @Test("Repeating body must preserve dimension")
    func repeatingPreservesDimension() throws {
        let component = Group {
            TokenEmbedding(vocabSize: 100, embeddingSize: 64)
            Repeat(count: 4) {
                Residual {
                    RMSNorm(dimension: 64)
                    Attention(hiddenSize: 64, headCount: 4, kvHeadCount: 4, headDimension: 16)
                }
                Residual {
                    RMSNorm(dimension: 64)
                    MLP(inputSize: 64, intermediateSize: 256)
                }
            }
            OutputHead(inputSize: 64, vocabSize: 100)
        }

        let graph = try graphFrom(component)
        try DimensionValidator.validate(graph)
    }

    @Test("Parallel(.add) branches must all produce same dimension")
    func parallelAddBranchesSameDimension() throws {
        let component = Group {
            TokenEmbedding(vocabSize: 100, embeddingSize: 64)
            Residual {
                RMSNorm(dimension: 64)
                Parallel(merge: .add) {
                    Attention(hiddenSize: 64, headCount: 4, kvHeadCount: 4, headDimension: 16)
                    MLP(inputSize: 64, intermediateSize: 256)
                }
            }
            OutputHead(inputSize: 64, vocabSize: 100)
        }

        let graph = try graphFrom(component)
        try DimensionValidator.validate(graph)
    }
}

// MARK: - Real Architecture Tests

@Suite("DimensionValidator — Real Architectures", .tags(.unit))
struct RealArchitectureDimensionTests {

    @Test("TinyLlama passes dimension validation")
    func tinyLlama() throws {
        let model = TinyLlama(
            vocabSize: 100, hiddenSize: 64, headCount: 4,
            kvHeadCount: 2, intermediateSize: 256, layerCount: 2
        )
        let graph = try ModelGraph(model)
        try DimensionValidator.validate(graph)
    }

    @Test("TinyCohere passes dimension validation")
    func tinyCohere() throws {
        let model = TinyCohere(
            vocabSize: 100, hiddenSize: 64, headCount: 4,
            kvHeadCount: 2, intermediateSize: 256, layerCount: 2
        )
        let graph = try ModelGraph(model)
        try DimensionValidator.validate(graph)
    }

    @Test("TinyMixtral passes dimension validation")
    func tinyMixtral() throws {
        let model = TinyMixtral(
            vocabSize: 100, hiddenSize: 64, headCount: 4,
            kvHeadCount: 2, intermediateSize: 256,
            expertCount: 4, expertsPerToken: 2, layerCount: 2
        )
        let graph = try ModelGraph(model)
        try DimensionValidator.validate(graph)
    }

    @Test("TinyJamba (Mamba + Attention hybrid) passes dimension validation")
    func tinyJamba() throws {
        let model = TinyJamba(
            vocabSize: 100, hiddenSize: 64, headCount: 4,
            kvHeadCount: 2, intermediateSize: 256, stateSize: 16
        )
        let graph = try ModelGraph(model)
        try DimensionValidator.validate(graph)
    }

    @Test("DeltaNet with numHeads * valueHeadDim != hiddenSize is valid (rectangular output projection)")
    func asymmetricDeltaNetOutputProjection() throws {
        // Asymmetric DeltaNet: output projection is [numHeads * valueHeadDim] → [hiddenSize] matmul.
        // The two dimensions need not be equal.
        let graph = graphWith(embeddingSize: 4, operation: .stateSpace(
            StateSpaceAttributes(
                hiddenSize: 4,
                numHeads: 1,
                keyHeadDim: 2,
                valueHeadDim: 2,  // 1 * 2 = 2 != 4 — valid
                variant: "deltanet"
            )
        ))

        try DimensionValidator.validate(graph)
    }

    @Test("Realistic DeltaNet dimensions (Qwen3.5 0.8B style) pass")
    func realisticDeltaNetDimensions() throws {
        // numHeads=16, groupCount=16, dk=128, dv=128, hiddenSize=2048
        let hiddenSize = 16 * 128
        let model = Group {
            TokenEmbedding(vocabSize: 32000, embeddingSize: hiddenSize)
            Residual {
                RMSNorm(dimension: hiddenSize)
                StateSpace(
                    hiddenSize: hiddenSize,
                    numHeads: 16,
                    groupCount: 16,
                    keyHeadDim: 128,
                    valueHeadDim: 128,
                    variant: "deltanet"
                )
            }
            Residual {
                RMSNorm(dimension: hiddenSize)
                MLP(inputSize: hiddenSize, intermediateSize: hiddenSize * 4)
            }
            RMSNorm(dimension: hiddenSize)
            OutputHead(inputSize: hiddenSize, vocabSize: 32000)
        }

        let graph = try graphFrom(model)
        try DimensionValidator.validate(graph)
    }

    @Test("Realistic asymmetric DeltaNet dimensions (Qwen3.5 4B style) pass")
    func realisticAsymmetricDeltaNetDimensions() throws {
        // Qwen3.5-4B: numHeads=32, groupCount=16, dk=128, dv=128, hiddenSize=2560
        // numHeads * valueHeadDim = 4096 != hiddenSize = 2560 (rectangular output projection)
        let hiddenSize = 2560
        let model = Group {
            TokenEmbedding(vocabSize: 32000, embeddingSize: hiddenSize)
            Residual {
                RMSNorm(dimension: hiddenSize)
                StateSpace(
                    hiddenSize: hiddenSize,
                    numHeads: 32,
                    groupCount: 16,
                    keyHeadDim: 128,
                    valueHeadDim: 128,
                    variant: "deltanet"
                )
            }
            RMSNorm(dimension: hiddenSize)
            OutputHead(inputSize: hiddenSize, vocabSize: 32000)
        }

        let graph = try graphFrom(model)
        try DimensionValidator.validate(graph)
    }
}

// Tiny model types are defined in DSLLoweringTests.swift (TinyLlama, TinyCohere)
// and IRInvariantTests.swift (TinyMixtral, TinyJamba) — reused here directly.
