import LMIR

extension TokenEmbeddingAttributes: MetalComponent {

    public var dispatchDeclarations: [MetalDispatchDeclaration] {
        [.compute(EmbeddingLookupOperation(vocabularySize: vocabSize, embeddingDimension: embeddingSize))]
    }

    public var weightSlots: [MetalWeightSlot] {
        [MetalWeightSlot(role: .embeddingTable)]
    }
}
