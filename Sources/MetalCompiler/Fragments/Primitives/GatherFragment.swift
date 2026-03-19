import Metal

/// Token ID → embedding vector lookup.
public struct GatherFragment: PrimitiveMetalKernelFragment {
    public let vocabularySize: Int
    public let embeddingDimension: Int

    public init(vocabularySize: Int, embeddingDimension: Int) {
        self.vocabularySize = vocabularySize
        self.embeddingDimension = embeddingDimension
    }

    public var isFusable: Bool { false }
    public func kernelName(context: KernelContext) -> String {
        let bf16 = context.weightFormat == .bfloat16
        if context.bufferPrecision == .float32 {
            return bf16 ? "embedding_lookup_seq_bf16_f32" : "embedding_lookup_seq_f32"
        }
        return bf16 ? "embedding_lookup_bf16" : "embedding_lookup"
    }
    public var dispatchDimension: MetalDispatchDimension { .gather(count: embeddingDimension) }
    public var weightSlots: [MetalWeightSlot] { [MetalWeightSlot(field: nil, role: .weight)] }

    public func kernelSource(name: String, bufferPrecision: BufferPrecision, weightFormat: WeightFormat) -> String {
        MetalSourceGenerator.generateEmbeddingLookup(name: name, bufferPrecision: bufferPrecision, weightFormat: weightFormat, isSequence: bufferPrecision == .float32)
    }

    public func decodeBindings(context: BufferBindingContext) -> FragmentBindings {
        let (weightBuffer, weightOffset) = context.resolveWeight("embedding_table")
        return FragmentBindings(
            buffers: [
                (0, context.bufferSet.tokenIn, 0),
                (1, weightBuffer, weightOffset),
                (2, context.bufferSet.hidden, 0),
            ],
            bytes: [
                uint32Binding(3, UInt32(embeddingDimension)),
            ],
            outputIsHidden: true
        )
    }

    public func prefillSteps(context: PrefillBindingContext) throws -> FragmentPrefillSteps {
        let (weightBuffer, weightOffset) = context.resolveWeight("embedding_table")
        let kernelName = kernelName(context: context.kernelContext)
        let pipeline = try context.getPipeline(kernelName)
        let tgSize = min(256, pipeline.maxTotalThreadsPerThreadgroup)
        let gridX = (embeddingDimension + tgSize - 1) / tgSize
        return FragmentPrefillSteps(
            steps: [MetalPrefillStep(
                pipeline: pipeline,
                gridSize: MTLSize(width: gridX, height: context.maximumSequenceLength, depth: 1),
                threadgroupSize: MTLSize(width: tgSize, height: 1, depth: 1),
                bufferBindings: [
                    (0, context.buffers.tokenIDs, 0),
                    (1, weightBuffer, weightOffset),
                    (2, context.buffers.hidden, 0),
                ],
                bytesBindings: [
                    uint32Binding(3, UInt32(embeddingDimension)),
                    uint32Binding(4, UInt32(context.maximumSequenceLength)),
                ],
                threadgroupMemoryLength: 0,
                sync: .bufferBarrier,
                mode: .batch,
                sequenceLengthPolicy: .bindAndAdjustGridHeight(index: 4),
                positionBufferIndex: nil,
                perPositionStrides: [:]
            )],
            outputIsHidden: true
        )
    }
}
