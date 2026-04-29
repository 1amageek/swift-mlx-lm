import Metal

/// Token ID → embedding vector lookup.
public struct GatherFragment: PrimitiveMetalKernelFragment {
    public let vocabularySize: Int
    public let embeddingDimension: Int
    public let embeddingScale: Float?

    public init(vocabularySize: Int, embeddingDimension: Int, embeddingScale: Float? = nil) {
        self.vocabularySize = vocabularySize
        self.embeddingDimension = embeddingDimension
        self.embeddingScale = embeddingScale
    }

    public var isFusable: Bool { false }
    public func kernelName(context: KernelContext) -> String {
        let scaled = embeddingScale != nil ? "_scaled" : ""
        let token = context.weightFormat.embeddingLookupToken
        let tokenPart = token.isEmpty ? "" : "_\(token)"
        let isSeq = context.bufferPrecision.isPrefillSequencePrecision
        return isSeq
            ? "embedding_lookup_seq\(tokenPart)_f32\(scaled)"
            : "embedding_lookup\(tokenPart)\(scaled)\(context.bufferPrecision.decodeKernelNameSuffix)"
    }
    public var dispatchDimension: MetalDispatchDimension { .gather(count: embeddingDimension) }
    public var weightSlots: [MetalWeightSlot] { [MetalWeightSlot(field: nil, role: .weight)] }

    public func kernelSource(name: String, bufferPrecision: BufferPrecision, weightFormat: WeightFormat) -> String {
        weightFormat.embeddingLookupKernelSource(
            name: name,
            bufferPrecision: bufferPrecision,
            isSequence: bufferPrecision.isPrefillSequencePrecision,
            embeddingScale: embeddingScale
        )
    }

    public func requiredFallbackBufferSize(for role: String, bytesPerScalar: Int) -> Int {
        vocabularySize * embeddingDimension * bytesPerScalar
    }

    public func decodeBindings(context: BufferBindingContext) -> FragmentBindings {
        let (weightBuffer, weightOffset) = context.resolveWeight("embedding_table")
        var bytes: [(Int, [UInt8])] = [
            uint32Binding(3, UInt32(embeddingDimension)),
        ]
        if let scale = embeddingScale {
            bytes.append(floatBinding(4, scale))
        }
        return FragmentBindings(
            buffers: [
                (0, context.bufferSet.tokenIn, 0),
                (1, weightBuffer, weightOffset),
                (2, context.bufferSet.hidden, 0),
            ],
            bytes: bytes,
            outputIsHidden: true,
            writeBufferIndices: Set<Int>([2])
        )
    }

    public func prefillSteps(context: PrefillBindingContext) throws -> FragmentPrefillSteps {
        let (weightBuffer, weightOffset) = context.resolveWeight("embedding_table")
        let kernelName = kernelName(context: context.kernelContext)
        let pipeline = try context.getPipeline(kernelName)
        let tgSize = min(256, pipeline.maxTotalThreadsPerThreadgroup)
        let gridX = (embeddingDimension + tgSize - 1) / tgSize
        var bytes: [(Int, [UInt8])] = [
            uint32Binding(3, UInt32(embeddingDimension)),
            uint32Binding(4, UInt32(context.maximumSequenceLength)),
        ]
        if let scale = embeddingScale {
            bytes.append(floatBinding(5, scale))
        }
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
                bytesBindings: bytes,
                threadgroupMemoryLength: 0,
                sync: .bufferBarrier,
                mode: .batch,
                sequenceLengthPolicy: .bindAndAdjustGridHeight(index: 4),
                positionBufferIndex: nil,
                perPositionStrides: [:],
                metadata: .init(
                    kernelName: kernelName,
                    bufferAccessPattern: .init(reads: [0, 1], writes: [2])
                )
            )],
            outputIsHidden: true
        )
    }
}
