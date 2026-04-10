import Metal

/// Extracts the query half from a head-interleaved packed Q projection layout.
///
/// Source layout per head is `[query(headDim), gate(headDim)]`.
/// Output layout is contiguous queries `[query0, query1, ...]`.
public struct PackedQueryExtractFragment: PrimitiveMetalKernelFragment {
    public let headCount: Int
    public let headDimension: Int
    public let packedSourceSlotIndex: Int
    public let outputSlotIndex: Int

    public init(
        headCount: Int,
        headDimension: Int,
        packedSourceSlotIndex: Int,
        outputSlotIndex: Int
    ) {
        self.headCount = headCount
        self.headDimension = headDimension
        self.packedSourceSlotIndex = packedSourceSlotIndex
        self.outputSlotIndex = outputSlotIndex
    }

    public var isFusable: Bool { true }

    public func kernelName(context: KernelContext) -> String {
        context.bufferPrecision == .float32 ? "packed_query_extract_seq_f32" : "packed_query_extract"
    }

    public var dispatchDimension: MetalDispatchDimension {
        .elementwise(count: headCount * headDimension)
    }

    public func kernelSource(name: String, bufferPrecision: BufferPrecision, weightFormat: WeightFormat) -> String {
        MetalSourceGenerator.generatePackedQueryExtract(
            name: name,
            bufferPrecision: bufferPrecision,
            isSequence: bufferPrecision == .float32
        )
    }

    public func decodeBindings(context: BufferBindingContext) -> FragmentBindings {
        let slotBytes = context.slotDimension * context.elementSize
        return FragmentBindings(
            buffers: [
                (0, context.bufferSet.scratch, packedSourceSlotIndex * slotBytes),
                (1, context.bufferSet.scratch, outputSlotIndex * slotBytes),
            ],
            bytes: [
                uint32Binding(2, UInt32(headCount)),
                uint32Binding(3, UInt32(headDimension)),
            ],
            outputIsHidden: false,
            writeBufferIndices: Set<Int>([1])
        )
    }

    public func prefillSteps(context: PrefillBindingContext) throws -> FragmentPrefillSteps {
        let pipeline = try context.getPipeline(kernelName(context: context.kernelContext))
        let tgSize = min(256, pipeline.maxTotalThreadsPerThreadgroup)
        let elementCount = headCount * headDimension
        let gridX = (elementCount + tgSize - 1) / tgSize
        let scratchSlotSize = context.slotDimension * context.scratchElementSize * context.maximumSequenceLength

        return FragmentPrefillSteps(
            steps: [MetalPrefillStep(
                pipeline: pipeline,
                gridSize: MTLSize(width: gridX, height: context.maximumSequenceLength, depth: 1),
                threadgroupSize: MTLSize(width: tgSize, height: 1, depth: 1),
                bufferBindings: [
                    (0, context.buffers.scratch, packedSourceSlotIndex * scratchSlotSize),
                    (1, context.buffers.scratch, outputSlotIndex * scratchSlotSize),
                ],
                bytesBindings: [
                    uint32Binding(2, UInt32(headCount)),
                    uint32Binding(3, UInt32(headDimension)),
                    uint32Binding(4, UInt32(headCount * headDimension * 2)),
                    uint32Binding(5, UInt32(headCount * headDimension)),
                    uint32Binding(6, UInt32(context.maximumSequenceLength)),
                ],
                threadgroupMemoryLength: 0,
                sync: .bufferBarrier,
                mode: .batch,
                sequenceLengthPolicy: .bindAndAdjustGridHeight(index: 6),
                positionBufferIndex: nil,
                perPositionStrides: [:]
            )],
            outputIsHidden: false
        )
    }
}
