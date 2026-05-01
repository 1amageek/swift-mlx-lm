import Metal

/// Applies `sigmoid(gate) * input` when gate values are packed after the query
/// projection in the same scratch slot.
public struct PackedSigmoidGateFragment: PrimitiveMetalKernelFragment {
    public let dimension: Int
    public let headDimension: Int
    public let packedSourceSlotIndex: Int
    public let packedHeadStride: Int
    public let gateHeadOffset: Int

    public init(
        dimension: Int,
        headDimension: Int,
        packedSourceSlotIndex: Int,
        packedHeadStride: Int,
        gateHeadOffset: Int
    ) {
        self.dimension = dimension
        self.headDimension = headDimension
        self.packedSourceSlotIndex = packedSourceSlotIndex
        self.packedHeadStride = packedHeadStride
        self.gateHeadOffset = gateHeadOffset
    }

    /// Not fusable: packed head-interleaved layout requires non-uniform buffer
    /// stride (packedHeadStride) incompatible with perElement scaffold's uniform
    /// element indexing.
    public var isFusable: Bool { false }

    public func kernelName(context: KernelContext) -> String {
        context.bufferPrecision == .float32 ? "packed_sigmoid_gate_seq_f32" : "packed_sigmoid_gate"
    }

    public var dispatchDimension: MetalDispatchDimension { .elementwise(count: dimension) }

    public func kernelSource(name: String, bufferPrecision: BufferPrecision, weightFormat: WeightFormat) -> String {
        MetalSourceGenerator.generatePackedSigmoidGate(
            name: name,
            bufferPrecision: bufferPrecision,
            isSequence: bufferPrecision == .float32
        )
    }

    public func decodeBindings(context: BufferBindingContext) -> FragmentBindings {
        let slotBytes = context.slotDimension * context.elementSize
        return FragmentBindings(
            buffers: [
                (0, context.bufferSet.scratch, 0),
                (1, context.bufferSet.scratch, packedSourceSlotIndex * slotBytes),
                (2, context.bufferSet.scratch, 0),
            ],
            bytes: [
                uint32Binding(3, UInt32(dimension)),
                uint32Binding(4, UInt32(headDimension)),
                uint32Binding(5, UInt32(packedHeadStride)),
                uint32Binding(6, UInt32(gateHeadOffset)),
            ],
            outputIsHidden: false,
            writeBufferIndices: Set<Int>([2])
        )
    }

    public func prefillSteps(context: PrefillBindingContext) throws -> FragmentPrefillSteps {
        let pipeline = try context.getPipeline(kernelName(context: context.kernelContext))
        let tgSize = min(256, pipeline.maxTotalThreadsPerThreadgroup)
        let gridX = (dimension + tgSize - 1) / tgSize
        let scratchSlotSize = context.slotDimension * context.scratchElementSize * context.maximumSequenceLength

        return FragmentPrefillSteps(
            steps: [MetalPrefillStep(
                pipeline: pipeline,
                gridSize: MTLSize(width: gridX, height: context.maximumSequenceLength, depth: 1),
                threadgroupSize: MTLSize(width: tgSize, height: 1, depth: 1),
                bufferBindings: [
                    (0, context.buffers.scratch, 0),
                    (1, context.buffers.scratch, packedSourceSlotIndex * scratchSlotSize),
                    (2, context.buffers.scratch, 0),
                ],
                bytesBindings: [
                    uint32Binding(3, UInt32(dimension)),
                    uint32Binding(4, UInt32(headDimension)),
                    uint32Binding(5, UInt32(packedHeadStride)),
                    uint32Binding(6, UInt32(gateHeadOffset)),
                    // Source (packed Q slot) and destination (attention output slot 0)
                    // are both slotDimension-strided in scratch — the packed Q was
                    // written by BatchedSequenceGEMV at slotDimension stride, and the
                    // attention output (slot 0) was written by flash_attn_batch_f32 at
                    // activationRowStride = slotDimension. Reading/writing at the
                    // narrower attention dimension misaligns positions >= 1.
                    uint32Binding(7, UInt32(context.slotDimension)),
                    uint32Binding(8, UInt32(context.slotDimension)),
                    uint32Binding(9, UInt32(context.maximumSequenceLength)),
                ],
                threadgroupMemoryLength: 0,
                sync: .bufferBarrier,
                mode: .batch,
                sequenceLengthPolicy: .bindAndAdjustGridHeight(index: 9),
                positionBufferIndex: nil,
                perPositionStrides: [:],
                metadata: .init(
                    kernelName: kernelName(context: context.kernelContext),
                    bufferAccessPattern: .init(reads: [0, 1], writes: [2])
                )
            )],
            outputIsHidden: false
        )
    }
}
