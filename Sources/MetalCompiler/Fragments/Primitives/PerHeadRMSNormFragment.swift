import Metal

/// Per-head RMS normalization without a learned scale.
public struct PerHeadRMSNormFragment: PrimitiveMetalKernelFragment {
    public let headCount: Int
    public let headDimension: Int
    public let epsilon: Float
    public let scratchSlotIndex: Int

    public init(
        headCount: Int,
        headDimension: Int,
        epsilon: Float,
        scratchSlotIndex: Int
    ) {
        self.headCount = headCount
        self.headDimension = headDimension
        self.epsilon = epsilon
        self.scratchSlotIndex = scratchSlotIndex
    }

    public var isFusable: Bool { false }
    public func kernelName(context: KernelContext) -> String {
        context.bufferPrecision == .float32
            ? "per_head_rms_norm_seq_f32"
            : "per_head_rms_norm"
    }
    public var dispatchDimension: MetalDispatchDimension { .perHead(headCount: headCount) }

    public func kernelSource(
        name: String,
        bufferPrecision: BufferPrecision,
        weightFormat: WeightFormat
    ) -> String {
        MetalSourceGenerator.generatePerHeadRMSNorm(
            name: name,
            bufferPrecision: bufferPrecision,
            isSequence: bufferPrecision == .float32
        )
    }

    public func decodeBindings(context: BufferBindingContext) -> FragmentBindings {
        let slotBytes = context.slotDimension * context.elementSize
        return FragmentBindings(
            buffers: [
                (0, context.bufferSet.scratch, scratchSlotIndex * slotBytes),
            ],
            bytes: [
                uint32Binding(1, UInt32(headCount)),
                uint32Binding(2, UInt32(headDimension)),
                floatBinding(3, epsilon),
            ],
            outputIsHidden: false,
            writeBufferIndices: Set([0])
        )
    }

    public func prefillSteps(context: PrefillBindingContext) throws -> FragmentPrefillSteps {
        let pipeline = try context.getPipeline(kernelName(context: context.kernelContext))
        let scratchSlotSize = context.slotDimension * context.scratchElementSize * context.maximumSequenceLength
        let threads = min(32, pipeline.maxTotalThreadsPerThreadgroup)
        return FragmentPrefillSteps(
            steps: [MetalPrefillStep(
                pipeline: pipeline,
                gridSize: MTLSize(width: headCount, height: context.maximumSequenceLength, depth: 1),
                threadgroupSize: MTLSize(width: threads, height: 1, depth: 1),
                bufferBindings: [
                    (0, context.buffers.scratch, scratchSlotIndex * scratchSlotSize),
                ],
                bytesBindings: [
                    uint32Binding(1, UInt32(headCount)),
                    uint32Binding(2, UInt32(headDimension)),
                    floatBinding(3, epsilon),
                    uint32Binding(4, UInt32(context.maximumSequenceLength)),
                    uint32Binding(5, UInt32(headCount * headDimension)),
                ],
                threadgroupMemoryLength: 0,
                sync: .bufferBarrier,
                mode: .batch,
                sequenceLengthPolicy: .bindAndAdjustGridHeight(index: 4),
                positionBufferIndex: nil,
                perPositionStrides: [:]
            )],
            outputIsHidden: false
        )
    }
}
