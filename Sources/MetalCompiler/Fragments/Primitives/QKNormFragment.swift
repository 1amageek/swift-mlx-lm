import Metal

/// Per-head RMS normalization for Q or K projections.
public struct QKNormFragment: PrimitiveMetalKernelFragment {
    public let headCount: Int
    public let headDimension: Int
    public let epsilon: Float
    public let weightRole: String  // "q_layernorm" or "k_layernorm"

    public init(headCount: Int, headDimension: Int, epsilon: Float, weightRole: String) {
        self.headCount = headCount
        self.headDimension = headDimension
        self.epsilon = epsilon
        self.weightRole = weightRole
    }

    public var isFusable: Bool { true }
    public var isInPlace: Bool { true }
    public var normEpsilon: Float? { epsilon }
    public func kernelName(context: KernelContext) -> String {
        if context.bufferPrecision == .float32 { return "qk_rms_norm_seq_f32" }
        return context.weightFormat == .bfloat16 ? "qk_rms_norm_bf16" : "qk_rms_norm"
    }
    public var dispatchDimension: MetalDispatchDimension { .perHead(headCount: headCount) }
    public var weightSlots: [MetalWeightSlot] { [MetalWeightSlot(field: weightRole, role: .weight)] }

    public func kernelSource(name: String, bufferPrecision: BufferPrecision, weightFormat: WeightFormat) -> String {
        bufferPrecision == .float32
            ? MetalSourceGenerator.generateQKNormSeq(name: name, bufferPrecision: bufferPrecision, weightFormat: weightFormat)
            : MetalSourceGenerator.generateQKNorm(name: name, bufferPrecision: bufferPrecision, weightFormat: weightFormat)
    }

    public func decodeBindings(context: BufferBindingContext) -> FragmentBindings {
        let slotBytes = context.slotDimension * context.elementSize
        let scratchSlotIndex = weightRole == "q_layernorm" ? 1 : 2
        let (weightBuffer, weightOffset) = context.resolveWeight(weightRole)
        return FragmentBindings(
            buffers: [
                (0, context.bufferSet.scratch, scratchSlotIndex * slotBytes),
                (1, weightBuffer, weightOffset),
            ],
            bytes: [
                uint32Binding(2, UInt32(headCount)),
                uint32Binding(3, UInt32(headDimension)),
                floatBinding(4, epsilon),
            ],
            outputIsHidden: false
        )
    }

    public func prefillSteps(context: PrefillBindingContext) throws -> FragmentPrefillSteps {
        let kernelName = kernelName(context: context.kernelContext)
        let pipeline = try context.getPipeline(kernelName)
        let scratchSlotSize = context.slotDimension * context.scratchElementSize * context.maximumSequenceLength
        let scratchSlotIndex = weightRole == "q_layernorm" ? 1 : 2
        let (weightBuffer, weightOffset) = context.resolveWeight(weightRole)
        let totalDimension = headCount * headDimension
        let threads = min(32, pipeline.maxTotalThreadsPerThreadgroup)
        return FragmentPrefillSteps(
            steps: [MetalPrefillStep(
                pipeline: pipeline,
                gridSize: MTLSize(width: headCount, height: context.maximumSequenceLength, depth: 1),
                threadgroupSize: MTLSize(width: threads, height: 1, depth: 1),
                bufferBindings: [
                    (0, context.buffers.scratch, scratchSlotIndex * scratchSlotSize),
                    (1, weightBuffer, weightOffset),
                ],
                bytesBindings: [
                    uint32Binding(2, UInt32(headCount)),
                    uint32Binding(3, UInt32(headDimension)),
                    floatBinding(4, epsilon),
                    uint32Binding(5, UInt32(context.maximumSequenceLength)),
                    uint32Binding(6, UInt32(totalDimension)),
                ],
                threadgroupMemoryLength: 0,
                sync: .bufferBarrier,
                mode: .batch,
                sequenceLengthPolicy: .bindAndAdjustGridHeight(index: 5),
                positionBufferIndex: nil,
                perPositionStrides: [:]
            )],
            outputIsHidden: false
        )
    }
}
