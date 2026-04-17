import Metal

/// DeltaNet/Mamba state-space model recurrence step.
public struct SSMRecurrenceFragment: PrimitiveMetalKernelFragment {
    public let headCount: Int
    public let groupCount: Int
    public let keyHeadDimension: Int
    public let valueHeadDimension: Int
    public let convKernelSize: Int

    public init(
        headCount: Int,
        groupCount: Int,
        keyHeadDimension: Int,
        valueHeadDimension: Int,
        convKernelSize: Int
    ) {
        self.headCount = headCount
        self.groupCount = groupCount
        self.keyHeadDimension = keyHeadDimension
        self.valueHeadDimension = valueHeadDimension
        self.convKernelSize = convKernelSize
    }

    public var isFusable: Bool { false }
    public func kernelName(context: KernelContext) -> String {
        Self.kernelName(
            bufferPrecision: context.bufferPrecision,
            weightFormat: context.weightFormat
        )
    }
    public var dispatchDimension: MetalDispatchDimension {
        .reduction(dimension: convDimension)
    }
    public var weightSlots: [MetalWeightSlot] {
        [
            MetalWeightSlot(field: "conv_weight", role: .weight),
            MetalWeightSlot(field: "scale", role: .scale),
        ]
    }
    public var cacheSlots: [MetalCacheSlot] {
        [
            MetalCacheSlot(name: "linear_conv_cache", kind: .conv, temporalSize: convKernelSize),
            MetalCacheSlot(name: "linear_recurrent_state", kind: .recurrent),
        ]
    }

    public var convDimension: Int {
        2 * groupCount * keyHeadDimension + headCount * valueHeadDimension
    }

    /// Upper bound for threadgroup size used in SSM kernels.
    /// Matches the normPartials array size in the generated kernel source.
    /// At dispatch time, actual threads = min(this, pipeline.maxTotalThreadsPerThreadgroup).
    public static let maxThreadgroupSize = 1024

    static func kernelName(
        bufferPrecision: BufferPrecision,
        weightFormat: WeightFormat
    ) -> String {
        let bf16 = weightFormat == .bfloat16
        if bufferPrecision == .float32 {
            return bf16 ? "ssm_recurrence_bf16_f32" : "ssm_recurrence_f32"
        }
        return bf16 ? "ssm_recurrence_bf16" : "ssm_recurrence"
    }

    static func sequenceKernelName(
        bufferPrecision: BufferPrecision,
        weightFormat: WeightFormat
    ) -> String {
        let bf16 = weightFormat == .bfloat16
        if bufferPrecision == .float32 {
            return bf16 ? "ssm_recurrence_seq_bf16_f32" : "ssm_recurrence_seq_f32"
        }
        return bf16 ? "ssm_recurrence_seq_bf16" : "ssm_recurrence_seq"
    }

    public func requiredFallbackBufferSize(for role: String, bytesPerScalar: Int) -> Int {
        switch role {
        case "conv_weight":
            return convDimension * convKernelSize * bytesPerScalar
        default:
            return convDimension * bytesPerScalar
        }
    }

    public func kernelSource(name: String, bufferPrecision: BufferPrecision, weightFormat: WeightFormat) -> String {
        MetalSourceGenerator.generateSSMRecurrence(
            name: name,
            bufferPrecision: bufferPrecision,
            weightFormat: weightFormat,
            convDimension: convDimension,
            maxThreadgroupSize: Self.maxThreadgroupSize
        )
    }

    public func decodeBindings(context: BufferBindingContext) -> FragmentBindings {
        let scratchSlotBytes = context.slotDimension * context.elementSize

        let (convWeightBuffer, convWeightOffset) = context.resolveWeight("conv_weight")
        let (normWeightBuffer, normWeightOffset) = context.resolveWeight("scale")
        let (dtBiasBuffer, dtBiasOffset) = context.resolveWeight("dt_bias")
        let (aLogBuffer, aLogOffset) = context.resolveWeight("A_log")

        guard let recurrentState = context.bufferSet.recurrentState else {
            fatalError("[Compiler] SSMRecurrenceFragment requires recurrent state buffer")
        }
        guard let convState = context.bufferSet.convState else {
            fatalError("[Compiler] SSMRecurrenceFragment requires conv state buffer")
        }

        let recurrentLayerOffset = context.recurrentLayerIndex * context.bufferSet.recurrentStateBytesPerLayer
        let convStateElementSize = MemoryLayout<Float16>.size
        let convLayerOffset = context.convLayerIndex
            * context.bufferSet.convStateKernelSize
            * context.bufferSet.convStateDimension
            * convStateElementSize

        return FragmentBindings(
            buffers: [
                (0, context.bufferSet.scratch, 1 * scratchSlotBytes),
                (1, context.bufferSet.scratch, 2 * scratchSlotBytes),
                (2, context.bufferSet.scratch, 3 * scratchSlotBytes),
                (3, context.bufferSet.scratch, 4 * scratchSlotBytes),
                (4, convWeightBuffer, convWeightOffset),
                (5, normWeightBuffer, normWeightOffset),
                (6, dtBiasBuffer, dtBiasOffset),
                (7, aLogBuffer, aLogOffset),
                (8, recurrentState, recurrentLayerOffset),
                (9, convState, convLayerOffset),
                (10, context.bufferSet.scratch, 0),
            ],
            bytes: [
                uint32Binding(11, UInt32(headCount)),
                uint32Binding(12, UInt32(groupCount)),
                uint32Binding(13, UInt32(keyHeadDimension)),
                uint32Binding(14, UInt32(valueHeadDimension)),
                uint32Binding(15, UInt32(convKernelSize)),
            ],
            outputIsHidden: false,
            resetsProjectionIndex: true,
            consumesConvLayer: true,
            consumesRecurrentLayer: true,
            writeBufferIndices: Set<Int>([3, 8, 9, 10])
        )
    }

    public func prefillSteps(context: PrefillBindingContext) throws -> FragmentPrefillSteps {
        let scratchSlotSize = context.slotDimension * context.scratchElementSize * context.maximumSequenceLength

        let (convWeightBuffer, convWeightOffset) = context.resolveWeight("conv_weight")
        let (normWeightBuffer, normWeightOffset) = context.resolveWeight("scale")
        let (dtBiasBuffer, dtBiasOffset) = context.resolveWeight("dt_bias")
        let (aLogBuffer, aLogOffset) = context.resolveWeight("A_log")

        guard let recurrentState = context.buffers.recurrentState else {
            fatalError("[Compiler] SSMRecurrenceFragment requires recurrent state buffer")
        }
        guard let convState = context.buffers.convState else {
            fatalError("[Compiler] SSMRecurrenceFragment requires conv state buffer")
        }

        let recurrentLayerOffset = context.recurrentLayerIndex * context.buffers.recurrentStateBytesPerLayer
        let convLayerOffset = context.convLayerIndex
            * context.buffers.convStateKernelSize
            * context.buffers.convStateDimension
            * MemoryLayout<Float16>.size
        let kernelName = Self.sequenceKernelName(
            bufferPrecision: context.kernelContext.bufferPrecision,
            weightFormat: context.kernelContext.weightFormat
        )
        let pipeline = try context.getPipeline(kernelName)
        // Size threadgroup to cover Phase 1 (localDim channels) and Phase 2 (headsPerGroup × dv threads).
        // Each threadgroup owns one key-group and runs independently on its own GPU core.
        let safeGroupCount = max(groupCount, 1)
        let headsPerGroup = max(1, headCount / safeGroupCount)
        let localDim = 2 * keyHeadDimension + headsPerGroup * valueHeadDimension
        let phase2Threads = headsPerGroup * min(valueHeadDimension, 256)
        let desiredThreads = max(localDim, phase2Threads)
        let threads = min(
            min(Self.maxThreadgroupSize, desiredThreads),
            pipeline.maxTotalThreadsPerThreadgroup
        )

        let step = MetalPrefillStep(
            pipeline: pipeline,
            gridSize: MTLSize(width: safeGroupCount, height: 1, depth: 1),
            threadgroupSize: MTLSize(width: threads, height: 1, depth: 1),
            bufferBindings: [
                (0, context.buffers.scratch, 1 * scratchSlotSize),
                (1, context.buffers.scratch, 2 * scratchSlotSize),
                (2, context.buffers.scratch, 3 * scratchSlotSize),
                (3, context.buffers.scratch, 4 * scratchSlotSize),
                (4, convWeightBuffer, convWeightOffset),
                (5, normWeightBuffer, normWeightOffset),
                (6, dtBiasBuffer, dtBiasOffset),
                (7, aLogBuffer, aLogOffset),
                (8, recurrentState, recurrentLayerOffset),
                (9, convState, convLayerOffset),
                (10, context.buffers.scratch, 0),
            ],
            bytesBindings: [
                uint32Binding(11, UInt32(headCount)),
                uint32Binding(12, UInt32(groupCount)),
                uint32Binding(13, UInt32(keyHeadDimension)),
                uint32Binding(14, UInt32(valueHeadDimension)),
                uint32Binding(15, UInt32(convKernelSize)),
                uint32Binding(16, UInt32(context.maximumSequenceLength)),
            ],
            threadgroupMemoryLength: 0,
            sync: .bufferBarrier,
            mode: .batch,
            sequenceLengthPolicy: .bind(index: 16),
            positionBufferIndex: nil,
            perPositionStrides: [:]
        )

        return FragmentPrefillSteps(
            steps: [step],
            outputIsHidden: false,
            resetsProjectionIndex: true,
            consumesConvLayer: true,
            consumesRecurrentLayer: true
        )
    }
}

extension SSMRecurrenceFragment: ConvStateRequiring {
    public var convStateDimension: Int { convDimension }
}

extension SSMRecurrenceFragment: RecurrentStateRequiring {
    public var recurrentStateBytesPerLayer: Int {
        headCount * keyHeadDimension * valueHeadDimension * MemoryLayout<Float>.size
    }
}
