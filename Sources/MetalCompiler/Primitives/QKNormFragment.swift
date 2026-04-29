import Metal

/// Per-head RMS normalization for Q or K projections.
public struct QKNormFragment: PrimitiveMetalKernelFragment {
    public let headCount: Int
    public let headDimension: Int
    public let epsilon: Float
    public let weightRole: String  // "q_layernorm" or "k_layernorm"
    public let weightBias: Float
    public let scratchSlotIndex: Int

    public init(
        headCount: Int,
        headDimension: Int,
        epsilon: Float,
        weightRole: String,
        weightBias: Float = 0,
        scratchSlotIndex: Int? = nil
    ) {
        self.headCount = headCount
        self.headDimension = headDimension
        self.epsilon = epsilon
        self.weightRole = weightRole
        self.weightBias = weightBias
        self.scratchSlotIndex = scratchSlotIndex ?? (weightRole == "q_layernorm" ? 1 : 2)
    }

    public var isFusable: Bool { true }
    public var isInPlace: Bool { true }
    public var perHeadDimension: Int? { headDimension }

    public func requiredFallbackBufferSize(for role: String, bytesPerScalar: Int) -> Int {
        headCount * headDimension * bytesPerScalar
    }
    public var normEpsilon: Float? { epsilon }
    public var supportsInPlaceBatching: Bool { true }

    // MARK: - Fusion Contract

    public var fusionContract: FusionContract? {
        FusionContract(
            ports: [
                FusionPort(name: "data", direction: .input, role: .buffer, accessPattern: .multiPass),
                FusionPort(name: "weight", direction: .input, role: .weight(field: weightRole)),
                FusionPort(name: "output", direction: .output, role: .buffer),
            ],
            scalarConstants: [
                ScalarConstant(name: "epsilon", metalType: "float"),
                ScalarConstant(name: "weightBias", metalType: "float"),
            ],
            parallelism: .perHead(headCount: headCount, headDimension: headDimension),
            threadgroupMemoryBytes: 32 * MemoryLayout<Float>.size,
            requiresSIMDReduction: true
        )
    }

    public var scalarConstantValues: [String: ScalarConstantValue] {
        ["epsilon": .float(epsilon), "weightBias": .float(weightBias)]
    }

    public func kernelBody(bufferPrecision: BufferPrecision, weightFormat: WeightFormat) -> String? {
        let readWeight = weightFormat.readExpression("weight[i]")
        let bt = bufferPrecision.metalType
        let value = "float(data[i]) * _rms_scale * (\(readWeight) + weightBias)"
        let stored = bufferPrecision.isPrefillSequencePrecision
            ? MetalSourceGenerator.sequenceStorageValue(value, weightFormat: weightFormat)
            : value
        return """
        float sumSq = 0.0f;
        for (uint i = tid; i < dimension; i += threadgroupSize) {
            float v = float(data[i]);
            sumSq += v * v;
        }
        sumSq = simd_sum(sumSq);

        uint simdIndex = tid / SIMD_WIDTH;
        if (tid % SIMD_WIDTH == 0) shared[simdIndex] = sumSq;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid == 0) {
            float total = 0.0f;
            uint sgCount = (threadgroupSize + SIMD_WIDTH - 1) / SIMD_WIDTH;
            for (uint s = 0; s < sgCount; s++) total += shared[s];
            shared[0] = rsqrt(total / float(dimension) + epsilon);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float _rms_scale = shared[0];
        for (uint i = tid; i < dimension; i += threadgroupSize) {
            output[i] = \(bt)(\(stored));
        }
        """
    }

    public func kernelName(context: KernelContext) -> String {
        if context.bufferPrecision.isPrefillSequencePrecision { return "qk_rms_norm_seq_f32" }
        return (context.weightFormat.isBFloat16 ? "qk_rms_norm_bf16" : "qk_rms_norm")
            + context.bufferPrecision.decodeKernelNameSuffix
    }
    public var dispatchDimension: MetalDispatchDimension { .perHead(headCount: headCount) }
    public var weightSlots: [MetalWeightSlot] { [MetalWeightSlot(field: weightRole, role: .weight)] }

    public func kernelSource(name: String, bufferPrecision: BufferPrecision, weightFormat: WeightFormat) -> String {
        bufferPrecision.isPrefillSequencePrecision
            ? MetalSourceGenerator.generateQKNormSeq(
                name: name,
                bufferPrecision: bufferPrecision,
                weightFormat: weightFormat,
                weightBias: weightBias
            )
            : MetalSourceGenerator.generateQKNorm(
                name: name,
                bufferPrecision: bufferPrecision,
                weightFormat: weightFormat,
                weightBias: weightBias
            )
    }

    public func decodeBindings(context: BufferBindingContext) -> FragmentBindings {
        let slotBytes = context.slotDimension * context.elementSize
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
                floatBinding(5, weightBias),
            ],
            outputIsHidden: false,
            writeBufferIndices: Set<Int>([0])
        )
    }

    public func prefillSteps(context: PrefillBindingContext) throws -> FragmentPrefillSteps {
        let kernelName = kernelName(context: context.kernelContext)
        let pipeline = try context.getPipeline(kernelName)
        let scratchSlotSize = context.slotDimension * context.scratchElementSize * context.maximumSequenceLength
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
                    floatBinding(5, weightBias),
                    uint32Binding(6, UInt32(context.maximumSequenceLength)),
                    uint32Binding(7, UInt32(totalDimension)),
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
