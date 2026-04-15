import Metal

/// Reduction: all threads cooperate to reduce across a dimension.
/// Used by: RMSNorm, LayerNorm.
public struct Reduction: PrimitiveMetalKernelFragment {
    public let dimension: Int
    public let epsilon: Float
    public let weightRole: String
    public let weightBias: Float

    public init(
        dimension: Int,
        epsilon: Float = 0,
        weightRole: String = "scale",
        weightBias: Float = 0
    ) {
        self.dimension = dimension
        self.epsilon = epsilon
        self.weightRole = weightRole
        self.weightBias = weightBias
    }

    public var isFusable: Bool { true }
    public var normEpsilon: Float? { epsilon }
    public var normWeightBias: Float? { weightBias }
    public var scalarConstantValues: [String: ScalarConstantValue] {
        ["epsilon": .float(epsilon), "weightBias": .float(weightBias)]
    }
    public func kernelName(context: KernelContext) -> String {
        let bf16 = context.weightFormat == .bfloat16
        if context.bufferPrecision == .float32 {
            return bf16 ? "rms_norm_seq_bf16_f32_inplace" : "rms_norm_seq_f32_inplace"
        }
        return bf16 ? "rms_norm_bf16" : "rms_norm"
    }
    public var dispatchDimension: MetalDispatchDimension { .reduction(dimension: dimension) }
    public var weightSlots: [MetalWeightSlot] { [MetalWeightSlot(field: weightRole, role: .weight)] }

    public func requiredFallbackBufferSize(for role: String, bytesPerScalar: Int) -> Int {
        dimension * bytesPerScalar
    }

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
            parallelism: .perRow(dimension: dimension),
            threadgroupMemoryBytes: 32 * MemoryLayout<Float>.size,
            requiresSIMDReduction: true
        )
    }

    public func kernelBody(bufferPrecision: BufferPrecision, weightFormat: WeightFormat) -> String? {
        let readWeight = { (expr: String) in weightFormat.readExpression(expr) }
        let bt = bufferPrecision.metalType

        return """
        float sumSquared = 0.0f;
        for (uint i = tid; i < dimension; i += threadgroupSize) {
            float v = float(data[i]);
            sumSquared += v * v;
        }
        sumSquared = simd_sum(sumSquared);

        uint simdIndex = tid / SIMD_WIDTH;
        if (tid % SIMD_WIDTH == 0) shared[simdIndex] = sumSquared;
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
            output[i] = \(bt)(float(data[i]) * _rms_scale * (\(readWeight("weight[i]")) + weightBias));
        }
        """
    }

    public func kernelSource(name: String, bufferPrecision: BufferPrecision, weightFormat: WeightFormat) -> String {
        guard let contract = fusionContract,
              let body = kernelBody(bufferPrecision: bufferPrecision, weightFormat: weightFormat) else {
            fatalError("Reduction must provide fusionContract and kernelBody")
        }
        return KernelScaffold.generate(
            name: name,
            body: body,
            contract: contract,
            bufferPrecision: bufferPrecision,
            weightFormats: [weightRole: weightFormat],
            isSequence: bufferPrecision == .float32
        )
    }

    public func decodeBindings(context: BufferBindingContext) -> FragmentBindings {
        let (weightBuffer, weightOffset) = context.resolveWeight(weightRole)
        return FragmentBindings(
            buffers: [
                (0, context.currentInputBuffer, context.currentInputOffset),
                (1, weightBuffer, weightOffset),
                (2, context.bufferSet.hidden, 0),
            ],
            bytes: [
                uint32Binding(3, UInt32(dimension)),
                floatBinding(4, epsilon),
                floatBinding(5, weightBias),
            ],
            outputIsHidden: true,
            resetsProjectionIndex: true,
            writeBufferIndices: Set<Int>([2])
        )
    }

    public func prefillSteps(context: PrefillBindingContext) throws -> FragmentPrefillSteps {
        let (weightBuffer, weightOffset) = context.resolveWeight(weightRole)
        let kernelName = kernelName(context: context.kernelContext)
        let pipeline = try context.getPipeline(kernelName)
        let simdWidth = pipeline.threadExecutionWidth
        let clamped = min(max(dimension, 1), 1024)
        let rounded = ((clamped + simdWidth - 1) / simdWidth) * simdWidth
        let threads = min(rounded, pipeline.maxTotalThreadsPerThreadgroup)
        return FragmentPrefillSteps(
            steps: [MetalPrefillStep(
                pipeline: pipeline,
                gridSize: MTLSize(width: context.maximumSequenceLength, height: 1, depth: 1),
                threadgroupSize: MTLSize(width: threads, height: 1, depth: 1),
                bufferBindings: [
                    (0, context.currentInputBuffer, context.currentInputOffset),
                    (1, weightBuffer, weightOffset),
                    (2, context.buffers.hidden, 0),
                ],
                bytesBindings: [
                    uint32Binding(3, UInt32(dimension)),
                    floatBinding(4, epsilon),
                    floatBinding(5, weightBias),
                    uint32Binding(6, UInt32(context.maximumSequenceLength)),
                ],
                threadgroupMemoryLength: 0,
                sync: .bufferBarrier,
                mode: .batch,
                sequenceLengthPolicy: .bind(index: 6),
                positionBufferIndex: nil,
                perPositionStrides: [:]
            )],
            outputIsHidden: true,
            resetsProjectionIndex: true
        )
    }
}
