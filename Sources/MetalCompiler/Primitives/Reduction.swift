import Metal

/// Reduction: all threads cooperate to reduce across a dimension.
/// Used by: RMSNorm, LayerNorm.
public struct Reduction: PrimitiveMetalKernelFragment {
    public let dimension: Int
    public let epsilon: Float
    public let weightRole: String
    public let weightBias: Float
    public let withScale: Bool

    public init(
        dimension: Int,
        epsilon: Float = 0,
        weightRole: String = "scale",
        weightBias: Float = 0,
        withScale: Bool = true
    ) {
        self.dimension = dimension
        self.epsilon = epsilon
        self.weightRole = weightRole
        self.weightBias = weightBias
        self.withScale = withScale
    }

    public var isFusable: Bool { true }
    public var normEpsilon: Float? { epsilon }
    public var normWeightBias: Float? { withScale ? weightBias : nil }
    public var scalarConstantValues: [String: ScalarConstantValue] {
        if withScale {
            return ["epsilon": .float(epsilon), "weightBias": .float(weightBias)]
        }
        return ["epsilon": .float(epsilon)]
    }
    public func kernelName(context: KernelContext) -> String {
        let bf16 = context.weightFormat.isBFloat16
        if context.bufferPrecision.isPrefillSequencePrecision {
            return bf16 ? "rms_norm_seq_bf16_f32_inplace" : "rms_norm_seq_f32_inplace"
        }
        return (bf16 ? "rms_norm_bf16" : "rms_norm") + context.bufferPrecision.decodeKernelNameSuffix
    }
    public var dispatchDimension: MetalDispatchDimension { .reduction(dimension: dimension) }
    public var weightSlots: [MetalWeightSlot] {
        withScale ? [MetalWeightSlot(field: weightRole, role: .weight)] : []
    }

    public func requiredFallbackBufferSize(for role: String, bytesPerScalar: Int) -> Int {
        withScale ? dimension * bytesPerScalar : 0
    }

    public var fusionContract: FusionContract? {
        var ports = [
            FusionPort(name: "data", direction: .input, role: .buffer, accessPattern: .multiPass),
        ]
        if withScale {
            ports.append(FusionPort(name: "weight", direction: .input, role: .weight(field: weightRole)))
        }
        ports.append(FusionPort(name: "output", direction: .output, role: .buffer))

        var constants = [ScalarConstant(name: "epsilon", metalType: "float")]
        if withScale {
            constants.append(ScalarConstant(name: "weightBias", metalType: "float"))
        }

        return FusionContract(
            ports: ports,
            scalarConstants: constants,
            parallelism: .perRow(dimension: dimension),
            threadgroupMemoryBytes: 32 * MemoryLayout<Float>.size,
            requiresSIMDReduction: true
        )
    }

    public func kernelBody(bufferPrecision: BufferPrecision, weightFormat: WeightFormat) -> String? {
        let bt = bufferPrecision.metalType
        let storeValue: (String) -> String = { expression in
            bufferPrecision.isPrefillSequencePrecision
                ? MetalSourceGenerator.sequenceStorageValue(expression, weightFormat: weightFormat)
                : expression
        }

        let outputExpression: String
        if withScale {
            let readWeight = weightFormat.readExpression("weight[i]")
            outputExpression = "\(bt)(\(storeValue("float(data[i]) * _rms_scale * (\(readWeight) + weightBias)")))"
        } else {
            outputExpression = "\(bt)(\(storeValue("float(data[i]) * _rms_scale")))"
        }

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
            output[i] = \(outputExpression);
        }
        """
    }

    public func kernelSource(name: String, bufferPrecision: BufferPrecision, weightFormat: WeightFormat) -> String {
        guard let contract = fusionContract,
              let body = kernelBody(bufferPrecision: bufferPrecision, weightFormat: weightFormat) else {
            fatalError("Reduction must provide fusionContract and kernelBody")
        }
        let formats: [String: WeightFormat] = withScale ? [weightRole: weightFormat] : [:]
        return KernelScaffold.generate(
            name: name,
            body: body,
            contract: contract,
            bufferPrecision: bufferPrecision,
            weightFormats: formats,
            isSequence: bufferPrecision.isPrefillSequencePrecision
        )
    }

    public func decodeBindings(context: BufferBindingContext) -> FragmentBindings {
        if withScale {
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
        } else {
            return FragmentBindings(
                buffers: [
                    (0, context.currentInputBuffer, context.currentInputOffset),
                    (1, context.bufferSet.hidden, 0),
                ],
                bytes: [
                    uint32Binding(2, UInt32(dimension)),
                    floatBinding(3, epsilon),
                ],
                outputIsHidden: true,
                resetsProjectionIndex: true,
                writeBufferIndices: Set<Int>([1])
            )
        }
    }

    public func prefillSteps(context: PrefillBindingContext) throws -> FragmentPrefillSteps {
        let kernelName = kernelName(context: context.kernelContext)
        let pipeline = try context.getPipeline(kernelName)
        let simdWidth = pipeline.threadExecutionWidth
        let clamped = min(max(dimension, 1), 1024)
        let rounded = ((clamped + simdWidth - 1) / simdWidth) * simdWidth
        let threads = min(rounded, pipeline.maxTotalThreadsPerThreadgroup)

        let bufferBindings: [(Int, MTLBuffer, Int)]
        let bytesBindings: [(index: Int, value: [UInt8])]
        let bufferAccessPattern: MetalDispatchStepMetadata.BufferAccessPattern

        if withScale {
            let (weightBuffer, weightOffset) = context.resolveWeight(weightRole)
            bufferBindings = [
                (0, context.currentInputBuffer, context.currentInputOffset),
                (1, weightBuffer, weightOffset),
                (2, context.buffers.hidden, 0),
            ]
            bytesBindings = [
                uint32Binding(3, UInt32(dimension)),
                floatBinding(4, epsilon),
                floatBinding(5, weightBias),
                uint32Binding(6, UInt32(context.maximumSequenceLength)),
            ]
            bufferAccessPattern = .init(reads: [0, 1], writes: [2])
        } else {
            bufferBindings = [
                (0, context.currentInputBuffer, context.currentInputOffset),
                (1, context.buffers.hidden, 0),
            ]
            bytesBindings = [
                uint32Binding(2, UInt32(dimension)),
                floatBinding(3, epsilon),
                uint32Binding(4, UInt32(context.maximumSequenceLength)),
            ]
            bufferAccessPattern = .init(reads: [0], writes: [1])
        }

        return FragmentPrefillSteps(
            steps: [MetalPrefillStep(
                pipeline: pipeline,
                gridSize: MTLSize(width: context.maximumSequenceLength, height: 1, depth: 1),
                threadgroupSize: MTLSize(width: threads, height: 1, depth: 1),
                bufferBindings: bufferBindings,
                bytesBindings: bytesBindings,
                threadgroupMemoryLength: 0,
                sync: .bufferBarrier,
                mode: .batch,
                sequenceLengthPolicy: .bind(index: withScale ? 6 : 4),
                positionBufferIndex: nil,
                perPositionStrides: [:],
                metadata: .init(
                    kernelName: kernelName,
                    bufferAccessPattern: bufferAccessPattern
                )
            )],
            outputIsHidden: true,
            resetsProjectionIndex: true
        )
    }
}
