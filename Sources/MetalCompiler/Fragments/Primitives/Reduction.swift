import Metal

/// Reduction: all threads cooperate to reduce across a dimension.
/// Used by: RMSNorm, LayerNorm.
public struct Reduction: PrimitiveMetalKernelFragment {
    public let dimension: Int
    public let epsilon: Float
    public let weightRole: String

    public init(dimension: Int, epsilon: Float = 0, weightRole: String = "scale") {
        self.dimension = dimension
        self.epsilon = epsilon
        self.weightRole = weightRole
    }

    public var isFusable: Bool { true }
    public var normEpsilon: Float? { epsilon }
    public func kernelName(context: KernelContext) -> String {
        let bf16 = context.weightFormat == .bfloat16
        if context.bufferPrecision == .float32 {
            return bf16 ? "rms_norm_seq_bf16_f32_inplace" : "rms_norm_seq_f32_inplace"
        }
        return bf16 ? "rms_norm_bf16" : "rms_norm"
    }
    public var dispatchDimension: MetalDispatchDimension { .reduction(dimension: dimension) }
    public var weightSlots: [MetalWeightSlot] { [MetalWeightSlot(field: weightRole, role: .weight)] }

    public func kernelSource(name: String, bufferPrecision: BufferPrecision, weightFormat: WeightFormat) -> String {
        MetalSourceGenerator.generateReduction(name: name, dimension: 0, epsilon: 0,
            bufferPrecision: bufferPrecision, weightFormat: weightFormat, isSequence: bufferPrecision == .float32)
    }

    public func decodeBindings(context: BufferBindingContext) -> FragmentBindings {
        let (weightBuffer, weightOffset) = context.resolveWeight(weightRole)
        return FragmentBindings(
            buffers: [
                (0, context.bufferSet.hidden, 0),
                (1, weightBuffer, weightOffset),
            ],
            bytes: [
                uint32Binding(2, UInt32(dimension)),
                floatBinding(3, epsilon),
            ],
            outputIsHidden: true,
            resetsProjectionIndex: true
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
                    (0, context.buffers.hidden, 0),
                    (1, weightBuffer, weightOffset),
                    (2, context.buffers.hidden, 0),
                ],
                bytesBindings: [
                    uint32Binding(3, UInt32(dimension)),
                    floatBinding(4, epsilon),
                    uint32Binding(5, UInt32(context.maximumSequenceLength)),
                ],
                threadgroupMemoryLength: 0,
                sync: .bufferBarrier,
                mode: .batch,
                sequenceLengthPolicy: .bind(index: 5),
                positionBufferIndex: nil,
                perPositionStrides: [:]
            )],
            outputIsHidden: true,
            resetsProjectionIndex: true
        )
    }
}
