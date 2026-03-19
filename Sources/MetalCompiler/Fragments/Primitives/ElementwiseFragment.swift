import Metal

/// Elementwise: one thread per element, trivially parallel.
/// Used by: SwiGLU, SigmoidGate.
public struct ElementwiseFragment: PrimitiveMetalKernelFragment {
    public let count: Int
    public let kind: ElementwiseKind

    public enum ElementwiseKind: Sendable {
        case swiglu
        case sigmoidGate
    }

    public init(count: Int, kind: ElementwiseKind = .swiglu) {
        self.count = count
        self.kind = kind
    }

    public var isFusable: Bool { true }
    public func kernelName(context: KernelContext) -> String {
        switch kind {
        case .swiglu:
            return context.bufferPrecision == .float32 ? "swiglu_seq_f32" : "swiglu"
        case .sigmoidGate:
            return "sigmoid_gate"
        }
    }
    public var dispatchDimension: MetalDispatchDimension { .elementwise(count: count) }

    public func kernelSource(name: String, bufferPrecision: BufferPrecision, weightFormat: WeightFormat) -> String {
        MetalSourceGenerator.generateSwiGLU(name: name, bufferPrecision: bufferPrecision, isSequence: bufferPrecision == .float32)
    }

    public func decodeBindings(context: BufferBindingContext) -> FragmentBindings {
        let slotBytes = context.slotDimension * context.elementSize
        return FragmentBindings(
            buffers: [
                (0, context.bufferSet.scratch, 1 * slotBytes),
                (1, context.bufferSet.scratch, 2 * slotBytes),
                (2, context.bufferSet.scratch, 0),
            ],
            bytes: [
                uint32Binding(3, UInt32(count)),
            ],
            outputIsHidden: false,
            resetsProjectionIndex: true
        )
    }

    public func prefillSteps(context: PrefillBindingContext) throws -> FragmentPrefillSteps {
        let kernelName = kernelName(context: context.kernelContext)
        let pipeline = try context.getPipeline(kernelName)
        let scratchSlotSize = context.slotDimension * context.scratchElementSize * context.maximumSequenceLength
        let tgSize = min(256, pipeline.maxTotalThreadsPerThreadgroup)
        let gridX = (count + tgSize - 1) / tgSize
        return FragmentPrefillSteps(
            steps: [MetalPrefillStep(
                pipeline: pipeline,
                gridSize: MTLSize(width: gridX, height: context.maximumSequenceLength, depth: 1),
                threadgroupSize: MTLSize(width: tgSize, height: 1, depth: 1),
                bufferBindings: [
                    (0, context.buffers.scratch, 1 * scratchSlotSize),
                    (1, context.buffers.scratch, 2 * scratchSlotSize),
                    (2, context.buffers.scratch, 0),
                ],
                bytesBindings: [
                    uint32Binding(3, UInt32(count)),
                    uint32Binding(4, UInt32(context.maximumSequenceLength)),
                ],
                threadgroupMemoryLength: 0,
                sync: .bufferBarrier,
                mode: .batch,
                sequenceLengthPolicy: .bindAndAdjustGridHeight(index: 4),
                positionBufferIndex: nil,
                perPositionStrides: [:]
            )],
            outputIsHidden: false,
            resetsProjectionIndex: true
        )
    }
}
