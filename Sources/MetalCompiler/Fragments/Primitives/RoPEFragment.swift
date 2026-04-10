import Metal
import LMIR

/// Rotary position embedding (in-place on Q and K).
public struct RoPEFragment: PrimitiveMetalKernelFragment {
    public let headCount: Int
    public let kvHeadCount: Int
    public let headDimension: Int
    public let ropeDimension: Int
    public let base: Float
    public let scaling: RoPEScaling?
    public let mropeAxes: MRoPEAxes?

    public init(headCount: Int, kvHeadCount: Int, headDimension: Int,
                ropeDimension: Int, base: Float, scaling: RoPEScaling? = nil, mropeAxes: MRoPEAxes? = nil) {
        self.headCount = headCount
        self.kvHeadCount = kvHeadCount
        self.headDimension = headDimension
        self.ropeDimension = ropeDimension
        self.base = base
        self.scaling = scaling
        self.mropeAxes = mropeAxes
    }

    public var isFusable: Bool { false }
    public var isInPlace: Bool { true }
    public func kernelName(context: KernelContext) -> String {
        context.bufferPrecision == .float32 ? "rope_seq_f32" : "rope"
    }
    public var dispatchDimension: MetalDispatchDimension {
        .perHead(headCount: max(headCount, kvHeadCount))
    }

    public func kernelSource(name: String, bufferPrecision: BufferPrecision, weightFormat: WeightFormat) -> String {
        bufferPrecision == .float32
            ? MetalSourceGenerator.generateRoPESeq(name: name, bufferPrecision: bufferPrecision)
            : MetalSourceGenerator.generateRoPE(name: name, bufferPrecision: bufferPrecision)
    }

    public func decodeBindings(context: BufferBindingContext) -> FragmentBindings {
        let slotBytes = context.slotDimension * context.elementSize
        return FragmentBindings(
            buffers: [
                (0, context.bufferSet.scratch, 1 * slotBytes),
                (1, context.bufferSet.scratch, 2 * slotBytes),
                (2, context.bufferSet.ropePositionAxes, 0),
            ],
            bytes: [
                uint32Binding(3, UInt32(headCount)),
                uint32Binding(4, UInt32(kvHeadCount)),
                uint32Binding(5, UInt32(headDimension)),
                uint32Binding(6, UInt32(ropeDimension)),
                floatBinding(7, base),
                uint32Binding(8, UInt32(sectionCount(at: 0))),
                uint32Binding(9, UInt32(sectionCount(at: 1))),
                uint32Binding(10, UInt32(sectionCount(at: 2))),
                uint32Binding(11, UInt32(mropeAxes?.interleaved == true ? 1 : 0)),
                uint32Binding(12, UInt32(usesProportionalRoPE ? 1 : 0)),
            ],
            outputIsHidden: false,
            writeBufferIndices: Set<Int>([0, 1])
        )
    }

    public func prefillSteps(context: PrefillBindingContext) throws -> FragmentPrefillSteps {
        let kernelName = kernelName(context: context.kernelContext)
        let pipeline = try context.getPipeline(kernelName)
        let threads = min(32, pipeline.maxTotalThreadsPerThreadgroup)
        let totalHeads = headCount + kvHeadCount
        let scratchSlotSize = context.slotDimension * context.scratchElementSize * context.maximumSequenceLength
        return FragmentPrefillSteps(
            steps: [MetalPrefillStep(
                pipeline: pipeline,
                gridSize: MTLSize(width: totalHeads, height: context.maximumSequenceLength, depth: 1),
                threadgroupSize: MTLSize(width: threads, height: 1, depth: 1),
                bufferBindings: [
                    (0, context.buffers.scratch, 1 * scratchSlotSize),
                    (1, context.buffers.scratch, 2 * scratchSlotSize),
                    (2, context.buffers.ropePositionAxes, 0),
                ],
                bytesBindings: [
                    uint32Binding(3, UInt32(headCount)),
                    uint32Binding(4, UInt32(kvHeadCount)),
                    uint32Binding(5, UInt32(headDimension)),
                    uint32Binding(6, UInt32(ropeDimension)),
                    floatBinding(7, base),
                    uint32Binding(8, UInt32(sectionCount(at: 0))),
                    uint32Binding(9, UInt32(sectionCount(at: 1))),
                    uint32Binding(10, UInt32(sectionCount(at: 2))),
                    uint32Binding(11, UInt32(mropeAxes?.interleaved == true ? 1 : 0)),
                    uint32Binding(12, UInt32(context.maximumSequenceLength)),
                    uint32Binding(13, UInt32(usesProportionalRoPE ? 1 : 0)),
                ],
                threadgroupMemoryLength: 0,
                sync: .bufferBarrier,
                mode: .batch,
                sequenceLengthPolicy: .bindAndAdjustGridHeight(index: 12),
                positionBufferIndex: nil,
                perPositionStrides: [:]
            )],
            outputIsHidden: false
        )
    }

    private func sectionCount(at index: Int) -> Int {
        guard let mropeAxes, index < mropeAxes.sections.count else {
            return 0
        }
        return mropeAxes.sections[index]
    }

    private var usesProportionalRoPE: Bool {
        scaling?.kind == .custom("proportional")
    }
}
