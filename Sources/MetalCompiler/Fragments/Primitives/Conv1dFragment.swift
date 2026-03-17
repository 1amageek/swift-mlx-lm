import Metal

/// Depthwise temporal convolution with double gating (decode: state update).
public struct Conv1dFragment: PrimitiveMetalKernelFragment {
    public let dimension: Int
    public let kernelSize: Int

    public init(dimension: Int, kernelSize: Int) {
        self.dimension = dimension
        self.kernelSize = kernelSize
    }

    public var isFusable: Bool { false }
    public func kernelName(context: KernelContext) -> String {
        if context.bufferPrecision == .float32 { return "conv1d_causal_seq_f32" }
        return context.weightFormat == .bfloat16 ? "conv_state_update_bf16" : "conv_state_update"
    }
    public var dispatchDimension: MetalDispatchDimension { .elementwise(count: dimension) }
    public var weightSlots: [MetalWeightSlot] { [MetalWeightSlot(field: "conv_weight", role: .weight)] }
    public var cacheSlots: [MetalCacheSlot] { [MetalCacheSlot(name: "conv_cache", kind: .conv, temporalSize: kernelSize)] }

    public func decodeBindings(context: BufferBindingContext) -> FragmentBindings {
        let (weightBuffer, weightOffset) = context.resolveWeight("conv_weight")
        let slotBytes = context.slotDimension * context.elementSize

        guard let convState = context.bufferSet.convState else {
            fatalError("[Compiler] Conv1dFragment requires conv_state buffer")
        }
        let convLayerOffset = context.convLayerIndex
            * context.bufferSet.convStateKernelSize * context.bufferSet.convStateDimension * context.elementSize

        return FragmentBindings(
            buffers: [
                (0, convState, convLayerOffset),
                (1, context.bufferSet.scratch, 1 * slotBytes),
                (2, weightBuffer, weightOffset),
                (3, context.bufferSet.scratch, 0),
            ],
            bytes: [
                uint32Binding(4, UInt32(dimension)),
                uint32Binding(5, UInt32(kernelSize)),
            ],
            outputIsHidden: false,
            resetsProjectionIndex: true,
            consumesConvLayer: true
        )
    }

    public func prefillSteps(context: PrefillBindingContext) throws -> FragmentPrefillSteps {
        let (weightBuffer, weightOffset) = context.resolveWeight("conv_weight")
        let scratchSlotBytes = context.slotDimension * context.scratchElementSize * context.maximumSequenceLength
        // in_proj GEMM output dimension = 3 * convDim (for ShortConv: in_proj, gate, up)
        let inProjDim = dimension * 3

        // Step 1: Causal temporal conv1d across all positions (F32 I/O)
        let kernelName = kernelName(context: context.kernelContext)
        let causalPipeline = try context.getPipeline(kernelName)
        let causalTgSize = min(256, causalPipeline.maxTotalThreadsPerThreadgroup)
        let causalGridX = (dimension + causalTgSize - 1) / causalTgSize
        var steps: [MetalPrefillStep] = []
        steps.append(MetalPrefillStep(
            pipeline: causalPipeline,
            gridSize: MTLSize(width: causalGridX, height: context.maximumSequenceLength, depth: 1),
            threadgroupSize: MTLSize(width: causalTgSize, height: 1, depth: 1),
            bufferBindings: [
                (0, context.buffers.scratch, 1 * scratchSlotBytes),  // in_proj output [seqLen x inProjDim]
                (1, weightBuffer, weightOffset),                     // conv weight [convDim x kernelSize]
                (2, context.buffers.scratch, 0),                     // output [seqLen x convDim]
            ],
            bytesBindings: [
                uint32Binding(3, UInt32(dimension)),
                uint32Binding(4, UInt32(inProjDim)),
                uint32Binding(5, UInt32(kernelSize)),
                uint32Binding(6, UInt32(context.maximumSequenceLength)),
            ],
            threadgroupMemoryLength: 0,
            sync: .bufferBarrier,
            mode: .batch,
            sequenceLengthBindingIndex: 6,
            positionBufferIndex: nil,
            perPositionStrides: [:]
        ))

        // Step 2: Extract conv_state from in_proj output (last kernelSize positions)
        // conv_state is always F16 — use F16 element size for offset calculation
        let f16ElementSize = MemoryLayout<Float16>.size
        if let convState = context.buffers.convState {
            let extractPipeline = try context.getPipeline("extract_conv_state_f32")
            let extractTgSize = min(256, extractPipeline.maxTotalThreadsPerThreadgroup)
            let extractGridX = (dimension + extractTgSize - 1) / extractTgSize
            let convLayerOffset = context.convLayerIndex
                * context.buffers.convStateKernelSize * context.buffers.convStateDimension * f16ElementSize
            steps.append(MetalPrefillStep(
                pipeline: extractPipeline,
                gridSize: MTLSize(width: extractGridX, height: kernelSize, depth: 1),
                threadgroupSize: MTLSize(width: extractTgSize, height: 1, depth: 1),
                bufferBindings: [
                    (0, context.buffers.scratch, 1 * scratchSlotBytes),  // in_proj output
                    (1, convState, convLayerOffset),                     // conv_state for this layer
                ],
                bytesBindings: [
                    uint32Binding(2, UInt32(dimension)),
                    uint32Binding(3, UInt32(inProjDim)),
                    uint32Binding(4, UInt32(kernelSize)),
                    uint32Binding(5, UInt32(context.maximumSequenceLength)),
                ],
                threadgroupMemoryLength: 0,
                sync: .bufferBarrier,
                mode: .batch,
                sequenceLengthBindingIndex: 5,
                positionBufferIndex: nil,
                perPositionStrides: [:]
            ))
        }

        return FragmentPrefillSteps(
            steps: steps,
            outputIsHidden: false,
            resetsProjectionIndex: true,
            consumesConvLayer: context.buffers.convState != nil
        )
    }
}
