import Metal

/// Elementwise: one thread per element, trivially parallel.
/// Used by: SwiGLU, GEGLU.
public struct ElementwiseFragment: PrimitiveMetalKernelFragment {
    public let count: Int
    public let kind: ElementwiseKind

    public enum ElementwiseKind: Sendable, Equatable {
        /// gate * sigmoid(gate) * up — SiLU-gated (Llama, LFM2)
        case swiglu
        /// gelu_tanh(gate) * up — GELU-gated (Gemma4)
        case geluGated
    }

    public init(count: Int, kind: ElementwiseKind = .swiglu) {
        self.count = count
        self.kind = kind
    }

    public var isFusable: Bool { true }

    /// Fusion contract with two buffer inputs.
    ///
    /// Note: SynthesizedFragment.decodeBindings binds all `.dataFlow` input
    /// ports to `currentInputBuffer`. With two buffer inputs, the non-primary
    /// input would be bound incorrectly if this fragment were fused as a
    /// consumer. In practice, neighbors are LinearFragment (non-fusable),
    /// so this fragment is never fused.
    public var fusionContract: FusionContract? {
        FusionContract(
            ports: [
                FusionPort(name: "gate", direction: .input, role: .buffer, accessPattern: .singlePass),
                FusionPort(name: "up", direction: .input, role: .buffer, accessPattern: .singlePass),
                FusionPort(name: "output", direction: .output, role: .buffer),
            ],
            parallelism: .perElement(count: count)
        )
    }

    public func kernelBody(bufferPrecision: BufferPrecision, weightFormat: WeightFormat) -> String? {
        let bt = bufferPrecision.metalType
        let storeValue: (String) -> String = { expression in
            bufferPrecision.isPrefillSequencePrecision
                ? MetalSourceGenerator.sequenceStorageValue(expression, weightFormat: weightFormat)
                : expression
        }
        switch kind {
        case .swiglu:
            return """
            float g = float(gate[idx]);
            output[idx] = \(bt)(\(storeValue("g * (1.0f / (1.0f + exp(-g))) * float(up[idx])")));
            """
        case .geluGated:
            return """
            float g = float(gate[idx]);
            output[idx] = \(bt)(\(storeValue("0.5f * g * (1.0f + precise::tanh(0.7978845608f * (g + 0.044715f * g * g * g))) * float(up[idx])")));
            """
        }
    }

    public func kernelName(context: KernelContext) -> String {
        switch kind {
        case .swiglu:
            return context.bufferPrecision.isPrefillSequencePrecision
                ? "swiglu_seq_f32"
                : "swiglu\(context.bufferPrecision.decodeKernelNameSuffix)"
        case .geluGated:
            return context.bufferPrecision.isPrefillSequencePrecision
                ? "geglu_seq_f32"
                : "geglu\(context.bufferPrecision.decodeKernelNameSuffix)"
        }
    }
    public var dispatchDimension: MetalDispatchDimension { .elementwise(count: count) }

    public func kernelSource(name: String, bufferPrecision: BufferPrecision, weightFormat: WeightFormat) -> String {
        MetalSourceGenerator.generateGatedActivation(
            name: name, bufferPrecision: bufferPrecision,
            activation: gatedActivation, isSequence: bufferPrecision.isPrefillSequencePrecision)
    }

    /// The activation function used by this elementwise operation.
    public var gatedActivation: MetalSourceGenerator.GatedActivation {
        switch kind {
        case .swiglu: return .silu
        case .geluGated: return .geluTanh
        }
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
            resetsProjectionIndex: true,
            writeBufferIndices: Set<Int>([2])
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
                    uint32Binding(5, UInt32(context.slotDimension)),
                ],
                threadgroupMemoryLength: 0,
                sync: .bufferBarrier,
                mode: .batch,
                sequenceLengthPolicy: .bindAndAdjustGridHeight(index: 4),
                positionBufferIndex: nil,
                perPositionStrides: [:],
                metadata: .init(
                    kernelName: kernelName,
                    bufferAccessPattern: .init(reads: [0, 1], writes: [2])
                )
            )],
            outputIsHidden: false,
            resetsProjectionIndex: true
        )
    }
}
